# Copyright (C) 2023 Raphael Javaux
# raphaeljavaux@gmail.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import datetime
import enum
import math
import os.path

from typing import Callable, List, Optional, Tuple

import numpy as np

from libhum.types import Signal, Match


MIN_MATCH_DURATION = datetime.timedelta(minutes=1)
MIN_MATCH_FRACTION = 0.05 # At least 5% of the samples should be present in the two signals.
MIN_MATCH_CORR_COEFF = 0.5
MIN_MATCH_RMSE = 10

# Runs a first a first approximation match at a lower signal frequency.
APPROX_TARGET_FREQUENCY = 0.1
APPROX_MIN_MATCH_DURATION = MIN_MATCH_DURATION * 0.75
APPROX_MIN_MATCH_CORR_COEFF = MIN_MATCH_CORR_COEFF + 0.1
APPROX_MIN_MATCH_RMSE = MIN_MATCH_RMSE - 0.1
APPROX_SEARCH_WINDOW = datetime.timedelta(seconds=1.0 / APPROX_TARGET_FREQUENCY * 2.0)

# Will ignore matches that have a better match within `MIN_MATCH_LOCAL_MAXIMUM`.
MIN_MATCH_LOCAL_MAXIMUM = datetime.timedelta(minutes=10)

MAX_BUFFER_SIZE = 64 * 1024 * 1024 # 64 MB

THREADS_PER_BLOCK = 256 # CUDA threads per bloc or OpenCL workers per group

KERNEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "match.kernel")


class MatchBackend(enum.Enum):
    CUDA = "cuda"
    NUMPY = "numpy"
    OPENCL = "opencl"


class MatchMethod(enum.Enum):
    CORR_COEFF = "corr_coeff"
    RMSE = "rmse"

    @property
    def as_int(self):
        return {
            MatchMethod.CORR_COEFF: 1,
            MatchMethod.RMSE: 2
        }[self]

    def backend_instance(self):
        return

    def min_match_result(self, approx: bool = False):
        return {
            MatchMethod.CORR_COEFF: APPROX_MIN_MATCH_CORR_COEFF if approx else MIN_MATCH_CORR_COEFF,
            MatchMethod.RMSE: APPROX_MIN_MATCH_RMSE if approx else MIN_MATCH_RMSE,
        }[self]


def match_signals(
    ref: Signal,
    target: Signal,
    max_matches: Optional[int],
    step: datetime.timedelta = datetime.timedelta(seconds=1),
    backend: MatchBackend = MatchBackend.NUMPY,
    method: MatchMethod = MatchMethod.RMSE,
) -> List[Match]:
    if ref.signal_frequency != target.signal_frequency:
        raise ValueError("signal frequencies should be identical.")

    if ref.network_frequency != target.network_frequency:
        raise ValueError("network frequencies should be identical.")

    frequency = ref.signal_frequency

    backend_instance = {
        MatchBackend.CUDA: _cuda_compute_results,
        MatchBackend.NUMPY: _numpy_compute_results,
        MatchBackend.OPENCL: _opencl_compute_results,
    }[backend]

    # Approximates the matching algorithm on downsampled signals.

    approx_q = math.floor(frequency / APPROX_TARGET_FREQUENCY)
    approx_frequency = frequency / approx_q
    assert approx_frequency >= APPROX_TARGET_FREQUENCY

    ref_approx = _decimated_masked_array(ref.signal, approx_q)
    target_approx = _decimated_masked_array(target.signal, approx_q)

    min_approx_match_len = math.ceil(APPROX_MIN_MATCH_DURATION.total_seconds() * approx_frequency)

    offset_begin = - len(target_approx) + min_approx_match_len
    offset_end = len(ref_approx) - min_approx_match_len + 1
    approx_step_offset = math.ceil(step.total_seconds() * approx_frequency)

    approx_offsets = np.arange(offset_begin, offset_end, approx_step_offset, dtype=np.int32)

    approx_offsets, _approx_results, _approx_match_lens = _compute_filtered_results(
        backend_instance, method, approx_offsets, ref_approx, target_approx,
        min_approx_match_len, method.min_match_result(approx=True),
    )

    # Builds the lookup offsets from the approximated matches

    if len(approx_offsets) < 1:
        return []

    assert np.all(approx_offsets[:-1] <= approx_offsets[1:]), "offsets must be sorted."

    min_match_len = math.ceil(MIN_MATCH_DURATION.total_seconds() * frequency)
    search_window_size = math.ceil(APPROX_SEARCH_WINDOW.total_seconds() * frequency)

    min_offset = - len(target.signal) + min_match_len
    max_offset = len(ref.signal) - min_match_len

    offset_windows = []
    for approx_offset in approx_offsets:
        offset = approx_q * approx_offset

        window_begin = max(min_offset, offset - search_window_size)
        window_end = min(max_offset, offset + search_window_size + 1)

        if len(offset_windows) > 0 and window_begin <= offset_windows[-1][1]:
            offset_windows[-1][1] = window_end
        else:
            offset_windows.append([window_begin, window_end])

    step_offset = math.ceil(step.total_seconds() * frequency)

    offsets = np.concatenate([
        np.arange(window_begin, window_end, step_offset)
        for window_begin, window_end in offset_windows
    ])

    # Search the approximated windows in the actual signal.

    offsets, results, match_lens = _compute_filtered_results(
        backend_instance, method, offsets, ref.signal, target.signal,
        min_match_len, method.min_match_result(approx=False),
    )

    # Sorts the resulting coefficients
    return _build_matches(frequency, method, offsets, results, match_lens, max_matches)


def _decimated_masked_array(array: np.ma.masked_array, q: int) -> np.ma.masked_array:
    # This is a very naïve downsampling method, but it is fast and works on masked arrays.
    idxs = np.arange(0, len(array), q)
    return array[idxs]


def _compute_filtered_results(
    backend_instance: Callable, method: MatchMethod,
    offsets: np.ndarray, a: np.ma.masked_array, b: np.ma.masked_array,
    min_match_len: int, min_match_result: float,
):
    """
    Computes the results for all the requested offsets using the requested backend instance.
    """

    offsets_chunks = []
    result_chunks = []
    match_lens_chunks = []

    # Processes the offset domain by chunks so that buffers do not exceed MAX_BUFFER_SIZE.
    chunk_size = MAX_BUFFER_SIZE // np.int32(0).nbytes
    for chunk_begin in range(0, len(offsets), chunk_size):
        chunk_end = min(len(offsets), chunk_begin + chunk_size)

        chunk_offsets = offsets[chunk_begin:chunk_end]

        # Computes the coefficients using the selected backend
        results, match_lens = backend_instance(method, chunk_offsets, a, b)
        assert len(results) == len(match_lens)

        # Reduces the memory usage by immediatly removing bad coefficients
        chunk_offsets, results, match_lens = _filter_results(
            chunk_offsets, results, match_lens,
            min_match_len, min_match_result,
        )

        offsets_chunks.append(chunk_offsets)
        result_chunks.append(results)
        match_lens_chunks.append(match_lens)


    # Combines the chunked results and match_lens
    offsets = np.concatenate(offsets_chunks)
    results = np.concatenate(result_chunks)
    match_lens = np.concatenate(match_lens_chunks)

    return offsets, results, match_lens


def _numpy_compute_results(
    method: MatchMethod, offsets: np.ndarray, a: np.ma.masked_array, b: np.ma.masked_array
) -> Tuple[np.ndarray, np.ndarray]:
    results = np.empty(len(offsets), dtype=np.float32)
    match_lens = np.empty(len(offsets), dtype=np.int32)

    if method == MatchMethod.CORR_COEFF:
        method_function = _numpy_corr_coeff
    else:
        method_function = _numpy_rmse

    for i, offset in enumerate(offsets):
        results[i], match_lens[i] = method_function(
            a[max(0, offset):offset + len(b)], b[max(0, -offset):len(a) - offset]
        )

    return results, match_lens


def _numpy_corr_coeff(a: np.ma.masked_array, b: np.ma.masked_array) -> Tuple[float, int]:
    """
    Computes the Pearson's correlation coefficient of two masked arrays.

    Ignore the masked samples and returns the total number of non-masked samples in the two signals.
    """

    assert len(a) == len(b)

    common_mask = a.mask | b.mask
    match_len = len(a) - np.sum(common_mask)

    if match_len < 1:
        return np.nan, match_len

    masked_a = np.ma.masked_where(common_mask, a)
    masked_b = np.ma.masked_where(common_mask, b)

    mean_a, mean_b = np.mean(masked_a), np.mean(masked_b)
    std_a, std_b = np.std(masked_a), np.std(masked_b)

    numerator = np.sum((masked_a - mean_a) * (masked_b - mean_b))
    denominator = std_a * std_b * match_len

    if denominator == 0:
        return np.nan, match_len

    return numerator / denominator, match_len


def _numpy_rmse(a: np.ma.masked_array, b: np.ma.masked_array) -> Tuple[float, int]:
    """
    Computes the Root-Mean-Square error of two masked arrays.

    Ignore the masked samples and returns the total number of non-masked samples in the two signals.
    """

    assert len(a) == len(b)

    common_mask = a.mask | b.mask
    match_len = len(a) - np.sum(common_mask)

    if match_len / len(a) < MIN_MATCH_FRACTION:
        return np.nan, match_len

    masked_a = np.ma.masked_where(common_mask, a)
    masked_b = np.ma.masked_where(common_mask, b)

    rmse = np.sqrt(np.mean((masked_a - masked_b)**2))

    return rmse, match_len


_opencl_ctx = None
_opencl_queue = None
_opencl_program = None
_opencl_buffer_dtype = None


def _opencl_compute_results(
    method: MatchMethod, offsets: np.ndarray, a: np.ma.masked_array, b: np.ma.masked_array
) -> Tuple[np.ndarray, np.ndarray]:
    import pyopencl as cl

    _opencl_initialize()

    # Selects only the section of `a` that will be computed against `b`.

    a_begin = max(0, offsets.min())
    a_end = min(len(a), offsets.max() + len(b))

    # Prepares Numpy buffers

    offsets_shifted = (offsets - a_begin).astype(np.int32)

    a_float = a[a_begin:a_end].astype(_opencl_buffer_dtype).data
    b_float = b.astype(_opencl_buffer_dtype).data

    a_float[np.isnan(a_float)] = 0.0
    b_float[np.isnan(b_float)] = 0.0

    mask_a_int8 = np.logical_not(a[a_begin:a_end].mask).astype(np.int8)
    mask_b_int8 = np.logical_not(b.mask).astype(np.int8)

    results = np.empty(len(offsets), dtype=np.float32)
    match_lens = np.empty(len(offsets), dtype=np.int32)

    # Prepares OpenCL buffers

    mf = cl.mem_flags

    offsets_gpu = cl.Buffer(_opencl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=offsets_shifted)
    a_gpu = cl.Buffer(_opencl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=a_float)
    b_gpu = cl.Buffer(_opencl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=b_float)
    mask_a_gpu = cl.Buffer(_opencl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=mask_a_int8)
    mask_b_gpu = cl.Buffer(_opencl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=mask_b_int8)

    results_gpu = cl.Buffer(_opencl_ctx, mf.WRITE_ONLY, results.nbytes)
    match_lens_gpu = cl.Buffer(_opencl_ctx, mf.WRITE_ONLY, match_lens.nbytes)

    # Runs the OpenCL kernel

    n_blocks = math.ceil(len(offsets) / THREADS_PER_BLOCK)

    _opencl_program.match_signals(
        _opencl_queue, (THREADS_PER_BLOCK * n_blocks,), (THREADS_PER_BLOCK,),
        np.int32(method.as_int),
        offsets_gpu, np.int32(len(offsets_shifted)),
        a_gpu, mask_a_gpu, np.int32(len(a_float)),
        b_gpu, mask_b_gpu, np.int32(len(b_float)),
        results_gpu, match_lens_gpu,
    )

    cl.enqueue_copy(_opencl_queue, results, results_gpu)
    cl.enqueue_copy(_opencl_queue, match_lens, match_lens_gpu)

    return results, match_lens


def _opencl_initialize():
    global _opencl_ctx, _opencl_queue, _opencl_program, _opencl_buffer_dtype

    if _opencl_ctx is None:
        import pyopencl as cl

        _opencl_ctx = cl.create_some_context()
        _opencl_queue = cl.CommandQueue(_opencl_ctx)

        compiler_flags = ["-DBACKEND_IS_OPENCL"]

        # Detects the optimal buffer item size

        supports_float16 = all(
            "cl_khr_fp16" in d.extensions.split(" ")
            for d in _opencl_ctx.devices
        )

        if supports_float16:
            _opencl_buffer_dtype = np.float16
            compiler_flags.append("-DUSE_FLOAT16_BUFFERS")
        else:
            _opencl_buffer_dtype = np.float32

        # Builds the kernel

        _opencl_program = cl.Program(
            _opencl_ctx, _read_kernel_source()
        ).build(options=compiler_flags)


_cuda_program = None
_cuda_buffer_dtype = None


def _cuda_compute_results(
    method: MatchMethod, offsets: np.ndarray, a: np.ma.masked_array, b: np.ma.masked_array
) -> Tuple[np.ndarray, np.ndarray]:
    import pycuda.driver as cuda

    _cuda_initialize()

    # Selects only the section of `a` that will be computed against `b`.

    a_begin = max(0, offsets.min())
    a_end = min(len(a), offsets.max() + len(b))

    # Prepares Numpy buffers

    offsets_shifted = (offsets - a_begin).astype(np.int32)

    a_float = a[a_begin:a_end].astype(_cuda_buffer_dtype).data
    b_float = b.astype(_cuda_buffer_dtype).data

    a_float[np.isnan(a_float)] = 0.0
    b_float[np.isnan(b_float)] = 0.0

    mask_a_int8 = np.logical_not(a[a_begin:a_end].mask).astype(np.int8)
    mask_b_int8 = np.logical_not(b.mask).astype(np.int8)

    results = np.zeros(len(offsets), dtype=np.float32)
    match_lens = np.zeros(len(offsets), dtype=np.int32)

    # Runs the kernel

    n_blocks = math.ceil(len(offsets) / THREADS_PER_BLOCK)

    _cuda_program.get_function("corr_coeffs")(
        np.int32(method.as_int),
        cuda.In(offsets_shifted), np.int32(len(offsets_shifted)),
        cuda.In(a_float), cuda.In(mask_a_int8), np.int32(len(a_float)),
        cuda.In(b_float), cuda.In(mask_b_int8), np.int32(len(b_float)),
        cuda.Out(results), cuda.InOut(match_lens),
        grid=(n_blocks, 1, 1), block=(THREADS_PER_BLOCK, 1, 1),
    )

    return results, match_lens


def _cuda_initialize():
    global _cuda_program, _cuda_buffer_dtype

    if _cuda_program is None:
        import pycuda.autoinit
        from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS

        compiler_flags = DEFAULT_NVCC_FLAGS + ["-DBACKEND_IS_CUDA"]

        # Detects the optimal buffer item size

        supports_float16 = pycuda.autoinit.device.compute_capability() >= (5, 3)

        if supports_float16:
            _cuda_buffer_dtype = np.float16
            compiler_flags.append("-DUSE_FLOAT16_BUFFERS")
        else:
            _cuda_buffer_dtype = np.float32

        # Builds the kernel

        _cuda_program = SourceModule(
            _read_kernel_source(), no_extern_c=True, options=compiler_flags
        )


def _filter_results(
    offsets: np.ndarray, results: np.ndarray, match_lens: np.ndarray,
    min_match_len: int, min_match_result: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filters out bad matches based on their values."""

    assert len(offsets) == len(results)
    assert len(offsets) == len(match_lens)

    valids = (results <= min_match_result) & (match_lens >= min_match_len)
    offsets = offsets[valids].copy()
    results = results[valids].copy()
    match_lens = match_lens[valids].copy()

    return offsets, results, match_lens


def _build_matches(
    frequency: float, method: MatchMethod, offsets: np.ndarray, results: np.ndarray,
    match_lens: np.ndarray, max_matches: Optional[int],
) -> List[Match]:
    """
    Post-processes the matches's results by filtering poor matches and by merging adjacent matches.
    """

    assert len(offsets) == len(results)
    assert len(offsets) == len(match_lens)

    # Filters out matches that are not local maximas.

    search_window_size = math.ceil(MIN_MATCH_LOCAL_MAXIMUM.total_seconds() * frequency)

    match_mask = np.full(len(offsets), True)

    i = 0
    while i < len(offsets):
        offset = offsets[i]
        corr_coeff = results[i]

        max_search_offset = offset + search_window_size
        j = i + 1
        while j < len(offsets) and offsets[j] < max_search_offset:
            if results[j] < corr_coeff:
                match_mask[j] = False
                j += 1
            else:
                match_mask[i] = False
                break
        i = j

    offsets = offsets[match_mask]
    results = results[match_mask]
    match_lens = match_lens[match_mask]
    scores = _match_scores(frequency, method, results, match_lens)

    # Sorts the matches

    matches = sorted(zip(scores, results, match_lens, offsets), reverse=False)

    if max_matches is not None:
        matches = matches[:max_matches]

    # Builds the ENFMatch objects.
    return [
        Match(
            offset=datetime.timedelta(seconds=int(offset / frequency)),
            duration=datetime.timedelta(seconds=int(match_len / frequency)),
            result=result,
            score=score,
        )
        for score, result, match_len, offset
        in matches
        if score > 0
    ]


def _match_scores(
    frequency: float, method: MatchMethod, result: np.ndarray, match_lens: np.ndarray
) -> np.ndarray:
    """Estimates a probabilistic score ([0..1]) of a match."""

    if method == MatchMethod.CORR_COEFF:
        # See https://gist.github.com/RaphaelJ/d5bb2a9e597fb60b0117b8eaaa8425f6 for the estimation
        # of the regression coefficients.

        sqrt_sec_duration = np.sqrt(match_lens / frequency)
        score = -0.056555 * sqrt_sec_duration + 0.099599 * result * sqrt_sec_duration + -0.307455
        return np.clip(score, 0, 1)
    else:
        return result


def _read_kernel_source() -> str:
    with open(KERNEL_PATH) as f:
        return f.read()