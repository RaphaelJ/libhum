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

from typing import List, Optional, Tuple

import numpy as np

from libhum.types import Signal, Match


MIN_MATCH_DURATION = datetime.timedelta(minutes=3)
MIN_MATCH_RATIO = 0.2
MIN_MATCH_CORR_COEFF = 0.8

# Will ignore matches that have a better match within `MIN_MATCH_LOCAL_MAXIMUM`.
MIN_MATCH_LOCAL_MAXIMUM = datetime.timedelta(minutes=10)

MAX_BUFFER_SIZE = 64 * 1024 * 1024 # 64 MB

THREADS_PER_BLOCK = 256 # CUDA threads per bloc or OpenCL workers per group

KERNEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "match.kernel")


class MatchBackend(enum.Enum):
    CUDA = "cuda"
    NUMPY = "numpy"
    OPENCL = "opencl"


def match_signals(
    ref: Signal,
    target: Signal,
    max_matches: Optional[int],
    step: datetime.timedelta = datetime.timedelta(seconds=1),
    backend: MatchBackend = MatchBackend.NUMPY,
) -> List[Match]:
    if ref.signal_frequency != target.signal_frequency:
        raise ValueError("signal frequencies should be identical.")

    if ref.network_frequency != target.network_frequency:
        raise ValueError("network frequencies should be identical.")

    frequency = ref.signal_frequency

    backend_instance = {
        MatchBackend.CUDA: _cuda_compute_corr_coeffs,
        MatchBackend.NUMPY: _compute_corr_coeffs_numpy,
        MatchBackend.OPENCL: _opencl_compute_corr_coeffs,
    }[backend]

    ref_len = len(ref.signal)
    target_len = len(target.signal)

    min_matching_len = math.ceil(MIN_MATCH_DURATION.total_seconds() * frequency)

    offset_begin = - target_len + min_matching_len
    offset_end = ref_len - min_matching_len + 1
    step_offset = math.ceil(step.total_seconds() * frequency)

    offsets_chunks = []
    corr_coeffs_chunks = []
    match_lens_chunks = []

    # Processes the offset domain by chunks so that buffers do not exceed MAX_BUFFER_SIZE.
    chunk_size = MAX_BUFFER_SIZE // np.int32(0).nbytes * step_offset
    for chunk_begin in range(offset_begin, offset_end, chunk_size):
        chunk_end = min(offset_end, chunk_begin + chunk_size)

        offsets = np.arange(chunk_begin, chunk_end, step_offset, dtype=np.int32)

        # Computes the coefficients using the selected backend
        corr_coeffs, match_lens = backend_instance(offsets, ref.signal, target.signal)
        assert len(corr_coeffs) == len(match_lens)

        # Reduces the memory usage by immediatly removing bad coefficients
        offsets, corr_coeffs, match_lens = _filter_coeffs(
            frequency, offsets, corr_coeffs, match_lens
        )

        offsets_chunks.append(offsets)
        corr_coeffs_chunks.append(corr_coeffs)
        match_lens_chunks.append(match_lens)

    # Combines the chunked corr_coeffs and match_lens
    offsets = np.concatenate(offsets_chunks)
    corr_coeffs = np.concatenate(corr_coeffs_chunks)
    match_lens = np.concatenate(match_lens_chunks)

    # Sorts the resulting coefficients
    return _build_matches(frequency, offsets, corr_coeffs, match_lens, max_matches)


def _compute_corr_coeffs_numpy(
    offsets: np.ndarray, a: np.ma.masked_array, b: np.ma.masked_array
) -> Tuple[np.ndarray, np.ndarray]:
    corr_coeffs = np.empty(len(offsets), dtype=np.float32)
    match_lens = np.empty(len(offsets), dtype=np.int32)

    for i, offset in enumerate(offsets):
        corr_coeffs[i], match_lens[i] = _corr_coeff(
            a[max(0, offset):offset + len(b)], b[max(0, -offset):len(a) - offset]
        )

    return corr_coeffs, match_lens


def _corr_coeff(a: np.ma.masked_array, b: np.ma.masked_array) -> Tuple[float, int]:
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


_opencl_ctx = None
_opencl_queue = None
_opencl_program = None
_opencl_buffer_dtype = None


def _opencl_compute_corr_coeffs(
    offsets: np.ndarray, a: np.ma.masked_array, b: np.ma.masked_array
) -> Tuple[np.ndarray, np.ndarray]:
    import pyopencl as cl

    _opencl_initialize()

    # Selects only the section of `a` that will be computed against `b`.

    a_begin = max(0, offsets.min())
    a_end = min(len(a), offsets.max() + len(b))

    # Prepares Numpy buffers

    a_float = a[a_begin:a_end].astype(_opencl_buffer_dtype).data
    b_float = b.astype(_opencl_buffer_dtype).data

    a_float[np.isnan(a_float)] = 0.0
    b_float[np.isnan(b_float)] = 0.0

    mask_a_int8 = np.logical_not(a[a_begin:a_end].mask).astype(np.int8)
    mask_b_int8 = np.logical_not(b.mask).astype(np.int8)

    corr_coeffs = np.empty(len(offsets), dtype=np.float32)
    match_lens = np.empty(len(offsets), dtype=np.int32)

    # Prepares OpenCL buffers

    mf = cl.mem_flags

    offsets_gpu = cl.Buffer(_opencl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=offsets - a_begin)
    a_gpu = cl.Buffer(_opencl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=a_float)
    b_gpu = cl.Buffer(_opencl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=b_float)
    mask_a_gpu = cl.Buffer(_opencl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=mask_a_int8)
    mask_b_gpu = cl.Buffer(_opencl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=mask_b_int8)

    corr_coeffs_gpu = cl.Buffer(_opencl_ctx, mf.WRITE_ONLY, corr_coeffs.nbytes)
    match_lens_gpu = cl.Buffer(_opencl_ctx, mf.WRITE_ONLY, match_lens.nbytes)

    # Runs the OpenCL kernel

    n_blocks = math.ceil(len(offsets) / THREADS_PER_BLOCK)

    _opencl_program.corr_coeffs(
        _opencl_queue, (THREADS_PER_BLOCK * n_blocks,), (THREADS_PER_BLOCK,),
        offsets_gpu, np.int32(len(offsets)),
        a_gpu, mask_a_gpu, np.int32(len(a_float)),
        b_gpu, mask_b_gpu, np.int32(len(b_float)),
        corr_coeffs_gpu, match_lens_gpu,
    )

    cl.enqueue_copy(_opencl_queue, corr_coeffs, corr_coeffs_gpu)
    cl.enqueue_copy(_opencl_queue, match_lens, match_lens_gpu)

    print(match_lens)
    print(np.count_nonzero(match_lens))

    return corr_coeffs, match_lens


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


def _cuda_compute_corr_coeffs(
    offsets: np.ndarray, a: np.ma.masked_array, b: np.ma.masked_array
) -> Tuple[np.ndarray, np.ndarray]:
    import pycuda.driver as cuda
    from pycuda.tools import make_default_context

    cuda.init()

    context = make_default_context()
    device = context.get_device()

    from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS

    compiler_flags = DEFAULT_NVCC_FLAGS + ["-DBACKEND_IS_CUDA"]

    # Detects the optimal buffer item size

    supports_float16 = device.compute_capability() >= (5, 3)

    if supports_float16:
        _cuda_buffer_dtype = np.float16
        compiler_flags.append("-DUSE_FLOAT16_BUFFERS")
    else:
        _cuda_buffer_dtype = np.float32

    # Builds the kernel

    _cuda_program = SourceModule(
        _read_kernel_source(), no_extern_c=True, options=compiler_flags
    )




    # Selects only the section of `a` that will be computed against `b`.

    a_begin = max(0, offsets.min())
    a_end = min(len(a), offsets.max() + len(b))

    # Prepares Numpy buffers

    a_float = a[a_begin:a_end].astype(_cuda_buffer_dtype).data
    b_float = b.astype(_cuda_buffer_dtype).data

    a_float[np.isnan(a_float)] = 0.0
    b_float[np.isnan(b_float)] = 0.0

    mask_a_int8 = np.logical_not(a[a_begin:a_end].mask).astype(np.int8)
    mask_b_int8 = np.logical_not(b.mask).astype(np.int8)

    corr_coeffs = np.zeros(len(offsets), dtype=np.float32)
    match_lens = np.zeros(len(offsets), dtype=np.int32)

    # Runs the kernel

    n_blocks = math.ceil(len(offsets) / THREADS_PER_BLOCK)

    print(len(offsets))

    _cuda_program.get_function("corr_coeffs")(
        cuda.In(offsets), np.int32(len(offsets)),
        cuda.In(a_float), cuda.In(mask_a_int8), np.int32(len(a_float)),
        cuda.In(b_float), cuda.In(mask_b_int8), np.int32(len(b_float)),
        cuda.Out(corr_coeffs), cuda.InOut(match_lens),
        grid=(n_blocks, 1, 1), block=(THREADS_PER_BLOCK, 1, 1),
    )

    context.pop()
    context = None

    from pycuda.tools import clear_context_caches

    clear_context_caches()

    return corr_coeffs, match_lens


def _cuda_initialize():
    global _cuda_program, _cuda_buffer_dtype

    if _cuda_program is None:
        import pycuda.tools
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


def _filter_coeffs(
    frequency: float, offsets: np.ndarray, corr_coeffs: np.ndarray, match_lens: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filters out bad matches based on their coefficients."""

    assert len(offsets) == len(corr_coeffs)
    assert len(offsets) == len(match_lens)

    min_matching_len = math.ceil(MIN_MATCH_DURATION.total_seconds() * frequency)

    valids = (corr_coeffs >= MIN_MATCH_CORR_COEFF) & (match_lens >= min_matching_len)
    offsets = offsets[valids].copy()
    corr_coeffs = corr_coeffs[valids].copy()
    match_lens = match_lens[valids].copy()

    return offsets, corr_coeffs, match_lens


def _build_matches(
    frequency: float, offsets: np.ndarray, corr_coeffs: np.ndarray, match_lens: np.ndarray,
    max_matches: Optional[int],
) -> List[Match]:
    """
    Post-processes the matches's coefficients by filtering poor matches and by merging adjacent
    matches.
    """

    assert len(offsets) == len(corr_coeffs)
    assert len(offsets) == len(match_lens)

    # Filters out bad matches.

    min_matching_len = math.ceil(MIN_MATCH_DURATION.total_seconds() * frequency)

    valids = (corr_coeffs >= MIN_MATCH_CORR_COEFF) & (match_lens >= min_matching_len)
    offsets = offsets[valids]
    corr_coeffs = corr_coeffs[valids]
    match_lens = match_lens[valids]

    # Filters out matches that are not local maximas.

    search_window_size = math.ceil(MIN_MATCH_LOCAL_MAXIMUM.total_seconds() * frequency)

    match_mask = np.full(len(offsets), True)

    i = 0
    while i < len(offsets):
        offset = offsets[i]
        corr_coeff = corr_coeffs[i]

        max_search_offset = offset + search_window_size
        j = i + 1
        while j < len(offsets) and offsets[j] < max_search_offset:
            if corr_coeffs[j] < corr_coeff:
                match_mask[j] = False
                j += 1
            else:
                match_mask[i] = False
                break
        i = j

    offsets = offsets[match_mask]
    corr_coeffs = corr_coeffs[match_mask]
    match_lens = match_lens[match_mask]

    # Sorts the matches

    matches = sorted(zip(corr_coeffs, match_lens, offsets), reverse=True)

    if max_matches is not None:
        matches = matches[:max_matches]

    # Builds the ENFMatch objects.
    return [
        Match(
            offset=datetime.timedelta(seconds=int(offset)),
            duration=datetime.timedelta(seconds=int(frequency * match_len)),
            corr_coeff=corr_coeff,
        )
        for corr_coeff, match_len, offset
        in matches
    ]


def _read_kernel_source() -> str:
    with open(KERNEL_PATH) as f:
        return f.read()