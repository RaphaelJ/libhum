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

# Some of the filtering functions have been heavily inspired by Robert Heaton's implementation of
# ENF analysis:
# https://github.com/robert/enf-matching

"""
Provides DSP algorithms to extract ENF signal from sound recordings.
"""

import datetime
import enum
import math
from typing import List, Tuple

import numpy as np
import scipy

from libhum.types import Signal, AnalysisResult


# Parameters obtained on training data using
# https://gist.github.com/RaphaelJ/254d7f0868ffb8eed9c43a17ad71f0bd

# Tries to decimate the original signal so that it is at least that frequency.
MIN_DECIMATED_FREQUENCY = 1000.0

SPECTRUM_BAND_SIZE = 0.25 # e.g. 49.75 to 50.25 for 50Hz ENF.
STFT_WINDOW_SIZE = datetime.timedelta(seconds=18)

# Post-filters the spectrum with a running normalization filter of the specified window size.
# Disabled if None.
NORMALIZE_WINDOW_SIZE = datetime.timedelta(seconds=12)

ENF_OUTPUT_FREQUENCY = 1.0 # Detects the source ENF at 1Hz

# Post-filters the detected ENF with a Gaussian filter. Disabled if None.
ENF_GAUSSIAN_SIGMA = 2.5

# Post-filters the detected ENF by selecting good sections with an high S/N and a minimum duration,
# and by expanding these to neighboring lower S/N sections if the signal's gradient is within the
# expected range. Allows short invalid sections of up to ENF_ARTIFACT_MAX_DURATION.
ENF_HIGH_SNR_THRES = 3.0
ENF_HIGH_SNR_MIN_DURATION = datetime.timedelta(seconds=5)
ENF_LOW_SNR_THRES = 1.5
ENF_MAX_GRADIENT = 0.004
ENF_ARTIFACT_MAX_DURATION = datetime.timedelta(seconds=0)


def compute_enf(
    signal: np.array, signal_frequency: float, network_frequency: float = 50.0,
    frequency_harmonics: List[int] = [1, 2],
) -> AnalysisResult:
    """Detects the ENF signal in the provided audio signal."""

    decimated_signal, decimated_frequency = _signal_decimate(signal, signal_frequency)

    spectrum = _signal_spectrum(
        decimated_signal, decimated_frequency, network_frequency, frequency_harmonics
    )

    # Computes the ENF signal for each harmonic, and keeps the best one.

    results = (
        _detect_enf(harmonic_spectrum, network_frequency, frequency_harmonic)
        for frequency_harmonic, harmonic_spectrum in zip(frequency_harmonics, spectrum)
    )

    return max(results, key=lambda result: result.enf.quality())


def _signal_decimate(signal: np.ndarray, signal_frequency: float) -> Tuple[np.ndarray, float]:
    """
    Decimates the input signal to at least ``MIN_DECIMATED_FREQUENCY``.

    Returns the new signal frequency and the decimated signal.
    """

    if signal_frequency <= MIN_DECIMATED_FREQUENCY:
        return signal, signal_frequency

    decimation_q = int(signal_frequency // MIN_DECIMATED_FREQUENCY)

    if decimation_q <= 1:
        return signal, signal_frequency

    downsampled_frequency = signal_frequency / decimation_q
    assert downsampled_frequency >= MIN_DECIMATED_FREQUENCY

    return scipy.signal.decimate(signal, decimation_q, ftype="fir", n=16), downsampled_frequency


def _signal_spectrum(
    signal: np.ndarray, signal_frequency: float, network_frequency: float,
    frequency_harmonics: List[int],
) -> List[Tuple[np.array, np.array, np.array]]:
    """
    Computes the normalized STFT spectrum for the given network frequency's harmonics.

    Returns the frequencies, timestamp, and spectrum for every requested harmonic's frequency band.
    """

    lo_hi_cuts = [
        (
            harmonic * (network_frequency - SPECTRUM_BAND_SIZE),
            harmonic * (network_frequency + SPECTRUM_BAND_SIZE)
        )
        for harmonic in frequency_harmonics
    ]

    filtered_data = _bandpass_filter(signal, signal_frequency, lo_hi_cuts, order=10)

    return [
        (f, t, _spectrum_normalize(Zxx, ENF_OUTPUT_FREQUENCY))
        for f, t, Zxx in _stft(filtered_data, signal_frequency, lo_hi_cuts)
    ]


def _bandpass_filter(
    signal: np.ndarray, frequency: float, lo_hi_cuts: List[Tuple[int, int]], order: int
) -> np.array:
    """
    Passes input data through a Butterworth bandpass filter. Code borrowed from
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """

    filtered_data = None

    for lo_cut, hi_cut in lo_hi_cuts:
        nyq = 0.5 * frequency
        low = lo_cut / nyq
        high = hi_cut / nyq

        sos = scipy.signal.butter(order, [low, high], analog=False, btype='band', output='sos')

        harmonic_filtered_data = scipy.signal.sosfilt(sos, signal)

        if filtered_data is None:
            filtered_data = harmonic_filtered_data
        else:
            filtered_data = np.maximum(filtered_data, harmonic_filtered_data)

    return filtered_data


def _stft(
    signal: np.ndarray, frequency: float, lo_hi_cuts: List[Tuple[int, int]],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Performs a Short-time Fourier Transform (STFT) on the input signal for the given bands

    Return the frequencies, timestamps and spectrum for each requested frequency band.
    """

    STFT_CHUNK_SIZE = 256

    signal_duration = len(signal) / frequency

    window_size_seconds = min(signal_duration, STFT_WINDOW_SIZE.total_seconds())
    n_perseg = int(frequency * window_size_seconds)
    n_overlap = int(frequency * window_size_seconds - frequency / ENF_OUTPUT_FREQUENCY)

    assert STFT_CHUNK_SIZE / ENF_OUTPUT_FREQUENCY >= window_size_seconds

    # Computes the STFT on small chunks to reduce the memory usage.

    f = None
    ts = []
    Zxxs = []

    hopsize = n_perseg - n_overlap
    output_len = math.ceil(len(signal) / hopsize) + 1

    window_half_size = n_perseg // 2

    for chunk_begin in range(0, output_len, STFT_CHUNK_SIZE):
        chunk_end = chunk_begin + STFT_CHUNK_SIZE

        signal_begin = max(
            0,
            int(chunk_begin / ENF_OUTPUT_FREQUENCY * frequency - window_half_size)
        )
        signal_end = int(chunk_end / ENF_OUTPUT_FREQUENCY * frequency + window_half_size)

        chunk = signal[signal_begin:signal_end]

        if len(chunk) < n_perseg:
            # FIXME: do not skip the last chunk when it's shorter than STFT_WINDOW_SIZE
            break

        if signal_begin == 0:
            output_begin = 0
        else:
            assert signal_begin >= window_size_seconds
            output_begin = int(window_half_size / frequency * ENF_OUTPUT_FREQUENCY)

        f, t, Zxx = scipy.signal.stft(chunk, frequency, nperseg=n_perseg, noverlap=n_overlap)

        band_f_idxs = None

        for lo_cut, hi_cut in lo_hi_cuts:
            harmonic_f_idxs = (f >= lo_cut) & (f <= hi_cut)

            if band_f_idxs is None:
                band_f_idxs = harmonic_f_idxs
            else:
                band_f_idxs |= harmonic_f_idxs

        f = f[band_f_idxs]
        ts.append(t[output_begin:output_begin+STFT_CHUNK_SIZE] + chunk_begin / ENF_OUTPUT_FREQUENCY)
        Zxxs.append(Zxx[band_f_idxs, output_begin:output_begin+STFT_CHUNK_SIZE])

    t = np.concatenate(ts)
    Zxx = np.concatenate(Zxxs, axis=1)

    if len(f) < 2 or len(t) < 1:
        raise ValueError(f"unable to compute spectrum on signal of length {len(signal)}.")

    return [
        (f[(f >= lo_cut) & (f <= hi_cut)], t, Zxx[(f >= lo_cut) & (f <= hi_cut)])
        for lo_cut, hi_cut in lo_hi_cuts
    ]


def _spectrum_normalize(spectrum: np.ndarray, signal_frequency: float) -> np.ndarray:
    """Normalizes to the mean and stddev over NORMALIZE_WINDOW_SIZE."""

    if NORMALIZE_WINDOW_SIZE is None:
        # Normalizes over the whole signal.
        mean = np.mean(spectrum)
        std = np.std(spectrum)

        return np.abs((spectrum - mean) / std)

    window_size = round(NORMALIZE_WINDOW_SIZE.total_seconds() * signal_frequency)

    spectrum = np.abs(spectrum).transpose()

    normalized = np.empty(spectrum.shape)

    for i in range(0, len(spectrum)):
        window_begin = max(0, i - window_size // 2)
        window_end = min(len(spectrum), i + window_size // 2)

        window = spectrum[window_begin:window_end]

        mean = np.mean(window)
        std = np.std(window)

        normalized[i] = (spectrum[i] - mean) / std

    return np.abs(normalized).transpose()


def _detect_enf(
    spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray], network_frequency: float,
    frequency_harmonic: int,
) -> AnalysisResult:
    """Detects the ENF signal at ENF_OUTPUT_FREQUENCY in the normalized spectrum."""

    f, t, Zxx = spectrum

    bin_size = f[1] - f[0]

    enf = np.empty(len(t), dtype=np.float16)
    snrs = np.empty(len(t), dtype=np.float32)

    for i, sub_spectrum in enumerate(Zxx.transpose()):
        max_amp = np.amax(sub_spectrum)

        max_amp_idx = np.where(sub_spectrum == max_amp)[0][0]
        max_amp_freq = f[0] + _quadratic_interpolation(sub_spectrum, max_amp_idx, bin_size)

        enf[i] = max_amp_freq / frequency_harmonic - network_frequency
        snrs[i] = max_amp / np.mean(sub_spectrum)

    enf = _post_process_enf(enf, snrs)

    enf_signal = Signal(
        network_frequency=network_frequency,
        signal=enf,
        signal_frequency=ENF_OUTPUT_FREQUENCY,
    )

    return AnalysisResult(
        enf=enf_signal,
        spectrum=spectrum,
        snr=snrs,
        frequency_harmonic=frequency_harmonic,
    )


def _quadratic_interpolation(spectrum, max_amp_idx, bin_size):
    """
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    """

    center = spectrum[max_amp_idx]
    left = spectrum[max_amp_idx - 1] if max_amp_idx > 0 else center
    right = spectrum[max_amp_idx + 1] if max_amp_idx + 1 < len(spectrum) else center

    p = 0.5 * (left - right) / (left - 2 * center + right)
    interpolated = (max_amp_idx + p) * bin_size

    return interpolated


def _post_process_enf(enf: np.ndarray, snrs: np.ndarray) -> np.ma.masked_array:
    smoothed = _gaussian_filter_enf(enf)

    # Clips any value out of the network's frequency band
    clipped = np.ma.masked_where(
        (smoothed < -SPECTRUM_BAND_SIZE) | (smoothed > SPECTRUM_BAND_SIZE),
        smoothed
    )

    thresholded = _threshold_enf(clipped, snrs)

    return thresholded


def _gaussian_filter_enf(enf: np.ma.masked_array) -> np.ma.masked_array:
    if ENF_GAUSSIAN_SIGMA is None:
        return enf

    return scipy.ndimage.gaussian_filter1d(
        enf.astype(np.float64), sigma=ENF_GAUSSIAN_SIGMA
    ).astype(np.float16)


def _threshold_enf(enf: np.ma.masked_array, snrs: np.ndarray) -> np.ma.masked_array:
    """
    Finds "good" ENF sections that:

    - have at least ENF_HIGH_SNR_MIN_DURATION of continuous ENF signal with a S/N higher than
      ENF_HIGH_SNR_THRES;

    - do not have any clipped sample;

    - do not have any sample with a S/N lower than ENF_LOW_SNR_THRES;

    - do not have any sample derivative higher than ENF_MAX_DERIV_THRES
    """

    min_section_len = ENF_OUTPUT_FREQUENCY * ENF_HIGH_SNR_MIN_DURATION.total_seconds()
    max_artifact_len = ENF_OUTPUT_FREQUENCY * ENF_ARTIFACT_MAX_DURATION.total_seconds()
    max_gradient = ENF_MAX_GRADIENT / ENF_OUTPUT_FREQUENCY

    def clipped(i: int) -> bool:
        return np.ma.is_masked(enf) and enf.mask[i]

    def above_low_threshold(i: int) -> bool:
        return snrs[i] >= ENF_LOW_SNR_THRES

    def above_high_threshold(i: int) -> bool:
        return snrs[i] >= ENF_HIGH_SNR_THRES

    def above_min_section_duration(section_duration: int) -> bool:
        return section_duration >= min_section_len

    gradient = np.abs(np.gradient(enf))

    def below_max_gradient(i: int) -> bool:
        return gradient[i] <= max_gradient

    class FilterState(enum.Enum):
        # Initial state, the filter is looking for a sample with S/N above ENF_HIGH_SNR_THRES.
        SEARCHING = 1

        # S/N is above ENF_HIGH_SNR_THRES, but the section isn't ENF_HIGH_SNR_MIN_DURATION long yet.
        ABOVE_HIGH_THRESHOLD = 2

        # The current signal is a valid ENF signal. We collect samples until we get lower than
        # ENF_LOW_SNR_THRES.
        IN_VALID_SECTION = 3

        # The current signal went below the ENF_LOW_SNR_THRES, or above the ENF_MAX_GRADIENT.
        IN_ARTIFACT = 4

    state = FilterState.SEARCHING
    section_len = None
    artifact_len = None
    artifact_total_gradient = None

    thres_mask = np.ma.masked_all(len(enf)).mask

    i = 0
    while i < len(enf):
        if state == FilterState.SEARCHING:
            if not clipped(i) and above_high_threshold(i) and below_max_gradient(i):
                section_len = 1
                state = FilterState.ABOVE_HIGH_THRESHOLD
        elif state == FilterState.ABOVE_HIGH_THRESHOLD:
            if not clipped(i) and above_high_threshold(i) and below_max_gradient(i):
                section_len += 1

                if above_min_section_duration(section_len):
                    state = FilterState.IN_VALID_SECTION
                    thres_mask[i] = False

                    # Extends the section left-wise to any previously ignored low threshold section.
                    artifact_len = 0
                    artifact_total_gradient = 0
                    j = i - 1
                    while j >= 0 and thres_mask[j] and artifact_len <= max_artifact_len:
                        artifact_len += 1
                        artifact_total_gradient += gradient[i]

                        if (
                            not clipped(j) and
                            above_low_threshold(j) and
                            below_max_gradient(j) and
                            artifact_total_gradient / artifact_len < max_gradient
                        ):
                            thres_mask[j] = False

                            artifact_len = 0
                            artifact_total_gradient = 0

                        j -= 1
            else:
                state = FilterState.SEARCHING
        elif state == FilterState.IN_VALID_SECTION:
            if not clipped(i) and above_low_threshold(i) and below_max_gradient(i):
                thres_mask[i] = False
            else:
                state = FilterState.IN_ARTIFACT
                artifact_len = 1
                artifact_total_gradient = gradient[i]
        elif state == FilterState.IN_ARTIFACT:
            if (
                not clipped(i) and
                above_low_threshold(i) and
                below_max_gradient(i) and
                (artifact_total_gradient / artifact_len < max_gradient or above_high_threshold(i))
            ):
                thres_mask[i] = False
                state = FilterState.IN_VALID_SECTION
            else:
                artifact_len += 1
                artifact_total_gradient += gradient[i]

                if artifact_len > max_artifact_len:
                    thres_mask[i] = False
                    state = FilterState.SEARCHING

        i += 1

    return np.ma.array(enf, mask=enf.mask | thres_mask)

