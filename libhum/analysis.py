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
from typing import Tuple

import numpy as np
import scipy

from libhum.types import Signal, AnalysisResult


# Tries to decimate the original signal so that it is at least that frequency.
MIN_DECIMATED_FREQUENCY = 1000.0

FREQUENCY_HARMONIC = 2 # e.g. look a the 100Hz signal for 50Hz ENF

SPECTRUM_BAND_SIZE = 0.2 # e.g. 49.8 to 50.2 for 50Hz ENF.
STFT_WINDOW_SIZE = datetime.timedelta(seconds=18)

# Post-filters the spectrum with a running normalization filter of the specified window size.
NORMALIZE_WINDOW_SIZE = datetime.timedelta(seconds=30)

ENF_OUTPUT_FREQUENCY = 1.0 # Detects the source ENF at 1Hz

# Post-filters the detected ENF with a Gaussian filter.
ENF_GAUSSIAN_SIGMA = 2.0

# Post-filters the detected ENF by selecting good sections with an high S/N and a minimum duration,
# and by expanding these to neighboring lower S/N sections if the signal's gradient is within the
# expected range.
ENF_HIGH_SNR_THRES = 3.0
ENF_HIGH_SNR_MIN_DURATION = datetime.timedelta(seconds=5)
ENF_LOW_SNR_THRES = 2.0
ENF_MAX_GRADIENT = 0.0075

import datetime


def compute_enf(
    signal: np.array, signal_frequency: float, network_frequency: float = 50.0
) -> AnalysisResult:
    """Detects the ENF signal in the provided audio signal."""

    decimated_signal, decimated_frequency = _signal_decimate(signal, signal_frequency)

    spectrum = _signal_spectrum(decimated_signal, decimated_frequency, network_frequency)

    f, t, Zxx = spectrum
    enf, snr = _detect_enf(f, t, Zxx, network_frequency)

    enf_signal = Signal(
        network_frequency=network_frequency,
        signal=enf,
        signal_frequency=ENF_OUTPUT_FREQUENCY,
    )

    return AnalysisResult(
        enf=enf_signal,
        spectrum=spectrum,
        snr=snr,
        frequency_harmonic=FREQUENCY_HARMONIC,
    )


def _signal_decimate(signal: np.ndarray, signal_frequency: float) -> Tuple[np.ndarray, float]:
    """
    Decimates the input signal to at least ``MIN_DECIMATED_FREQUENCY``.

    Returns the new signal frequency and the decimated signal.
    """

    decimation_q = int(signal_frequency // MIN_DECIMATED_FREQUENCY)

    if decimation_q <= 1:
        return signal, signal_frequency

    downsampled_frequency = signal_frequency / decimation_q
    assert downsampled_frequency >= MIN_DECIMATED_FREQUENCY

    return scipy.signal.decimate(signal, decimation_q, ftype="fir", n=16), downsampled_frequency


def _signal_spectrum(
    signal: np.ndarray, signal_frequency: float, network_frequency: float
) -> np.array:
    """
    Computes the normalized STFT spectrum for the given network frequency's harmonic.

    Returns the frequencies, timestamp, and the target frequency band's spectrum.
    """

    locut = FREQUENCY_HARMONIC * (network_frequency - SPECTRUM_BAND_SIZE)
    hicut = FREQUENCY_HARMONIC * (network_frequency + SPECTRUM_BAND_SIZE)

    filtered_data = _bandpass_filter(signal, signal_frequency, locut, hicut, order=10)

    f, t, Zxx = _stft(filtered_data, signal_frequency)

    band_f_idx = (f >= locut) & (f <= hicut)

    return f[band_f_idx], t, _spectrum_normalize(Zxx[band_f_idx], ENF_OUTPUT_FREQUENCY)


def _bandpass_filter(
    signal: np.ndarray, frequency: float, locut: float, hicut: float, order: int
) -> np.array:
    """
    Passes input data through a Butterworth bandpass filter. Code borrowed from
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """

    nyq = 0.5 * frequency
    low = locut / nyq
    high = hicut / nyq

    sos = scipy.signal.butter(order, [low, high], analog=False, btype='band', output='sos')

    return scipy.signal.sosfilt(sos, signal)


def _stft(signal: np.ndarray, frequency: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs a Short-time Fourier Transform (STFT) on the input signal."""

    window_size_seconds = STFT_WINDOW_SIZE.total_seconds()
    nperseg = int(frequency * window_size_seconds / ENF_OUTPUT_FREQUENCY)
    noverlap = int(frequency * (window_size_seconds - 1) / ENF_OUTPUT_FREQUENCY)

    return scipy.signal.stft(signal, frequency, nperseg=nperseg, noverlap=noverlap)


def _spectrum_normalize(spectrum: np.ndarray, frequency: float) -> np.ndarray:
    """Normalizes to the mean and stddev over NORMALIZE_WINDOW_SIZE."""

    window_size = round(NORMALIZE_WINDOW_SIZE.total_seconds() * frequency)

    spectrum = spectrum.transpose()

    for window_begin in range(0, len(spectrum), window_size):
        window_end = window_begin + window_size

        window = spectrum[window_begin:window_end]

        mean = np.mean(window)
        std = np.std(window)

        spectrum[window_begin:window_end] = (window - mean) / std

    return np.abs(spectrum).transpose()


def _detect_enf(
    f: np.ndarray, t: np.ndarray, spectrum: np.ndarray, network_frequency: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detects the ENF signal at 1Hz in the normalized spectrum, with its SNR vector."""

    bin_size = f[1] - f[0]

    enf = np.empty(len(t), dtype=np.float16)
    snrs = np.empty(len(t), dtype=np.float32)

    for i, sub_spectrum in enumerate(spectrum.transpose()):
        max_amp = np.amax(sub_spectrum)

        max_amp_idx = np.where(sub_spectrum == max_amp)[0][0]
        max_amp_freq = f[0] + _quadratic_interpolation(sub_spectrum, max_amp_idx, bin_size)

        enf[i] = max_amp_freq / FREQUENCY_HARMONIC - network_frequency
        snrs[i] = max_amp / np.mean(sub_spectrum)

    enf = _post_process_enf(enf, snrs)

    return enf, snrs


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

    min_section_duration = ENF_OUTPUT_FREQUENCY * ENF_HIGH_SNR_MIN_DURATION.total_seconds()

    def clipped(i: int) -> bool:
        return np.ma.is_masked(enf) and enf.mask[i]

    def above_low_threshold(i: int) -> bool:
        return snrs[i] >= ENF_LOW_SNR_THRES

    def above_high_threshold(i: int) -> bool:
        return snrs[i] >= ENF_HIGH_SNR_THRES

    def above_min_section_duration(section_duration: int) -> bool:
        return section_duration >= min_section_duration

    gradient = np.abs(np.gradient(enf))

    def below_max_gradient(i: int) -> bool:
        return gradient[i] <= (ENF_MAX_GRADIENT / ENF_OUTPUT_FREQUENCY)

    class FilterState(enum.Enum):
        # Initial state, the filter is looking for a sample with S/N above ENF_HIGH_SNR_THRES.
        SEARCHING = 1

        # S/N is above ENF_HIGH_SNR_THRES, but the section isn't ENF_HIGH_SNR_MIN_DURATION long yet.
        ABOVE_HIGH_THRESHOLD = 2

        # The current signal is a valid ENF signal. We collect samples until we get lower than
        # ENF_LOW_SNR_THRES.
        IN_VALID_SECTION = 3

    state = FilterState.SEARCHING
    section_duration = None

    thres_mask = np.ma.masked_all(len(enf)).mask

    i = 0
    while i < len(enf):
        if state == FilterState.SEARCHING:
            if not clipped(i) and above_high_threshold(i) and below_max_gradient(i):
                section_duration = 1
                state = FilterState.ABOVE_HIGH_THRESHOLD
        elif state == FilterState.ABOVE_HIGH_THRESHOLD:
            if not clipped(i) and above_high_threshold(i) and below_max_gradient(i):
                section_duration += 1

                if above_min_section_duration(section_duration):
                    state = FilterState.IN_VALID_SECTION

                    # Extends the section left-wise to any previously ignored low threshold section.
                    j = i
                    while (
                        j >= 0 and
                        not clipped(j) and
                        thres_mask[j] and
                        above_low_threshold(j) and
                        below_max_gradient(j)
                    ):
                        thres_mask[j] = False
                        j -= 1
            else:
                state = FilterState.SEARCHING
        elif state == FilterState.IN_VALID_SECTION:
            if not clipped(i) and above_low_threshold(i) and below_max_gradient(i):
                thres_mask[i] = False
            else:
                state = FilterState.SEARCHING

        i += 1

    return np.ma.array(enf, mask=enf.mask | thres_mask)
