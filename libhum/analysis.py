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
from typing import Tuple

import attrs
import matplotlib.pyplot as plt
import numpy as np
import scipy

from libhum.signal import ENFSignal

# Tries to decimate the original signal so that it is at least that frequency.
MIN_DECIMATED_FREQUENCY = 1000.0

TARGET_FREQUENCY_HARMONIC = 2 # e.g. look a the 100Hz signal for 50Hz ENF

SPECTRUM_BAND_SIZE = 0.15 # e.g. 49.85 to 50.15 for 50Hz ENF.

STFT_WINDOW_SIZE = datetime.timedelta(seconds=16)
STFT_OUTPUT_FREQUENCY = 1.0

ENF_MEDIAN_FILTER_SIZE = 3 # samples

ENF_LOWER_MIN_SNR = 1.6
ENF_HIGHER_MIN_SNR = 2.0


@attrs.define
class ENFAnalysisResult:
    enf: ENFSignal = attrs.field()

    # The filtered spectrum for the signal's band, with its frequency and timestamps.
    spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray] = attrs.field()

    snr: np.ndarray = attrs.field()

    def plot(self):
        _, ax = plt.subplots(1, 1, figsize=(15, 5))

        f, t, Zxx = self.spectrum

        ax.pcolormesh(t, f / TARGET_FREQUENCY_HARMONIC, Zxx, shading='gouraud')
        ax.plot(t, self.enf.signal.astype(np.float64) + self.enf.network_frequency, color="red")

        ax_snr = ax.twinx()
        ax_snr.plot(t, self.snr, color="orange", alpha=0.25)

        plt.show()


def compute_enf(
    signal: np.array, signal_frequency: float, network_frequency: float = 50.0
) -> ENFAnalysisResult:
    decimated_signal, decimated_frequency = _signal_decimate(signal, signal_frequency)

    spectrum = _signal_spectrum(decimated_signal, decimated_frequency, network_frequency)

    f, t, Zxx = spectrum
    enf, snr = _detect_enf(f, t, Zxx, network_frequency)

    enf_signal = ENFSignal(
        network_frequency=network_frequency,
        signal=enf,
        signal_frequency=signal_frequency
    )

    return ENFAnalysisResult(
        enf_signal,
        spectrum,
        snr,
    )


def _signal_decimate(signal: np.ndarray, signal_frequency: float) -> Tuple[np.ndarray, float]:
    """
    Decimates the input signal to at least ``MIN_DECIMATED_FREQUENCY``.

    Returns the new signal frequency and the decimated signal.
    """

    decimation_q = int(signal_frequency // MIN_DECIMATED_FREQUENCY)

    downsampled_frequency = signal_frequency / decimation_q
    assert downsampled_frequency >= MIN_DECIMATED_FREQUENCY

    return scipy.signal.decimate(signal, decimation_q), downsampled_frequency


def _signal_spectrum(
    signal: np.ndarray, signal_frequency: float, network_frequency: float
) -> np.array:
    """
    Computes the normalized STFT spectrum for the given network frequency's harmonic.

    Returns the frequencies, timestamp, and the target frequency band's spectrum.
    """

    locut = TARGET_FREQUENCY_HARMONIC * (network_frequency - SPECTRUM_BAND_SIZE)
    hicut = TARGET_FREQUENCY_HARMONIC * (network_frequency + SPECTRUM_BAND_SIZE)

    filtered_data = _bandpass_filter(signal, signal_frequency, locut, hicut, order=10)

    f, t, Zxx = _stft(filtered_data, signal_frequency)

    band_f_idx = (f >= locut) & (f <= hicut)

    return f[band_f_idx], t, _spectrum_normalize(Zxx[band_f_idx])


def _spectrum_normalize(spectrum: np.ndarray) -> np.ndarray:
    """Normalizes to the mean and stddev."""

    # Normalizes to the mean over the whole signal
    spectrum = (spectrum - np.mean(spectrum)) / np.std(spectrum)

    return np.abs(spectrum)


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
    nperseg = int(frequency * window_size_seconds / STFT_OUTPUT_FREQUENCY)
    noverlap = int(frequency * (window_size_seconds - 1) / STFT_OUTPUT_FREQUENCY)

    return scipy.signal.stft(signal, frequency, nperseg=nperseg, noverlap=noverlap)


def _detect_enf(
    f: np.ndarray, t: np.ndarray, spectrum: np.ndarray, network_frequency: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detects the ENF signal at 1Hz in the normalized spectrum, with its SNR vector."""

    bin_size = f[1] - f[0]

    enf = np.empty(len(t), dtype=np.float16)
    snrs = np.empty(len(t), dtype=np.float32)

    for i, sub_spectrum in enumerate(np.transpose(spectrum)):
        max_amp = np.amax(sub_spectrum)

        max_amp_idx = np.where(sub_spectrum == max_amp)[0][0]
        max_amp_freq = f[0] + _quadratic_interpolation(sub_spectrum, max_amp_idx, bin_size)

        enf[i] = max_amp_freq / TARGET_FREQUENCY_HARMONIC - network_frequency
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


def _post_process_enf(enf: np.ndarray, snrs: np.ndarray) -> np.ndarray:


    return enf
