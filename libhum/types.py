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
import math
import pickle

from typing import Optional, Tuple

import attrs
import lz4.frame
import matplotlib.pyplot as plt
import numpy as np
import scipy


@attrs.define
class Signal:
    """An ENF signal with its associated attributes."""

    network_frequency: float = attrs.field() # e.g. 50Hz or 60Hz

    signal_frequency: float = attrs.field() # e.g. 1Hz or 0.1Hz

    # The ENF signal, relative to the network's frequency.
    signal: np.ma.masked_array = attrs.field()

    begins_at: Optional[datetime.datetime] = attrs.field(default=None)

    @property
    def signal_sampling_rate(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=1.0 / self.signal_frequency)

    @property
    def duration(self) -> datetime.timedelta:
        return self.signal_sampling_rate * len(self.signal)

    @property
    def ends_at(self) -> Optional[datetime.datetime]:
        if self.begins_at is not None:
            return self.begins_at + self.duration
        else:
            return None

    def plot(self):
        sampling_rate = self.signal_sampling_rate
        ts = [(sampling_rate * i).total_seconds() for i in range(0, len(self.signal))]

        plt.plot(ts, self.signal.astype(np.float64) + self.network_frequency)
        plt.ylim(self.network_frequency - 0.1, self.network_frequency + 0.1)
        plt.show()

    def serialize(self) -> bytes:
        # Pickles a regular Python directory so that it stays somewhat backward compatible.
        pickled = pickle.dumps({
            "network_frequency": self.network_frequency,
            "signal_frequency": self.signal_frequency,
            "signal": self.signal.astype(np.float16).tobytes(fill_value=np.nan),
            "begins_at": self.begins_at.isoformat() if self.begins_at is not None else None,
        })

        return lz4.frame.compress(pickled)

    @staticmethod
    def deserialize(data: bytes) -> "Signal":
        decompressed = lz4.frame.decompress(data)
        unpickled = pickle.loads(decompressed)

        signal = np.ma.masked_invalid(np.frombuffer(unpickled["signal"], dtype=np.float16))

        if unpickled["begins_at"] is not None:
            begins_at = datetime.datetime.fromisoformat(unpickled["begins_at"])
        else:
            begins_at = None

        return Signal(
            network_frequency=unpickled["network_frequency"],
            signal_frequency=unpickled["signal_frequency"],
            signal=signal,
            begins_at=begins_at
        )

@attrs.define
class Match:
    """A single result of a time matching comparison between two ENF signals."""

    # The offset to apply to the target signal to match the reference.
    offset: datetime.timedelta

    # The total duration of the matching signals (i.e. invalid values are ignored during the
    # computation of the correlation coefficient).
    duration: datetime.timedelta

    # The Pearson's correlation coefficient for this match.
    corr_coeff: float

    def plot(self, ref: Signal, target: Signal):
        if ref.signal_frequency != target.signal_frequency:
            raise ValueError("signal frequencies should be identical.")

        if ref.network_frequency != target.network_frequency:
            raise ValueError("network frequencies should be identical.")

        signal_frequency = ref.signal_frequency

        sampling_rate = ref.signal_sampling_rate

        match_len = math.floor(self.duration.total_seconds() * signal_frequency)

        if ref.begins_at:
            match_begins_at = ref.begins_at + self.offset
            t = [match_begins_at + i * sampling_rate for i in range(0, match_len)]
        else:
            t = [(self.offset + i * sampling_rate).total_seconds() for i in range(0, match_len)]

        offset_sample = math.floor(self.offset.total_seconds() * signal_frequency)

        ref_samples = np.ma.empty(match_len, dtype=np.float32)
        ref_samples[:] = np.ma.masked

        ref_begin = max(0, offset_sample)
        ref_samples[0:min(len(ref.signal), match_len)] = ref.signal[ref_begin:ref_begin + match_len]

        target_samples = target.signal[0:match_len].astype(np.float32)

        _, ax = plt.subplots()

        ax.plot(t, ref_samples + ref.network_frequency, color="blue", label="Reference ENF")
        ax.plot(t, target_samples + target.network_frequency, color="red", label="Target ENF")

        ax.set_xlabel("Time")
        ax.set_ylabel("Network frequency (Hz)")
        ax.legend()

        plt.show()


@attrs.define
class AnalysisResult:
    """The result of the detection of an ENF signal from an audio file."""

    enf: Signal = attrs.field()

    # The filtered spectrum for the signal's band, with its frequency and timestamps.
    spectrum: Tuple[np.ndarray, np.ndarray, np.ndarray] = attrs.field()

    snr: np.ndarray = attrs.field()

    frequency_harmonic: int = attrs.field()

    def plot(self):
        _, ax = plt.subplots(1, 1, figsize=(18, 4))

        f, t, Zxx = self.spectrum

        ax.pcolormesh(t, f / self.frequency_harmonic, Zxx, shading='gouraud')
        ax.plot(
            t,
            self.enf.signal.astype(np.float64) + self.enf.network_frequency,
            color="blue",
            label="Detected ENF"
        )

        ax_snr = ax.twinx()
        ax_snr.plot(t, self.snr, color="grey", alpha=0.25, label="S/N")

        h, l = ax.get_legend_handles_labels()
        h_snr, l_snr = ax_snr.get_legend_handles_labels()
        ax.legend(h + h_snr, l + l_snr, loc=2)

        plt.show()