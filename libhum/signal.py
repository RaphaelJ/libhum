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

from typing import Optional

import attrs
import matplotlib.pyplot as plt
import numpy as np
import scipy


@attrs.define
class ENFSignal:
    network_frequency: float = attrs.field() # e.g. 50Hz or 60Hz

    signal_frequency: float = attrs.field() # e.g. 1Hz or 0.1Hz

    # The ENF signal, relative to the network's frequency.
    signal: np.ma.masked_array = attrs.field()

    begins_at: Optional[datetime.datetime] = attrs.field(default=None)

    @property
    def signal_sampling_rate(self) -> datetime.timedelta:
        return datetime.timedelta(1 / self.signal_frequency)

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

    def downsample(self, new_frequency: float) -> "ENFSignal":
        if new_frequency > self.frequency:
            raise ValueError("only downsampling is supported.")

        if self.frequency % new_frequency != 0:
            raise ValueError("new frequency should be a multiple of the signal frequency.")

        q = self.frequency // new_frequency
        new_signal = scipy.signal.decimate(self.signal, q)

        return attrs.evolve(self, signal_frequency=new_frequency, signal=new_signal)

