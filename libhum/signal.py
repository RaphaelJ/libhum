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

import attrs
import numpy as np
import scipy

@attrs.define
class ENFSignal:
    network_frequency: float = attrs.field() # e.g. 50Hz or 60Hz

    begins_at: datetime.datetime = attrs.field()
    signal_frequency: float = attrs.field() # e.g. 1Hz or 0.1Hz
    signal: np.ma.MaskedArray = attrs.field()

    @property
    def duration(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=len(self.signal) / self.signal_frequency)

    @property
    def ends_at(self):
        return self.begins_at + self.duration

    def downsample(self, new_frequency: float) -> "ENFSignal":
        if new_frequency > self.frequency:
            raise ValueError("only downsampling is supported.")

        if self.frequency % new_frequency != 0:
            raise ValueError("new frequency should be a multiple of the signal frequency.")

        q = self.frequency // new_frequency
        new_signal = scipy.signal.decimate(self.signal, q)

        return ENFSignal(self.begin_at, new_frequency, new_signal)
