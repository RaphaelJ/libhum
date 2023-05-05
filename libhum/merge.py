
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


import enum
import math

import numpy as np

from libhum.types import Signal


class MergeStrategy(enum.Enum):
    """How the merge algorithm handles overlapping values."""

    # Only keep the value of the first or second signal.
    USE_FIRST = "first"
    USE_SECOND = "second"

    # Mean of the two values.
    MEAN = "mean"

    # Masks the value in the resulting signal.
    MASKED = "masked"


def merge_signals(
    first: Signal, second: "Signal", merge_strategy: MergeStrategy = MergeStrategy.MEAN
) -> "Signal":
    """
    Combines the two ENF signal in a single continuous signal, based on their respective
    time stamps.
    """

    if first.begins_at is None or second.begins_at is None:
        raise ValueError("cannot only merge signals with `begins_at` values.")

    if first.signal_frequency != second.signal_frequency:
        raise ValueError("signal frequencies should be identical.")

    if first.network_frequency != second.network_frequency:
        raise ValueError("network frequencies should be identical.")

    frequency = first.signal_frequency
    network_frequency = first.network_frequency

    begins_at = min(first.begins_at, second.begins_at)
    ends_at = max(first.ends_at, second.ends_at)
    duration = ends_at - begins_at

    length = math.ceil(duration.total_seconds() * frequency)

    merged = np.ma.empty(length, dtype=np.float16)
    merged[:] = np.ma.masked

    # Adds the first signal.
    first_offset = round((first.begins_at - begins_at).total_seconds() * frequency)
    merged[first_offset:first_offset + len(first.signal)] = first.signal

    # Adds the non-conflicting values of the second signal.

    second_offset = round((second.begins_at - begins_at).total_seconds() * frequency)
    second_in_merged = merged[second_offset:second_offset + len(second.signal)]

    # True if no conflict
    non_conflict_mask = second_in_merged.mask & np.invert(second.signal.mask)
    second_in_merged[non_conflict_mask] = second.signal[non_conflict_mask]

    # Adds the conflicting values of the second signal.

    conflict_mask = np.invert(second_in_merged.mask) & np.invert(second.signal.mask)

    if merge_strategy == MergeStrategy.USE_SECOND:
        second_in_merged[conflict_mask] = second.signal[conflict_mask]
    elif merge_strategy == MergeStrategy.MASKED:
        second_in_merged[conflict_mask] = np.ma.masked
    elif merge_strategy == MergeStrategy.MEAN:
        first_values = second_in_merged[conflict_mask].astype(np.float32)
        second_values = second.signal[conflict_mask].astype(np.float32)
        second_in_merged[conflict_mask] = (first_values + second_values) / 2.0
    else:
        assert merge_strategy == MergeStrategy.USE_FIRST

    return Signal(
        network_frequency=network_frequency,
        signal_frequency=first.signal_frequency,
        signal=merged,
        begins_at=begins_at,
    )
