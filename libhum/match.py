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

from typing import List, Optional, Tuple

import attrs

import numpy as np

from libhum.signal import ENFSignal


MIN_MATCHING_DURATION = datetime.timedelta(minutes=3)
MIN_MATCHING_RATIO = 0.2
MIN_MATCHING_COEFF = 0.8


@attrs.define
class ENFMatch:
    # The offset to apply to the target signal to match the reference.
    offset: datetime.timedelta

    # The total duration of the matching signals (i.e. invalid values are ignored during the
    # computation of the correlation coefficient).
    matching_duration: datetime.timedelta

    # The Pearson's correlation coefficient for this match.
    corr_coeff: float


def match_signals(
    ref: ENFSignal,
    target: ENFSignal,
    max_matches: Optional[int],
    step: datetime.timedelta = datetime.timedelta(seconds=1),
) -> List[ENFMatch]:
    if ref.signal_frequency != target.signal_frequency:
        raise ValueError("signal frequencies should be identical.")

    frequency = ref.signal_frequency

    ref_len = len(ref.signal)
    target_len = len(target.signal)

    min_matching_len = math.ceil(MIN_MATCHING_DURATION.total_seconds() * frequency)

    min_offset = - target_len + min_matching_len
    max_offset = ref_len - min_matching_len

    step_offset = math.ceil(step.total_seconds() * frequency)

    coefficients = (
        _corr_coeff(
            ref.signal[max(0, offset):offset + target_len],
            target.signal[max(0, -offset):ref_len - offset]
        )
        for offset in range(min_offset, max_offset + 1, step_offset)
    )

    matches = (
        ENFMatch(
            offset=datetime.timedelta(seconds=min_offset + i * step_offset / frequency),
            matching_duration=datetime.timedelta(seconds=matching_len / frequency),
            corr_coeff=coefficient
        )
        for i, (coefficient, matching_len)
        in enumerate(coefficients)
        if coefficient >= MIN_MATCHING_COEFF and matching_len >= min_matching_len
    )

    sorted_matches = sorted(matches, key=lambda match: match.corr_coeff, reverse=True)

    if max_matches is None:
        return sorted_matches
    else:
        return sorted_matches[:max_matches]


def _corr_coeff(a: np.ma.masked_array, b: np.ma.masked_array) -> Tuple[float, int]:
    """
    Computes the Pearson's correlation coefficient of two masked arrays.

    Ignore the masked samples and returns the total number of non-masked samples in the two signals.
    """

    assert len(a) == len(b)

    common_mask = a.mask | b.mask
    n_non_masked = len(a) - np.sum(common_mask)

    if n_non_masked < 1:
        return np.nan, n_non_masked

    masked_a = np.ma.masked_where(common_mask, a)
    masked_b = np.ma.masked_where(common_mask, b)

    mean_a, mean_b = np.mean(masked_a), np.mean(masked_b)
    std_a, std_b = np.std(masked_a), np.std(masked_b)

    numerator = np.sum((masked_a - mean_a) * (masked_b - mean_b))
    denominator = std_a * std_b * n_non_masked

    if denominator == 0:
        return np.nan, n_non_masked

    return numerator / denominator, n_non_masked