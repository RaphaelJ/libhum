import datetime
import csv

from typing import List, Tuple

import numpy as np

from sortedcontainers import SortedDict

from libhum.signal import ENFSignal

def read_weti(path: str, frequency: float) -> ENFSignal:
    """
    Reads a reference ENF signal from a CSV file produced by the Wind Energy Technology Instiute.

    See https://osf.io/jbk82/.
    """

    def parse_weti_value(value: List[str]) -> Tuple[datetime.datetime, float]:
        year, month, day, hour, minute = tuple(int(v) for v in value[:5])

        second_fract = float(value[5])
        if second_fract >= 60:
            dt_offset = datetime.timedelta(seconds=60)
            second_fract -= 60
        else:
            dt_offset = datetime.timedelta(0)

        second = int(second_fract)
        microsecond = int((second_fract - second) * 1_000_000)

        freq = float(value[6])

        dt = datetime.datetime(year, month, day, hour, minute, second, microsecond) + dt_offset
        return dt, freq

    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=";")
        values = SortedDict(parse_weti_value(value) for value in reader)

    # Samples the sparse signal at the requested frequency.

    def interpolate_value(at: datetime.datetime) -> float:
        """Linear interpolates the signal value based on the 2 closest values."""
        left_dt, left_val = values.peekitem(values.bisect_left(at))
        right_dt, right_val = values.peekitem(values.bisect_right(at))
        return np.interp(
            0,
            [(left_dt - at).total_seconds(), (right_dt - at).total_seconds()],
            [left_val, right_val]
        )

    def round_datetime(dt: datetime.datetime) -> datetime.datetime:
        if dt.microsecond < 500_000:
            return dt.replace(microsecond=0)
        else:
            return dt.replace(microsecond=0) + datetime.timedelta(seconds=1)

    begins_at = round_datetime(values.peekitem(0)[0])
    ends_at = round_datetime(values.peekitem(-1)[0])

    duration = ends_at - begins_at

    sampling_rate = datetime.timedelta(seconds=1.0 / frequency)
    n_samples = duration // sampling_rate

    signal = np.fromiter(
        (interpolate_value(begins_at + sampling_rate * i) for i in range(0, n_samples)),
        dtype=np.half,
    )

    return ENFSignal(50.0, begins_at, frequency, signal)
