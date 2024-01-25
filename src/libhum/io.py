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
import csv
import re
import tempfile
import zipfile

from datetime import date, datetime, timedelta, timezone
from typing import List, Tuple
from zoneinfo import ZoneInfo

import audiofile
import numpy as np
import requests
import scipy

from sortedcontainers import SortedDict

from libhum.types import Signal


def read_signal(path: str) -> Signal:
    """Reads an ENF signal from a file."""

    with open(path, "rb") as f:
        return Signal.deserialize(f.read())


def write_signal(path: str, signal: Signal):
    """Writes an ENF signal to a file."""

    with open(path, "wb") as f:
        return f.write(signal.serialize())


def read_audio(path: str) -> Tuple[np.array, float]:
    """Reads an audio file and returns its single channel buffer with its sampling frequency."""

    data, frequency = audiofile.read(path, always_2d=True)
    return np.mean(data, axis=0), float(frequency)


def read_weti(path: str, frequency: float = 1.0) -> Signal:
    """
    Reads a reference ENF signal from a CSV file produced by the Wind Energy Technology Instiute.

    See https://osf.io/jbk82/.
    """

    def parse_weti_value(value: List[str]) -> Tuple[datetime, float]:
        year, month, day, hour, minute = tuple(int(v) for v in value[:5])

        second_fract = float(value[5])
        if second_fract >= 60:
            dt_offset = timedelta(seconds=60)
            second_fract -= 60
        else:
            dt_offset = timedelta(0)

        second = int(second_fract)
        microsecond = int((second_fract - second) * 1_000_000)

        freq = float(value[6])

        dt = datetime(
            year, month, day, hour, minute, second, microsecond,
            tzinfo=timezone.utc
        ) + dt_offset
        return dt, freq

    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=";")
        sparse_signal = SortedDict(parse_weti_value(value) for value in reader)

    signal, begins_at = _resample_sparse_signal(sparse_signal, frequency)

    return Signal(
        network_frequency=50.0,
        signal=signal,
        signal_frequency=frequency,
        begins_at=begins_at
    )


DEFAULT_SWISS_GRID_URL = "https://www.swissgrid.ch/content/swissgrid/en/home/operation/regulation/frequency.apicache.html?path=/content/swissgrid/en/home/operation/regulation/frequency/jcr:content/parsys/chart_copy"


def fetch_swiss_grid(url: str = DEFAULT_SWISS_GRID_URL, frequency: float = 0.1):
    """
    Fetches the latest 10 minute data from the Swiss Grid operator.

    See https://www.swissgrid.ch/en/home/operation/grid-data/current-data.html
    """

    swiss_tz = ZoneInfo("Europe/Zurich")
    network_frequency = 50.0

    resp = requests.get(url)
    resp.raise_for_status()

    values = resp.json()["data"]["series"][0]["data"]

    def parse_swiss_grid_value(timestamp: int, enf: float):
        local_dt = datetime.utcfromtimestamp(timestamp / 1000).replace(tzinfo=swiss_tz)
        utc_dt = local_dt.astimezone(timezone.utc)
        return utc_dt, enf - network_frequency

    sparse_signal = SortedDict(parse_swiss_grid_value(*v) for v in values)

    signal, begins_at = _resample_sparse_signal(sparse_signal, frequency)

    return Signal(
        network_frequency=network_frequency,
        signal=signal,
        signal_frequency=frequency,
        begins_at=begins_at
    )


DEFAULT_UK_GRID_INDEX_URL = "https://data.nationalgrideso.com/system/system-frequency-data/datapackage.json"


def fetch_uk_grid(
    month: date, index_url: str = DEFAULT_UK_GRID_INDEX_URL, frequency: float = 1.0
) -> Signal:
    """
    Fetches one month of data from the UK National Grid operator.

    See https://data.nationalgrideso.com/system/system-frequency-data
    """

    MONTHS = {
        m: i + 1
        for i, m in enumerate([
            "january", "february", "march", "april", "may", "june", "july", "august", "september",
            "october", "november", "december"
        ])
    }

    network_frequency = 50.0

    # Downloads the index

    def parse_resource_name(resource_name: str) -> date:

        match = re.match(r"([a-z]*)_([0-9]{4})_[-–]_historic_frequency_data", resource_name)

        month = MONTHS[match.group(1)]
        year = int(match.group(2))

        return date(year, month, 1)

    resp = requests.get(index_url)
    resp.raise_for_status()

    resources_map = {
        parse_resource_name(r["name"]): r
        for r in resp.json()["result"]["resources"]
    }

    # Downloads the ENF file.

    resource = resources_map[month.replace(day=1)]

    with tempfile.NamedTemporaryFile("wb") as downloaded_file:
        with requests.get(resource["path"], stream=True) as resp:
            resp.raise_for_status()

            for chunk in resp.iter_content(chunk_size=256 * 1024):
                downloaded_file.write(chunk)

        downloaded_file.flush()

        if resource["mediatype"] == "application/zip":
            with zipfile.ZipFile(downloaded_file.name) as zip_file:
                content = zip_file.read(zip_file.filelist[0]).decode("utf-8")
        else:
            assert resource["mediatype"] == "text/csv"
            with open(downloaded_file.name, "r") as text_file:
                content = text_file.read()

    # Parses the ENF file.

    csv_file = csv.reader(content.splitlines()[1:], delimiter=",")

    def parse_uk_grid_value(dt_str: str, enf_str: str) -> datetime:
        try:
            dt = datetime.fromisoformat(dt_str)
        except ValueError:
            dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S").replace(tzinfo=timezone.utc)

        dt = dt.replace(tzinfo=timezone.utc)

        enf = float(enf_str) - network_frequency

        return dt, enf

    sparse_signal = SortedDict(parse_uk_grid_value(*line) for line in csv_file)

    signal, begins_at = _resample_sparse_signal(sparse_signal, frequency)

    return Signal(
        network_frequency=network_frequency,
        signal=signal,
        signal_frequency=frequency,
        begins_at=begins_at
    )


DEFAULT_SWEDISH_GRID_URL = "https://www.svk.se/services/controlroom/v2/freq?fromDateTime={from_dt}&toDateTime={to_dt}"


def fetch_swedish_grid(
    url: str = DEFAULT_SWEDISH_GRID_URL, duration: timedelta = timedelta(hours=1),
    frequency: float = 1.0
) -> Signal:
    """
    Fetches up to one hour of data from the Swedish national grid operator.

    See https://www.svk.se/en/national-grid/the-control-room/
    """

    network_frequency = 50.0

    if duration > timedelta(hours=1):
        raise ValueError("Swedish grid does not allow fetching more than 1h of data.")

    to_dt = datetime.now(tz=timezone.utc)
    from_dt = to_dt - duration

    endpoint = url.format(from_dt=from_dt.timestamp() * 1000, to_dt=to_dt.timestamp() * 1000)
    resp = requests.get(endpoint)
    resp.raise_for_status()

    values = resp.json()["Data"]

    def parse_swedish_grid_value(value: dict):
        dt = datetime.fromtimestamp(value["x"] / 1000, tz=timezone.utc)
        enf = value["y"] - network_frequency

        return dt, enf

    sparse_signal = SortedDict(parse_swedish_grid_value(v) for v in values)

    signal, begins_at = _resample_sparse_signal(sparse_signal, frequency)

    return Signal(
        network_frequency=network_frequency,
        signal=signal,
        signal_frequency=frequency,
        begins_at=begins_at
    )


def _resample_sparse_signal(
    sparse_signal: SortedDict[datetime, float], frequency
) -> Tuple[np.ma.masked_array, datetime]:
    """
    Samples the sparse signal at the requested frequency.

    Returns the resampled signal, and the begin time.
    """

    INTERP_RANGE = 2 # Interpolates the values up to 2x the output sampling rate

    sampling_rate = timedelta(seconds=1.0 / frequency)

    interp_range_td = INTERP_RANGE * sampling_rate

    def interp_value(at: datetime) -> float:
        """Linear interpolates the signal value based on the closest values."""

        # Collects all samples within INTERP_RANGE.
        xs = list(sparse_signal.irange(at - interp_range_td, at + interp_range_td))

        if len(xs) == 0:
            return np.nan
        else:
            kind = "cubic" if len(xs) >= 4 else "linear"

            return scipy.interpolate.interp1d(
                [(x - at).total_seconds() for x in xs],
                [sparse_signal[x] for x in xs],
                kind=kind,
                bounds_error=False,
            )(0.0)

    def round_datetime(dt: datetime) -> datetime:
        if dt.microsecond < 500_000:
            return dt.replace(microsecond=0)
        else:
            return dt.replace(microsecond=0) + timedelta(seconds=1)

    begins_at = round_datetime(sparse_signal.peekitem(0)[0])
    ends_at = round_datetime(sparse_signal.peekitem(-1)[0])

    duration = ends_at - begins_at

    n_samples = duration // sampling_rate

    signal = np.ma.masked_invalid(np.fromiter(
        (interp_value(begins_at + sampling_rate * i) for i in range(0, n_samples)),
        dtype=np.float16,
    ))

    return signal, begins_at