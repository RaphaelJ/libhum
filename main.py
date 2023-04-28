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


import argparse

from libhum.analysis import compute_enf
from libhum.io import read_audio


def main():
    parser = argparse.ArgumentParser(
        "libhum", description="ENF (Electric Network Frequency) computation and analysis."
    )

    subparsers = parser.add_subparsers(required=True)

    compute_enf = subparsers.add_parser("compute_enf")
    compute_enf.set_defaults(handler=_compute_enf_handler)
    compute_enf.add_argument("wav_file", type=str)
    compute_enf.add_argument("--network-frequency", "-f", type=float, default=50.0)

    args = parser.parse_args()

    args.handler(args)


def _compute_enf_handler(args: argparse.Namespace):
    compute_enf(*read_audio(args.wav_file), network_frequency=args.network_frequency).plot()


if __name__ == "__main__":
    main()