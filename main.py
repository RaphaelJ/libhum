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
from libhum.match import MatchBackend, match_signals


def main():
    parser = argparse.ArgumentParser(
        "libhum", description="ENF (Electric Network Frequency) computation and analysis."
    )

    subparsers = parser.add_subparsers(required=True)

    compute_enf = subparsers.add_parser(
        "compute_enf",
        description="computes the ENF signal of a single audio file."
    )
    compute_enf.set_defaults(handler=_compute_enf_handler)
    compute_enf.add_argument("wav_file", type=str)
    compute_enf.add_argument("--network-frequency", "-f", type=float, default=50.0)

    match_enf = subparsers.add_parser(
        "match_enf",
        description="match two audio signals using the signals' ENF."
    )
    match_enf.set_defaults(handler=_match_enf_handler)
    match_enf.add_argument("file_a", type=str)
    match_enf.add_argument("file_b", type=str)
    match_enf.add_argument("--network-frequency", "-f", type=float, default=50.0)
    match_enf.add_argument("--max-matches", type=int, default=1)
    match_enf.add_argument("--opencl", action="store_true", default=False)
    match_enf.add_argument("--plot", action="store_true", default=False)

    args = parser.parse_args()

    args.handler(args)


def _compute_enf_handler(args: argparse.Namespace):
    compute_enf(*read_audio(args.wav_file), network_frequency=args.network_frequency).plot()


def _match_enf_handler(args: argparse.Namespace):
    enf_a = compute_enf(*read_audio(args.file_a), network_frequency=args.network_frequency).enf
    enf_b = compute_enf(*read_audio(args.file_b), network_frequency=args.network_frequency).enf

    if args.opencl:
        backend = MatchBackend.OPENCL
    else:
        backend = MatchBackend.NUMPY

    matches = match_signals(enf_a, enf_b, max_matches=args.max_matches, backend=backend)

    for match in matches:
        print(
            f"Corr. coeff.: {match.corr_coeff:>6.3f}\t" +
            f"Offset: {match.offset}\t" +
            f"Duration: {match.duration}"
        )

        if args.plot:
            match.plot(enf_a, enf_b)


if __name__ == "__main__":
    main()