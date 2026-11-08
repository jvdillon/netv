#!/bin/sh
# ruff: noqa: EXE003, D300, D205, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P -m netv.tools.debug_remap "$@"

Debug remap scoring for one channel.

Prints the query variants remap_epg.py builds for a channel and the
highest-scoring guide candidates, scored exactly as remap_epg.py scores them.

Usage:
    python -m netv.tools.debug_remap --cache-dir ~/netv/cache "US| SYFY HD" Syfy.us
'''
# fmt: on

from __future__ import annotations

from typing import Protocol, cast

import argparse
import pathlib

from netv import cache
from netv.tools import remap_epg


def main() -> int:
    """Print query variants and the highest-scoring guide candidates.

    Returns:
        exit_code: Process exit status.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    variants = remap_epg.query_variants(flags.name, flags.current_id)
    for variant in variants:
        print("VARIANT", variant)
    candidates = remap_epg.load_candidates(flags.cache_dir / "epg.db", flags.cache_dir)
    available = frozenset(remap_epg.source_countries(flags.cache_dir).values())
    scored = remap_epg.candidate_scores(variants, candidates, available)
    for score, candidate, from_name, _ in scored[:8]:
        print(
            f"{score:6.0f} from_name={from_name} {candidate['id']:38s} "
            f"name_tokens={candidate['name_tokens']} id_tokens={candidate['id_tokens']} "
            f"nc={candidate['name_country']!r} ic={candidate['id_country']!r} "
            f"shift={candidate['timeshift']}",
        )
    return 0


class Flags(Protocol):
    """Channel identity supplied on the command line."""

    name: str
    current_id: str
    cache_dir: pathlib.Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    cache.initialize_paths()
    parser.add_argument("name", help="Playlist channel name")
    parser.add_argument("current_id", nargs="?", default="", help="Current guide ID, if any")
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=cache.CACHE_DIR,
        help="netv data directory containing epg.db and server_settings.json (default: %(default)s)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
