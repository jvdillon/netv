#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"

Debug remap scoring for one channel.

Run from the netv checkout root, where cache/epg.db lives, as a module so
the root is importable: python -m tools.debug_remap NAME [CURRENT_ID].
'''
# fmt: on

from __future__ import annotations

from typing import Protocol, cast

import argparse
import pathlib

from tools import remap_epg


def main() -> int:
    """Print query variants and the highest-scoring guide candidates.

    Returns:
        exit_code: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 2)[2] if __doc__ else None)
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    variants = remap_epg.query_variants(flags.name, flags.current_id)
    for variant in variants:
        print("VARIANT", variant)
    candidates = remap_epg.load_candidates(pathlib.Path("cache/epg.db"))
    scored = remap_epg.candidate_scores(variants, candidates)
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


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name")
    parser.add_argument("current_id", nargs="?", default="")


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
