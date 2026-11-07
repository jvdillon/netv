#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"

Align HDHomeRun M3U with XMLTV guide data.

Takes an M3U playlist from HDHomeRun and aligns channel IDs with an XMLTV
guide file (e.g., from zap2xml.py). Outputs a new M3U with tvg-id attributes
set for EPG matching.

Usage:
    # First, fetch your HDHomeRun lineup and generate XMLTV:
    wget http://YOUR_HDHR_IP/lineup.m3u -O lineup.m3u
    ./zap2xml.py --zip YOUR_ZIP

    # Then align them:
    ./alignm3u.py --input lineup.m3u --xmltv xmltv.xml --output ota.m3u

    # Optionally specify the URL where xmltv.xml will be served:
    ./alignm3u.py --input lineup.m3u --xmltv xmltv.xml --output ota.m3u \\
        --xmltv-url http://your-server/xmltv.xml
'''
# fmt: on

from __future__ import annotations

from typing import Protocol, cast

import argparse
import collections
import pathlib
import re
import xml.etree.ElementTree as ET


# https://en.wikipedia.org/wiki/Call_signs_in_the_United_States#Suffixes
_CALLSIGN_REGEX = re.compile(r"^([A-Z]+?)(LD|DT|CD|CA|LP|TV|FM|D)(\d*)$")


def parse_callsign(coded_callsign: str) -> tuple[str, str, int]:
    """Parse an FCC callsign.

    Args:
        coded_callsign: Callsign with its broadcast suffix and optional number.

    Returns:
        call: Station callsign.
        suffix: Broadcast suffix, or an empty string for an unknown format.
        number: Subchannel number, defaulting to one.
    """
    result = _CALLSIGN_REGEX.search(coded_callsign.upper())
    if not result:
        return coded_callsign, "", 1
    call, suffix, num = result.groups()
    if call == "KQS" and suffix == "LD":
        call, suffix = "KQSL", "LD"  # Known bug in some data.
    return call, suffix, int(num) if num else 1


def parse_m3u(path: pathlib.Path) -> list[list]:
    """Parse M3U entries without discarding incomplete records.

    Args:
        path: Playlist file to read.

    Returns:
        entries: Lists containing title, attributes, and any following URL.
    """
    with open(path) as f:
        first_line = f.readline().strip()
        if not first_line.startswith("#EXTM3U"):
            raise ValueError(f"Invalid M3U file: {path}")
        entries = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("#EXTINF:"):
                if entries:
                    entries[-1].append(line)
                continue
            attrs_str, title = line.split("#EXTINF:")[1].split(",", 1)
            attrs_str = attrs_str.split("-1 ", 1)[1] if "-1 " in attrs_str else attrs_str
            attrs_list = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', attrs_str)
            attrs = dict(s.replace('"', "").split("=", 1) for s in attrs_list if "=" in s)
            entries.append([title.strip(), attrs])
    return entries


def parse_xmltv_channels(path: pathlib.Path) -> dict[str, tuple[str, ...]]:
    """Read channel display names from XMLTV.

    Args:
        path: XMLTV guide file.

    Returns:
        channels: Channel IDs mapped to their display names.
    """
    channels = {}
    for elem in ET.parse(path).getroot():
        if elem.tag == "channel":
            channel_id = elem.get("id")
            names = tuple(v.text for v in elem if v.tag == "display-name" and v.text)
            if channel_id:
                channels[channel_id] = names
    return channels


def build_lookup(xmltv_channels: dict[str, tuple[str, ...]]) -> dict[str, set[str]]:
    """Build lookup from channel number/name to channel IDs."""
    lookup: dict[str, set[str]] = collections.defaultdict(set)
    for channel_id, names in xmltv_channels.items():
        for name in names:
            lookup[name].add(channel_id)
    return lookup


def align_channels(
    m3u: list[list],
    lookup: dict[str, set[str]],
) -> tuple[list[list], list[list]]:
    """Align M3U entries with XMLTV channel IDs in place.

    Args:
        m3u: Playlist entries containing titles, attributes, and URLs.
        lookup: Channel names and numbers mapped to guide IDs.

    Returns:
        aligned: Original playlist with matched guide IDs added.
        missing: Complete entries with no matching guide ID.
    """
    missing = []
    for entry in m3u:
        if len(entry) < 3:
            continue
        title, attrs, _ = entry
        chan_num = attrs.get("channel-number", "")
        chan_name = attrs.get("tvg-name", title)

        # Normalize channel number (some have leading digit for ATSC3)
        if chan_num and float(chan_num) > 100:
            chan_num = str(float(chan_num[1:]))

        candidates_num = tuple(lookup.get(chan_num, ()))
        candidates_name = tuple(lookup.get(chan_name, ()))

        # Priority: exact match on number > exact match on name > any match.
        if len(candidates_num) == 1:
            attrs["tvg-id"] = candidates_num[0]
        elif len(candidates_name) == 1:
            attrs["tvg-id"] = candidates_name[0]
        elif candidates_num:
            attrs["tvg-id"] = candidates_num[0]
        elif candidates_name:
            attrs["tvg-id"] = candidates_name[0]
        else:
            missing.append(entry)

    return m3u, missing


def write_m3u(
    m3u: list[list],
    output: pathlib.Path,
    xmltv_url: str,
    group_prefix: str = "OTA",
) -> None:
    """Write an aligned M3U file.

    Args:
        m3u: Playlist entries to serialize.
        output: Destination playlist path.
        xmltv_url: Guide URL included in the playlist header.
        group_prefix: Prefix prepended to each channel's groups.
    """
    with open(output, "w") as f:
        if xmltv_url:
            print(f'#EXTM3U url-tvg="{xmltv_url}" x-tvg-url="{xmltv_url}"', file=f)
        else:
            print("#EXTM3U", file=f)
        for entry in m3u:
            if len(entry) < 3:
                continue
            title, attrs, url = entry
            # Use tvg-name as title if available.
            title = attrs.get("tvg-name", title)
            # Build group-title.
            groups = [group_prefix]
            if "group-title" in attrs:
                groups.append(attrs["group-title"])
            # Mark ATSC3 channels.
            if "channel-id" in attrs and float(attrs["channel-id"]) >= 100:
                groups.append("ATSC3")
            attrs["group-title"] = " | ".join(groups)
            # Format attributes.
            attrs_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
            print(f"#EXTINF:-1 {attrs_str},{title}", file=f)
            print(url, file=f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align HDHomeRun M3U with XMLTV guide data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    wget http://192.168.1.100/lineup.m3u -O lineup.m3u
    ./zap2xml.py --zip 90210
    ./alignm3u.py --input lineup.m3u --xmltv xmltv.xml --output ota.m3u
        """,
    )
    _add_arguments(parser)
    args = cast(Flags, parser.parse_args())

    # Parse inputs.
    print(f"Reading M3U: {args.input}")
    m3u = parse_m3u(args.input)
    print(f"  Found {len(m3u)} channels")

    print(f"Reading XMLTV: {args.xmltv}")
    xmltv_channels = parse_xmltv_channels(args.xmltv)
    print(f"  Found {len(xmltv_channels)} channels")

    # Build lookup and align.
    lookup = build_lookup(xmltv_channels)
    m3u, missing = align_channels(m3u, lookup)

    if missing:
        print(f"\nUnable to align {len(missing)} channels:")
        for entry in missing:
            title, attrs = entry[0], entry[1]
            num = attrs.get("channel-number", "?")
            print(f"  {num}: {title}")

    # Write output.
    print(f"\nWriting: {args.output}")
    write_m3u(m3u, args.output, args.xmltv_url, args.group)
    aligned = len(m3u) - len(missing)
    print(f"  Aligned {aligned}/{len(m3u)} channels")
    return 0


class Flags(Protocol):
    """Playlist alignment command-line arguments."""

    input: pathlib.Path
    xmltv: pathlib.Path
    output: pathlib.Path
    xmltv_url: str
    group: str


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        "-i",
        type=pathlib.Path,
        required=True,
        help="Input M3U file from HDHomeRun",
    )
    parser.add_argument(
        "--xmltv",
        "-x",
        type=pathlib.Path,
        required=True,
        help="XMLTV guide file (e.g., from zap2xml.py)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        required=True,
        help="Output M3U file with aligned tvg-id",
    )
    parser.add_argument(
        "--xmltv-url",
        type=str,
        default="",
        help="URL where XMLTV file will be served (for M3U header)",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="OTA",
        help="Group prefix for channels (default: OTA)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
