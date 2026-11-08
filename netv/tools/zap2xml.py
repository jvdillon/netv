#!/bin/sh
# ruff: noqa: EXE003, D300, D205, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P -m netv.tools.zap2xml "$@"
Fetch TV guide data from zap2it/gracenote in XMLTV format.

Scrapes the internal JSON feed from zap2it/gracenote to generate XMLTV guide
data. The site occasionally returns 400 errors for certain time windows; this
tool ignores those and continues fetching available data.

Written with only standard library dependencies.

Writes xmltv.xml and a .zap2xml/ fetch cache into --path (default: the
current directory).

Usage:
    zap2xml --zip 90210 --days 7

Cron example:
    0 0 * * * cd /path/to/guide && /path/to/bin/zap2xml --zip 90210
'''
# fmt: on

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypedDict, cast
from urllib import error, parse, request

import argparse
import datetime
import gzip  # lzma(80%), bz2(78%), gzip(75%) but gzip was fastest.
import json
import math
import pathlib
import re
import time
import xml.etree.ElementTree as xml


# https://en.wikipedia.org/wiki/Call_signs_in_the_United_States#Suffixes
# Note: Doesn't correctly handly boosters.
_CALLSIGN_REGEX = re.compile(r"^([A-Z]+?)(LD|DT|CD|CA|LP|TV|FM|D)(\d*)$")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="This tool is noisy to stdout; with cron use chronic from moreutils.",
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    working_dir = flags.path

    cache_dir = working_dir / ".zap2xml"
    if not cache_dir.is_dir():
        cache_dir.mkdir()

    url_flags = {k[len("zap_") :]: v for k, v in vars(flags).items() if k.startswith("zap_")}
    url_flags["lineupId"] = f"{flags.zap_country}-{flags.zap_headendId}-DEFAULT"

    # Start time parameter is now rounded down to nearest `zap_timespan`, in s.
    zap_time = int(datetime.datetime.now().timestamp())
    print(f"Local time:     {zap_time}  {strf_time_int(zap_time)}")

    zap_time_window = flags.zap_timespan * 3_600
    zap_time = (zap_time // zap_time_window) * zap_time_window
    print(f"First zap time: {zap_time}  {strf_time_int(zap_time)}")

    remove_stale_cache(cache_dir, zap_time)

    # https://wiki.xmltv.org/index.php/XMLTVFormat
    # https://github.com/XMLTV/xmltv/blob/master/xmltv.dtd#L529

    out = add_xml_child(
        parent=None,
        tag="tv",
        attrib={
            "source-info-url": f"https://{flags.base_url}/grid-affiliates.html?aid=gapzap",
            "source-info-name": "zap2it",
            "generator-info-name": "zap2xml.py",
            "generator-info-url": "https://github.com/jvdillon/netv",
        },
    )

    channel_map = {}  # Only used for debugging.
    done_channels = False

    # Fetch data in `zap_timespan` chunks.
    if flags.days > 15:
        raise ValueError(f"Can only collect at most 15 days; {flags.days} too large.")
    num_fetch = math.ceil(flags.days * 24 / flags.zap_timespan)
    for i in range(num_fetch):
        i_time = zap_time + (i * zap_time_window)
        print(f"Getting data:   {i_time}  {strf_time_int(i_time)}")

        url = f"https://{flags.base_url}/api/grid?"
        url += parse.urlencode({**url_flags, "time": i_time})

        result = get_cached(cache_dir, i_time, flags.delay, url)
        json_result = json.loads(result)

        if not done_channels:
            done_channels = True
            for c_in in json_result["channels"]:
                # {'affiliateCallSign': 'null',
                #  'affiliateName': 'AMERICAN BROADCASTING COMPANY',
                #  'callSign': 'KXTVDT',
                #  'channelId': '20775',
                #  'channelNo': '10.1',
                #  'id': '2077555',
                #  'stationFilters': ['filter-sports'],
                #  'stationGenres': [False],
                #  'thumbnail': '//zap2it.tmsimg.com/h3/NowShowing/20775/s28708_ll_h15_ac.png?w=55'}.
                channel_key = get_channel_key(c_in)
                channel_display_name = " - ".join(
                    [
                        c_in["affiliateName"].title(),  # Eg, "CATCHY COMEDY"
                        parse_callsign(c_in["callSign"]),  # Eg, "KOVR-DT-5"
                        c_in["channelNo"],  # Eg., "13.5"
                    ],
                )
                channel_map[channel_key] = channel_display_name
                c_out = add_xml_child(
                    parent=out,
                    tag="channel",
                    id=channel_key,
                )
                _ = add_xml_child(
                    parent=c_out,
                    tag="display-name",
                    text=channel_display_name,
                )
                _ = add_xml_child(
                    parent=c_out,
                    tag="icon",
                    src=f"https:{c_in['thumbnail'].split('?')[0]}",
                )
            channel_map = dict(sorted(channel_map.items(), key=lambda kv: kv[0]))

        f = add_programme_tvimate if flags.tvimate else add_programme
        for c_in in json_result["channels"]:
            channel_key = get_channel_key(c_in)
            for event in c_in["events"]:
                f(out, event, channel_key)

    # https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.indent
    # Note: xml.indent must be done last.
    xml.indent(out, space="\t", level=0)
    with pathlib.Path.open((working_dir / "xmltv.xml").resolve(), "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(xml.tostring(out, encoding="UTF-8"))

    return 0


def get_cached(
    cache_dir: pathlib.Path,
    timestamp: int,
    delay: int,
    url: str,
) -> bytes:
    cache_path = (cache_dir / str(timestamp)).with_suffix(".json.gz")
    if cache_path.is_file():
        print(f"Cached: {url}")
        with gzip.open(cache_path, "rb") as f:
            return f.read()

    print(f"Fetching: '{url}'.")
    if not url.startswith(("http:", "https:")):
        raise ValueError(f"URL '{url}' must start with 'http:' or 'https:'") from None
    req = request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Sec-Ch-Ua": ('"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"'),
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Linux"',
            # "Accept-Encoding": "br, gzip, deflate, zstd, identity".
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "text/plain;charset=UTF-8",
            "Priority": "u=1, i",
        },
    )
    try:
        response = request.urlopen(req)
        # Typeshed types urlopen's return as Any.
        result = cast(bytes, response.read())
    except error.HTTPError as e:
        if e.code != 400:
            e.add_note(f'Url is "{url}".')
            raise
        with e:
            print("Got a 400 error! Ignoring it.")
            result = b'{"note": "Got a 400 error at this time, skipping.","channels": []}'
    with gzip.open(cache_path, "wb") as f:
        f.write(result)
    time.sleep(delay)
    return result


def remove_stale_cache(cache_dir: pathlib.Path, zap_time: int) -> None:
    for p in sorted(cache_dir.glob("*")):
        data_time = int(p.name.removesuffix("".join(p.suffixes)))
        file_time = int(p.stat().st_mtime)
        is_irrelevant = data_time < zap_time
        is_1day_expired = _expired(3, 1, data_time, file_time, zap_time)
        is_7day_expired = _expired(7, 7, data_time, file_time, zap_time)
        if is_irrelevant or is_1day_expired or is_7day_expired:
            print(
                f"Removing stale cache file: name={p.name} zap_time={zap_time} "
                f"data_time={data_time} file_time={file_time} "
                f"is_irrelevant={is_irrelevant} is_1day_expired={is_1day_expired} "
                f"is_7day_expired={is_7day_expired} "
                f"file_time_str={strf_time_int(file_time)} "
                f"data_time_str={strf_time_int(data_time)}",
            )
            p.unlink()


def add_programme(
    out: xml.Element,
    event: Mapping[str, Any],
    channel_key: str,
) -> None:
    # {'callSign': 'KCRADT2', 'duration': '30', 'startTime': '2025-04-20T18:00:00Z', 'endTime': '2025-04-20T18:30:00Z',
    # 'thumbnail': 'p1119901_e_v9_ab', 'channelNo': '3.2', 'filter': [], 'seriesId': 'SH00001996', 'rating': 'TV-G', 'flag': [], 'tags': ['CC'],
    # 'program': {
    #     'title': 'Happy Days',
    #     'id': 'EP000019960180',
    #     'tmsId': 'EP000019960180',
    #     'shortDesc': 'Richie is selected to become a contestant on a popular game show with a chance to win $3,200.',
    #     'season': '2', 'releaseYear': None, 'episode': '9', 'episodeTitle': 'Big Money', 'seriesId': 'SH00001996', 'isGeneric': '0'}
    # }.
    # https://tvlistings.gracenote.com/overview-affiliates.html?programSeriesId=SH00001996&tmsId=EP000019960180&aid=lat

    prog_out = add_xml_child(
        parent=out,
        tag="programme",
        start=strf_time_str(event["startTime"]),
        stop=strf_time_str(event["endTime"]),
        channel=channel_key,
    )

    prog_in = event["program"]

    if prog_in["title"]:
        _ = add_xml_child(
            parent=prog_out,
            tag="title",
            # lang="en".
            text=prog_in["title"],
        )

    year = toint(prog_in["releaseYear"])

    if prog_in["episodeTitle"]:
        _ = add_xml_child(
            parent=prog_out,
            tag="sub-title",
            # lang="en".
            text=prog_in["episodeTitle"],
        )
    elif "filter-movie" in event["filter"]:
        if prog_in["title"] == "Movie":
            text = "TBD"
        elif year:
            text = f"Movie ({year})"
        else:
            text = "Movie"

        _ = add_xml_child(
            parent=prog_out,
            tag="sub-title",
            # lang="en".
            text=text,
        )

    if prog_in["shortDesc"]:
        _ = add_xml_child(
            parent=prog_out,
            tag="desc",
            # lang="en".
            text=prog_in["shortDesc"],
        )

    if prog_in["season"] and prog_in["episode"]:
        # Format:
        # season_num/season_total.episode_num/episode_total.part_num/part_total
        # where "num" is zero indexed and "/total" is optional
        # and "num/total" is also optional.
        _ = add_xml_child(
            parent=prog_out,
            tag="episode-num",
            system="xmltv_ns",
            text=f"{int(prog_in['season']) - 1}.{int(prog_in['episode']) - 1}.",
        )

    if event["rating"]:
        r = add_xml_child(
            parent=prog_out,
            tag="rating",
            system="VCHIP",
        )
        _ = add_xml_child(
            parent=r,
            tag="value",
            text=event["rating"],
        )

    _ = add_xml_child(
        parent=prog_out,
        tag="length",
        units="minutes",
        text=event["duration"],
    )

    if year:
        _ = add_xml_child(
            parent=prog_out,
            tag="date",
            text=str(year),
        )

    if event["thumbnail"]:
        # Not part of xmltv spec but we're including it anyway.
        _ = add_xml_child(
            parent=prog_out,
            tag="icon",
            src=f"https://zap2it.tmsimg.com/assets/{event['thumbnail']}.jpg",
        )

    for f in event["filter"]:
        if f not in {
            "filter-family",
            "filter-movie",
            "filter-news",
            "filter-sports",
            "filter-talk",
        }:
            print(f"Novel filter '{f}'.")
            if not f.startswith("filter-"):
                continue
        _ = add_xml_child(
            parent=prog_out,
            tag="category",  # Was: "genre"
            # lang="en".
            text=f[len("filter-") :].title(),
        )

    if "Dolby Digital" in event["tags"]:
        audio = "dolby digital"
    elif "Dolby" in event["tags"]:
        audio = "dolby"
    elif "Surround" in event["tags"]:
        audio = "surround"
    elif "Stereo" in event["tags"]:
        audio = "stereo"
    elif "Mono" in event["tags"]:
        audio = "mono"
    else:
        audio = "stereo"
    r = add_xml_child(
        parent=prog_out,
        tag="audio",
    )
    _ = add_xml_child(
        parent=r,
        tag="present",
        text="yes",
    )
    _ = add_xml_child(
        parent=r,
        tag="stereo",
        text=audio,
    )
    if "DVS" in event["tags"]:
        _ = add_xml_child(
            parent=r,
            tag="stereo",
            text="bilingual",
        )
        # If False:
        #     a = strf_time_str(
        #         event["startTime"],
        #         format_str="%Y-%b-%d %_I:%M%P",
        #     )
        #     t = prog_in["title"]
        #     e = prog_in["episodeTitle"] if prog_in["episodeTitle"] else ""
        #     c = channel_map[channel_key]
        #     print(f"### {a:30s} {t:40s} {e:50s} {c:20s}")

    if "CC" in event["tags"]:
        r = add_xml_child(
            parent=prog_out,
            tag="subtitles",
            type="teletext",
        )
        _ = add_xml_child(
            parent=r,
            tag="language",
            text="English",
        )

    if "New" in event["flag"]:  # And "Live" not in event["flag"]:
        _ = add_xml_child(
            parent=prog_out,
            tag="new",
        )


def add_programme_tvimate(
    out: xml.Element,
    event: Mapping[str, Any],
    channel_key: str,
) -> None:
    prog_out = add_xml_child(
        parent=out,
        tag="programme",
        start=strf_time_str(event["startTime"]),
        stop=strf_time_str(event["endTime"]),
        channel=channel_key,
    )

    prog_in = event["program"]

    title = prog_in["title"]
    subtitle = prog_in["episodeTitle"]
    year = toint(prog_in["releaseYear"])
    season = toint(prog_in["season"])
    episode = toint(prog_in["episode"])
    description = prog_in["shortDesc"]

    if title and subtitle and "filter-sports" in event["filter"]:
        title = f"{title}: {subtitle}"
        subtitle = None
    elif not subtitle and "filter-movie" in event["filter"]:
        if title == "Movie":
            subtitle = None
        elif year:
            subtitle = f"Movie ({year})"
        else:
            subtitle = "Movie"

    if title:
        if "Live" in event["flag"]:
            if "filter-news" not in event["filter"]:
                title += " ᴸⁱᵛᵉ"
        elif "New" in event["flag"]:
            title += " ᴺᵉʷ"
        _ = add_xml_child(
            parent=prog_out,
            tag="title",
            # lang="en".
            text=title,
        )

    if season and episode:
        season_episode = f"S{season:02d}E{episode:02d}"
    else:
        season_episode = None

    short = " ".join([a_ for a_ in [season_episode, subtitle] if a_ is not None and a_])
    description = "\n".join([a_ for a_ in [short, description] if a_])
    if description:
        _ = add_xml_child(
            parent=prog_out,
            tag="desc",
            # lang="en".
            text=description,
        )

    # If event["rating"]:
    #     r = add_xml_child(
    #         parent=prog_out,
    #         tag="rating",
    #         system="VCHIP",
    #     )
    #     _ = add_xml_child(
    #         parent=r,
    #         tag="value",
    #         text=event["rating"],
    #     )

    for f in event["filter"]:
        if f not in {
            "filter-family",
            "filter-movie",
            "filter-news",
            "filter-sports",
            "filter-talk",
        }:
            print(f"Novel filter '{f}'.")
            if not f.startswith("filter-"):
                continue
        _ = add_xml_child(
            parent=prog_out,
            tag="category",  # Was: "genre"
            # lang="en".
            text=f[len("filter-") :].title(),
        )


def get_channel_key(c: CallSigned) -> str:
    # Old way:
    # return f"I{c['channelNo']}.{c['channelId']}.zap2it.com"
    return c["callSign"]


def parse_callsign(coded_callsign: str) -> str:
    result = _CALLSIGN_REGEX.search(coded_callsign.upper())
    if not result:
        raise ValueError("Expected result.")
    call, suffix, num = result.groups()
    if num == "1":
        raise ValueError('Expected num != "1".')
    if call == "KQS" and suffix == "LD":
        # Appears to be a bug in their coded callsign.
        call = "KQSL"
        suffix = "LD"
    if not num:
        num = "1"
    return f"{call}-{suffix}-{num}"


def strf_time_str(tm: str, format_str: str = "%Y%m%d%H%M%S %z") -> str:
    tm = tm.replace("Z", "+00:00")
    return parse_time_iso(tm).strftime(format_str)


def strf_time_int(timestamp: int, format_str: str = "%Y-%b-%d %_I:%M%P %z") -> str:
    return parse_time_int(timestamp).strftime(format_str)


def parse_time_iso(tm: str) -> datetime.datetime:
    tm = tm.replace("Z", "+00:00")
    return datetime.datetime.fromisoformat(tm).astimezone()


def parse_time_int(timestamp: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp).astimezone()


def add_xml_child(
    parent: xml.Element | None,
    tag: str,
    text: str | None = None,
    attrib: Mapping[str, str] | None = None,
    **extra: Any,
) -> xml.Element:
    attributes: dict[str, str] = {} if attrib is None else dict(attrib)
    if parent is None:
        # https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.Element
        el = xml.Element(tag, attributes, **extra)
    else:
        # https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.SubElement
        el = xml.SubElement(parent, tag, attributes, **extra)
    if text is not None:
        el.text = text
    return el


def toint(x: str | None, fail: int = 0) -> int:
    if x is None:
        return fail
    return int(x)


class CallSigned(TypedDict):
    """The one field of a grid channel that keys it."""

    callSign: str


class Flags(Protocol):
    """Guide fetch and output options."""

    delay: int
    base_url: str
    days: float
    path: pathlib.Path
    zap_aid: str
    zap_country: str
    zap_device: str
    zap_headendId: str
    zap_isOverride: bool
    zap_languagecode: str
    zap_pref: str
    zap_timespan: int
    zap_timezone: str
    zap_userId: str
    zap_postalCode: str
    tvimate: bool


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument(
        "--delay",
        dest="delay",
        type=int,
        default=5,
        help="Delay, in seconds, between server fetches.",
    )
    _ = parser.add_argument(
        "--url",
        dest="base_url",
        type=str,
        default="tvlistings.gracenote.com",
        # default="tvlistings.zap2it.com".
        help="Source url without http prefix.",
    )
    _ = parser.add_argument(
        "--days",
        dest="days",
        type=float,
        default=15,
        help="Num days to fetch.",
    )
    _ = parser.add_argument(
        "--path",
        dest="path",
        type=pathlib.Path,
        default=pathlib.Path(),
        help="Directory for xmltv.xml and the .zap2xml cache (default: cwd).",
    )
    _ = parser.add_argument(
        "--aid",
        dest="zap_aid",
        type=str,
        # Previously we used "gapzap" but redditors seem to have found this one.
        # https://www.reddit.com/r/cordcutters/comments/1m1iba0/zap2it_and_gracenote_listings_are_gone_again/
        default="orbebb",
        help="Raw zap2it input parameter. (Affiliate ID?)",
    )
    _ = parser.add_argument(
        "--country",
        dest="zap_country",
        type=str,
        default="USA",
        help="Country identifying the listings to fetch.",
    )
    _ = parser.add_argument(
        "--device",
        dest="zap_device",
        type=str,
        default="-",
        help="Raw zap2it input parameter.  (?)",
    )
    _ = parser.add_argument(
        "--headend-id",
        dest="zap_headendId",
        type=str,
        default="lineupId",
        help="Raw zap2it input parameter.  (?)",
    )
    _ = parser.add_argument(
        "--is-override",
        dest="zap_isOverride",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Raw zap2it input parameter.  (?)",
    )
    _ = parser.add_argument(
        "--language",
        dest="zap_languagecode",
        type=str,
        default="en",
        help="Raw zap2it input parameter.  (Language.)",
    )
    _ = parser.add_argument(
        "--pref",
        dest="zap_pref",
        type=str,
        default="",
        help="Raw zap2it input parameter.  (Preferences?)",
    )
    _ = parser.add_argument(
        "--timespan",
        dest="zap_timespan",
        type=int,
        default=3,
        help="Raw zap2it input parameter.  (Hours of data per fetch?)",
    )
    _ = parser.add_argument(
        "--timezone",
        dest="zap_timezone",
        type=str,
        default="",
        help="Raw zap2it input parameter.  (Time zone?)",
    )
    _ = parser.add_argument(
        "--user-id",
        dest="zap_userId",
        type=str,
        default="-",
        help="Raw zap2it input parameter.  (?)",
    )
    _ = parser.add_argument(
        "--zip",
        dest="zap_postalCode",
        type=str,
        required=True,
        help="The zip/postal code identifying the listings to fetch.",
    )
    _ = parser.add_argument(
        "--tvimate",
        dest="tvimate",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Guide formatted specifically for TViMate.",
    )


def _expired(
    data_days: float,
    file_days: float,
    data_time: int,
    file_time: int,
    zap_time: int,
) -> bool:
    data_time_within_limit = data_time < zap_time + data_days * 86_400
    file_time_within_limit = zap_time < file_time + file_days * 86_400
    return data_time_within_limit and not file_time_within_limit


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
