"""EPG parsing: Program, EPGData, XMLTV parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import contextlib
import gzip
import logging
import re
import time

import defusedxml.ElementTree as ET  # Safe XML parsing

from epg_db import Program
from util import safe_urlopen

import epg_db


log = logging.getLogger(__name__)


@dataclass(slots=True)
class EPGData:
    channels: dict[str, str] = field(default_factory=dict)  # id -> name
    icons: dict[str, str] = field(default_factory=dict)  # id -> icon url
    programs: dict[str, list[Program]] = field(default_factory=dict)  # channel_id -> programs


def parse_epg_time(s: str) -> datetime:
    """Parse XMLTV time format: 20241130120000 +0000 or 20241130120000+0530."""
    s = s.replace(" ", "")
    if len(s) >= 14:
        dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S")
        if len(s) > 14:
            tz_str = s[14:]
            sign = -1 if tz_str[0] == "-" else 1
            tz_hours = int(tz_str[1:3]) if len(tz_str) >= 3 else 0
            tz_mins = int(tz_str[3:5]) if len(tz_str) >= 5 else 0
            offset = timedelta(hours=tz_hours, minutes=tz_mins)
            dt = dt.replace(tzinfo=timezone(sign * offset))
        return dt
    return datetime.now(UTC)


def _sanitize_epg_xml(xml_str: str) -> str:
    """Try to fix corrupted EPG XML by extracting valid elements."""
    channels = re.findall(r"<channel\s+[^>]*>.*?</channel>", xml_str, re.DOTALL)
    programmes = re.findall(
        r'<programme\s+start="[^"<>]+"\s+stop="[^"<>]+"\s+channel="[^"<>]+"[^>]*>.*?</programme>',
        xml_str,
        re.DOTALL,
    )
    log.info("Sanitized EPG: extracted %d channels, %d programmes", len(channels), len(programmes))
    return '<?xml version="1.0"?>\n<tv>\n' + "\n".join(channels) + "\n".join(programmes) + "\n</tv>"


def fetch_epg(
    epg_url: str,
    cache_dir: Path,
    timeout: int = 120,
    source_id: str = "",
) -> int:
    """Fetch and parse XMLTV EPG data directly into sqlite.

    Returns number of programs inserted.
    """
    with safe_urlopen(epg_url, timeout=timeout) as resp:
        content = resp.read()
        with contextlib.suppress(Exception):
            content = gzip.decompress(content)
        xml_str = content.decode("utf-8")

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        debug_file = cache_dir / f"epg_debug_{int(time.time())}.xml"
        debug_file.write_text(xml_str)
        log.warning("EPG parse failed (%s), attempting sanitization...", e)
        try:
            sanitized = _sanitize_epg_xml(xml_str)
            root = ET.fromstring(sanitized)
            log.info("Sanitized EPG parsed successfully")
        except ET.ParseError as e2:
            log.error("Sanitized EPG also failed: %s", e2)
            raise

    # Parse channels directly into sqlite
    channel_ids: set[str] = set()
    for ch in root.findall("channel"):
        ch_id = ch.get("id", "")
        channel_ids.add(ch_id)
        name_el = ch.find("display-name")
        name = name_el.text if name_el is not None and name_el.text else ch_id
        epg_db.insert_channel(ch_id, name, source_id)
        icon_el = ch.find("icon")
        if icon_el is not None:
            epg_db.insert_icon(ch_id, icon_el.get("src", ""))

    # Parse programs in batches
    batch: list[tuple[str, str, float, float, str, str]] = []
    batch_size = 10000
    program_count = 0
    program_channel_ids: set[str] = set()

    for prog in root.findall("programme"):
        ch_id = prog.get("channel", "")
        program_channel_ids.add(ch_id)
        start_str = prog.get("start", "")
        stop_str = prog.get("stop", "")

        title_el = prog.find("title")
        title = title_el.text if title_el is not None and title_el.text else "Unknown"

        desc_el = prog.find("desc")
        desc = desc_el.text if desc_el is not None and desc_el.text else ""

        try:
            start = parse_epg_time(start_str)
            stop = parse_epg_time(stop_str)
        except Exception:
            continue

        batch.append((ch_id, title, start.timestamp(), stop.timestamp(), desc, source_id))
        program_count += 1

        if len(batch) >= batch_size:
            epg_db.insert_programs(batch)
            batch.clear()

    if batch:
        epg_db.insert_programs(batch)

    epg_db.commit()
    log.debug(
        "EPG parsed: %d channels, %d unique program channel IDs, %d programs",
        len(channel_ids),
        len(program_channel_ids),
        program_count,
    )
    return program_count


def get_programs_in_range(
    epg: EPGData, channel_id: str, start: datetime, end: datetime, preferred_source_id: str = ""
) -> list[Program]:
    """Get programs for a channel within a time range.

    Args:
        epg: EPG data
        channel_id: Channel to get programs for
        start: Start of time range
        end: End of time range
        preferred_source_id: If set, prefer programs from this source when there are duplicates
    """
    programs = epg.programs.get(channel_id, [])
    matching = [p for p in programs if p.stop > start and p.start < end]

    if not preferred_source_id or len(matching) <= 1:
        return matching

    # Deduplicate overlapping programs, preferring the preferred source
    result: list[Program] = []
    for p in matching:
        dominated = False
        for i, existing in enumerate(result):
            if p.start < existing.stop and p.stop > existing.start:
                if p.source_id == preferred_source_id and existing.source_id != preferred_source_id:
                    result[i] = p
                dominated = True
                break
        if not dominated:
            result.append(p)
    return sorted(result, key=lambda p: p.start)
