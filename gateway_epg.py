"""Filtered XMLTV generation for native-player users."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pathlib
import sqlite3
import xml.etree.ElementTree as ET

from gateway_catalog import GatewayStream


_SQL_CHUNK_SIZE = 400
_GUIDE_HISTORY_HOURS = 12
_GUIDE_DAYS = 7


class EpgUnavailableError(RuntimeError):
    """Raised when the shared neTV EPG database is not ready."""


def _xml_text(value: object) -> str:
    text = str(value or "")
    return "".join(
        char
        for char in text
        if char in ("\t", "\n", "\r") or ord(char) >= 0x20
    )


def _xmltv_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y%m%d%H%M%S +0000")


def _chunks(values: list[str]) -> list[list[str]]:
    return [
        values[index : index + _SQL_CHUNK_SIZE]
        for index in range(0, len(values), _SQL_CHUNK_SIZE)
    ]


def build_filtered_xmltv(
    database_path: pathlib.Path,
    streams: list[GatewayStream],
    now: datetime | None = None,
) -> bytes:
    """Build XMLTV containing only guide entries for the supplied streams."""
    stream_channels: dict[str, dict[str, object]] = {}
    for stream in streams:
        channel_id = str(stream.public.get("epg_channel_id") or "")
        if not channel_id:
            continue
        channel = stream_channels.setdefault(
            channel_id,
            {
                "name": stream.public.get("name") or channel_id,
                "icon": stream.public.get("stream_icon") or "",
                "source_ids": set(),
            },
        )
        source_ids = channel["source_ids"]
        if isinstance(source_ids, set):
            source_ids.add(stream.source_id)

    root = ET.Element("tv", {"generator-info-name": "neTV"})
    if not stream_channels:
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if not database_path.exists():
        raise EpgUnavailableError("EPG database is not ready")

    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise EpgUnavailableError("EPG database is not available") from exc

    channel_ids = list(stream_channels)
    current = now or datetime.now(UTC)
    start_timestamp = (current - timedelta(hours=_GUIDE_HISTORY_HOURS)).timestamp()
    stop_timestamp = (current + timedelta(days=_GUIDE_DAYS)).timestamp()
    programs: list[sqlite3.Row] = []
    try:
        connection.execute("PRAGMA query_only=ON")
        for chunk in _chunks(channel_ids):
            placeholders = ",".join("?" for _ in chunk)
            channel_rows = connection.execute(
                f"""
                SELECT channels.id, channels.name, icons.url
                FROM channels
                LEFT JOIN icons ON icons.channel_id = channels.id
                WHERE channels.id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in channel_rows:
                stream_channels[row["id"]]["name"] = row["name"] or row["id"]
                if row["url"]:
                    stream_channels[row["id"]]["icon"] = row["url"]

            programs.extend(
                connection.execute(
                    f"""
                    SELECT channel_id, title, start_ts, stop_ts, desc, source_id
                    FROM programs
                    WHERE channel_id IN ({placeholders})
                      AND stop_ts > ?
                      AND start_ts < ?
                    ORDER BY channel_id, start_ts
                    """,
                    [*chunk, start_timestamp, stop_timestamp],
                ).fetchall()
            )
    except sqlite3.Error as exc:
        raise EpgUnavailableError("EPG database schema is not ready") from exc
    finally:
        connection.close()

    for channel_id, metadata in stream_channels.items():
        channel_element = ET.SubElement(root, "channel", {"id": _xml_text(channel_id)})
        ET.SubElement(channel_element, "display-name").text = _xml_text(metadata["name"])
        icon = _xml_text(metadata["icon"])
        if icon:
            ET.SubElement(channel_element, "icon", {"src": icon})

    selected_programs: dict[str, list[sqlite3.Row]] = {}
    for row in programs:
        channel = stream_channels.get(row["channel_id"])
        if channel is None:
            continue
        preferred_sources = channel["source_ids"]
        channel_programs = selected_programs.setdefault(row["channel_id"], [])
        overlap_index = next(
            (
                index
                for index, existing in enumerate(channel_programs)
                if row["start_ts"] < existing["stop_ts"]
                and row["stop_ts"] > existing["start_ts"]
            ),
            None,
        )
        if overlap_index is not None:
            overlapping = [
                existing
                for existing in channel_programs
                if row["start_ts"] < existing["stop_ts"]
                and row["stop_ts"] > existing["start_ts"]
            ]
            if (
                isinstance(preferred_sources, set)
                and row["source_id"] in preferred_sources
                and not any(
                    existing["source_id"] in preferred_sources
                    for existing in overlapping
                )
            ):
                channel_programs[:] = [
                    existing
                    for existing in channel_programs
                    if existing not in overlapping
                ]
                channel_programs.append(row)
            continue
        channel_programs.append(row)

    for channel_programs in selected_programs.values():
        for row in sorted(channel_programs, key=lambda program: program["start_ts"]):
            programme = ET.SubElement(
                root,
                "programme",
                {
                    "start": _xmltv_time(row["start_ts"]),
                    "stop": _xmltv_time(row["stop_ts"]),
                    "channel": _xml_text(row["channel_id"]),
                },
            )
            ET.SubElement(programme, "title").text = _xml_text(row["title"] or "Unknown")
            if row["desc"]:
                ET.SubElement(programme, "desc").text = _xml_text(row["desc"])

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
