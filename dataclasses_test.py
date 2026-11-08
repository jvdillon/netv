"""Netv records reject field reassignment; owners publish replacements."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from cache import Source
from chromecast import CastSession
from epg import Program
from ffmpeg_command import MediaInfo, SubtitleStream
from ffmpeg_session import _FastLiveStartup, _SeekSessionInfo, _SessionSnapshot, _StartupGuard
from gateway_catalog import CatalogSnapshot, GatewayStream
from handoff import Channel, HandoffPolicy, _Watch
from main import PlayerInfo
from playback_policy import LiveRecoveryPolicy, PlaybackPolicy
from xtream import XtreamClient


@pytest.mark.parametrize(
    ("record", "attribute"),
    [
        (Source(id="s", name="Source", type="m3u", url="http://source"), "id"),
        (
            CastSession(
                username="u",
                host="h",
                session_id="s",
                url="url",
                source_url="src",
                name="TV",
            ),
            "starting",
        ),
        (
            Program(
                channel_id="c",
                title="Program",
                start=datetime.fromtimestamp(0, timezone.utc),
                stop=datetime.fromtimestamp(1, timezone.utc),
            ),
            "title",
        ),
        (MediaInfo(video_codec="", audio_codec="", pix_fmt=""), "height"),
        (SubtitleStream(index=0, lang="eng", name="English"), "name"),
        (
            _SessionSnapshot(output_dir="d", process=None, seek_offset=0, subtitles=[], duration=0),
            "seek_offset",
        ),
        (
            _SeekSessionInfo(
                url="url",
                output_dir="d",
                process=None,
                subtitles=[],
                series_id=None,
                episode_id=None,
            ),
            "url",
        ),
        (
            _FastLiveStartup(
                session_id="s",
                directory="d",
                processes=[],
                started=0,
                is_disconnected=None,
            ),
            "started",
        ),
        (_StartupGuard(), "users"),
        (
            GatewayStream(
                local_id=1,
                source_id="s",
                source_type="m3u",
                upstream_id="1",
                direct_url="url",
                access_group_ids=(),
                public={},
            ),
            "local_id",
        ),
        (
            CatalogSnapshot(categories=[], category_access_ids={}, streams=[], streams_by_id={}),
            "streams",
        ),
        (Channel(stream_id="s", name="Channel", url="url"), "name"),
        (HandoffPolicy(), "min_watch_secs"),
        (_Watch(stream_id="s", position=0, first_seen=0, last_seen=0), "position"),
        (PlayerInfo(), "url"),
        (PlaybackPolicy(), "started"),
        (LiveRecoveryPolicy(), "restart_limit"),
        (XtreamClient(base_url="http://source/", username="u", password="p"), "base_url"),
    ],
)
def test_record_fields_cannot_be_reassigned(record: object, attribute: str):
    with pytest.raises(FrozenInstanceError):
        setattr(record, attribute, None)


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
