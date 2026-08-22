"""Tests for m3u_export.py and the /get.php, /xmltv.php, /live/ routes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import json

import pytest

import cache as cache_module


XTREAM_STREAM = {
    "stream_id": 42,
    "name": "News HD",
    "stream_icon": "http://logo/news.png",
    "epg_channel_id": "news.example",
    "category_ids": ["src1_5"],
    "source_id": "src1",
    "source_type": "xtream",
    "source_url": "http://provider.example",
    "source_username": "puser",
    "source_password": "ppass",
}

M3U_STREAM = {
    "stream_id": "src2_1",
    "name": "Local OTA",
    "stream_icon": "",
    "epg_channel_id": "ota.example",
    "category_ids": ["src2_ota"],
    "direct_url": "http://192.168.1.87:5004/auto/v2.1",
    "source_id": "src2",
}

CATEGORIES = [
    {"category_id": "src1_5", "category_name": "News", "source_id": "src1"},
    {"category_id": "src2_ota", "category_name": "OTA", "source_id": "src2"},
]


@pytest.fixture
def export_module(tmp_path: Path):
    """Import m3u_export with temp auth settings and a clean state."""
    import auth
    import m3u_export

    with (
        patch.object(auth, "SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch.object(auth, "USERS_DIR", tmp_path / "users"),
    ):
        (tmp_path / "users").mkdir(exist_ok=True)
        m3u_export.clear_credential_cache()
        m3u_export._active_streams.clear()
        yield m3u_export


class TestVerifyCredentials:
    def test_valid_and_cached(self, export_module):
        import auth

        auth.create_user("alice", "password123")
        assert export_module.verify_credentials("alice", "password123")
        # Second call must hit the cache, not PBKDF2
        with patch.object(auth, "verify_password", side_effect=AssertionError) as _:
            assert export_module.verify_credentials("alice", "password123")

    def test_invalid_password(self, export_module):
        import auth

        auth.create_user("alice", "password123")
        assert not export_module.verify_credentials("alice", "wrong")

    def test_wrong_password_not_served_from_cache(self, export_module):
        import auth

        auth.create_user("alice", "password123")
        assert export_module.verify_credentials("alice", "password123")
        assert not export_module.verify_credentials("alice", "different")

    def test_clear_cache_forces_reverify(self, export_module):
        import auth

        auth.create_user("alice", "password123")
        assert export_module.verify_credentials("alice", "password123")
        auth.change_password("alice", "newpassword1")
        export_module.clear_credential_cache("alice")
        assert not export_module.verify_credentials("alice", "password123")
        assert export_module.verify_credentials("alice", "newpassword1")


class TestStreamFiltering:
    def test_allowed_when_no_restrictions(self, export_module):
        with patch("auth.get_user_limits", return_value={"unavailable_groups": []}):
            streams = export_module.allowed_live_streams([XTREAM_STREAM, M3U_STREAM], "alice")
        assert len(streams) == 2

    def test_blocked_category_filtered(self, export_module):
        with patch(
            "auth.get_user_limits", return_value={"unavailable_groups": ["cat:src1_5"]}
        ):
            streams = export_module.allowed_live_streams([XTREAM_STREAM, M3U_STREAM], "alice")
        assert [s["name"] for s in streams] == ["Local OTA"]

    def test_stream_allowed_rule(self, export_module):
        assert not export_module.stream_allowed(XTREAM_STREAM, {"cat:src1_5"})
        assert export_module.stream_allowed(XTREAM_STREAM, {"cat:other"})


class TestUpstreamUrl:
    def test_m3u_direct_url(self, export_module):
        assert export_module.upstream_url(M3U_STREAM) == "http://192.168.1.87:5004/auto/v2.1"

    def test_xtream_ts_url(self, export_module):
        assert (
            export_module.upstream_url(XTREAM_STREAM)
            == "http://provider.example/live/puser/ppass/42.ts"
        )

    def test_xtream_credentials_urlencoded(self, export_module):
        stream = {**XTREAM_STREAM, "source_password": "p#ss"}
        assert "p%23ss" in export_module.upstream_url(stream)

    def test_unknown_stream(self, export_module):
        assert export_module.upstream_url({"stream_id": 1}) == ""


class TestStripStreamExt:
    def test_strips_known_extensions(self, export_module):
        assert export_module.strip_stream_ext("42.ts") == "42"
        assert export_module.strip_stream_ext("42.m3u8") == "42"
        assert export_module.strip_stream_ext("src2_1.ts") == "src2_1"

    def test_keeps_other_ids(self, export_module):
        assert export_module.strip_stream_ext("42") == "42"
        assert export_module.strip_stream_ext("src2_1") == "src2_1"


class TestBuildPlaylist:
    def test_playlist_format(self, export_module):
        playlist = export_module.build_playlist(
            CATEGORIES, [XTREAM_STREAM, M3U_STREAM], "https://tv.example", "alice", "pw"
        )
        lines = playlist.strip().split("\n")
        assert lines[0] == "#EXTM3U"
        assert 'tvg-id="news.example"' in lines[1]
        assert 'group-title="News"' in lines[1]
        assert lines[1].endswith(",News HD")
        assert lines[2] == "https://tv.example/live/alice/pw/42.ts"
        assert 'group-title="OTA"' in lines[3]
        assert lines[4] == "https://tv.example/live/alice/pw/src2_1.ts"

    def test_credentials_urlencoded_in_urls(self, export_module):
        playlist = export_module.build_playlist(
            CATEGORIES, [XTREAM_STREAM], "https://tv.example", "alice", "p w#1"
        )
        assert "/live/alice/p%20w%231/42.ts" in playlist


class TestBuildXmltv:
    def test_channels_and_programs(self, export_module, tmp_path):
        from datetime import UTC, datetime, timedelta

        import epg

        epg.init(tmp_path)
        now = datetime.now(UTC)
        epg.insert_programs(
            [
                (
                    "news.example",
                    "Evening News",
                    now.timestamp(),
                    (now + timedelta(hours=1)).timestamp(),
                    "Daily news & <updates>",
                    "src1",
                )
            ]
        )
        epg.commit()

        xml = export_module.build_xmltv([XTREAM_STREAM, M3U_STREAM])
        assert '<channel id="news.example">' in xml
        assert "<display-name>News HD</display-name>" in xml
        assert '<channel id="ota.example">' in xml
        assert "<title>Evening News</title>" in xml
        assert "Daily news &amp; &lt;updates&gt;" in xml

    def test_streams_without_epg_id_skipped(self, export_module, tmp_path):
        import epg

        epg.init(tmp_path)
        xml = export_module.build_xmltv([{**M3U_STREAM, "epg_channel_id": ""}])
        assert "<channel" not in xml


class TestStreamLimits:
    def test_unlimited_by_default(self, export_module):
        with (
            patch("auth.get_user_limits", return_value={"max_streams_per_source": {}}),
            patch("m3u_export.get_sources", return_value=[]),
        ):
            for _ in range(5):
                assert export_module.try_acquire_stream("alice", "src1")

    def test_user_limit_enforced(self, export_module):
        with (
            patch(
                "auth.get_user_limits",
                return_value={"max_streams_per_source": {"src1": 1}},
            ),
            patch("m3u_export.get_sources", return_value=[]),
        ):
            assert export_module.try_acquire_stream("alice", "src1")
            assert not export_module.try_acquire_stream("alice", "src1")
            export_module.release_stream("alice", "src1")
            assert export_module.try_acquire_stream("alice", "src1")

    def test_source_limit_enforced_across_users(self, export_module):
        source = MagicMock()
        source.id = "src1"
        source.max_streams = 2
        with (
            patch("auth.get_user_limits", return_value={"max_streams_per_source": {}}),
            patch("m3u_export.get_sources", return_value=[source]),
        ):
            assert export_module.try_acquire_stream("alice", "src1")
            assert export_module.try_acquire_stream("bob", "src1")
            assert not export_module.try_acquire_stream("carol", "src1")


@pytest.fixture
def export_client(tmp_path: Path):
    """Test client with a user, live data cache, and M3U export enabled."""
    from fastapi.testclient import TestClient

    with (
        patch.dict(
            "sys.modules", {"defusedxml": MagicMock(), "defusedxml.ElementTree": MagicMock()}
        ),
        patch("cache.CACHE_DIR", tmp_path),
        patch("cache.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("cache.USERS_DIR", tmp_path / "users"),
        patch("auth.CACHE_DIR", tmp_path),
        patch("auth.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("auth.USERS_DIR", tmp_path / "users"),
        patch("epg.init"),
        patch("ffmpeg_command.init"),
        patch("ffmpeg_session.cleanup_and_recover_sessions"),
    ):
        (tmp_path / "users").mkdir(exist_ok=True)
        import auth
        import m3u_export
        import main

        cache_module.get_cache().clear()
        m3u_export.clear_credential_cache()
        m3u_export._active_streams.clear()
        main._login_attempts.clear()

        auth.create_user("alice", "password123")
        settings = cache_module.load_server_settings()
        settings["m3u_export_enabled"] = True
        cache_module.save_server_settings(settings)

        (tmp_path / "live_data.json").write_text(
            json.dumps(
                {
                    "data": {
                        "cats": CATEGORIES,
                        "streams": [XTREAM_STREAM, M3U_STREAM],
                        "epg_urls": [],
                    },
                    "timestamp": 9999999999,
                }
            )
        )

        yield TestClient(main.app)


class TestExportRoutes:
    def test_disabled_returns_404(self, export_client, tmp_path):
        settings = cache_module.load_server_settings()
        settings["m3u_export_enabled"] = False
        cache_module.save_server_settings(settings)

        resp = export_client.get("/get.php?username=alice&password=password123")
        assert resp.status_code == 404

    def test_invalid_credentials_rejected(self, export_client):
        resp = export_client.get("/get.php?username=alice&password=wrong")
        assert resp.status_code == 401

    def test_playlist_served(self, export_client):
        resp = export_client.get("/get.php?username=alice&password=password123")
        assert resp.status_code == 200
        assert resp.text.startswith("#EXTM3U")
        assert "News HD" in resp.text
        assert "/live/alice/password123/42.ts" in resp.text

    def test_playlist_respects_group_restrictions(self, export_client):
        import auth

        auth.set_user_limits("alice", unavailable_groups=["cat:src1_5"])
        resp = export_client.get("/get.php?username=alice&password=password123")
        assert "News HD" not in resp.text
        assert "Local OTA" in resp.text

    def test_live_redirects_to_upstream(self, export_client):
        resp = export_client.get(
            "/live/alice/password123/42.ts", follow_redirects=False
        )
        assert resp.status_code == 302
        assert resp.headers["location"] == "http://provider.example/live/puser/ppass/42.ts"

    def test_live_unknown_channel_404(self, export_client):
        resp = export_client.get(
            "/live/alice/password123/999.ts", follow_redirects=False
        )
        assert resp.status_code == 404

    def test_live_restricted_channel_403(self, export_client):
        import auth

        auth.set_user_limits("alice", unavailable_groups=["cat:src1_5"])
        resp = export_client.get(
            "/live/alice/password123/42.ts", follow_redirects=False
        )
        assert resp.status_code == 403

    def test_live_proxy_mode_streams_bytes(self, export_client):
        settings = cache_module.load_server_settings()
        settings["m3u_export_mode"] = "proxy"
        cache_module.save_server_settings(settings)

        chunks = [b"tsdata1", b"tsdata2"]

        def fake_iter(url, username, source_id, timeout=15):
            yield from chunks

        with patch("m3u_export.iter_proxy_stream", side_effect=fake_iter):
            resp = export_client.get(
                "/live/alice/password123/42.ts", follow_redirects=False
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp2t"
        assert resp.content == b"tsdata1tsdata2"

    def test_live_hls_direct_url_always_redirects(self, export_client):
        settings = cache_module.load_server_settings()
        settings["m3u_export_mode"] = "proxy"
        cache_module.save_server_settings(settings)

        # Rewrite the m3u stream's direct_url to an HLS playlist
        data = json.loads((cache_module.CACHE_DIR / "live_data.json").read_text())
        data["data"]["streams"][1]["direct_url"] = "http://cam.example/stream.m3u8"
        (cache_module.CACHE_DIR / "live_data.json").write_text(json.dumps(data))
        cache_module.get_cache().clear()

        resp = export_client.get(
            "/live/alice/password123/src2_1.ts", follow_redirects=False
        )
        assert resp.status_code == 302
        assert resp.headers["location"] == "http://cam.example/stream.m3u8"

    def test_xmltv_served(self, export_client):
        with (
            patch("epg.get_programs_batch", return_value={}),
            patch("epg.get_icons_batch", return_value={}),
        ):
            resp = export_client.get("/xmltv.php?username=alice&password=password123")
        assert resp.status_code == 200
        assert "<tv" in resp.text
        assert 'id="news.example"' in resp.text

    def test_failed_attempts_rate_limited(self, export_client):
        import main

        for _ in range(main._LOGIN_MAX_ATTEMPTS):
            resp = export_client.get("/get.php?username=alice&password=wrong")
            assert resp.status_code == 401
        resp = export_client.get("/get.php?username=alice&password=wrong")
        assert resp.status_code == 429


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)
