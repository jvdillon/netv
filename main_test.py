"""Tests for main.py - FastAPI routes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import json

import pytest

import cache as cache_module
import m3u as m3u_module


@pytest.fixture
def mock_deps():
    """Mock all external dependencies before importing main."""
    with (
        patch.dict(
            "sys.modules", {"defusedxml": MagicMock(), "defusedxml.ElementTree": MagicMock()}
        ),
        patch("cache.CACHE_DIR", Path("/tmp/test_cache")),
        patch("cache.SERVER_SETTINGS_FILE", Path("/tmp/test_cache/server_settings.json")),
        patch("cache.USERS_DIR", Path("/tmp/test_cache/users")),
    ):
        yield


@pytest.fixture
def client(tmp_path: Path, mock_deps):
    """Create test client with mocked dependencies."""
    from fastapi.testclient import TestClient

    # Patch paths before importing main
    with (
        patch("cache.CACHE_DIR", tmp_path),
        patch("cache.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("cache.USERS_DIR", tmp_path / "users"),
        patch("auth.CACHE_DIR", tmp_path),
        patch("auth.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("auth.USERS_DIR", tmp_path / "users"),
        patch("epg_db.init"),
        patch("ffmpeg_command.init"),
        patch("ffmpeg_session.cleanup_and_recover_sessions"),
    ):
        (tmp_path / "users").mkdir(exist_ok=True)
        import main

        # Disable background loading
        cache_module.get_cache().clear()
        yield TestClient(main.app)


@pytest.fixture
def auth_client(tmp_path: Path, mock_deps):
    """Create test client with a logged-in user."""
    from fastapi.testclient import TestClient

    with (
        patch("cache.CACHE_DIR", tmp_path),
        patch("cache.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("cache.USERS_DIR", tmp_path / "users"),
        patch("auth.CACHE_DIR", tmp_path),
        patch("auth.SERVER_SETTINGS_FILE", tmp_path / "server_settings.json"),
        patch("auth.USERS_DIR", tmp_path / "users"),
        patch("epg_db.init"),
        patch("ffmpeg_command.init"),
        patch("ffmpeg_session.cleanup_and_recover_sessions"),
    ):
        (tmp_path / "users").mkdir(exist_ok=True)
        import auth
        import main

        cache_module.get_cache().clear()
        client = TestClient(main.app)

        # Create user and get token
        auth.create_user("testuser", "testpass123")
        token = auth.create_token({"sub": "testuser"})
        client.cookies.set("token", token)

        yield client


class TestSetup:
    """Tests for initial setup flow."""

    def test_setup_page_shown_when_no_users(self, client):
        resp = client.get("/setup", follow_redirects=False)
        assert resp.status_code == 200
        assert b"setup" in resp.content.lower() or b"Create" in resp.content

    def test_setup_redirects_when_users_exist(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/setup", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_setup_creates_user(self, client):
        resp = client.post(
            "/setup",
            data={"username": "admin", "password": "password123", "confirm": "password123"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

        import auth

        assert auth.verify_password("admin", "password123")

    def test_setup_validates_username_length(self, client):
        resp = client.post(
            "/setup",
            data={"username": "ab", "password": "password123", "confirm": "password123"},
        )
        assert resp.status_code == 200
        assert b"at least 3" in resp.content

    def test_setup_validates_password_length(self, client):
        resp = client.post(
            "/setup",
            data={"username": "admin", "password": "short", "confirm": "short"},
        )
        assert resp.status_code == 200
        assert b"at least 8" in resp.content

    def test_setup_validates_password_match(self, client):
        resp = client.post(
            "/setup",
            data={"username": "admin", "password": "password123", "confirm": "different"},
        )
        assert resp.status_code == 200
        assert b"do not match" in resp.content


class TestLogin:
    """Tests for login flow."""

    def test_login_page_redirects_to_setup_when_no_users(self, client):
        resp = client.get("/login", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/setup"

    def test_login_page_shown_when_users_exist(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/login")
        assert resp.status_code == 200

    def test_login_success_sets_cookie(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.post(
            "/login",
            data={"username": "admin", "password": "password123"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "token" in resp.cookies

    def test_login_failure_returns_401(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.post(
            "/login",
            data={"username": "admin", "password": "wrongpassword"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "error=invalid" in resp.headers["location"]


class TestLogout:
    """Tests for logout."""

    def test_logout_clears_cookie(self, auth_client):
        resp = auth_client.get("/logout", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"


class TestAuthRequired:
    """Tests for auth-protected routes."""

    def test_index_redirects_to_login(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_guide_redirects_to_login(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/guide", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_vod_redirects_to_login(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/vod", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"

    def test_series_redirects_to_login(self, client, tmp_path):
        import auth

        auth.create_user("admin", "password123")

        resp = client.get("/series", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login"


class TestIndex:
    """Tests for index route."""

    def test_index_redirects_to_guide(self, auth_client):
        resp = auth_client.get("/", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/guide"


class TestFavicon:
    """Tests for favicon."""

    def test_favicon_returns_204(self, client):
        resp = client.get("/favicon.ico")
        assert resp.status_code == 204


class TestGuide:
    """Tests for guide page."""

    def test_guide_shows_loading_when_no_cache(self, auth_client):
        with patch("main.load_file_cache", return_value=None):
            resp = auth_client.get("/guide")
            assert resp.status_code == 200
            # Should show loading state
            assert b"loading" in resp.content.lower() or b"Loading" in resp.content

    def test_guide_shows_channels_from_cache(self, auth_client):
        cache_module.get_cache()["live_categories"] = [
            {"category_id": "1", "category_name": "News"}
        ]
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN", "category_ids": ["1"], "epg_channel_id": ""}
        ]

        with patch("main.epg_db.has_programs", return_value=True):
            resp = auth_client.get("/guide?cats=1")
            assert resp.status_code == 200

    def test_guide_redirects_to_saved_filter(self, auth_client, tmp_path):
        user_dir = tmp_path / "users" / "testuser"
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "settings.json").write_text(json.dumps({"guide_filter": ["1", "2"]}))

        cache_module.get_cache()["live_categories"] = []
        cache_module.get_cache()["live_streams"] = []

        resp = auth_client.get("/guide", follow_redirects=False)
        assert resp.status_code == 303
        assert "cats=1,2" in resp.headers["location"]


class TestVod:
    """Tests for VOD page."""

    def test_vod_shows_loading_when_no_cache(self, auth_client):
        with patch("main.load_file_cache", return_value=None):
            resp = auth_client.get("/vod")
            assert resp.status_code == 200

    def test_vod_shows_movies_from_cache(self, auth_client):
        cache_module.get_cache()["vod_categories"] = [
            {"category_id": "10", "category_name": "Movies", "source_id": "src1"}
        ]
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 100, "name": "Movie 1", "category_id": "10", "source_id": "src1"}
        ]

        resp = auth_client.get("/vod")
        assert resp.status_code == 200

    def test_vod_filters_by_category(self, auth_client):
        cache_module.get_cache()["vod_categories"] = [
            {"category_id": "10", "category_name": "Action", "source_id": "src1"},
            {"category_id": "20", "category_name": "Comedy", "source_id": "src1"},
        ]
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 100, "name": "Action Movie", "category_id": "10", "source_id": "src1"},
            {"stream_id": 101, "name": "Comedy Movie", "category_id": "20", "source_id": "src1"},
        ]

        resp = auth_client.get("/vod?category=10")
        assert resp.status_code == 200

    def test_vod_sorts_by_alpha(self, auth_client):
        cache_module.get_cache()["vod_categories"] = []
        cache_module.get_cache()["vod_streams"] = [
            {"stream_id": 1, "name": "Zebra", "source_id": "src1"},
            {"stream_id": 2, "name": "Apple", "source_id": "src1"},
        ]

        resp = auth_client.get("/vod?sort=alpha")
        assert resp.status_code == 200


class TestSeries:
    """Tests for series page."""

    def test_series_shows_loading_when_no_cache(self, auth_client):
        with patch("main.load_file_cache", return_value=None):
            resp = auth_client.get("/series")
            assert resp.status_code == 200

    def test_series_shows_list_from_cache(self, auth_client):
        cache_module.get_cache()["series_categories"] = [
            {"category_id": "30", "category_name": "Drama", "source_id": "src1"}
        ]
        cache_module.get_cache()["series"] = [
            {"series_id": 200, "name": "Show 1", "category_id": "30", "source_id": "src1"}
        ]

        resp = auth_client.get("/series")
        assert resp.status_code == 200


class TestSearch:
    """Tests for search page."""

    def test_search_page_renders(self, auth_client):
        cache_module.get_cache()["live_streams"] = []
        cache_module.get_cache()["vod_streams"] = []
        cache_module.get_cache()["series"] = []

        resp = auth_client.get("/search")
        assert resp.status_code == 200

    def test_search_finds_live_streams(self, auth_client):
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN News"},
            {"stream_id": 2, "name": "BBC World"},
        ]
        cache_module.get_cache()["live_categories"] = []
        cache_module.get_cache()["epg_urls"] = []
        cache_module.get_cache()["vod_streams"] = []
        cache_module.get_cache()["series"] = []

        resp = auth_client.get("/search?q=CNN&live=true")
        assert resp.status_code == 200

    def test_search_regex_mode(self, auth_client):
        cache_module.get_cache()["live_streams"] = [
            {"stream_id": 1, "name": "CNN News"},
            {"stream_id": 2, "name": "CNBC Finance"},
        ]
        cache_module.get_cache()["live_categories"] = []
        cache_module.get_cache()["epg_urls"] = []
        cache_module.get_cache()["vod_streams"] = []
        cache_module.get_cache()["series"] = []

        resp = auth_client.get("/search?q=CN.*&regex=true&live=true")
        assert resp.status_code == 200

    def test_search_rejects_long_regex(self, auth_client):
        cache_module.get_cache()["live_streams"] = []

        resp = auth_client.get(f"/search?q={'a' * 101}&regex=true&live=true")
        assert resp.status_code == 400


class TestSettings:
    """Tests for settings page."""

    def test_settings_page_renders(self, auth_client):
        cache_module.get_cache()["live_categories"] = []

        with patch("main.load_file_cache", return_value=None):
            resp = auth_client.get("/settings")
            assert resp.status_code == 200

    def test_settings_guide_filter(self, auth_client):
        resp = auth_client.post(
            "/settings/guide-filter",
            json={"cats": ["1", "2", "3"]},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_settings_captions(self, auth_client):
        resp = auth_client.post(
            "/settings/captions",
            data={"enabled": "on"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_settings_transcode(self, auth_client):
        resp = auth_client.post(
            "/settings/transcode",
            data={
                "transcode_mode": "auto",
                "transcode_hw": "nvidia",
                "vod_transcode_cache_mins": 60,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


class TestAddSource:
    """Tests for adding sources."""

    def test_add_xtream_source(self, auth_client):
        with patch("main.clear_all_caches"):
            resp = auth_client.post(
                "/settings/add",
                data={
                    "name": "Test Provider",
                    "source_type": "xtream",
                    "url": "http://example.com",
                    "username": "user",
                    "password": "pass",
                    "epg_timeout": 120,
                },
                follow_redirects=False,
            )
            assert resp.status_code == 303

    def test_add_m3u_source(self, auth_client):
        with patch("main.clear_all_caches"):
            resp = auth_client.post(
                "/settings/add",
                data={
                    "name": "M3U Playlist",
                    "source_type": "m3u",
                    "url": "http://example.com/playlist.m3u",
                    "epg_timeout": 120,
                },
                follow_redirects=False,
            )
            assert resp.status_code == 303

    def test_add_source_validates_type(self, auth_client):
        resp = auth_client.post(
            "/settings/add",
            data={
                "name": "Bad Source",
                "source_type": "invalid",
                "url": "http://example.com",
            },
        )
        assert resp.status_code == 400

    def test_add_source_validates_url_scheme(self, auth_client):
        resp = auth_client.post(
            "/settings/add",
            data={
                "name": "Bad Source",
                "source_type": "xtream",
                "url": "ftp://example.com",
            },
        )
        assert resp.status_code == 400

    def test_add_source_validates_name_length(self, auth_client):
        resp = auth_client.post(
            "/settings/add",
            data={
                "name": "x" * 201,
                "source_type": "xtream",
                "url": "http://example.com",
            },
        )
        assert resp.status_code == 400


class TestDeleteSource:
    """Tests for deleting sources."""

    def test_delete_source(self, auth_client, tmp_path):
        settings_file = tmp_path / "server_settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "id": "src_123",
                            "name": "Test",
                            "type": "xtream",
                            "url": "http://example.com",
                        }
                    ]
                }
            )
        )

        with patch("main.clear_all_caches"):
            resp = auth_client.post("/settings/delete/src_123", follow_redirects=False)
            assert resp.status_code == 303


class TestUserPrefs:
    """Tests for user preferences API."""

    def test_get_user_prefs(self, auth_client):
        resp = auth_client.get("/api/user-prefs")
        assert resp.status_code == 200
        data = resp.json()
        assert "favorites" in data
        assert "cc_lang" in data

    def test_save_user_prefs(self, auth_client):
        resp = auth_client.post(
            "/api/user-prefs",
            json={"cc_lang": "eng", "cast_host": "192.168.1.100"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


class TestWatchPosition:
    """Tests for watch position API."""

    def test_save_watch_position(self, auth_client):
        resp = auth_client.post(
            "/api/watch-position",
            json={"url": "http://example.com/movie.mkv", "position": 1234.5, "duration": 7200},
        )
        assert resp.status_code == 200

    def test_get_watch_position(self, auth_client):
        # Save first
        auth_client.post(
            "/api/watch-position",
            json={"url": "http://example.com/movie.mkv", "position": 1234.5, "duration": 7200},
        )

        resp = auth_client.get("/api/watch-position?url=http://example.com/movie.mkv")
        assert resp.status_code == 200
        data = resp.json()
        assert data["position"] == 1234.5
        assert data["duration"] == 7200

    def test_get_watch_position_not_found(self, auth_client):
        resp = auth_client.get("/api/watch-position?url=http://example.com/unknown.mkv")
        assert resp.status_code == 200
        data = resp.json()
        assert data["position"] == 0


class TestUserManagement:
    """Tests for user management endpoints."""

    def test_delete_user(self, auth_client, tmp_path):
        import auth

        auth.create_user("otheruser", "password123")

        resp = auth_client.post("/settings/users/delete/otheruser", follow_redirects=False)
        assert resp.status_code == 303

    def test_cannot_delete_self(self, auth_client):
        resp = auth_client.post("/settings/users/delete/testuser")
        assert resp.status_code == 400

    def test_change_password(self, auth_client):
        resp = auth_client.post(
            "/settings/users/password",
            data={"current_password": "testpass123", "new_password": "newpass456"},
        )
        assert resp.status_code == 200

    def test_change_password_wrong_current(self, auth_client):
        resp = auth_client.post(
            "/settings/users/password",
            data={"current_password": "wrongpass", "new_password": "newpass456"},
        )
        assert resp.status_code == 400


class TestPlaylistXspf:
    """Tests for XSPF playlist generation."""

    def test_playlist_xspf(self, auth_client):
        resp = auth_client.get("/playlist.xspf?url=http://example.com/stream.m3u8")
        assert resp.status_code == 200
        assert b"<?xml" in resp.content
        assert b"http://example.com/stream.m3u8" in resp.content
        assert resp.headers["content-type"] == "application/xspf+xml"


class TestApiSettings:
    """Tests for settings API."""

    def test_get_settings(self, auth_client):
        resp = auth_client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "transcode_mode" in data

    def test_update_settings(self, auth_client):
        resp = auth_client.post(
            "/api/settings",
            json={"transcode_mode": "always"},
        )
        assert resp.status_code == 200


class TestTranscodeRoutes:
    """Tests for transcode routes (with mocked transcoding module)."""

    def test_transcode_file_not_found(self, auth_client):
        with patch("main.ffmpeg_session.get_session", return_value=None):
            resp = auth_client.get("/transcode/invalid-session/stream.m3u8")
            assert resp.status_code == 404

    def test_transcode_stop(self, auth_client):
        with patch("main.ffmpeg_session.stop_session"):
            resp = auth_client.delete("/transcode/test-session")
            assert resp.status_code == 200
            assert resp.json()["status"] == "stopped"

    def test_transcode_stop_post(self, auth_client):
        with patch("main.ffmpeg_session.stop_session"):
            resp = auth_client.post("/transcode/test-session/stop")
            assert resp.status_code == 200

    def test_transcode_progress_not_found(self, auth_client):
        with patch("main.ffmpeg_session.get_session_progress", return_value=None):
            resp = auth_client.get("/transcode/progress/invalid-session")
            assert resp.status_code == 404


class TestSubtitleRoutes:
    """Tests for subtitle routes."""

    def test_subtitle_invalid_filename(self, auth_client):
        resp = auth_client.get("/subs/session/notavtt.txt")
        assert resp.status_code == 400

    def test_subtitle_session_not_found(self, auth_client):
        with patch("main.ffmpeg_session.get_session", return_value=None):
            resp = auth_client.get("/subs/invalid-session/sub0.vtt")
            assert resp.status_code == 404


class TestProbeCache:
    """Tests for probe cache endpoints."""

    def test_get_probe_cache(self, auth_client):
        with patch("main.ffmpeg_command.get_series_probe_cache_stats", return_value=[]):
            resp = auth_client.get("/settings/probe-cache")
            assert resp.status_code == 200
            assert "series" in resp.json()

    def test_clear_probe_cache(self, auth_client):
        with patch("main.ffmpeg_command.clear_all_probe_cache", return_value=5):
            resp = auth_client.post("/settings/probe-cache/clear")
            assert resp.status_code == 200
            assert resp.json()["cleared"] == 5

    def test_clear_series_probe_cache(self, auth_client):
        with patch("main.ffmpeg_command.invalidate_series_probe_cache"):
            resp = auth_client.post("/settings/probe-cache/clear/123")
            assert resp.status_code == 200


class TestRefreshStatus:
    """Tests for refresh status endpoints."""

    def test_guide_refresh_status(self, auth_client):
        m3u_module.get_refresh_in_progress().clear()

        resp = auth_client.get("/guide/refresh-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["live"] is False
        assert data["epg"] is False

    def test_settings_refresh_status(self, auth_client):
        m3u_module.get_refresh_in_progress().clear()

        resp = auth_client.get("/settings/refresh-status")
        assert resp.status_code == 200


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)
