"""Tests for ffmpeg session management."""

from unittest.mock import patch

import json
import pathlib
import tempfile
import time

from ffmpeg_session import (
    _HEARTBEAT_TIMEOUT_SEC,
    _build_subtitle_tracks,
    _calc_hls_duration,
    _DeadProcess,
    _is_process_alive,
    _kill_process,
    _regenerate_playlist,
    _transcode_lock,
    _transcode_sessions,
    _url_to_session,
    cleanup_and_recover_sessions,
    cleanup_expired_sessions,
    clear_url_session,
    enforce_stream_limits,
    get_live_cache_timeout,
    get_session,
    get_session_progress,
    get_source_sessions,
    get_user_sessions,
    get_vod_cache_timeout,
    is_session_valid,
    shutdown,
    stop_session,
    touch_session,
)


class FakeProcess:
    """Fake async process for testing."""

    def __init__(self, alive: bool = True, killed: bool = False):
        self.returncode = None if alive else 0
        self._killed = killed

    def kill(self) -> None:
        if self._killed:
            raise ProcessLookupError("No such process")
        self.returncode = -9


def _clear_session_state():
    """Clear all session state for test isolation."""
    with _transcode_lock:
        _transcode_sessions.clear()
        _url_to_session.clear()


# =============================================================================
# Process Lifecycle Tests
# =============================================================================


class TestIsProcessAlive:
    """Tests for _is_process_alive."""

    def test_none_is_dead(self):
        assert _is_process_alive(None) is False

    def test_dead_process_placeholder(self):
        assert _is_process_alive(_DeadProcess()) is False

    def test_alive_process(self):
        proc = FakeProcess(alive=True)
        assert _is_process_alive(proc) is True

    def test_dead_process(self):
        proc = FakeProcess(alive=False)
        assert _is_process_alive(proc) is False


class TestKillProcess:
    """Tests for _kill_process."""

    def test_kill_alive_process(self):
        proc = FakeProcess(alive=True)
        assert _kill_process(proc) is True
        assert proc.returncode == -9

    def test_kill_already_dead(self):
        proc = FakeProcess(alive=True, killed=True)
        assert _kill_process(proc) is False


# =============================================================================
# Session Validity Tests
# =============================================================================


class TestIsSessionValid:
    """Tests for is_session_valid with heartbeat timeout."""

    def test_active_process_with_recent_heartbeat(self):
        """Active process + recent heartbeat = valid."""
        session = {
            "process": FakeProcess(alive=True),
            "started": time.time(),
            "last_access": time.time(),
            "is_vod": False,
        }
        assert is_session_valid(session) is True

    def test_active_process_stale_heartbeat(self):
        """Active process but no heartbeat in 5+ min = invalid."""
        session = {
            "process": FakeProcess(alive=True),
            "started": time.time() - 400,
            "last_access": time.time() - 400,  # 6+ min ago
            "is_vod": False,
        }
        assert is_session_valid(session) is False

    def test_dead_process_live_session_no_cache(self):
        """Dead process, live session, cache=0 = invalid."""
        with patch("ffmpeg_session.get_live_cache_timeout", return_value=0):
            session = {
                "process": FakeProcess(alive=False),
                "started": time.time(),
                "last_access": time.time(),
                "is_vod": False,
            }
            assert is_session_valid(session) is False

    def test_dead_process_vod_session_within_cache(self):
        """Dead process, VOD session, within cache timeout = valid."""
        with patch("ffmpeg_session.get_vod_cache_timeout", return_value=3600):
            session = {
                "process": FakeProcess(alive=False),
                "started": time.time() - 60,
                "last_access": time.time() - 60,  # 1 min ago
                "is_vod": True,
            }
            assert is_session_valid(session) is True

    def test_dead_process_vod_session_expired_cache(self):
        """Dead process, VOD session, past cache timeout = invalid."""
        with patch("ffmpeg_session.get_vod_cache_timeout", return_value=60):
            session = {
                "process": FakeProcess(alive=False),
                "started": time.time() - 120,
                "last_access": time.time() - 120,  # 2 min ago, cache is 1 min
                "is_vod": True,
            }
            assert is_session_valid(session) is False

    def test_heartbeat_timeout_boundary(self):
        """Test exactly at heartbeat timeout boundary."""
        # Just under timeout = valid (if process alive)
        session = {
            "process": FakeProcess(alive=True),
            "started": time.time() - (_HEARTBEAT_TIMEOUT_SEC - 1),
            "last_access": time.time() - (_HEARTBEAT_TIMEOUT_SEC - 1),
            "is_vod": False,
        }
        assert is_session_valid(session) is True

        # Just over timeout = invalid
        session["last_access"] = time.time() - (_HEARTBEAT_TIMEOUT_SEC + 1)
        assert is_session_valid(session) is False

    def test_missing_last_access_uses_started(self):
        """If last_access missing, falls back to started time."""
        session = {
            "process": FakeProcess(alive=True),
            "started": time.time(),
            "is_vod": False,
        }
        assert is_session_valid(session) is True


# =============================================================================
# Cache Timeout Tests
# =============================================================================


class TestCacheTimeouts:
    """Tests for cache timeout getters."""

    def test_vod_cache_timeout_default(self):
        """VOD cache default is 60 min = 3600 sec."""
        with patch("ffmpeg_session.get_settings", return_value={}):
            assert get_vod_cache_timeout() == 3600

    def test_vod_cache_timeout_custom(self):
        """VOD cache from settings."""
        with patch("ffmpeg_session.get_settings", return_value={"vod_transcode_cache_mins": 30}):
            assert get_vod_cache_timeout() == 1800

    def test_live_cache_timeout_default(self):
        """Live cache default is 0 (no caching)."""
        with patch("ffmpeg_session.get_settings", return_value={}):
            assert get_live_cache_timeout() == 0

    def test_live_cache_timeout_custom(self):
        """Live cache from settings."""
        with patch("ffmpeg_session.get_settings", return_value={"live_transcode_cache_secs": 30}):
            assert get_live_cache_timeout() == 30


# =============================================================================
# Session Start/Stop Tests
# =============================================================================


class TestStopSession:
    """Tests for stop_session."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_stop_nonexistent_session(self):
        """Stopping nonexistent session is a no-op."""
        stop_session("nonexistent")  # Should not raise

    def test_stop_session_force(self):
        """Force stop removes session."""
        with tempfile.TemporaryDirectory() as tmp:
            session_id = "test-123"
            with _transcode_lock:
                _transcode_sessions[session_id] = {
                    "process": FakeProcess(alive=True),
                    "dir": tmp,
                    "url": "http://test",
                    "last_access": time.time(),
                }
                _url_to_session["http://test"] = session_id

            stop_session(session_id, force=True)

            assert session_id not in _transcode_sessions
            assert "http://test" not in _url_to_session

    def test_stop_session_skip_recent(self):
        """Skip stop for recently-accessed session (race protection)."""
        session_id = "test-456"
        with _transcode_lock:
            _transcode_sessions[session_id] = {
                "process": FakeProcess(alive=True),
                "dir": "/tmp/test",
                "url": "http://test",
                "last_access": time.time(),  # Just now
            }

        stop_session(session_id, force=False)

        # Session should still exist because it was recently accessed
        assert session_id in _transcode_sessions

    def test_stop_session_caches_vod(self):
        """Stop caches VOD session instead of removing it."""
        with tempfile.TemporaryDirectory() as tmp:
            session_id = "test-vod"
            with _transcode_lock:
                _transcode_sessions[session_id] = {
                    "process": FakeProcess(alive=True),
                    "dir": tmp,
                    "url": "http://vod",
                    "is_vod": True,
                    "last_access": time.time() - 10,  # Old enough to stop
                }
                _url_to_session["http://vod"] = session_id

            with patch("ffmpeg_session.get_vod_cache_timeout", return_value=3600):
                stop_session(session_id, force=False)

            # Session should still exist (cached)
            assert session_id in _transcode_sessions


class TestCleanupExpiredSessions:
    """Tests for cleanup_expired_sessions."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_cleanup_removes_expired(self):
        """Cleanup removes expired sessions."""
        with tempfile.TemporaryDirectory() as tmp:
            session_id = "expired-session"
            with _transcode_lock:
                _transcode_sessions[session_id] = {
                    "process": FakeProcess(alive=False),
                    "dir": tmp,
                    "url": "http://expired",
                    "is_vod": False,
                    "started": time.time() - 400,
                    "last_access": time.time() - 400,  # Expired
                }

            with patch("ffmpeg_session.get_live_cache_timeout", return_value=0):
                cleanup_expired_sessions()

            assert session_id not in _transcode_sessions

    def test_cleanup_keeps_valid(self):
        """Cleanup keeps valid sessions."""
        session_id = "valid-session"
        with _transcode_lock:
            _transcode_sessions[session_id] = {
                "process": FakeProcess(alive=True),
                "dir": "/tmp/test",
                "url": "http://valid",
                "is_vod": False,
                "started": time.time(),
                "last_access": time.time(),
            }

        cleanup_expired_sessions()

        assert session_id in _transcode_sessions


class TestShutdown:
    """Tests for shutdown."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_shutdown_kills_all_processes(self):
        """Shutdown kills all processes and clears sessions."""
        proc1 = FakeProcess(alive=True)
        proc2 = FakeProcess(alive=True)

        with _transcode_lock:
            _transcode_sessions["s1"] = {"process": proc1, "dir": "/tmp/1"}
            _transcode_sessions["s2"] = {"process": proc2, "dir": "/tmp/2"}

        shutdown()

        assert proc1.returncode == -9
        assert proc2.returncode == -9
        assert len(_transcode_sessions) == 0


# =============================================================================
# Stream Limits Tests
# =============================================================================


class TestGetUserSessions:
    """Tests for get_user_sessions."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_get_user_sessions_filters_by_username(self):
        """Returns only sessions for specified user."""
        with _transcode_lock:
            _transcode_sessions["s1"] = {"username": "alice", "started": 1}
            _transcode_sessions["s2"] = {"username": "bob", "started": 2}
            _transcode_sessions["s3"] = {"username": "alice", "started": 3}

        sessions = get_user_sessions("alice")
        assert len(sessions) == 2
        assert sessions[0][0] == "s1"  # Sorted by start time
        assert sessions[1][0] == "s3"

    def test_get_user_sessions_empty(self):
        """Returns empty list for user with no sessions."""
        sessions = get_user_sessions("nobody")
        assert sessions == []


class TestGetSourceSessions:
    """Tests for get_source_sessions."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_get_source_sessions_filters_by_source(self):
        """Returns only sessions for specified source."""
        with _transcode_lock:
            _transcode_sessions["s1"] = {"source_id": "src1", "started": 1}
            _transcode_sessions["s2"] = {"source_id": "src2", "started": 2}
            _transcode_sessions["s3"] = {"source_id": "src1", "started": 3}

        sessions = get_source_sessions("src1")
        assert len(sessions) == 2
        assert sessions[0][0] == "s1"
        assert sessions[1][0] == "s3"


class TestEnforceStreamLimits:
    """Tests for enforce_stream_limits."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_no_limits_returns_none(self):
        """No limits set = no error."""
        result = enforce_stream_limits("alice", None, 0, 0)
        assert result is None

    def test_user_limit_stops_oldest(self):
        """User at limit stops their oldest session."""
        with tempfile.TemporaryDirectory() as tmp:
            with _transcode_lock:
                _transcode_sessions["s1"] = {
                    "username": "alice",
                    "started": 1,
                    "process": FakeProcess(alive=True),
                    "dir": tmp,
                    "url": "http://1",
                    "last_access": 0,  # Old enough to stop
                }
                _transcode_sessions["s2"] = {
                    "username": "alice",
                    "started": 2,
                    "process": FakeProcess(alive=True),
                    "dir": "/tmp/2",
                    "url": "http://2",
                    "last_access": time.time(),
                }

            result = enforce_stream_limits("alice", None, 2, 0)

            assert result is None
            assert "s1" not in _transcode_sessions
            assert "s2" in _transcode_sessions

    def test_source_limit_stops_user_session(self):
        """Source at limit stops user's oldest session on that source."""
        with tempfile.TemporaryDirectory() as tmp:
            with _transcode_lock:
                _transcode_sessions["s1"] = {
                    "username": "alice",
                    "source_id": "src1",
                    "started": 1,
                    "process": FakeProcess(alive=True),
                    "dir": tmp,
                    "url": "http://1",
                    "last_access": 0,
                }

            result = enforce_stream_limits("alice", "src1", 0, 1)

            assert result is None
            assert "s1" not in _transcode_sessions

    def test_source_limit_returns_error_for_other_user(self):
        """Source at limit with other user's session returns error."""
        with _transcode_lock:
            _transcode_sessions["s1"] = {
                "username": "bob",
                "source_id": "src1",
                "started": 1,
                "process": FakeProcess(alive=True),
                "dir": "/tmp/1",
                "url": "http://1",
            }

        result = enforce_stream_limits("alice", "src1", 0, 1)

        assert result == "Source at capacity (1 streams)"
        assert "s1" in _transcode_sessions  # Not stopped


# =============================================================================
# Session Query/Update Tests
# =============================================================================


class TestGetSession:
    """Tests for get_session."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_get_existing_session(self):
        """Returns copy of session dict."""
        with _transcode_lock:
            _transcode_sessions["test"] = {"dir": "/tmp", "url": "http://test"}

        session = get_session("test")
        assert session is not None
        assert session["dir"] == "/tmp"

    def test_get_nonexistent_session(self):
        """Returns None for nonexistent session."""
        assert get_session("nonexistent") is None


class TestTouchSession:
    """Tests for touch_session (heartbeat)."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_touch_updates_last_access(self):
        """Touch updates last_access timestamp."""
        old_time = time.time() - 100
        with _transcode_lock:
            _transcode_sessions["test"] = {"last_access": old_time}

        result = touch_session("test")

        assert result is True
        assert _transcode_sessions["test"]["last_access"] > old_time

    def test_touch_nonexistent_returns_false(self):
        """Touch returns False for nonexistent session."""
        assert touch_session("nonexistent") is False


class TestGetSessionProgress:
    """Tests for get_session_progress."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_progress_with_playlist(self):
        """Returns progress from playlist."""
        with tempfile.TemporaryDirectory() as tmp:
            playlist = pathlib.Path(tmp) / "stream.m3u8"
            playlist.write_text("#EXTM3U\n#EXTINF:3.0,\nseg0.ts\n#EXTINF:3.0,\nseg1.ts\n")

            with _transcode_lock:
                _transcode_sessions["test"] = {"dir": tmp, "last_access": 0}

            progress = get_session_progress("test")

            assert progress is not None
            assert progress["segment_count"] == 2
            assert progress["duration"] == 6.0

    def test_progress_no_playlist(self):
        """Returns zero progress without playlist."""
        with tempfile.TemporaryDirectory() as tmp:
            with _transcode_lock:
                _transcode_sessions["test"] = {"dir": tmp, "last_access": 0}

            progress = get_session_progress("test")

            assert progress == {"segment_count": 0, "duration": 0.0}

    def test_progress_nonexistent_session(self):
        """Returns None for nonexistent session."""
        assert get_session_progress("nonexistent") is None


class TestClearUrlSession:
    """Tests for clear_url_session."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_clear_existing_url(self):
        """Clears existing URL mapping."""
        with _transcode_lock:
            _url_to_session["http://test"] = "session-123"

        result = clear_url_session("http://test")

        assert result == "session-123"
        assert "http://test" not in _url_to_session

    def test_clear_nonexistent_url(self):
        """Returns None for nonexistent URL."""
        result = clear_url_session("http://nonexistent")
        assert result is None


# =============================================================================
# Playlist Helper Tests
# =============================================================================


class TestCalcHlsDuration:
    """Tests for _calc_hls_duration."""

    def test_duration_from_playlist(self):
        """Calculates duration from EXTINF entries."""
        with tempfile.TemporaryDirectory() as tmp:
            playlist = pathlib.Path(tmp) / "stream.m3u8"
            playlist.write_text("#EXTM3U\n#EXTINF:3.5,\nseg0.ts\n#EXTINF:3.0,\nseg1.ts\n")

            duration = _calc_hls_duration(playlist, 2)

            assert duration == 6.5

    def test_duration_estimate_from_segments(self):
        """Estimates duration when playlist missing."""
        with patch("ffmpeg_session.get_hls_segment_duration", return_value=3.0):
            playlist = pathlib.Path("/nonexistent/stream.m3u8")
            duration = _calc_hls_duration(playlist, 5)

            assert duration == 15.0


class TestBuildSubtitleTracks:
    """Tests for _build_subtitle_tracks."""

    def test_builds_track_list(self):
        """Builds subtitle track list."""
        sub_info = [
            {"index": 2, "lang": "eng", "name": "English"},
            {"index": 3, "lang": "jpn", "name": "Japanese"},
        ]

        tracks = _build_subtitle_tracks("session-123", sub_info)

        assert len(tracks) == 2
        assert tracks[0]["url"] == "/subs/session-123/sub0.vtt"
        assert tracks[0]["lang"] == "eng"
        assert tracks[0]["label"] == "English"
        assert tracks[0]["default"] is True
        assert tracks[1]["default"] is False

    def test_empty_sub_info(self):
        """Returns empty list for no subtitles."""
        assert _build_subtitle_tracks("s", []) == []
        assert _build_subtitle_tracks("s", None) == []  # type: ignore[arg-type]

    def test_non_dict_sub_info(self):
        """Returns empty list for old format (indices only)."""
        assert _build_subtitle_tracks("s", [2, 3]) == []  # type: ignore[arg-type]


class TestRegeneratePlaylist:
    """Tests for _regenerate_playlist."""

    def test_regenerates_playlist_from_segments(self):
        """Regenerates playlist from segment files."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = pathlib.Path(tmp)
            # Create segment files
            (output_dir / "seg000.ts").write_bytes(b"x" * 2000)
            (output_dir / "seg001.ts").write_bytes(b"x" * 2000)
            (output_dir / "seg002.ts").write_bytes(b"x" * 2000)

            with patch("ffmpeg_session.get_hls_segment_duration", return_value=3.0):
                _regenerate_playlist(output_dir, start_segment=1)

            playlist = output_dir / "stream.m3u8"
            assert playlist.exists()
            content = playlist.read_text()
            assert "#EXT-X-MEDIA-SEQUENCE:1" in content
            assert "seg001.ts" in content
            assert "seg002.ts" in content
            assert "seg000.ts" not in content  # Before start_segment

    def test_regenerate_skips_small_segments(self):
        """Skips segments smaller than threshold."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = pathlib.Path(tmp)
            (output_dir / "seg000.ts").write_bytes(b"x" * 500)  # Too small
            (output_dir / "seg001.ts").write_bytes(b"x" * 2000)  # OK

            with patch("ffmpeg_session.get_hls_segment_duration", return_value=3.0):
                _regenerate_playlist(output_dir, start_segment=0)

            content = (output_dir / "stream.m3u8").read_text()
            assert "seg001.ts" in content
            assert "seg000.ts" not in content


# =============================================================================
# Session Recovery Tests
# =============================================================================


class TestCleanupAndRecoverSessions:
    """Tests for cleanup_and_recover_sessions."""

    def setup_method(self):
        _clear_session_state()

    def teardown_method(self):
        _clear_session_state()

    def test_removes_orphaned_dirs(self):
        """Removes dirs without session.json."""
        with tempfile.TemporaryDirectory() as tmp:
            transcode_dir = pathlib.Path(tmp)
            orphan = transcode_dir / "netv_transcode_orphan"
            orphan.mkdir()
            (orphan / "seg000.ts").write_bytes(b"data")

            with (
                patch("ffmpeg_session.get_transcode_dir", return_value=transcode_dir),
                patch("ffmpeg_session.get_vod_cache_timeout", return_value=3600),
            ):
                cleanup_and_recover_sessions()

            assert not orphan.exists()

    def test_recovers_valid_vod_session(self):
        """Recovers valid VOD session with segments."""
        with tempfile.TemporaryDirectory() as tmp:
            transcode_dir = pathlib.Path(tmp)
            vod_dir = transcode_dir / "netv_transcode_vod123"
            vod_dir.mkdir()

            # Create session.json
            session_info = {
                "session_id": "vod123",
                "url": "http://movie.mp4",
                "is_vod": True,
                "started": time.time(),
                "duration": 3600,
            }
            (vod_dir / "session.json").write_text(json.dumps(session_info))
            (vod_dir / "seg000.ts").write_bytes(b"x" * 2000)

            with (
                patch("ffmpeg_session.get_transcode_dir", return_value=transcode_dir),
                patch("ffmpeg_session.get_vod_cache_timeout", return_value=3600),
            ):
                cleanup_and_recover_sessions()

            assert "vod123" in _transcode_sessions
            assert _url_to_session.get("http://movie.mp4") == "vod123"

    def test_removes_expired_vod_session(self):
        """Removes expired VOD session (older than cache timeout)."""
        with tempfile.TemporaryDirectory() as tmp:
            transcode_dir = pathlib.Path(tmp)
            vod_dir = transcode_dir / "netv_transcode_expired"
            vod_dir.mkdir()

            session_info = {
                "session_id": "expired",
                "url": "http://old.mp4",
                "is_vod": True,
            }
            (vod_dir / "session.json").write_text(json.dumps(session_info))
            (vod_dir / "seg000.ts").write_bytes(b"x" * 2000)

            # Very short cache timeout
            with (
                patch("ffmpeg_session.get_transcode_dir", return_value=transcode_dir),
                patch("ffmpeg_session.get_vod_cache_timeout", return_value=0),
            ):
                cleanup_and_recover_sessions()

            assert not vod_dir.exists()
            assert "expired" not in _transcode_sessions


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)
