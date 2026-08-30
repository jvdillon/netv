"""Tests for the native-player gateway."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import json
import urllib.error

from fastapi.testclient import TestClient

import pytest

from gateway_catalog import CatalogSnapshot, GatewayCatalog, GatewayStream, StreamIdRegistry

import auth
import cache
import gateway
import gateway_catalog


class _FakeCatalog:
    def __init__(self, snapshot: CatalogSnapshot) -> None:
        self.snapshot = snapshot

    def get(self) -> CatalogSnapshot:
        return self.snapshot


@pytest.fixture
def gateway_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings_file = tmp_path / "server_settings.json"
    users_dir = tmp_path / "users"
    users_dir.mkdir()
    monkeypatch.setattr(auth, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(auth, "SERVER_SETTINGS_FILE", settings_file)
    monkeypatch.setattr(auth, "USERS_DIR", users_dir)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "SERVER_SETTINGS_FILE", settings_file)
    monkeypatch.setattr(cache, "USERS_DIR", users_dir)
    auth.create_user("player", "local-pass")
    monkeypatch.setattr(gateway, "authenticator", gateway.GatewayAuthenticator())

    public = {
        "num": 1,
        "name": "News One",
        "stream_type": "live",
        "stream_id": 41,
        "stream_icon": "https://images.example/news.png",
        "epg_channel_id": "news.one",
        "added": "",
        "category_id": "src_1_news",
        "category_ids": ["src_1_news"],
        "custom_sid": "",
        "tv_archive": 0,
        "tv_archive_duration": 0,
    }
    stream = GatewayStream(
        local_id=41,
        source_id="src_1",
        source_type="xtream",
        upstream_id="987",
        direct_url="",
        access_group_ids=("src_1_news",),
        public=public,
    )
    snapshot = CatalogSnapshot(
        categories=[
            {
                "category_id": "src_1_news",
                "category_name": "News",
                "parent_id": 0,
            }
        ],
        category_access_ids={"src_1_news": "src_1_news"},
        streams=[stream],
        streams_by_id={41: stream},
    )
    monkeypatch.setattr(gateway, "catalog", _FakeCatalog(snapshot))
    return TestClient(gateway.app), snapshot


def test_healthcheck_does_not_require_auth(gateway_client):
    client, _ = gateway_client
    assert client.get("/healthz").json() == {"status": "ok"}


def test_player_api_rejects_invalid_credentials(gateway_client):
    client, _ = gateway_client
    response = client.get(
        "/player_api.php",
        params={"username": "player", "password": "wrong"},
    )
    assert response.status_code == 200
    assert response.json()["user_info"]["auth"] == 0


def test_player_api_exposes_active_server_profile(gateway_client):
    client, _ = gateway_client
    response = client.get(
        "/player_api.php",
        params={"username": "player", "password": "local-pass"},
    )
    assert response.status_code == 200
    assert response.json()["user_info"] == {
        "username": "player",
        "password": "local-pass",
        "message": "neTV native-player gateway",
        "auth": 1,
        "status": "Active",
        "exp_date": None,
        "is_trial": "0",
        "active_cons": "0",
        "created_at": "0",
        "max_connections": "1",
        "allowed_output_formats": ["ts"],
    }


def test_server_profile_uses_configured_public_url(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    monkeypatch.setenv("NETV_GATEWAY_PUBLIC_URL", "https://tv.example:8443")

    response = client.get(
        "/player_api.php",
        params={"username": "player", "password": "local-pass"},
    )

    assert response.json()["server_info"] == {
        "url": "tv.example",
        "port": "8443",
        "https_port": "8443",
        "server_protocol": "https",
        "rtmp_port": "0",
        "timezone": "UTC",
        "timestamp_now": response.json()["server_info"]["timestamp_now"],
        "time_now": response.json()["server_info"]["time_now"],
    }


def test_successful_authentication_is_cached(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    verify_password = MagicMock(return_value=True)
    monkeypatch.setattr(auth, "verify_password", verify_password)

    for _ in range(2):
        response = client.get(
            "/player_api.php",
            params={"username": "player", "password": "local-pass"},
        )
        assert response.status_code == 200

    verify_password.assert_called_once_with("player", "local-pass")


def test_failed_authentication_is_rate_limited(gateway_client):
    client, _ = gateway_client
    params = {"username": "player", "password": "wrong"}

    for _ in range(10):
        assert client.get("/player_api.php", params=params).status_code == 200

    assert client.get("/player_api.php", params=params).status_code == 429


def test_success_cache_is_checked_before_client_rate_limit(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    valid = {"username": "player", "password": "local-pass"}
    assert client.get("/player_api.php", params=valid).status_code == 200
    monkeypatch.setattr(gateway, "_MAX_LOGIN_FAILURES", 1)
    assert client.get(
        "/player_api.php",
        params={"username": "player", "password": "wrong"},
    ).status_code == 200

    assert client.get("/player_api.php", params=valid).status_code == 200


def test_player_api_exposes_local_live_catalog(gateway_client):
    client, _ = gateway_client
    response = client.get(
        "/player_api.php",
        params={
            "username": "player",
            "password": "local-pass",
            "action": "get_live_streams",
        },
    )
    assert response.status_code == 200
    assert response.json() == [
        {
            "num": 1,
            "name": "News One",
            "stream_type": "live",
            "stream_id": 41,
            "stream_icon": "https://images.example/news.png",
            "epg_channel_id": "news.one",
            "added": "",
            "category_id": "src_1_news",
            "category_ids": ["src_1_news"],
            "custom_sid": "",
            "tv_archive": 0,
            "tv_archive_duration": 0,
        }
    ]


def test_playlist_contains_only_local_playback_urls(gateway_client):
    client, _ = gateway_client
    response = client.get(
        "/get.php",
        params={"username": "player", "password": "local-pass", "output": "ts"},
    )
    assert response.status_code == 200
    assert "http://testserver/live/player/local-pass/41.ts" in response.text
    assert "provider.example/live" not in response.text
    assert 'group-title="News"' in response.text


def test_user_category_restrictions_apply_to_gateway(gateway_client):
    client, _ = gateway_client
    assert auth.set_user_limits("player", unavailable_groups=["cat:src_1_news"])
    response = client.get(
        "/player_api.php",
        params={
            "username": "player",
            "password": "local-pass",
            "action": "get_live_streams",
        },
    )
    assert response.json() == []


def test_uncategorized_streams_hidden_for_restricted_user(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    _, snapshot = gateway_client
    stream = snapshot.streams[0]
    uncategorized = GatewayStream(
        local_id=stream.local_id,
        source_id=stream.source_id,
        source_type=stream.source_type,
        upstream_id=stream.upstream_id,
        direct_url=stream.direct_url,
        access_group_ids=(),
        public={**stream.public, "category_id": "99", "category_ids": ["99"]},
    )
    restricted_snapshot = CatalogSnapshot(
        categories=[{"category_id": "99", "category_name": "Uncategorized", "parent_id": 0}],
        category_access_ids={},
        streams=[uncategorized],
        streams_by_id={uncategorized.local_id: uncategorized},
    )
    monkeypatch.setattr(
        auth,
        "get_user_limits",
        lambda _username: {"unavailable_groups": ["cat:some-other-category"]},
    )

    assert gateway._allowed_category_ids("player", restricted_snapshot) == set()
    assert gateway._visible_streams("player", restricted_snapshot) == []


def test_live_stream_proxies_resolved_upstream(gateway_client, monkeypatch: pytest.MonkeyPatch):
    client, _ = gateway_client
    upstream = MagicMock()
    upstream.headers = {"Content-Type": "video/mp2t"}
    upstream.read.side_effect = [b"video-data", b""]
    fake_xtream = MagicMock()
    fake_xtream.build_stream_url.return_value = (
        "https://provider.example/live/remote-user/remote-pass/987.ts"
    )
    monkeypatch.setattr(gateway, "get_xtream_client_by_source", lambda _source_id: fake_xtream)
    open_mock = MagicMock(return_value=upstream)
    monkeypatch.setattr(gateway, "safe_urlopen", open_mock)

    response = client.get("/live/player/local-pass/41.ts")

    assert response.status_code == 200
    assert response.content == b"video-data"
    fake_xtream.build_stream_url.assert_called_once_with("live", 987, "ts")
    assert open_mock.call_args.args[0].endswith("/987.ts")
    upstream.close.assert_called_once()


def test_live_stream_enforces_category_restrictions(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    assert auth.set_user_limits("player", unavailable_groups=["cat:src_1_news"])
    open_mock = MagicMock()
    monkeypatch.setattr(gateway, "safe_urlopen", open_mock)

    response = client.get("/live/player/local-pass/41.ts")

    assert response.status_code == 404
    open_mock.assert_not_called()


def test_xmltv_builds_feed_for_visible_streams(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, snapshot = gateway_client
    build_mock = MagicMock(return_value=b'<?xml version="1.0"?><tv />')
    monkeypatch.setattr(gateway, "build_filtered_xmltv", build_mock)

    response = client.get(
        "/xmltv.php",
        params={"username": "player", "password": "local-pass"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert build_mock.call_args.args[1] == snapshot.streams


def test_xmltv_enforces_category_restrictions(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    assert auth.set_user_limits("player", unavailable_groups=["cat:src_1_news"])
    build_mock = MagicMock(return_value=b'<?xml version="1.0"?><tv />')
    monkeypatch.setattr(gateway, "build_filtered_xmltv", build_mock)

    response = client.get(
        "/xmltv.php",
        params={"username": "player", "password": "local-pass"},
    )

    assert response.status_code == 200
    assert build_mock.call_args.args[1] == []


def test_stream_id_registry_is_stable(tmp_path: Path):
    path = tmp_path / "stream_ids.json"
    first = StreamIdRegistry(path)
    assigned = first.get_or_create_many(["source-a:live:7", "source-b:live:7"])

    second = StreamIdRegistry(path)
    assert second.get_or_create("source-a:live:7") == assigned["source-a:live:7"]
    assert second.get_or_create("source-b:live:7") == assigned["source-b:live:7"]
    assert assigned["source-a:live:7"] != assigned["source-b:live:7"]


def test_stream_id_registry_rejects_invalid_existing_file(tmp_path: Path):
    path = tmp_path / "stream_ids.json"
    path.write_text("[]")

    with pytest.raises(RuntimeError, match="Invalid gateway stream ID registry"):
        StreamIdRegistry(path).get_or_create("source-a:live:7")

    assert path.read_text() == "[]"


def test_catalog_removes_upstream_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = SimpleNamespace(
        id="source-a",
        name="Provider",
        type="xtream",
        url="https://provider.example",
        username="remote-user",
        password="remote-password",
        epg_enabled=True,
    )
    monkeypatch.setattr(cache, "get_sources", lambda: [source])
    monkeypatch.setattr(
        gateway_catalog,
        "fetch_source_live_data",
        lambda _source, **_kwargs: (
            [{"category_id": "source-a_9", "category_name": "News", "parent_id": 0}],
            [
                {
                    "stream_id": 7,
                    "name": "News",
                    "category_ids": ["source-a_9"],
                    "source_username": "remote-user",
                    "source_password": "remote-password",
                    "source_url": "https://provider.example",
                }
            ],
            "https://provider.example/xmltv.php",
            120,
        ),
    )
    snapshot = GatewayCatalog(
        registry=StreamIdRegistry(tmp_path / "ids.json"),
        ttl_seconds=60,
    ).get()

    encoded = json.dumps(snapshot.streams[0].public)
    assert snapshot.categories[0]["category_id"].isdigit()
    assert snapshot.streams[0].public["category_id"] == snapshot.categories[0]["category_id"]
    assert "remote-user" not in encoded
    assert "remote-password" not in encoded
    assert "provider.example" not in encoded


def test_catalog_assigns_uncategorized_streams_numeric_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = SimpleNamespace(
        id="source-a",
        name="Provider",
        type="xtream",
        epg_enabled=False,
    )
    monkeypatch.setattr(cache, "get_sources", lambda: [source])
    monkeypatch.setattr(
        gateway_catalog,
        "fetch_source_live_data",
        lambda _source, **_kwargs: (
            [],
            [{"stream_id": 7, "name": "Uncategorized", "category_ids": []}],
            None,
            120,
        ),
    )
    snapshot = GatewayCatalog(
        registry=StreamIdRegistry(tmp_path / "ids.json"),
        ttl_seconds=60,
    ).get()

    category_id = snapshot.categories[0]["category_id"]
    assert category_id.isdigit()
    assert snapshot.streams[0].public["category_id"] == category_id
    assert snapshot.streams[0].public["category_ids"] == [category_id]


def test_m3u_registry_key_uses_url_instead_of_playlist_position(tmp_path: Path):
    catalog = GatewayCatalog(
        registry=StreamIdRegistry(tmp_path / "ids.json"),
        ttl_seconds=60,
    )
    first = catalog._stream_registry_key(
        "source-a",
        "m3u",
        {
            "stream_id": "source-a_1",
            "direct_url": "https://streams.example/news?token=first",
            "epg_channel_id": "news.example",
            "name": "News",
        },
    )
    reordered = catalog._stream_registry_key(
        "source-a",
        "m3u",
        {
            "stream_id": "source-a_42",
            "direct_url": "https://streams.example/news?token=rotated",
            "epg_channel_id": "news.example",
            "name": "News",
        },
    )
    different = catalog._stream_registry_key(
        "source-a",
        "m3u",
        {
            "stream_id": "source-a_1",
            "direct_url": "https://streams.example/sports",
            "epg_channel_id": "sports.example",
            "name": "Sports",
        },
    )

    assert first == reordered
    assert first != different


def test_catalog_assigns_distinct_ids_to_colliding_m3u_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = SimpleNamespace(
        id="source-a",
        name="Provider",
        type="m3u",
        epg_enabled=False,
    )
    monkeypatch.setattr(cache, "get_sources", lambda: [source])
    monkeypatch.setattr(
        gateway_catalog,
        "fetch_source_live_data",
        lambda _source, **_kwargs: (
            [],
            [
                {
                    "stream_id": "source-a_1",
                    "name": "News",
                    "direct_url": "https://streams.example/play?id=1",
                    "category_ids": [],
                },
                {
                    "stream_id": "source-a_2",
                    "name": "News",
                    "direct_url": "https://streams.example/play?id=2",
                    "category_ids": [],
                },
            ],
            None,
            120,
        ),
    )

    snapshot = GatewayCatalog(
        registry=StreamIdRegistry(tmp_path / "ids.json"),
        ttl_seconds=60,
    ).get()

    assert len(snapshot.streams) == 2
    assert len(snapshot.streams_by_id) == 2
    assert snapshot.streams[0].local_id != snapshot.streams[1].local_id


def test_catalog_retains_previous_snapshot_when_refresh_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = SimpleNamespace(
        id="source-a",
        name="Provider",
        type="xtream",
        epg_enabled=False,
    )
    monkeypatch.setattr(cache, "get_sources", lambda: [source])
    source_data = (
        [{"category_id": "source-a_9", "category_name": "News", "parent_id": 0}],
        [{"stream_id": 7, "name": "News", "category_ids": ["source-a_9"]}],
        None,
        120,
    )
    fetch = MagicMock(return_value=source_data)
    monkeypatch.setattr(gateway_catalog, "fetch_source_live_data", fetch)
    catalog = GatewayCatalog(
        registry=StreamIdRegistry(tmp_path / "ids.json"),
        ttl_seconds=60,
    )
    original = catalog.get()
    fetch.side_effect = urllib.error.URLError("temporary failure")

    refreshed = catalog.get(force=True)

    assert refreshed is original
    assert len(refreshed.streams) == 1


def test_playlist_accepts_mpegts_output(gateway_client):
    client, _ = gateway_client
    response = client.get(
        "/get.php",
        params={
            "username": "player",
            "password": "local-pass",
            "output": "mpegts",
        },
    )
    assert response.status_code == 200


def test_live_stream_password_may_contain_slash(
    gateway_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _ = gateway_client
    upstream = MagicMock()
    upstream.headers = {"Content-Type": "video/mp2t"}
    upstream.read.side_effect = [b"video-data", b""]
    monkeypatch.setattr(
        gateway.authenticator,
        "verify",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(gateway, "_resolve_upstream", lambda _stream, _extension: "http://upstream")
    monkeypatch.setattr(gateway, "safe_urlopen", MagicMock(return_value=upstream))

    response = client.get("/live/player/local%2Fpass/41.ts")

    assert response.status_code == 200
    assert response.content == b"video-data"
