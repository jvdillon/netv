"""Tests for credentials.py."""

from __future__ import annotations

from pathlib import Path

import json
import stat

from cryptography.fernet import Fernet

import pytest

from netv import cache, credentials


@pytest.fixture(autouse=True)
def no_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(credentials.KEY_ENV, raising=False)
    monkeypatch.delenv(credentials.KEY_FILE_ENV, raising=False)
    monkeypatch.setattr(credentials, "_warned_default_key", False)


@pytest.fixture
def settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point cache at a temp settings file, as cache_test does."""
    monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", {"nvenc": False})
    monkeypatch.setattr(cache, "SERVER_SETTINGS_FILE", tmp_path / "server_settings.json")
    monkeypatch.setattr(cache, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache, "LOGOS_DIR", tmp_path / "logos")
    return tmp_path / "server_settings.json"


SOURCE = {
    "id": "s1",
    "name": "Example",
    "type": "xtream",
    "url": "http://provider.example.test:8080",
    "username": "alice",
    "password": "hunter2",
    "epg_url": "http://provider.example.test/xmltv.php?username=alice&password=hunter2",
}


def _save(source: dict[str, object]) -> None:
    settings = cache.load_server_settings()
    settings["sources"] = [source]  # type: ignore[list-item]
    cache.save_server_settings(settings)


class TestValues:
    def test_round_trip_and_plain_text_passes_through(self, tmp_path: Path):
        token = credentials.encrypt("hunter2", tmp_path)
        assert token.startswith("enc:v1:")
        assert "hunter2" not in token
        assert credentials.decrypt(token, tmp_path) == "hunter2"
        assert credentials.decrypt("plain", tmp_path) == "plain"

    def test_empty_values_stay_empty(self, tmp_path: Path):
        assert credentials.encrypt("", tmp_path) == ""
        assert not (tmp_path / "credentials.key").exists()

    def test_a_value_that_looks_encrypted_is_still_encrypted(self, tmp_path: Path):
        lookalike = "enc:v1:my-password"
        token = credentials.encrypt(lookalike, tmp_path)
        assert token != lookalike
        assert credentials.decrypt(token, tmp_path) == lookalike

    def test_an_empty_key_file_left_by_an_interrupted_write_is_replaced(self, tmp_path: Path):
        (tmp_path / "credentials.key").write_bytes(b"")
        with pytest.raises(credentials.CredentialsKeyError, match="no key is configured"):
            credentials.decrypt("enc:v1:token", tmp_path)
        token = credentials.encrypt("x", tmp_path)
        assert (tmp_path / "credentials.key").read_bytes().strip()
        assert credentials.decrypt(token, tmp_path) == "x"
        assert not list(tmp_path.glob(".credentials.key.*"))

    def test_default_key_file_is_private(self, tmp_path: Path):
        credentials.encrypt("x", tmp_path)
        mode = stat.S_IMODE((tmp_path / "credentials.key").stat().st_mode)
        assert mode == 0o600

    def test_missing_key_fails_instead_of_returning_nothing(self, tmp_path: Path):
        token = credentials.encrypt("x", tmp_path)
        (tmp_path / "credentials.key").unlink()
        with pytest.raises(credentials.CredentialsKeyError, match="no key is configured"):
            credentials.decrypt(token, tmp_path)
        # Reading must not create a new, unrelated key.
        assert not (tmp_path / "credentials.key").exists()

    def test_wrong_key_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        token = credentials.encrypt("x", tmp_path)
        monkeypatch.setenv(credentials.KEY_ENV, Fernet.generate_key().decode())
        with pytest.raises(credentials.CredentialsKeyError, match="cannot be decrypted"):
            credentials.decrypt(token, tmp_path)

    def test_key_from_environment_writes_no_key_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(credentials.KEY_ENV, Fernet.generate_key().decode())
        token = credentials.encrypt("x", tmp_path)
        assert credentials.decrypt(token, tmp_path) == "x"
        assert not (tmp_path / "credentials.key").exists()

    def test_key_file_outside_the_data_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        secret = tmp_path / "secrets" / "netv.key"
        secret.parent.mkdir()
        secret.write_bytes(Fernet.generate_key() + b"\n")
        monkeypatch.setenv(credentials.KEY_FILE_ENV, str(secret))
        data = tmp_path / "data"
        token = credentials.encrypt("x", data)
        assert credentials.decrypt(token, data) == "x"
        assert not (data / "credentials.key").exists()


class TestSettingsFile:
    def test_saved_file_holds_no_plain_credentials(self, settings_file: Path):
        _save(dict(SOURCE))
        text = settings_file.read_text()
        assert "hunter2" not in text
        assert "alice" not in text
        assert "provider.example.test" not in text
        stored = json.loads(text)["sources"][0]
        assert stored["name"] == "Example"
        assert stored["type"] == "xtream"
        loaded = cache.load_server_settings()["sources"][0]
        assert {k: loaded.get(k) for k in SOURCE} == SOURCE

    def test_plain_text_from_older_releases_is_read_then_encrypted_on_save(
        self, settings_file: Path
    ):
        settings_file.write_text(json.dumps({"sources": [SOURCE], "secret_key": "s"}))
        settings = cache.load_server_settings()
        assert settings["sources"][0].get("password") == "hunter2"
        cache.save_server_settings(settings)
        stored = json.loads(settings_file.read_text())
        assert stored["secret_key"] == "s"
        assert credentials.is_encrypted(stored["sources"][0]["password"])
        assert cache.load_server_settings()["sources"][0].get("password") == "hunter2"

    def test_saving_twice_keeps_the_values(self, settings_file: Path):
        _save(dict(SOURCE))
        cache.save_server_settings(cache.load_server_settings())
        loaded = cache.load_server_settings()["sources"][0]
        assert loaded.get("password") == "hunter2"
        assert loaded.get("epg_url") == SOURCE["epg_url"]

    def test_a_password_that_looks_encrypted_survives_save_and_load(self, settings_file: Path):
        _save({**SOURCE, "password": "enc:v1:not-a-token"})
        assert cache.load_server_settings()["sources"][0].get("password") == "enc:v1:not-a-token"

    def test_lost_key_stops_loading_instead_of_blanking_credentials(self, settings_file: Path):
        _save(dict(SOURCE))
        before = settings_file.read_text()
        (settings_file.parent / "credentials.key").unlink()
        with pytest.raises(credentials.CredentialsKeyError):
            cache.load_server_settings()
        assert settings_file.read_text() == before
