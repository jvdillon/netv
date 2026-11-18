"""Tests for credentials.py."""

from __future__ import annotations

from pathlib import Path

import json

from cryptography.fernet import Fernet

import pytest

from netv import cache, credentials


@pytest.fixture(autouse=True)
def no_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(credentials.KEY_ENV, raising=False)
    monkeypatch.delenv(credentials.KEY_FILE_ENV, raising=False)


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Enable encryption with a fresh key from the environment."""
    value = Fernet.generate_key().decode()
    monkeypatch.setenv(credentials.KEY_ENV, value)
    return value


@pytest.fixture
def settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point cache at a temp settings file, as cache_test does."""
    monkeypatch.setattr(cache, "AVAILABLE_ENCODERS", {"nvenc": False})
    monkeypatch.setattr(cache, "SERVER_SETTINGS_FILE", tmp_path / "server_settings.json")
    monkeypatch.setattr(cache, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cache, "LOGOS_DIR", tmp_path / "logos")
    return tmp_path / "server_settings.json"


SOURCE: cache.SourceConfig = {
    "id": "s1",
    "name": "Example",
    "type": "xtream",
    "url": "http://provider.example.test:8080",
    "username": "alice",
    "password": "hunter2",
    "epg_url": "http://provider.example.test/xmltv.php?username=alice&password=hunter2",
}


def _save(source: cache.SourceConfig) -> None:
    settings = cache.load_server_settings()
    settings["sources"] = [source]
    cache.save_server_settings(settings)


class TestValues:
    def test_without_a_key_values_stay_plain(self):
        assert credentials.encrypt("hunter2") == "hunter2"
        assert credentials.decrypt("hunter2") == "hunter2"

    @pytest.mark.usefixtures("key")
    def test_round_trip_and_plain_text_passes_through(self):
        token = credentials.encrypt("hunter2")
        assert token.startswith("enc:v1:")
        assert "hunter2" not in token
        assert credentials.decrypt(token) == "hunter2"
        assert credentials.decrypt("plain") == "plain"

    @pytest.mark.usefixtures("key")
    def test_empty_values_stay_empty(self):
        assert credentials.encrypt("") == ""

    @pytest.mark.usefixtures("key")
    def test_a_value_that_looks_encrypted_is_still_encrypted(self):
        lookalike = "enc:v1:my-password"
        token = credentials.encrypt(lookalike)
        assert token != lookalike
        assert credentials.decrypt(token) == lookalike

    def test_a_value_that_looks_encrypted_needs_a_key(self):
        # As plain text it would read back as encrypted and stop the settings loading.
        with pytest.raises(credentials.CredentialsKeyError, match="can only be saved with"):
            credentials.encrypt("enc:v1:my-password")

    def test_missing_key_fails_instead_of_returning_nothing(
        self,
        key: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        del key
        token = credentials.encrypt("x")
        monkeypatch.delenv(credentials.KEY_ENV)
        with pytest.raises(credentials.CredentialsKeyError, match="no key is configured"):
            credentials.decrypt(token)

    def test_wrong_key_fails(self, key: str, monkeypatch: pytest.MonkeyPatch):
        del key
        token = credentials.encrypt("x")
        monkeypatch.setenv(credentials.KEY_ENV, Fernet.generate_key().decode())
        message = r"^Source credentials cannot be decrypted with the configured key\.$"
        with pytest.raises(credentials.CredentialsKeyError, match=message):
            credentials.decrypt(token)

    @pytest.mark.parametrize("bad", ["not-a-key", "c2hvcnQ=", "é" * 44])
    def test_invalid_key_in_environment_is_a_key_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bad: str,
    ):
        monkeypatch.setenv(credentials.KEY_ENV, bad)
        for call in (lambda: credentials.encrypt("x"), lambda: credentials.decrypt("enc:v1:t")):
            with pytest.raises(
                credentials.CredentialsKeyError,
                match="not a valid Fernet key",
            ) as e:
                call()
            assert bad not in str(e.value)

    def test_key_file_outside_the_data_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        secret = tmp_path / "secrets" / "netv.key"
        secret.parent.mkdir()
        secret.write_bytes(Fernet.generate_key() + b"\n")
        monkeypatch.setenv(credentials.KEY_FILE_ENV, str(secret))
        token = credentials.encrypt("x")
        assert token.startswith("enc:v1:")
        assert credentials.decrypt(token) == "x"

    def test_unreadable_key_file_is_a_key_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv(credentials.KEY_FILE_ENV, str(tmp_path / "missing.key"))
        with pytest.raises(credentials.CredentialsKeyError, match="cannot be read"):
            credentials.encrypt("x")

    def test_invalid_key_file_is_a_key_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        secret = tmp_path / "netv.key"
        secret.write_bytes(b"not-a-key\n")
        monkeypatch.setenv(credentials.KEY_FILE_ENV, str(secret))
        with pytest.raises(credentials.CredentialsKeyError, match="not a valid Fernet key"):
            credentials.decrypt("enc:v1:t")


class TestSettingsFile:
    def test_without_a_key_the_file_is_plain_as_before(self, settings_file: Path):
        _save(SOURCE)
        stored = json.loads(settings_file.read_text())["sources"][0]
        assert {k: stored.get(k) for k in SOURCE} == SOURCE
        assert sorted(p.name for p in settings_file.parent.iterdir()) == ["server_settings.json"]

    @pytest.mark.usefixtures("key")
    def test_saved_file_holds_no_plain_credentials(self, settings_file: Path):
        _save(SOURCE)
        text = settings_file.read_text()
        assert "hunter2" not in text
        assert "alice" not in text
        assert "provider.example.test" not in text
        stored = json.loads(text)["sources"][0]
        assert stored["name"] == "Example"
        assert stored["type"] == "xtream"
        loaded = cache.load_server_settings()["sources"][0]
        assert {k: loaded.get(k) for k in SOURCE} == SOURCE

    def test_plain_settings_are_encrypted_on_the_first_save_with_a_key(
        self,
        settings_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        settings_file.write_text(json.dumps({"sources": [SOURCE], "secret_key": "s"}))
        monkeypatch.setenv(credentials.KEY_ENV, Fernet.generate_key().decode())
        settings = cache.load_server_settings()
        assert settings["sources"][0].get("password") == "hunter2"
        cache.save_server_settings(settings)
        stored = json.loads(settings_file.read_text())
        assert stored["secret_key"] == "s"
        assert credentials.is_encrypted(stored["sources"][0]["password"])
        assert cache.load_server_settings()["sources"][0].get("password") == "hunter2"

    @pytest.mark.usefixtures("key")
    def test_saving_twice_keeps_the_values(self, settings_file: Path):
        _save(SOURCE)
        cache.save_server_settings(cache.load_server_settings())
        loaded = cache.load_server_settings()["sources"][0]
        assert loaded.get("password") == "hunter2"
        assert loaded.get("epg_url") == SOURCE.get("epg_url")

    @pytest.mark.usefixtures("key")
    def test_a_password_that_looks_encrypted_survives_save_and_load(self, settings_file: Path):
        _save({**SOURCE, "password": "enc:v1:not-a-token"})
        assert cache.load_server_settings()["sources"][0].get("password") == "enc:v1:not-a-token"

    def test_lost_key_stops_loading_instead_of_blanking_credentials(
        self,
        settings_file: Path,
        key: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        del key
        _save(SOURCE)
        before = settings_file.read_text()
        monkeypatch.delenv(credentials.KEY_ENV)
        with pytest.raises(credentials.CredentialsKeyError):
            cache.load_server_settings()
        assert settings_file.read_text() == before


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
