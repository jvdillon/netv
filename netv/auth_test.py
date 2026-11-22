"""Tests for auth.py."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import hashlib
import hmac
import json

import pytest

from netv import auth


@pytest.fixture
def auth_module(tmp_path: Path) -> Iterator[None]:
    """Point auth at a temp settings file."""
    original_server = auth.SERVER_SETTINGS_FILE
    original_users = auth.USERS_DIR
    auth.SERVER_SETTINGS_FILE = tmp_path / "server_settings.json"
    auth.USERS_DIR = tmp_path / "users"
    auth.USERS_DIR.mkdir(exist_ok=True)
    yield
    auth.SERVER_SETTINGS_FILE = original_server
    auth.USERS_DIR = original_users


@pytest.mark.usefixtures("auth_module")
class TestPasswordHashing:
    def test_hash_password_creates_salt(self):
        hashed = auth._hash_password("mypassword")
        assert ":" in hashed
        salt, key = hashed.split(":")
        assert len(salt) == 32  # 16 bytes hex.
        assert len(key) == 64  # 32 bytes hex.

    def test_hash_password_with_salt_deterministic(self):
        salt = "a" * 32
        h1 = auth._hash_password("test", salt)
        h2 = auth._hash_password("test", salt)
        assert h1 == h2

    def test_verify_hashed_password_correct(self):
        hashed = auth._hash_password("secret")
        assert auth._verify_hashed_password("secret", hashed)

    def test_verify_hashed_password_wrong(self):
        hashed = auth._hash_password("secret")
        assert not auth._verify_hashed_password("wrong", hashed)


@pytest.mark.usefixtures("auth_module")
class TestUserManagement:
    def test_is_setup_required_no_users(self):
        assert auth.is_setup_required()

    def test_create_user_and_verify(self):
        auth.create_user("admin", "password123")
        assert not auth.is_setup_required()
        assert auth.verify_password("admin", "password123")
        assert not auth.verify_password("admin", "wrongpass")
        assert not auth.verify_password("nobody", "password123")

    def test_second_user_not_admin_and_usernames(self):
        auth.create_user("alice", "pw")
        auth.create_user("bob", "pw")
        assert auth.get_all_usernames() == ["alice", "bob"]
        assert auth.is_admin("alice")
        assert not auth.is_admin("bob")
        assert not auth.is_admin("nobody")
        assert (auth.USERS_DIR / "bob").is_dir()

    def test_invalid_stored_hash_rejected(self):
        assert not auth._verify_hashed_password("pw", "nocolon")


@pytest.mark.usefixtures("auth_module")
class TestDeleteUser:
    def test_no_settings_file(self):
        assert not auth.delete_user("alice")

    def test_unknown_user(self):
        auth.create_user("alice", "pw")
        assert not auth.delete_user("bob")

    def test_delete_promotes_admin_and_scrubs_sources(self):
        auth.create_user("alice", "pw")
        auth.create_user("bob", "pw")
        auth.create_user("carol", "pw")
        f = auth.SERVER_SETTINGS_FILE
        settings = json.loads(f.read_text())
        settings["sources"] = [
            {"name": "a", "users": ["alice", "bob"]},
            {"name": "b", "users": ["bob"]},
            {"name": "c"},
            {"name": "d", "users": "alice"},
            "not-a-source",
        ]
        f.write_text(json.dumps(settings))
        assert auth.delete_user("alice")
        assert auth.get_all_usernames() == ["bob", "carol"]
        assert auth.is_admin("bob")
        assert not auth.is_admin("carol")
        sources = json.loads(f.read_text())["sources"]
        assert sources[0]["users"] == ["bob"]
        assert sources[1]["users"] == ["bob"]
        assert "users" not in sources[2]
        assert sources[3]["users"] == "alice"

    def test_delete_ignores_malformed_sources(self):
        auth.create_user("alice", "pw")
        auth.create_user("bob", "pw")
        f = auth.SERVER_SETTINGS_FILE
        settings = json.loads(f.read_text())
        settings["sources"] = {"alice": True}
        f.write_text(json.dumps(settings))
        assert auth.delete_user("alice")
        assert json.loads(f.read_text())["sources"] == {"alice": True}

    def test_delete_last_user(self):
        auth.create_user("alice", "pw")
        assert auth.delete_user("alice")
        assert auth.is_setup_required()


@pytest.mark.usefixtures("auth_module")
class TestChangePassword:
    def test_no_settings_file(self):
        assert not auth.change_password("alice", "new")

    def test_unknown_user(self):
        auth.create_user("alice", "pw")
        assert not auth.change_password("bob", "new")

    def test_changes_password(self):
        auth.create_user("alice", "pw")
        assert auth.change_password("alice", "new")
        assert auth.verify_password("alice", "new")
        assert not auth.verify_password("alice", "pw")

    def test_current_password_must_match(self):
        auth.create_user("alice", "pw")
        assert not auth.change_password("alice", "new", current_password="wrong")
        # Still "pw": the rejected change wrote nothing.
        assert auth.change_password("alice", "new", current_password="pw")


@pytest.mark.usefixtures("auth_module")
class TestSetAdmin:
    def test_no_settings_file(self):
        assert not auth.set_admin("alice", True)

    def test_unknown_user(self):
        auth.create_user("alice", "pw")
        assert not auth.set_admin("bob", True)

    def test_grant_and_revoke(self):
        auth.create_user("alice", "pw")
        auth.create_user("bob", "pw")
        assert auth.set_admin("bob", True)
        assert auth.is_admin("bob")
        assert auth.set_admin("alice", False)
        assert not auth.is_admin("alice")

    def test_revoking_last_admin_keeps_one(self):
        auth.create_user("alice", "pw")
        auth.create_user("bob", "pw")
        assert auth.set_admin("alice", False)
        assert auth.is_admin("alice")
        assert not auth.is_admin("bob")


@pytest.mark.usefixtures("auth_module")
class TestUserLimits:
    def test_defaults(self):
        auth.create_user("alice", "pw")
        assert auth.get_user_limits("alice") == {
            "max_streams_per_source": {},
            "unavailable_groups": [],
        }
        assert auth.get_user_limits("nobody") == {
            "max_streams_per_source": {},
            "unavailable_groups": [],
        }

    def test_no_settings_file(self):
        assert not auth.set_user_limits("alice", {"s": 1})

    def test_unknown_user(self):
        auth.create_user("alice", "pw")
        assert not auth.set_user_limits("bob", {"s": 1})

    def test_set_and_preserve(self):
        auth.create_user("alice", "pw")
        assert auth.set_user_limits("alice", {"s": 2}, ["Adult"])
        assert auth.set_user_limits("alice")
        assert auth.get_user_limits("alice") == {
            "max_streams_per_source": {"s": 2},
            "unavailable_groups": ["Adult"],
        }
        assert auth.set_user_limits("alice", unavailable_groups=[])
        assert auth.get_users_with_admin() == [
            {
                "username": "alice",
                "admin": True,
                "max_streams_per_source": {"s": 2},
                "unavailable_groups": [],
                "lan_only": False,
            },
        ]


class TestGetSettingsFile:
    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("NETV_DATA_DIR", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(auth, "APP_DIR", tmp_path)
        monkeypatch.setattr(auth, "CACHE_DIR", tmp_path / ".cache")
        monkeypatch.setattr(
            auth,
            "SERVER_SETTINGS_FILE",
            tmp_path / ".cache" / "server_settings.json",
        )
        monkeypatch.setattr(auth, "USERS_DIR", tmp_path / ".cache" / "users")

    def test_legacy_dir_moved_to_user_data_dir(self, tmp_path: Path):
        (tmp_path / "cache").mkdir()
        (tmp_path / "cache" / "server_settings.json").write_text("{}")
        state = tmp_path / "xdg" / "netv"
        assert auth._get_settings_file() == state / "server_settings.json"
        assert state == auth.CACHE_DIR
        assert state / "users" == auth.USERS_DIR
        assert (state / "server_settings.json").exists()

    def test_default_dir_without_legacy(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").touch()
        state = tmp_path / "xdg" / "netv"
        assert auth._get_settings_file() == state / "server_settings.json"
        assert state / "users" == auth.USERS_DIR


@pytest.mark.usefixtures("auth_module")
class TestTokens:
    def test_create_and_verify_token(self):
        token = auth.create_token({"sub": "admin"})
        result = auth.verify_token(token)
        assert result is not None
        assert result["sub"] == "admin"
        assert "exp" in result

    def test_token_without_subject_rejected(self):
        settings_file = auth._get_settings_file()
        auth._get_secret_key()
        data = b'{"exp":4102444800}'
        secret = json.loads(settings_file.read_text())["secret_key"]
        sig = hmac.new(secret.encode(), data, hashlib.sha256).hexdigest()
        assert auth.verify_token(f"{data.hex()}.{sig}") is None

    def test_token_format(self):
        token = auth.create_token({"sub": "tester"})
        assert "." in token
        _, sig = token.split(".")
        assert len(sig) == 64  # sha256 hex.

    def test_invalid_token_rejected(self):
        assert auth.verify_token("invalid") is None
        assert auth.verify_token("abc.def") is None
        assert auth.verify_token("") is None

    def test_tampered_token_rejected(self):
        token = auth.create_token({"sub": "admin"})
        # Tamper with signature.
        data, _ = token.split(".")
        tampered = f"{data}.{'0' * 64}"
        assert auth.verify_token(tampered) is None

    def test_expired_token_rejected(self):
        # Create token with expired time.
        token = auth.create_token({"sub": "admin"}, expiry_seconds=-1)
        assert auth.verify_token(token) is None


@pytest.mark.usefixtures("auth_module")
class TestUrlSignatures:
    def test_logo_url_signature_verifies_only_its_url(self):
        sig = auth.logo_url_signature("http://x/a.png")
        assert auth.verify_logo_url_signature("http://x/a.png", sig)
        assert not auth.verify_logo_url_signature("http://x/b.png", sig)

    @pytest.mark.parametrize("bad", ["", "0" * 63, "A" * 64, "é" * 64])
    def test_logo_url_signature_rejects_malformed(self, bad: str):
        assert not auth.verify_logo_url_signature("http://x/a.png", bad)

    def test_stream_url_token_is_bound_to_user_and_url(self):
        token = auth.stream_url_token("ann", "http://x/live")
        assert auth.verify_stream_url_token("ann", "http://x/live", token)
        assert not auth.verify_stream_url_token("bob", "http://x/live", token)
        assert not auth.verify_stream_url_token("ann", "http://x/other", token)

    def test_stream_url_token_differs_from_logo_signature(self):
        assert auth.stream_url_token("", "u") != auth.logo_url_signature("u")

    @pytest.mark.parametrize("bad", ["", "0" * 63, "A" * 64, "é" * 64])
    def test_stream_url_token_rejects_malformed(self, bad: str):
        assert not auth.verify_stream_url_token("ann", "http://x/live", bad)


@pytest.mark.usefixtures("auth_module")
class TestSecretKey:
    def test_get_secret_key_generates_and_persists(self):
        key1 = auth._get_secret_key()
        assert len(key1) == 64  # 32 bytes hex.

        # Should return same key.
        key2 = auth._get_secret_key()
        assert key1 == key2

        # Should be persisted (uses _get_settings_file which returns legacy if server doesn't exist)
        settings_file = auth._get_settings_file()
        settings = json.loads(settings_file.read_text())
        assert settings["secret_key"] == key1

    def test_get_secret_key_reads_settings_once(self):
        key = auth._get_secret_key()
        auth._get_settings_file().write_text("{}")
        assert auth._get_secret_key() == key


@pytest.mark.usefixtures("auth_module")
def test_a_lan_only_user_is_allowed_only_from_private_addresses():
    # Before any account exists there is no one to limit.
    assert not auth._get_settings_file().exists()
    assert not auth.set_lan_only("kid", lan_only=True)
    auth.create_user("admin", "pw")
    auth.create_user("kid", "pw")
    # Anyone not marked LAN-only is allowed from anywhere.
    assert auth.allowed_from("kid", "8.8.8.8")
    assert auth.set_lan_only("kid", lan_only=True)
    assert not auth.set_lan_only("ghost", lan_only=True)
    for home in (
        "192.168.1.5",
        "10.0.0.2",
        "172.16.4.4",
        "127.0.0.1",
        "::1",
        "fd00::7",
        "fe80::1",
        "169.254.3.3",
        "::ffff:192.168.1.9",
    ):
        assert auth.allowed_from("kid", home), home
    # CGNAT (100.64/10) is a carrier's network, not the home's.
    for away in ("8.8.8.8", "2001:4860::8888", "100.64.0.1", "::ffff:8.8.8.8", "testclient", ""):
        assert not auth.allowed_from("kid", away), away
    assert auth.allowed_from("admin", "8.8.8.8")
    assert [u["lan_only"] for u in auth.get_users_with_admin()] == [False, True]
    assert auth.set_lan_only("kid", lan_only=False)
    assert auth.allowed_from("kid", "8.8.8.8")


@pytest.mark.usefixtures("auth_module")
def test_user_writes_keep_unrelated_settings():
    settings_file = auth._get_settings_file()
    settings_file.write_text(json.dumps({"transcode_mode": "always", "future": {"x": [1]}}))
    auth.create_user("alice", "pw")
    auth.set_user_limits("alice", {"s": 1})
    auth._get_secret_key()
    saved = json.loads(settings_file.read_text())
    assert saved["transcode_mode"] == "always"
    assert saved["future"] == {"x": [1]}
    assert saved["users"]["alice"]["max_streams_per_source"] == {"s": 1}


@pytest.mark.usefixtures("auth_module")
def test_user_writes_keep_unknown_record_keys():
    auth.create_user("alice", "pw")
    auth.create_user("bob", "pw")
    auth.set_user_limits("alice", {"s": 1})
    settings_file = auth._get_settings_file()
    settings = json.loads(settings_file.read_text())
    settings["users"]["alice"]["theme"] = "dark"
    settings["users"]["alice"]["max_streams_per_source"] = {"s": 1}
    settings_file.write_text(json.dumps(settings))
    auth.change_password("alice", "new")
    auth.set_admin("bob", True)
    auth.delete_user("bob")
    alice = json.loads(settings_file.read_text())["users"]["alice"]
    assert alice["theme"] == "dark"
    assert alice["max_streams_per_source"] == {"s": 1}
    assert auth.verify_password("alice", "new")
    assert list(json.loads(settings_file.read_text())["users"]) == ["alice"]


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
