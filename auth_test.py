"""Tests for auth.py."""

from __future__ import annotations

from pathlib import Path

import json

import pytest

import auth


@pytest.fixture
def auth_module(tmp_path: Path):
    """Import auth module with temp settings file."""

    # Patch settings files to temp location.
    original_server = auth.SERVER_SETTINGS_FILE
    original_users = auth.USERS_DIR
    auth.SERVER_SETTINGS_FILE = tmp_path / "server_settings.json"
    auth.USERS_DIR = tmp_path / "users"
    auth.USERS_DIR.mkdir(exist_ok=True)
    yield auth
    auth.SERVER_SETTINGS_FILE = original_server
    auth.USERS_DIR = original_users


class TestPasswordHashing:
    def test_hash_password_creates_salt(self, auth_module):
        hashed = auth_module._hash_password("mypassword")
        assert ":" in hashed
        salt, key = hashed.split(":")
        assert len(salt) == 32  # 16 bytes hex.
        assert len(key) == 64  # 32 bytes hex.

    def test_hash_password_with_salt_deterministic(self, auth_module):
        salt = "a" * 32
        h1 = auth_module._hash_password("test", salt)
        h2 = auth_module._hash_password("test", salt)
        assert h1 == h2

    def test_verify_hashed_password_correct(self, auth_module):
        hashed = auth_module._hash_password("secret")
        assert auth_module._verify_hashed_password("secret", hashed)

    def test_verify_hashed_password_wrong(self, auth_module):
        hashed = auth_module._hash_password("secret")
        assert not auth_module._verify_hashed_password("wrong", hashed)


class TestUserManagement:
    def test_is_setup_required_no_users(self, auth_module):
        assert auth_module.is_setup_required()

    def test_create_user_and_verify(self, auth_module):
        auth_module.create_user("admin", "password123")
        assert not auth_module.is_setup_required()
        assert auth_module.verify_password("admin", "password123")
        assert not auth_module.verify_password("admin", "wrongpass")
        assert not auth_module.verify_password("nobody", "password123")

    def test_second_user_not_admin_and_usernames(self, auth_module):
        auth_module.create_user("alice", "pw")
        auth_module.create_user("bob", "pw")
        assert auth_module.get_all_usernames() == ["alice", "bob"]
        assert auth_module.is_admin("alice")
        assert not auth_module.is_admin("bob")
        assert not auth_module.is_admin("nobody")
        assert (auth_module.USERS_DIR / "bob").is_dir()

    def test_invalid_stored_hash_rejected(self, auth_module):
        assert not auth_module._verify_hashed_password("pw", "nocolon")


class TestDeleteUser:
    def test_no_settings_file(self, auth_module):
        assert not auth_module.delete_user("alice")

    def test_unknown_user(self, auth_module):
        auth_module.create_user("alice", "pw")
        assert not auth_module.delete_user("bob")

    def test_delete_promotes_admin_and_scrubs_sources(self, auth_module):
        auth_module.create_user("alice", "pw")
        auth_module.create_user("bob", "pw")
        auth_module.create_user("carol", "pw")
        f = auth_module.SERVER_SETTINGS_FILE
        settings = json.loads(f.read_text())
        settings["sources"] = [
            {"name": "a", "users": ["alice", "bob"]},
            {"name": "b", "users": ["bob"]},
            {"name": "c"},
            {"name": "d", "users": "alice"},
        ]
        f.write_text(json.dumps(settings))
        assert auth_module.delete_user("alice")
        assert auth_module.get_all_usernames() == ["bob", "carol"]
        assert auth_module.is_admin("bob")
        assert not auth_module.is_admin("carol")
        sources = json.loads(f.read_text())["sources"]
        assert sources[0]["users"] == ["bob"]
        assert sources[1]["users"] == ["bob"]
        assert "users" not in sources[2]
        assert sources[3]["users"] == "alice"

    def test_delete_last_user(self, auth_module):
        auth_module.create_user("alice", "pw")
        assert auth_module.delete_user("alice")
        assert auth_module.is_setup_required()


class TestChangePassword:
    def test_no_settings_file(self, auth_module):
        assert not auth_module.change_password("alice", "new")

    def test_unknown_user(self, auth_module):
        auth_module.create_user("alice", "pw")
        assert not auth_module.change_password("bob", "new")

    def test_changes_password(self, auth_module):
        auth_module.create_user("alice", "pw")
        assert auth_module.change_password("alice", "new")
        assert auth_module.verify_password("alice", "new")
        assert not auth_module.verify_password("alice", "pw")


class TestSetAdmin:
    def test_no_settings_file(self, auth_module):
        assert not auth_module.set_admin("alice", True)

    def test_unknown_user(self, auth_module):
        auth_module.create_user("alice", "pw")
        assert not auth_module.set_admin("bob", True)

    def test_grant_and_revoke(self, auth_module):
        auth_module.create_user("alice", "pw")
        auth_module.create_user("bob", "pw")
        assert auth_module.set_admin("bob", True)
        assert auth_module.is_admin("bob")
        assert auth_module.set_admin("alice", False)
        assert not auth_module.is_admin("alice")

    def test_revoking_last_admin_keeps_one(self, auth_module):
        auth_module.create_user("alice", "pw")
        auth_module.create_user("bob", "pw")
        assert auth_module.set_admin("alice", False)
        assert auth_module.is_admin("alice")
        assert not auth_module.is_admin("bob")


class TestUserLimits:
    def test_defaults(self, auth_module):
        auth_module.create_user("alice", "pw")
        assert auth_module.get_user_limits("alice") == {
            "max_streams_per_source": {},
            "unavailable_groups": [],
        }
        assert auth_module.get_user_limits("nobody") == {
            "max_streams_per_source": {},
            "unavailable_groups": [],
        }

    def test_no_settings_file(self, auth_module):
        assert not auth_module.set_user_limits("alice", {"s": 1})

    def test_unknown_user(self, auth_module):
        auth_module.create_user("alice", "pw")
        assert not auth_module.set_user_limits("bob", {"s": 1})

    def test_set_and_preserve(self, auth_module):
        auth_module.create_user("alice", "pw")
        assert auth_module.set_user_limits("alice", {"s": 2}, ["Adult"])
        assert auth_module.set_user_limits("alice")
        assert auth_module.get_user_limits("alice") == {
            "max_streams_per_source": {"s": 2},
            "unavailable_groups": ["Adult"],
        }
        assert auth_module.set_user_limits("alice", unavailable_groups=[])
        assert auth_module.get_users_with_admin() == [
            {
                "username": "alice",
                "admin": True,
                "max_streams_per_source": {"s": 2},
                "unavailable_groups": [],
            },
        ]


class TestGetSettingsFile:
    def test_legacy_dir_used_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(auth, "APP_DIR", tmp_path)
        monkeypatch.setattr(auth, "CACHE_DIR", tmp_path / ".cache")
        monkeypatch.setattr(
            auth,
            "SERVER_SETTINGS_FILE",
            tmp_path / ".cache" / "server_settings.json",
        )
        monkeypatch.setattr(auth, "USERS_DIR", tmp_path / ".cache" / "users")
        (tmp_path / "cache").mkdir()
        assert auth._get_settings_file() == tmp_path / "cache" / "server_settings.json"
        assert tmp_path / "cache" == auth.CACHE_DIR
        assert tmp_path / "cache" / "users" == auth.USERS_DIR

    def test_default_dir_without_legacy(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").touch()
        monkeypatch.delenv("NETV_DATA_DIR", raising=False)
        monkeypatch.setattr(auth, "APP_DIR", tmp_path)
        monkeypatch.setattr(auth, "CACHE_DIR", tmp_path / ".cache")
        monkeypatch.setattr(
            auth,
            "SERVER_SETTINGS_FILE",
            tmp_path / ".cache" / "server_settings.json",
        )
        monkeypatch.setattr(auth, "USERS_DIR", tmp_path / ".cache" / "users")
        assert auth._get_settings_file() == tmp_path / ".cache" / "server_settings.json"
        assert tmp_path / ".cache" / "users" == auth.USERS_DIR


class TestTokens:
    def test_create_and_verify_token(self, auth_module):
        payload = {"user": "admin", "role": "admin"}
        token = auth_module.create_token(payload)
        result = auth_module.verify_token(token)
        assert result is not None
        assert result["user"] == "admin"
        assert result["role"] == "admin"
        assert "exp" in result

    def test_token_format(self, auth_module):
        token = auth_module.create_token({"test": 1})
        assert "." in token
        _, sig = token.split(".")
        assert len(sig) == 64  # sha256 hex.

    def test_invalid_token_rejected(self, auth_module):
        assert auth_module.verify_token("invalid") is None
        assert auth_module.verify_token("abc.def") is None
        assert auth_module.verify_token("") is None

    def test_tampered_token_rejected(self, auth_module):
        token = auth_module.create_token({"user": "admin"})
        # Tamper with signature.
        data, _ = token.split(".")
        tampered = f"{data}.{'0' * 64}"
        assert auth_module.verify_token(tampered) is None

    def test_expired_token_rejected(self, auth_module):
        # Create token with expired time.
        token = auth_module.create_token({"user": "admin"}, expiry_seconds=-1)
        assert auth_module.verify_token(token) is None


class TestSecretKey:
    def test_get_secret_key_generates_and_persists(self, auth_module):
        key1 = auth_module._get_secret_key()
        assert len(key1) == 64  # 32 bytes hex.

        # Should return same key.
        key2 = auth_module._get_secret_key()
        assert key1 == key2

        # Should be persisted (uses _get_settings_file which returns legacy if server doesn't exist)
        settings_file = auth_module._get_settings_file()
        settings = json.loads(settings_file.read_text())
        assert settings["secret_key"] == key1


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
