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
