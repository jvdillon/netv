"""Authentication: users, passwords, tokens, JWT."""

from __future__ import annotations

import hashlib
import hmac
import json
import pathlib
import secrets
import time
from typing import Any


APP_DIR = pathlib.Path(__file__).parent
CACHE_DIR = APP_DIR / "cache"
SERVER_SETTINGS_FILE = CACHE_DIR / "server_settings.json"
USERS_DIR = CACHE_DIR / "users"
TOKEN_EXPIRY = 86400 * 7  # 7 days


def _get_settings_file() -> pathlib.Path:
    """Get the settings file."""
    return SERVER_SETTINGS_FILE


def _get_secret_key() -> str:
    """Get or generate secret key (persisted in settings)."""
    settings_file = _get_settings_file()
    settings = {}
    if settings_file.exists():
        settings = json.loads(settings_file.read_text())
    if "secret_key" not in settings:
        settings["secret_key"] = secrets.token_hex(32)
        settings_file.write_text(json.dumps(settings, indent=2))
    return settings["secret_key"]


def _hash_password(password: str, salt: str | None = None) -> str:
    """Hash password with salt using PBKDF2."""
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return f"{salt}:{key.hex()}"


def _verify_hashed_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    if ":" not in hashed:
        return False  # Invalid hash format
    salt, _ = hashed.split(":", 1)
    return hmac.compare_digest(_hash_password(password, salt), hashed)


def _get_users() -> dict[str, dict[str, Any]]:
    """Get users from settings. Returns empty dict if no users configured.

    User format: {username: {password: str, admin: bool}}
    """
    settings_file = _get_settings_file()
    if settings_file.exists():
        settings = json.loads(settings_file.read_text())
        return settings.get("users", {})
    return {}


def get_all_usernames() -> list[str]:
    """Get list of all usernames."""
    return list(_get_users().keys())


def is_setup_required() -> bool:
    """Check if initial setup is required (no users configured)."""
    return len(_get_users()) == 0


def create_user(username: str, password: str, admin: bool = False) -> None:
    """Create a new user with hashed password."""
    settings_file = _get_settings_file()
    settings = {}
    if settings_file.exists():
        settings = json.loads(settings_file.read_text())
    users = settings.get("users", {})
    # First user is always admin
    if len(users) == 0:
        admin = True
    users[username] = {"password": _hash_password(password), "admin": admin}
    settings["users"] = users
    settings_file.write_text(json.dumps(settings, indent=2))
    # Create user directory for per-user settings
    user_dir = USERS_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)


def _ensure_one_admin(users: dict[str, dict[str, Any]]) -> None:
    """Ensure at least one user is admin. Promotes first user if needed."""
    if not users or any(u.get("admin") for u in users.values()):
        return
    next(iter(users.values()))["admin"] = True


def delete_user(username: str) -> bool:
    """Delete a user. Returns True if deleted, False if not found."""
    settings_file = _get_settings_file()
    if not settings_file.exists():
        return False
    settings = json.loads(settings_file.read_text())
    users = settings.get("users", {})
    if username not in users:
        return False
    del users[username]
    _ensure_one_admin(users)
    settings["users"] = users
    settings_file.write_text(json.dumps(settings, indent=2))
    return True


def verify_password(username: str, password: str) -> bool:
    """Verify username and password."""
    users = _get_users()
    user_data = users.get(username, {"password": _hash_password("dummy")})
    stored = user_data["password"]
    valid = _verify_hashed_password(password, stored)
    return valid and username in users


def change_password(username: str, new_password: str) -> bool:
    """Change a user's password. Returns True if successful."""
    settings_file = _get_settings_file()
    if not settings_file.exists():
        return False
    settings = json.loads(settings_file.read_text())
    users = settings.get("users", {})
    if username not in users:
        return False
    users[username]["password"] = _hash_password(new_password)
    settings["users"] = users
    settings_file.write_text(json.dumps(settings, indent=2))
    return True


def is_admin(username: str) -> bool:
    """Check if user is an admin."""
    users = _get_users()
    user_data = users.get(username, {})
    return user_data.get("admin", False)


def set_admin(username: str, admin: bool) -> bool:
    """Set admin status for a user. Returns True if successful."""
    settings_file = _get_settings_file()
    if not settings_file.exists():
        return False
    settings = json.loads(settings_file.read_text())
    users = settings.get("users", {})
    if username not in users:
        return False
    users[username]["admin"] = admin
    _ensure_one_admin(users)
    settings["users"] = users
    settings_file.write_text(json.dumps(settings, indent=2))
    return True


def get_users_with_admin() -> list[dict[str, Any]]:
    """Get list of users with their admin status."""
    users = _get_users()
    return [{"username": u, "admin": d.get("admin", False)} for u, d in users.items()]


def create_token(payload: dict[str, Any]) -> str:
    """Create a signed JWT-like token."""
    payload = {**payload, "exp": int(time.time()) + TOKEN_EXPIRY}
    data = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(_get_secret_key().encode(), data, hashlib.sha256).hexdigest()
    return f"{data.hex()}.{sig}"


def verify_token(token: str) -> dict[str, Any] | None:
    """Verify token and return payload, or None if invalid/expired."""
    try:
        data_hex, sig = token.split(".")
        data = bytes.fromhex(data_hex)
        expected = hmac.new(_get_secret_key().encode(), data, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(data)
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None
