"""Authentication: users, passwords, tokens, JWT."""

from __future__ import annotations

from typing import Any

import hashlib
import hmac
import json
import pathlib
import secrets
import time

from util import atomic_write_json


APP_DIR = pathlib.Path(__file__).parent
CACHE_DIR = APP_DIR / ".cache"
SERVER_SETTINGS_FILE = CACHE_DIR / "server_settings.json"
USERS_DIR = CACHE_DIR / "users"


def get_all_usernames() -> list[str]:
    """Get list of all usernames."""
    return list(_get_users().keys())


def is_setup_required() -> bool:
    """Check if initial setup is required (no users configured)."""
    return len(_get_users()) == 0


def create_user(username: str, password: str, admin: bool = False) -> None:
    """Create a new user with hashed password.

    Args:
        username: Account name.
        password: Plaintext password to hash.
        admin: Grant administrator access; the first account is always admin.
    """
    settings_file = _get_settings_file()
    settings = {}
    if settings_file.exists():
        settings = json.loads(settings_file.read_text())
    users = settings.get("users", {})
    # First user is always admin.
    if len(users) == 0:
        admin = True
    users[username] = {"password": _hash_password(password), "admin": admin}
    settings["users"] = users
    atomic_write_json(settings_file, settings)
    # Create user directory for per-user settings.
    user_dir = USERS_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)


def delete_user(username: str) -> bool:
    """Delete an account and its source assignments.

    Args:
        username: Account to remove.

    Returns:
        deleted: Whether the account existed.
    """
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
    # Drop the user from source assignments so a future account with the same
    # name does not inherit access.
    for source in settings.get("sources", []):
        if isinstance(source.get("users"), list) and username in source["users"]:
            source["users"] = [u for u in source["users"] if u != username]
    atomic_write_json(settings_file, settings)
    return True


def verify_password(username: str, password: str) -> bool:
    """Verify username and password."""
    users = _get_users()
    user_data = users.get(username, {"password": _hash_password("dummy")})
    stored = user_data["password"]
    valid = _verify_hashed_password(password, stored)
    return valid and username in users


def change_password(username: str, new_password: str) -> bool:
    """Change an existing user's password.

    Args:
        username: Account to update.
        new_password: Plaintext replacement password.

    Returns:
        changed: Whether the account existed and was updated.
    """
    settings_file = _get_settings_file()
    if not settings_file.exists():
        return False
    settings = json.loads(settings_file.read_text())
    users = settings.get("users", {})
    if username not in users:
        return False
    users[username]["password"] = _hash_password(new_password)
    settings["users"] = users
    atomic_write_json(settings_file, settings)
    return True


def is_admin(username: str) -> bool:
    """Check if user is an admin."""
    users = _get_users()
    user_data = users.get(username, {})
    return user_data.get("admin", False)


def set_admin(username: str, admin: bool) -> bool:
    """Set administrator access while retaining at least one administrator.

    Args:
        username: Account to update.
        admin: Requested administrator status.

    Returns:
        changed: Whether the account existed and was updated.
    """
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
    atomic_write_json(settings_file, settings)
    return True


def get_users_with_admin() -> list[dict[str, Any]]:
    """Get users with their administrator status and limits.

    Returns:
        users: Public account metadata without password hashes.
    """
    users = _get_users()
    return [
        {
            "username": u,
            "admin": d.get("admin", False),
            "max_streams_per_source": d.get("max_streams_per_source", {}),
            "unavailable_groups": d.get("unavailable_groups", []),
        }
        for u, d in users.items()
    ]


def get_user_limits(username: str) -> dict[str, Any]:
    """Get a user's stream limits and group restrictions.

    Args:
        username: Account to look up.

    Returns:
        limits: Per-source stream limits and unavailable groups.
    """
    users = _get_users()
    user_data = users.get(username, {})
    return {
        "max_streams_per_source": user_data.get("max_streams_per_source", {}),
        "unavailable_groups": user_data.get("unavailable_groups", []),
    }


def set_user_limits(
    username: str,
    max_streams_per_source: dict[str, int] | None = None,
    unavailable_groups: list[str] | None = None,
) -> bool:
    """Set a user's stream limits and group restrictions.

    Args:
        username: Account to update.
        max_streams_per_source: Replacement limits; None preserves existing limits.
        unavailable_groups: Replacement restrictions; None preserves existing groups.

    Returns:
        changed: Whether the account existed and was updated.
    """
    settings_file = _get_settings_file()
    if not settings_file.exists():
        return False
    settings = json.loads(settings_file.read_text())
    users = settings.get("users", {})
    if username not in users:
        return False
    if max_streams_per_source is not None:
        users[username]["max_streams_per_source"] = max_streams_per_source
    if unavailable_groups is not None:
        users[username]["unavailable_groups"] = unavailable_groups
    settings["users"] = users
    atomic_write_json(settings_file, settings)
    return True


def create_token(payload: dict[str, Any], *, expiry_seconds: int = 86_400 * 7) -> str:
    """Create a signed JWT-like token."""
    payload = {**payload, "exp": int(time.time()) + expiry_seconds}
    data = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(_get_secret_key().encode(), data, hashlib.sha256).hexdigest()
    return f"{data.hex()}.{sig}"


def verify_token(token: str) -> dict[str, Any] | None:
    """Verify a token's signature and expiry.

    Args:
        token: Hex-encoded payload and signature.

    Returns:
        payload: Authenticated payload, or None for an invalid or expired token.
    """
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


def _get_settings_file() -> pathlib.Path:
    """Resolve legacy storage without replacing caller-supplied paths."""
    global CACHE_DIR, SERVER_SETTINGS_FILE, USERS_DIR
    default_dir = APP_DIR / ".cache"
    legacy_dir = APP_DIR / "cache"
    if default_dir == CACHE_DIR and legacy_dir.exists():
        CACHE_DIR = legacy_dir
    if default_dir / "server_settings.json" == SERVER_SETTINGS_FILE:
        SERVER_SETTINGS_FILE = CACHE_DIR / "server_settings.json"
    if default_dir / "users" == USERS_DIR:
        USERS_DIR = CACHE_DIR / "users"
    return SERVER_SETTINGS_FILE


def _get_secret_key() -> str:
    """Get or generate secret key (persisted in settings)."""
    settings_file = _get_settings_file()
    settings = {}
    if settings_file.exists():
        settings = json.loads(settings_file.read_text())
    if "secret_key" not in settings:
        settings["secret_key"] = secrets.token_hex(32)
        atomic_write_json(settings_file, settings)
    return settings["secret_key"]


def _hash_password(password: str, salt: str | None = None) -> str:
    """Hash password with salt using PBKDF2."""
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}:{key.hex()}"


def _verify_hashed_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    if ":" not in hashed:
        return False  # Invalid hash format.
    salt, _ = hashed.split(":", 1)
    return hmac.compare_digest(_hash_password(password, salt), hashed)


# User format: {username: {password: str, admin: bool}}.
def _get_users() -> dict[str, dict[str, Any]]:
    """Get users from settings. Returns empty dict if no users configured."""
    settings_file = _get_settings_file()
    if settings_file.exists():
        settings = json.loads(settings_file.read_text())
        return settings.get("users", {})
    return {}


def _ensure_one_admin(users: dict[str, dict[str, Any]]) -> None:
    """Ensure at least one user is admin. Promotes first user if needed."""
    if not users or any(u.get("admin") for u in users.values()):
        return
    next(iter(users.values()))["admin"] = True
