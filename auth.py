"""Authentication: users, passwords, tokens, JWT."""

from __future__ import annotations

from typing import NotRequired, TypedDict

import hashlib
import hmac
import json
import pathlib
import secrets
import time

from treekle import MutablePlainTree, from_plain, json_loads, mutable, to_plain

from util import atomic_write_json, data_dir, merge_unknown, read_json_object


class UserRecord(TypedDict):
    """A stored account; ``password`` is a salted PBKDF2 hash."""

    password: str
    admin: bool
    max_streams_per_source: NotRequired[dict[str, int]]
    unavailable_groups: NotRequired[list[str]]


class UserLimits(TypedDict):
    """A user's per-source stream limits and blocked groups."""

    max_streams_per_source: dict[str, int]
    unavailable_groups: list[str]


class UserSummary(UserLimits):
    """Public account metadata, without the password hash."""

    username: str
    admin: bool


class TokenClaims(TypedDict):
    """Claims signed into a session token."""

    sub: str
    exp: NotRequired[int]


APP_DIR = pathlib.Path(__file__).parent
CACHE_DIR = APP_DIR / ".cache"
SERVER_SETTINGS_FILE = CACHE_DIR / "server_settings.json"
USERS_DIR = CACHE_DIR / "users"


def get_all_usernames() -> list[str]:
    """Get list of all usernames.

    Returns:
        usernames: Account names in stored order.
    """
    return list(
        from_plain(
            read_json_object(_get_settings_file()).get("users", {}),
            dict[str, UserRecord],
        ).keys(),
    )


def is_setup_required() -> bool:
    """Check if initial setup is required (no users configured).

    Returns:
        required: Whether no account exists yet.
    """
    return (
        len(
            from_plain(
                read_json_object(_get_settings_file()).get("users", {}),
                dict[str, UserRecord],
            ),
        )
        == 0
    )


def create_user(username: str, password: str, admin: bool = False) -> None:
    """Create a new user with hashed password.

    Args:
        username: Account name.
        password: Plaintext password to hash.
        admin: Grant administrator access; the first account is always admin.
    """
    settings = read_json_object(_get_settings_file())
    users = from_plain(settings.get("users", {}), dict[str, UserRecord])
    # First user is always admin.
    if len(users) == 0:
        admin = True
    users[username] = {"password": _hash_password(password), "admin": admin}
    _save_users(settings, users)
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
    settings = read_json_object(settings_file)
    users = from_plain(settings.get("users", {}), dict[str, UserRecord])
    if username not in users:
        return False
    del users[username]
    _ensure_one_admin(users)
    # Drop the user from source assignments so a future account with the same
    # name does not inherit access.
    sources = settings.get("sources", [])
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict):
                assigned = source.get("users")
                if isinstance(assigned, list) and username in assigned:
                    source["users"] = [u for u in assigned if u != username]
    _save_users(settings, users)
    return True


def verify_password(username: str, password: str) -> bool:
    """Verify username and password.

    Args:
        username: Account to check.
        password: Plaintext password to verify.

    Returns:
        valid: Whether the account exists and the password matches.
    """
    users = from_plain(
        read_json_object(_get_settings_file()).get("users", {}),
        dict[str, UserRecord],
    )
    stored = users[username]["password"] if username in users else _hash_password("dummy")
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
    settings = read_json_object(settings_file)
    users = from_plain(settings.get("users", {}), dict[str, UserRecord])
    if username not in users:
        return False
    users[username]["password"] = _hash_password(new_password)
    _save_users(settings, users)
    return True


def is_admin(username: str) -> bool:
    """Check if user is an admin."""
    users = from_plain(
        read_json_object(_get_settings_file()).get("users", {}),
        dict[str, UserRecord],
    )
    return username in users and users[username]["admin"]


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
    settings = read_json_object(settings_file)
    users = from_plain(settings.get("users", {}), dict[str, UserRecord])
    if username not in users:
        return False
    users[username]["admin"] = admin
    _ensure_one_admin(users)
    _save_users(settings, users)
    return True


def get_users_with_admin() -> list[UserSummary]:
    """Get users with their administrator status and limits.

    Returns:
        users: Public account metadata without password hashes.
    """
    return [
        {
            "username": u,
            "admin": d["admin"],
            "max_streams_per_source": d.get("max_streams_per_source", {}),
            "unavailable_groups": d.get("unavailable_groups", []),
        }
        for u, d in from_plain(
            read_json_object(_get_settings_file()).get("users", {}),
            dict[str, UserRecord],
        ).items()
    ]


def get_user_limits(username: str) -> UserLimits:
    """Get a user's stream limits and group restrictions.

    Args:
        username: Account to look up.

    Returns:
        limits: Per-source stream limits and unavailable groups.
    """
    user_data = from_plain(
        read_json_object(_get_settings_file()).get("users", {}),
        dict[str, UserRecord],
    ).get(username)
    if user_data is None:
        return {"max_streams_per_source": {}, "unavailable_groups": []}
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
    settings = read_json_object(settings_file)
    users = from_plain(settings.get("users", {}), dict[str, UserRecord])
    if username not in users:
        return False
    if max_streams_per_source is not None:
        users[username]["max_streams_per_source"] = max_streams_per_source
    if unavailable_groups is not None:
        users[username]["unavailable_groups"] = unavailable_groups
    _save_users(settings, users)
    return True


def create_token(payload: TokenClaims, *, expiry_seconds: int = 86_400 * 7) -> str:
    """Create a signed JWT-like token."""
    claims: TokenClaims = {**payload, "exp": int(time.time()) + expiry_seconds}
    data = json.dumps(claims, separators=(",", ":")).encode()
    sig = hmac.new(_get_secret_key().encode(), data, hashlib.sha256).hexdigest()
    return f"{data.hex()}.{sig}"


def verify_token(token: str) -> TokenClaims | None:
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
        payload = json_loads(data.decode(), TokenClaims)
    except (OSError, ValueError):
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload


def _get_settings_file() -> pathlib.Path:
    """Resolve legacy storage without replacing caller-supplied paths."""
    global CACHE_DIR, SERVER_SETTINGS_FILE, USERS_DIR
    default_dir = APP_DIR / ".cache"
    if default_dir == CACHE_DIR:
        CACHE_DIR = data_dir(APP_DIR)
    if default_dir / "server_settings.json" == SERVER_SETTINGS_FILE:
        SERVER_SETTINGS_FILE = CACHE_DIR / "server_settings.json"
    if default_dir / "users" == USERS_DIR:
        USERS_DIR = CACHE_DIR / "users"
    return SERVER_SETTINGS_FILE


def _get_secret_key() -> str:
    """Get or generate secret key (persisted in settings)."""
    settings_file = _get_settings_file()
    settings = read_json_object(settings_file)
    key = settings.get("secret_key")
    if isinstance(key, str):
        return key
    key = secrets.token_hex(32)
    settings["secret_key"] = key
    atomic_write_json(settings_file, settings)
    return key


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


# Unknown keys survive both beside ``users`` and inside each user's record; a deleted
# user, or an owned key the record no longer has, is removed.
def _save_users(settings: dict[str, MutablePlainTree], users: dict[str, UserRecord]) -> None:
    """Write ``users`` into the raw settings tree, keeping keys this release does not know."""
    settings["users"] = merge_unknown(
        settings.get("users"),
        mutable(to_plain(users)),
        dict[str, UserRecord],
    )
    atomic_write_json(_get_settings_file(), settings)


def _ensure_one_admin(users: dict[str, UserRecord]) -> None:
    """Ensure at least one user is admin. Promotes first user if needed."""
    if not users or any(u["admin"] for u in users.values()):
        return
    next(iter(users.values()))["admin"] = True
