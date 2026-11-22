"""Authentication: users, passwords, tokens, JWT."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final, NamedTuple, NotRequired, TypedDict

import base64
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import time

from cryptography.fernet import Fernet, InvalidToken
from treekle import MutablePlainTree, ReadError, from_plain, json_loads, mutable, to_plain

from netv.util import atomic_write_json, data_dir, merge_unknown, read_json_object


_CWD: Final[Path] = Path(__file__).resolve().parent


class UserRecord(TypedDict):
    """A stored account; ``password`` is a salted PBKDF2 hash."""

    password: str
    admin: bool
    max_streams_per_source: NotRequired[dict[str, int]]
    unavailable_groups: NotRequired[list[str]]
    lan_only: NotRequired[bool]  # Usable only from the home network.


class UserLimits(TypedDict):
    """A user's per-source stream limits and blocked groups."""

    max_streams_per_source: dict[str, int]
    unavailable_groups: list[str]


class UserSummary(UserLimits):
    """Public account metadata, without the password hash."""

    username: str
    admin: bool
    lan_only: bool


class TokenClaims(TypedDict):
    """Claims signed into a session token."""

    sub: str
    exp: NotRequired[int]
    pwd: NotRequired[str]  # Marks the password the session began with.


class LastAdminError(ValueError):
    """A change that would leave other accounts without an administrator."""


# The home network: private, loopback, link-local, and unique-local ranges. Not
# Python's is_private, which also counts routed tunnels (Teredo, 6to4) as private.
_HOME_NETWORKS: Final = tuple(
    ipaddress.ip_network(n)
    for n in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)

_SESSION_SECONDS: Final = 86_400 * 7
# PBKDF2-SHA256 iterations for new password hashes: OWASP's 2023 figure.
_ROUNDS: Final = 600_000


APP_DIR = _CWD
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
    _require_an_admin(users)
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
    settings = read_json_object(_get_settings_file())
    users = from_plain(settings.get("users", {}), dict[str, UserRecord])
    # An unknown name still costs one hash, so timing does not tell it apart.
    stored = users[username]["password"] if username in users else _hash_password(password)
    valid = _verify_hashed_password(password, stored) and username in users
    # An older, cheaper hash is replaced the first time its password is given.
    if valid and not stored.startswith(f"pbkdf2_sha256${_ROUNDS}$"):
        users[username]["password"] = _hash_password(password)
        _save_users(settings, users)
    return valid


def change_password(
    username: str,
    new_password: str,
    *,
    current_password: str | None = None,
) -> bool:
    """Change an existing user's password.

    Args:
        username: Account to update.
        new_password: Plaintext replacement password.
        current_password: When given, the change happens only if it matches,
            checked against the same read that is rewritten.

    Returns:
        changed: Whether the account existed (with a matching current
            password, when one was given) and was updated.
    """
    settings_file = _get_settings_file()
    if not settings_file.exists():
        return False
    settings = read_json_object(settings_file)
    users = from_plain(settings.get("users", {}), dict[str, UserRecord])
    if username not in users:
        return False
    if current_password is not None and not _verify_hashed_password(
        current_password,
        users[username]["password"],
    ):
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
    _require_an_admin(users)
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
            "lan_only": d.get("lan_only", False),
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


def set_lan_only(username: str, *, lan_only: bool) -> bool:
    """Limit an account to the home network, or lift the limit.

    Args:
        username: Account to update.
        lan_only: Whether the account may be used only from a private address.

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
    users[username]["lan_only"] = lan_only
    _save_users(settings, users)
    return True


def allowed_from(username: str, address: str) -> bool:
    """Whether ``username`` may sign in or stay signed in from ``address``.

    A LAN-only account is allowed only from a private, loopback, or link-local
    address. Carrier-grade NAT (100.64.0.0/10) is the carrier's, not the home's.

    Args:
        username: Account to check.
        address: The client's IP address, as the server sees it.

    Returns:
        allowed: False for an account that does not exist, and for a LAN-only
            one away from the home network.
    """
    record = _user_record(username)
    return record is not None and _allowed(record, address)


def is_home_address(address: str) -> bool:
    """Whether ``address`` is on the home network: private, loopback, or link-local.

    Args:
        address: An IP address as the server sees it.

    Returns:
        home: False for public addresses, carrier NAT, tunnels, and non-addresses.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    named = [ipaddress.ip_network(n, strict=False) for n in home_networks()]
    return any(ip in network for network in named or _HOME_NETWORKS)


def home_networks() -> list[str]:
    """The networks an admin named as home; empty for the private ranges."""
    named = read_json_object(_get_settings_file()).get("home_networks", [])
    return [str(n) for n in named] if isinstance(named, list) else []


def set_home_networks(networks: list[str]) -> bool:
    """Name the home network, as CIDRs; none for every private range.

    A proxy on this host makes every request arrive from 127.0.0.1, which the
    private ranges count as home; naming the LAN exactly keeps it away.

    Args:
        networks: CIDRs such as ``192.168.1.0/24``.

    Returns:
        saved: False, saving nothing, when any is not a network.
    """
    try:
        for network in networks:
            ipaddress.ip_network(network, strict=False)
    except ValueError:
        return False
    settings = read_json_object(_get_settings_file())
    settings["home_networks"] = list(networks)
    atomic_write_json(_get_settings_file(), settings)
    return True


def session_user(token: str, address: str) -> TokenClaims | None:
    """The claims of a session that is still good from ``address``.

    Args:
        token: Session token from the client.
        address: The client's IP address, as the server sees it.

    Returns:
        claims: None for a forged or expired token, a deleted account, a password
            changed since sign-in, or a LAN-only account away from home.
    """
    claims = verify_token(token)
    if claims is None:
        return None
    record = _user_record(claims["sub"])
    if record is None or not _allowed(record, address):
        return None
    mark = claims.get("pwd")
    if mark is None or not hmac.compare_digest(mark, _password_mark(record["password"])):
        return None
    return claims


_HEX_DIGITS = frozenset("0123456789abcdef")


def logo_url_signature(url: str) -> str:
    """HMAC of a logo URL the server chose to proxy through ``/api/logo``.

    Only signed URLs are fetched, so the endpoint cannot be pointed at
    arbitrary hosts (internal services included) by any signed-in viewer.

    Args:
        url: External logo URL the server put on a page.

    Returns:
        signature: Hex HMAC-SHA256 of the label and URL under the secret key.
    """
    data = b"logo-url\n" + url.encode()
    return hmac.new(_get_secret_key().encode(), data, hashlib.sha256).hexdigest()


def verify_logo_url_signature(url: str, signature: str) -> bool:
    """Whether ``signature`` is the server's ``logo_url_signature`` for ``url``."""
    # Reject anything but 64 lowercase hex digits before compare_digest, which
    # raises TypeError for non-ASCII strings.
    if len(signature) != 64 or not _HEX_DIGITS.issuperset(signature):
        return False
    return hmac.compare_digest(signature, logo_url_signature(url))


def create_token(payload: TokenClaims, *, expiry_seconds: int = _SESSION_SECONDS) -> str:
    """Create a signed JWT-like token, marked with its account's current password.

    Args:
        payload: Claims to sign; ``sub`` names the account.
        expiry_seconds: How long the token lasts.

    Returns:
        token: Hex-encoded claims and their HMAC-SHA256 signature.
    """
    claims: TokenClaims = {**payload, "exp": int(time.time()) + expiry_seconds}
    record = _user_record(payload["sub"])
    if record is not None:
        claims["pwd"] = _password_mark(record["password"])
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
        # compare_digest raises TypeError for a non-ASCII signature.
        if not hmac.compare_digest(sig.encode(), expected.encode()):
            return None
        payload = json_loads(data.decode(), TokenClaims)
    except (OSError, ValueError):
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload


class StreamTarget(NamedTuple):
    """What a ``stream_ref`` names."""

    username: str
    url: str
    lan: bool  # Whether the stream started on the home network.


def stream_ref(username: str, url: str, *, lan: bool = False) -> str:
    """An opaque reference to an upstream URL, issued to ``username``.

    Upstream URLs carry the provider's account and password, so browsers get this
    instead; the server alone turns it back into the URL. It is encrypted, not
    just signed, and only this server can make one. Like a session, it ends when
    its viewer's password changes or account goes.

    Args:
        username: Viewer the stream is issued to.
        url: Upstream media URL.
        lan: Whether the stream started on the home network.

    Returns:
        ref: URL-safe text.
    """
    return stream_ref_issuer(username, lan=lan)(url)


def stream_ref_issuer(username: str, *, lan: bool) -> Callable[[str], str]:
    """``stream_ref`` for many URLs, such as a playlist's, reading the account once.

    Args:
        username: Viewer the streams are issued to.
        lan: Whether the streams started on the home network.

    Returns:
        issue: Maps an upstream URL to its ref.
    """
    record = _user_record(username)
    mark = "" if record is None else _password_mark(record["password"])
    cipher = _stream_cipher()

    def issue(url: str) -> str:
        data = json.dumps([username, url, mark, lan], separators=(",", ":")).encode()
        return cipher.encrypt(data).decode().rstrip("=")

    return issue


def stream_ref_target(ref: str) -> StreamTarget | None:
    """What a ``stream_ref`` names, while it is still good.

    Args:
        ref: A reference from ``stream_ref``.

    Returns:
        target: None for a forged or expired reference, or one whose viewer's
            account is gone or password changed since.
    """
    try:
        padded = ref.encode("ascii") + b"=" * (-len(ref) % 4)
        data = _stream_cipher().decrypt(padded, ttl=_SESSION_SECONDS)
        # An earlier release's refs held fewer fields.
        username, url, mark, lan = json_loads(data.decode(), tuple[str, str, str, bool])
    except (InvalidToken, UnicodeEncodeError, ReadError):
        return None
    record = _user_record(username)
    if record is None or not hmac.compare_digest(mark, _password_mark(record["password"])):
        return None
    return StreamTarget(username, url, lan)


def _stream_cipher() -> Fernet:
    key = hmac.new(_get_secret_key().encode(), b"stream-ref", hashlib.sha256).digest()
    return Fernet(base64.urlsafe_b64encode(key))


def _get_settings_file() -> Path:
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


_secret_key_cache: tuple[Path, str] | None = None


def _get_secret_key() -> str:
    """Get or generate secret key (persisted in settings)."""
    global _secret_key_cache
    settings_file = _get_settings_file()
    # Pages sign every poster URL, so reading the settings file per call stalls renders.
    if _secret_key_cache is not None and _secret_key_cache[0] == settings_file:
        return _secret_key_cache[1]
    settings = read_json_object(settings_file)
    key = settings.get("secret_key")
    if not isinstance(key, str):
        key = secrets.token_hex(32)
        settings["secret_key"] = key
        atomic_write_json(settings_file, settings)
    _secret_key_cache = (settings_file, key)
    return key


def _hash_password(password: str, salt: str | None = None, *, rounds: int = _ROUNDS) -> str:
    """Hash a password with PBKDF2-SHA256, recording its cost beside its salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), rounds)
    return f"pbkdf2_sha256${rounds}${salt}${key.hex()}"


# Hashes from before costs were recorded, ``<salt>:<key hex>``, used 100,000 rounds.
def _verify_hashed_password(password: str, hashed: str) -> bool:
    """Whether ``password`` matches ``hashed``, at the cost it records."""
    parts = hashed.split("$")
    legacy = re.fullmatch(r"([0-9a-f]+):[0-9a-f]+", hashed)
    if len(parts) == 4 and parts[0] == "pbkdf2_sha256" and parts[1].isdigit():
        expected = _hash_password(password, parts[2], rounds=int(parts[1]))
    elif legacy is not None:
        salt = legacy.group(1)
        old = hashlib.pbkdf2_hmac(
            "sha256",  # pragma: no mutate[case-insensitive]
            password.encode(),
            salt.encode(),
            100_000,
        )
        expected = f"{salt}:{old.hex()}"
    else:
        return False
    return hmac.compare_digest(expected, hashed)


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


def _require_an_admin(users: dict[str, UserRecord]) -> None:
    """Refuse a change that leaves accounts without an administrator."""
    if users and not any(u["admin"] for u in users.values()):
        raise LastAdminError("Make another user an admin first")


def _allowed(record: UserRecord, address: str) -> bool:
    return not record.get("lan_only", False) or is_home_address(address)


def _user_record(username: str) -> UserRecord | None:
    return from_plain(
        read_json_object(_get_settings_file()).get("users", {}),
        dict[str, UserRecord],
    ).get(username)


def _password_mark(hashed: str) -> str:
    """A mark of a stored password hash; a new password has a new mark."""
    return hmac.new(
        _get_secret_key().encode(),
        b"session\n" + hashed.encode(),
        hashlib.sha256,
    ).hexdigest()
