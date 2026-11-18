"""Optional encryption of source credentials stored in ``server_settings.json``.

Off unless a key is configured, through either:

1. ``NETV_CREDENTIALS_KEY``: a Fernet key.
2. ``NETV_CREDENTIALS_KEY_FILE``: a file holding one, such as a Docker secret.

With a key, provider logins (the Xtream username and password, and URLs that
often carry them) are written as ``enc:v1:<Fernet token>`` and decrypted only in
memory. This protects the settings file when it is copied or shared without the
key, so keep the key somewhere else: losing it means adding every source again.

Without a key, values are stored as plain text, as in older releases. Plain text
is always readable, and is encrypted the next time the settings are saved once a
key is configured.
"""

from __future__ import annotations

import os
import pathlib

from cryptography.fernet import Fernet, InvalidToken
from treekle import MutablePlainTree


PREFIX = "enc:v1:"
KEY_ENV = "NETV_CREDENTIALS_KEY"
KEY_FILE_ENV = "NETV_CREDENTIALS_KEY_FILE"
# Source fields that hold provider logins or URLs that may embed them.
SECRET_FIELDS = ("url", "epg_url", "username", "password")


class CredentialsKeyError(RuntimeError):
    """The configured key is unusable, or stored credentials cannot be decrypted with it."""


def is_encrypted(value: object) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def _key() -> Fernet | None:
    """The configured key, or ``None`` when encryption is not enabled.

    Raises:
        CredentialsKeyError: A key is configured but cannot be read or is not a
            valid Fernet key.
    """
    configured = os.environ.get(KEY_ENV, "").strip()
    if configured:
        return _fernet(configured.encode(), KEY_ENV)
    key_file = os.environ.get(KEY_FILE_ENV, "").strip()
    if key_file:
        path = pathlib.Path(key_file).expanduser()
        try:
            key = path.read_bytes().strip()
        except OSError as exc:
            raise CredentialsKeyError(
                f"{KEY_FILE_ENV} names {path}, which cannot be read: {exc.strerror}."
            ) from None
        return _fernet(key, f"{KEY_FILE_ENV} ({path})")
    return None


def _fernet(key: bytes, origin: str) -> Fernet:
    try:
        return Fernet(key)
    except ValueError:
        # Fernet's own message can echo part of the key, so it is not passed on.
        raise CredentialsKeyError(
            f"{origin} is not a valid Fernet key (32 url-safe base64-encoded bytes)."
        ) from None


def encrypt(value: str) -> str:
    """Encrypt one in-memory value when a key is configured; otherwise return it as it is.

    Values reach this function decrypted, so a password that happens to start
    with ``enc:v1:`` is encrypted like any other; the prefix is only meaningful
    when reading the settings file.

    Raises:
        CredentialsKeyError: No key is configured and the value starts with
            ``enc:v1:``. Stored as plain text, it would read back as an
            encrypted value and stop the settings from loading.
    """
    if not value:
        return value
    fernet = _key()
    if fernet is None:
        if is_encrypted(value):
            raise CredentialsKeyError(
                f"A source value starting with {PREFIX!r} can only be saved with "
                f"{KEY_ENV} or {KEY_FILE_ENV} set."
            )
        return value
    return PREFIX + fernet.encrypt(value.encode()).decode()


def decrypt(value: str) -> str:
    """Decrypt one stored value; plain text passes through.

    Raises:
        CredentialsKeyError: The value is encrypted and no key, or the wrong
            key, is configured. Failing here keeps a later save from replacing
            the stored credentials with empty ones.
    """
    if not is_encrypted(value):
        return value
    fernet = _key()
    if fernet is None:
        raise CredentialsKeyError(
            f"Source credentials are encrypted but no key is configured: set {KEY_ENV} "
            f"or {KEY_FILE_ENV} to the key they were saved with."
        )
    try:
        return fernet.decrypt(value[len(PREFIX) :].encode()).decode()
    except InvalidToken:
        raise CredentialsKeyError(
            "Source credentials cannot be decrypted with the configured key."
        ) from None


def encrypt_source(source: dict[str, object]) -> dict[str, object]:
    """A copy of a stored source with its secret fields encrypted, if a key is configured."""
    out = dict(source)
    for field in SECRET_FIELDS:
        value = out.get(field)
        if isinstance(value, str):
            out[field] = encrypt(value)
    return out


def decrypt_source(source: dict[str, MutablePlainTree]) -> dict[str, MutablePlainTree]:
    """A copy of a stored source with its secret fields decrypted."""
    out = dict(source)
    for field in SECRET_FIELDS:
        value = out.get(field)
        if isinstance(value, str):
            out[field] = decrypt(value)
    return out
