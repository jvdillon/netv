"""Encryption of source credentials stored in ``server_settings.json``.

Provider logins (the Xtream username and password, and URLs that often carry
them) are written as ``enc:v1:<Fernet token>`` and decrypted only in memory.
The key comes from, in order:

1. ``NETV_CREDENTIALS_KEY``: a Fernet key.
2. ``NETV_CREDENTIALS_KEY_FILE``: a file holding one, such as a Docker secret.
3. ``credentials.key`` next to the settings file, created on the first save
   that needs it. A copy of the data directory then still holds the key, so a
   warning says so once; plain-text passwords at least stay out of copied or
   shared settings files.

Plain-text values from older releases are read as they are and encrypted the
next time the settings are saved.
"""

from __future__ import annotations

import logging
import os
import pathlib
import threading

from cryptography.fernet import Fernet, InvalidToken


log = logging.getLogger(__name__)

PREFIX = "enc:v1:"
KEY_ENV = "NETV_CREDENTIALS_KEY"
KEY_FILE_ENV = "NETV_CREDENTIALS_KEY_FILE"
DEFAULT_KEY_FILE = "credentials.key"
# Source fields that hold provider logins or URLs that may embed them.
SECRET_FIELDS = ("url", "epg_url", "username", "password")

_lock = threading.Lock()
_warned_default_key = False


class CredentialsKeyError(RuntimeError):
    """Stored credentials cannot be decrypted with the configured key."""


def is_encrypted(value: object) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def _read_key(path: pathlib.Path) -> bytes:
    return path.read_bytes().strip()


def _key(settings_dir: pathlib.Path, *, create: bool) -> Fernet | None:
    """The configured key; with ``create``, generate the default key file if needed."""
    global _warned_default_key
    configured = os.environ.get(KEY_ENV, "").strip()
    if configured:
        return Fernet(configured.encode())
    key_file = os.environ.get(KEY_FILE_ENV, "").strip()
    if key_file:
        return Fernet(_read_key(pathlib.Path(key_file).expanduser()))
    path = settings_dir / DEFAULT_KEY_FILE
    with _lock:
        if not path.exists():
            if not create:
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            # O_EXCL: two processes starting at once cannot each write a key.
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "wb") as file:
                    file.write(Fernet.generate_key())
        if not _warned_default_key:
            _warned_default_key = True
            log.warning(
                "Source credentials are encrypted with a key stored next to the settings (%s). "
                "Set %s or %s to keep the key out of copies of the data directory.",
                path,
                KEY_ENV,
                KEY_FILE_ENV,
            )
        return Fernet(_read_key(path))


def encrypt(value: str, settings_dir: pathlib.Path) -> str:
    """Encrypt one value; empty and already-encrypted values are returned as they are."""
    if not value or is_encrypted(value):
        return value
    fernet = _key(settings_dir, create=True)
    assert fernet is not None
    return PREFIX + fernet.encrypt(value.encode()).decode()


def decrypt(value: str, settings_dir: pathlib.Path) -> str:
    """Decrypt one stored value; plain text from older releases passes through.

    Raises:
        CredentialsKeyError: The value is encrypted and no key, or the wrong
            key, is configured. Failing here keeps a later save from replacing
            the stored credentials with empty ones.
    """
    if not is_encrypted(value):
        return value
    fernet = _key(settings_dir, create=False)
    if fernet is None:
        raise CredentialsKeyError(
            f"Source credentials are encrypted but no key is configured: set {KEY_ENV} "
            f"or {KEY_FILE_ENV}, or restore {settings_dir / DEFAULT_KEY_FILE}."
        )
    try:
        return fernet.decrypt(value[len(PREFIX) :].encode()).decode()
    except InvalidToken:
        raise CredentialsKeyError(
            "Source credentials cannot be decrypted with the configured key."
        ) from None


def encrypt_source(source: dict[str, object], settings_dir: pathlib.Path) -> dict[str, object]:
    """A copy of a stored source with its secret fields encrypted."""
    out = dict(source)
    for field in SECRET_FIELDS:
        value = out.get(field)
        if isinstance(value, str):
            out[field] = encrypt(value, settings_dir)
    return out


def decrypt_source(source: dict[str, object], settings_dir: pathlib.Path) -> dict[str, object]:
    """A copy of a stored source with its secret fields decrypted."""
    out = dict(source)
    for field in SECRET_FIELDS:
        value = out.get(field)
        if isinstance(value, str):
            out[field] = decrypt(value, settings_dir)
    return out
