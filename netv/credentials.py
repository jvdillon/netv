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

import contextlib
import logging
import os
import pathlib
import tempfile
import threading

from cryptography.fernet import Fernet, InvalidToken
from treekle import MutablePlainTree


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
        # An empty file is what an interrupted write by an older build could
        # leave; it holds no key, so it counts as missing.
        if not path.exists() or not _read_key(path):
            if not create:
                return None
            _create_key_file(path)
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


def _create_key_file(path: pathlib.Path) -> None:
    """Write a new key to ``path`` atomically, unless another process got there first.

    The key is written and synced in a private temporary file, then linked into
    place. ``os.link`` fails when the name already exists, so two processes
    starting at once cannot each publish a key, and no reader ever sees a
    partly written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp = pathlib.Path(name)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(Fernet.generate_key())
            file.flush()
            os.fsync(file.fileno())
        if path.exists() and not _read_key(path):
            path.unlink()
        # Another process may have published its key first; then use that one.
        with contextlib.suppress(FileExistsError):
            os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def encrypt(value: str, settings_dir: pathlib.Path) -> str:
    """Encrypt one in-memory value; an empty value is returned as it is.

    Values reach this function decrypted, so a password that happens to start
    with ``enc:v1:`` is encrypted like any other; the prefix is only meaningful
    when reading the settings file.
    """
    if not value:
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


def decrypt_source(
    source: dict[str, MutablePlainTree], settings_dir: pathlib.Path
) -> dict[str, MutablePlainTree]:
    """A copy of a stored source with its secret fields decrypted."""
    out = dict(source)
    for field in SECRET_FIELDS:
        value = out.get(field)
        if isinstance(value, str):
            out[field] = decrypt(value, settings_dir)
    return out
