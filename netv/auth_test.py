"""Tests for auth.py."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import hashlib
import hmac
import json
import re
import time

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
        # Each hash records its cost, so a stronger default can replace it later.
        scheme, rounds, salt, key = hashed.split("$")
        assert (scheme, rounds) == ("pbkdf2_sha256", "600000")
        assert len(salt) == 32  # 16 bytes hex.
        assert len(key) == 64  # 32 bytes hex.
        cheap = auth._hash_password("mypassword", salt, rounds=1000)
        assert cheap.split("$")[:3] == ["pbkdf2_sha256", "1000", salt]
        assert auth._verify_hashed_password("mypassword", cheap)
        assert not auth._verify_hashed_password("other", cheap)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "nodelimiter",
            "pbkdf2_sha256$x$salt$00",
            "md5$1000$salt$00",
            "a$b$c",
            "s$1$a$b$c",
            "aa:bb:cc",
            "AA:BB",
        ],
    )
    def test_unknown_hash_formats_never_verify(self, bad: str):
        assert not auth._verify_hashed_password("pw", bad)

    def test_an_unknown_name_never_verifies(self):
        # Even before any account exists.
        assert not auth.verify_password("nobody", "pw")
        auth.create_user("alice", "pw")
        assert not auth.verify_password("nobody", "pw")

    def test_a_password_from_before_costs_were_recorded_used_100000_rounds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # The real KDF, which tests otherwise swap for a one-round one: the old cost
        # is part of the old format.
        monkeypatch.setattr(auth, "hashlib", hashlib)
        salt = "a" * 32
        key = hashlib.pbkdf2_hmac("sha256", b"pw", salt.encode(), 100_000).hex()
        assert auth._verify_hashed_password("pw", f"{salt}:{key}")
        assert not auth._verify_hashed_password("px", f"{salt}:{key}")

    def test_a_password_from_before_costs_were_recorded_upgrades_at_sign_in(self):
        # As 0.6 stored it: salt and key, at 100,000 rounds.
        salt = "a" * 32
        # Through auth's own KDF, which tests swap for a one-round one.
        key = auth.hashlib.pbkdf2_hmac("sha256", b"pw", salt.encode(), 100_000).hex()
        auth.create_user("alice", "other")
        settings = json.loads(auth.SERVER_SETTINGS_FILE.read_text())
        settings["users"]["alice"]["password"] = f"{salt}:{key}"
        auth.SERVER_SETTINGS_FILE.write_text(json.dumps(settings))
        assert not auth.verify_password("alice", "wrong")
        assert (
            json.loads(auth.SERVER_SETTINGS_FILE.read_text())["users"]["alice"]["password"]
            == f"{salt}:{key}"
        )
        assert auth.verify_password("alice", "pw")
        stored = json.loads(auth.SERVER_SETTINGS_FILE.read_text())["users"]["alice"]["password"]
        assert stored.startswith("pbkdf2_sha256$600000$")
        assert auth.verify_password("alice", "pw")
        # A current hash is left alone.
        assert (
            json.loads(auth.SERVER_SETTINGS_FILE.read_text())["users"]["alice"]["password"]
            == stored
        )

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

    def test_the_last_admin_cannot_be_deleted_while_others_remain(self):
        auth.create_user("alice", "pw")
        auth.create_user("kid", "pw")
        auth.set_lan_only("kid", lan_only=True)
        # Promoting whoever is left could hand a LAN-only account the keys.
        with pytest.raises(auth.LastAdminError):
            auth.delete_user("alice")
        assert auth.get_all_usernames() == ["alice", "kid"]
        assert not auth.is_admin("kid")

    def test_delete_scrubs_sources(self):
        auth.create_user("alice", "pw")
        auth.create_user("bob", "pw", admin=True)
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
        auth.create_user("bob", "pw", admin=True)
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

    def test_the_last_admin_cannot_be_revoked(self):
        auth.create_user("alice", "pw")
        auth.create_user("bob", "pw")
        with pytest.raises(auth.LastAdminError):
            auth.set_admin("alice", False)
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

    @pytest.mark.parametrize("bad", ["00." + "é" * 64, "zz." + "0" * 64, "." + "0" * 64])
    def test_malformed_token_rejected_not_raised(self, bad: str):
        assert auth.verify_token(bad) is None

    def test_sessions_end_with_the_password_or_the_account(self):
        auth.create_user("alice", "pw")
        session = auth.create_token({"sub": "alice"})
        claims = auth.session_user(session, "8.8.8.8")
        assert claims is not None and claims["sub"] == "alice"
        # A new password signs every earlier session out, a stolen one included.
        assert auth.change_password("alice", "new")
        assert auth.session_user(session, "8.8.8.8") is None
        fresh = auth.create_token({"sub": "alice"})
        assert auth.session_user(fresh, "8.8.8.8") is not None
        # Nor does a deleted account's session work, even once its name is reused.
        auth.create_user("bob", "pw", admin=True)
        assert auth.delete_user("alice")
        assert auth.session_user(fresh, "8.8.8.8") is None
        auth.create_user("alice", "new")
        assert auth.session_user(fresh, "8.8.8.8") is None
        # A LAN-only account's session works only at home.
        auth.set_lan_only("alice", lan_only=True)
        home = auth.create_token({"sub": "alice"})
        assert auth.session_user(home, "192.168.1.5") is not None
        assert auth.session_user(home, "8.8.8.8") is None
        assert auth.session_user("garbage", "192.168.1.5") is None
        # A token signed for a name before its account existed marks no password.
        early = auth.create_token({"sub": "carol"})
        auth.create_user("carol", "pw")
        assert auth.session_user(early, "8.8.8.8") is None


@pytest.mark.usefixtures("auth_module")
class TestUrlSignatures:
    def test_logo_url_signature_verifies_only_its_url(self):
        sig = auth.logo_url_signature("http://x/a.png")
        assert auth.verify_logo_url_signature("http://x/a.png", sig)
        assert not auth.verify_logo_url_signature("http://x/b.png", sig)

    @pytest.mark.parametrize("bad", ["", "0" * 63, "A" * 64, "é" * 64])
    def test_logo_url_signature_rejects_malformed(self, bad: str):
        assert not auth.verify_logo_url_signature("http://x/a.png", bad)

    def test_stream_refs_hide_their_url_and_name_their_user(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        auth.create_user("ann", "pw")
        url = "http://provider/live/someone/s3cret/1.m3u8"
        ref = auth.stream_ref("ann", url)
        # A browser holds only the ref: no provider, account, or password in it.
        assert not {"provider", "someone", "s3cret", "http"} & set(re.split(r"[^a-z0-9]", ref))
        assert "s3cret" not in ref
        assert auth.stream_ref_target(ref) == ("ann", url, False)
        # Whether the stream is on the home network rides along, for its relay.
        assert auth.stream_ref_target(auth.stream_ref("ann", url, lan=True)) == ("ann", url, True)
        # One issuer makes many refs, reading the account once.
        issue = auth.stream_ref_issuer("ann", lan=False)
        assert auth.stream_ref_target(issue(url)) == ("ann", url, False)
        # Forged, mangled, or another server's refs name nothing.
        for bad in (
            "",
            "x",
            ref[:-2],
            ref + "x",
            "é" * 40,
            ref.replace(ref[10], "A" if ref[10] != "A" else "B", 1),
        ):
            assert auth.stream_ref_target(bad) is None, bad
        # A ref lasts as long as a session.
        now = time.time()
        monkeypatch.setattr(time, "time", lambda: now + 86_400 * 7 - 60)
        assert auth.stream_ref_target(ref) == ("ann", url, False)
        monkeypatch.setattr(time, "time", lambda: now + 86_400 * 7 + 60)
        assert auth.stream_ref_target(ref) is None

    def test_stream_refs_end_with_the_password_or_the_account(self):
        auth.create_user("ann", "pw")
        auth.create_user("bob", "pw", admin=True)
        url = "http://provider/1.ts"
        ref = auth.stream_ref("ann", url)
        # A shared link stops when its owner changes the password, like a session.
        assert auth.change_password("ann", "new")
        assert auth.stream_ref_target(ref) is None
        ref = auth.stream_ref("ann", url)
        # Nor does a deleted account's link come back when its name is reused.
        assert auth.delete_user("ann")
        assert auth.stream_ref_target(ref) is None
        auth.create_user("ann", "new")
        assert auth.stream_ref_target(ref) is None
        # A ref for no account names nothing.
        assert auth.stream_ref_target(auth.stream_ref("ghost", url)) is None
        # Nor does one in an earlier release's shape, left open in a page.
        old = auth._stream_cipher().encrypt(b'["bob","http://provider/1.ts"]').decode()
        assert auth.stream_ref_target(old) is None


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
    # CGNAT (100.64/10) is a carrier's network, not the home's; Teredo (2001::/32)
    # and 6to4 (2002::/16) tunnels and benchmark nets (198.18/15) are routed from
    # anywhere, though Python calls them private.
    for away in (
        "8.8.8.8",
        "2001:4860::8888",
        "100.64.0.1",
        "::ffff:8.8.8.8",
        "2001::1",
        "2002:808:808::1",
        "198.18.0.1",
        "testclient",
        "",
    ):
        assert not auth.allowed_from("kid", away), away
        assert not auth.is_home_address(away), away
    assert auth.is_home_address("192.168.1.5") and auth.is_home_address("::ffff:10.0.0.1")
    assert auth.allowed_from("admin", "8.8.8.8")
    # An account that no longer exists is allowed nowhere.
    assert not auth.allowed_from("ghost", "192.168.1.5")
    # An admin may name the home network exactly, so a proxy on this host (whose
    # every request arrives from 127.0.0.1) is not home.
    assert auth.set_home_networks(["192.168.1.0/24", "fd00::/8"])
    assert auth.home_networks() == ["192.168.1.0/24", "fd00::/8"]
    assert auth.allowed_from("kid", "192.168.1.5") and auth.allowed_from("kid", "fd00::7")
    for away in ("127.0.0.1", "10.0.0.2", "192.168.2.1", "::ffff:127.0.0.1"):
        assert not auth.allowed_from("kid", away), away
    assert not auth.set_home_networks(["192.168.1.0/24", "not a network"])
    assert auth.home_networks() == ["192.168.1.0/24", "fd00::/8"]
    # None named: the private, loopback, link-local, and unique-local ranges.
    assert auth.set_home_networks([])
    assert auth.allowed_from("kid", "127.0.0.1") and auth.allowed_from("kid", "10.0.0.2")
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
