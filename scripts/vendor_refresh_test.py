"""Tests for scripts/vendor_refresh.py, against a mocked npm registry."""

from __future__ import annotations

from pathlib import Path
from typing import override

import argparse
import base64
import hashlib
import io
import json
import re
import tarfile

import pytest

from scripts import vendor_refresh as vr


REGISTRY = "https://registry.npmjs.org/"


def _tarball(files: dict[str, bytes], *, symlink: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if symlink is not None:
            link = tarfile.TarInfo(symlink)
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive.addfile(link)
    return buffer.getvalue()


def _integrity(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()


class Registry:
    """A fake registry. `get` is a drop-in for vendor_refresh.fetch."""

    def __init__(self) -> None:
        self.responses: dict[str, bytes] = {}
        self.requested: list[str] = []

    def publish(
        self,
        npm: str,
        versions: dict[str, bytes],
        member: str,
        *,
        integrity: str | None = None,
        tarball_url: str | None = None,
    ) -> None:
        packument: dict[str, dict[str, str]] = {}
        for version, dist_file in versions.items():
            tarball = _tarball({member: dist_file})
            url = tarball_url or f"{REGISTRY}{npm}/-/{npm}-{version}.tgz"
            packument[version] = {}
            meta = {"dist": {"tarball": url, "integrity": integrity or _integrity(tarball)}}
            self.responses[f"{REGISTRY}{npm}/{version}"] = json.dumps(meta).encode()
            self.responses[f"{REGISTRY}{npm}/-/{npm}-{version}.tgz"] = tarball
        self.responses[f"{REGISTRY}{npm}"] = json.dumps({"versions": packument}).encode()

    def get(self, url: str) -> bytes:
        self.requested.append(url)
        return self.responses[url]


def _sri(data: bytes) -> str:
    return "sha384-" + base64.b64encode(hashlib.sha384(data).digest()).decode()


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A repo with htmx 2.0.4 and hls.js 1.7.4 vendored the way VENDOR.md describes."""
    vendor = tmp_path / "netv" / "static" / "vendor"
    vendor.mkdir(parents=True)
    (tmp_path / "netv" / "templates").mkdir()
    (vendor / "htmx-2.0.4.min.js").write_bytes(b"htmx old")
    (vendor / "hls-1.7.4.min.js").write_bytes(b"hls old")
    (vendor / "VENDOR.md").write_text(
        "| File | Version | Source | SHA-384 (SRI) | License |\n"
        "|------|---------|--------|---------------|---------|\n"
        f"| `htmx-2.0.4.min.js` | htmx 2.0.4 | https://unpkg.com/x | `{_sri(b'htmx old')}` | 0BSD (`htmx-LICENSE`) |\n"
        f"| `hls-1.7.4.min.js` | hls.js 1.7.4 | https://cdn.example/y | `{_sri(b'hls old')}` | Apache-2.0 (`hls.js-LICENSE`) |\n",
    )
    (tmp_path / "netv" / "templates" / "base.html").write_text(
        '<script src="/static/vendor/htmx-2.0.4.min.js"></script>\n'
        '<script src="/static/vendor/hls-1.7.4.min.js"></script>\n',
    )
    return tmp_path


@pytest.fixture
def registry() -> Registry:
    reg = Registry()
    reg.publish(
        "htmx.org",
        {"2.0.4": b"htmx old", "2.0.9": b"htmx new", "2.1.0-beta1": b"beta", "3.0.0": b"htmx 3"},
        "package/dist/htmx.min.js",
    )
    reg.publish(
        "hls.js",
        {"1.7.4": b"hls old", "1.8.0": b"hls new", "2.0.0": b"hls 2"},
        "package/dist/hls.min.js",
    )
    return reg


def test_latest_version_stays_in_major_and_skips_prereleases() -> None:
    versions = ["1.9.0", "1.10.2", "1.11.0-rc.1", "2.0.0", "0.9.9"]
    assert vr.latest_version(versions, 1) == "1.10.2"
    assert vr.latest_version(versions, 2) == "2.0.0"


def test_latest_version_without_candidates_fails() -> None:
    with pytest.raises(vr.RefreshError, match=r"no stable 2\.x release"):
        vr.latest_version(["1.0.0", "2.0.0-beta.1"], 2)


def test_parse_version_rejects_odd_strings() -> None:
    with pytest.raises(vr.RefreshError, match="not a stable version"):
        vr.parse_version("1.2.3-rc.1")
    with pytest.raises(vr.RefreshError, match="not a stable version"):
        vr.parse_version("../../etc")


def test_verify_integrity_accepts_matching_sha512() -> None:
    vr.verify_integrity(b"data", _integrity(b"data"))
    vr.verify_integrity(b"data", "sha256-AAAA " + _integrity(b"data"))
    with pytest.raises(vr.RefreshError, match="does not match"):
        vr.verify_integrity(b"data", "sha512-" + _integrity(b"data"))


def test_verify_integrity_rejects_mismatch_and_weak_hashes() -> None:
    with pytest.raises(vr.RefreshError, match="does not match"):
        vr.verify_integrity(b"data", _integrity(b"other"))
    with pytest.raises(vr.RefreshError, match="no sha512"):
        vr.verify_integrity(
            b"data",
            "sha256-" + base64.b64encode(hashlib.sha256(b"data").digest()).decode(),
        )


def test_extract_member_reads_one_file() -> None:
    tarball = _tarball({"package/dist/a.js": b"a", "package/other": b"b"})
    assert vr.extract_member(tarball, "package/dist/a.js") == b"a"


def test_extract_member_rejects_an_uncompressed_tar() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("package/dist/a.js")
        archive.addfile(info, io.BytesIO(b""))
    with pytest.raises(tarfile.ReadError):
        vr.extract_member(buffer.getvalue(), "package/dist/a.js")


def test_extract_member_rejects_missing_and_non_regular_files() -> None:
    with pytest.raises(vr.RefreshError, match="not in the tarball"):
        vr.extract_member(_tarball({"package/x": b"x"}), "package/dist/a.js")
    with pytest.raises(vr.RefreshError, match="not a regular file"):
        vr.extract_member(_tarball({}, symlink="package/dist/a.js"), "package/dist/a.js")


def test_fetch_refuses_hosts_other_than_the_registry() -> None:
    for url in (
        "https://unpkg.com/htmx.org",
        "http://registry.npmjs.org/x",
        "https://registry.npmjs.org.evil.test/x",
    ):
        with pytest.raises(vr.RefreshError, match="refusing"):
            vr.fetch(url)


def test_fetch_reads_the_body_and_caps_its_size(monkeypatch: pytest.MonkeyPatch) -> None:
    reads: list[int] = []
    calls: list[tuple[str, float]] = []
    body = b"ok"

    class Response(io.BytesIO):
        @override
        def __enter__(self) -> Response:
            return self

        @override
        def read(self, size: int | None = -1, /) -> bytes:
            reads.append(-1 if size is None else size)
            return super().read(size)

    def urlopen(url: str, timeout: float) -> Response:
        calls.append((url, timeout))
        return Response(body)

    monkeypatch.setattr(vr.request, "urlopen", urlopen)
    assert vr.fetch(REGISTRY + "x") == b"ok"
    assert calls == [(REGISTRY + "x", 60)]
    assert reads == [vr.MAX_DOWNLOAD + 1]

    # A body of exactly the cap is fine; one byte more is refused.
    monkeypatch.setattr(vr, "MAX_DOWNLOAD", 2)
    assert vr.fetch(REGISTRY + "x") == b"ok"
    body = b"okk"
    with pytest.raises(vr.RefreshError, match="larger"):
        vr.fetch(REGISTRY + "x")
    assert reads[-1] == 3


def test_registry_url_quotes_scoped_names_and_appends_a_version() -> None:
    assert vr.registry_url("htmx.org") == REGISTRY + "htmx.org"
    assert vr.registry_url("@scope/pkg") == REGISTRY + "@scope%2Fpkg"
    assert vr.registry_url("@scope/pkg", "1.2.3") == REGISTRY + "@scope%2Fpkg/1.2.3"


def test_refresh_updates_files_row_and_template(tree: Path, registry: Registry) -> None:
    changes = vr.refresh(tree, registry.get)
    assert changes == ["htmx 2.0.4 -> 2.0.9", "hls.js 1.7.4 -> 1.8.0"]

    vendor = tree / "netv" / "static" / "vendor"
    assert sorted(p.name for p in vendor.glob("*.js")) == ["hls-1.8.0.min.js", "htmx-2.0.9.min.js"]
    assert (vendor / "htmx-2.0.9.min.js").read_bytes() == b"htmx new"
    html = (tree / "netv" / "templates" / "base.html").read_text()
    assert "/static/vendor/htmx-2.0.9.min.js" in html
    assert "/static/vendor/hls-1.8.0.min.js" in html
    assert "2.0.4" not in html and "1.7.4" not in html

    # The shape vendor_test.py parses: file name, then a sha384 that matches the file.
    rows = dict(
        re.findall(
            r"^\| `([^`]+\.js)` \|.*?`(sha384-[^`]+)`",
            (vendor / "VENDOR.md").read_text(),
            re.M,
        ),
    )
    assert rows == {"htmx-2.0.9.min.js": _sri(b"htmx new"), "hls-1.8.0.min.js": _sri(b"hls new")}
    md = (vendor / "VENDOR.md").read_text()
    assert "htmx 2.0.9" in md and "hls.js 1.8.0" in md
    assert "(`htmx-LICENSE`)" in md and "(`hls.js-LICENSE`)" in md
    assert "unpkg" not in md and "cdn.example" not in md
    assert f"| {REGISTRY}htmx.org/-/htmx.org-2.0.9.tgz (`dist/htmx.min.js`) |" in md
    assert f"| {REGISTRY}hls.js/-/hls.js-1.8.0.tgz (`dist/hls.min.js`) |" in md


def test_refresh_is_a_noop_when_current(tree: Path, registry: Registry) -> None:
    registry.publish("htmx.org", {"2.0.4": b"htmx old"}, "package/dist/htmx.min.js")
    registry.publish("hls.js", {"1.7.4": b"hls old"}, "package/dist/hls.min.js")
    before = (tree / "netv" / "static" / "vendor" / "VENDOR.md").read_text()
    assert vr.refresh(tree, registry.get) == []
    assert (tree / "netv" / "static" / "vendor" / "VENDOR.md").read_text() == before
    assert not any("/-/" in url for url in registry.requested)  # No tarball downloaded.


def test_dry_run_reports_but_writes_nothing(tree: Path, registry: Registry) -> None:
    assert vr.refresh(tree, registry.get, dry_run=True) == [
        "htmx 2.0.4 -> 2.0.9",
        "hls.js 1.7.4 -> 1.8.0",
    ]
    assert (tree / "netv" / "static" / "vendor" / "htmx-2.0.4.min.js").exists()
    assert not any("/-/" in url for url in registry.requested)


def test_integrity_mismatch_leaves_the_tree_untouched(tree: Path, registry: Registry) -> None:
    registry.publish(
        "htmx.org",
        {"2.0.4": b"htmx old", "2.0.9": b"htmx new"},
        "package/dist/htmx.min.js",
        integrity=_integrity(b"nope"),
    )
    with pytest.raises(vr.RefreshError, match="does not match"):
        vr.refresh(tree, registry.get)
    assert (tree / "netv" / "static" / "vendor" / "htmx-2.0.4.min.js").read_bytes() == b"htmx old"
    assert "2.0.4" in (tree / "netv" / "templates" / "base.html").read_text()


def test_tarball_outside_the_registry_is_refused(tree: Path, registry: Registry) -> None:
    registry.publish(
        "htmx.org",
        {"2.0.4": b"htmx old", "2.0.9": b"htmx new"},
        "package/dist/htmx.min.js",
        tarball_url="https://evil.test/x.tgz",
    )
    with pytest.raises(vr.RefreshError, match="outside"):
        vr.refresh(tree, registry.get)


def test_missing_references_fail_before_writing(tree: Path, registry: Registry) -> None:
    template = tree / "netv" / "templates" / "base.html"
    template.write_text("<html></html>")
    with pytest.raises(vr.RefreshError, match="base.html"):
        vr.refresh(tree, registry.get)
    assert (tree / "netv" / "static" / "vendor" / "htmx-2.0.4.min.js").exists()


def test_odd_vendor_md_row_fails(tree: Path, registry: Registry) -> None:
    md = tree / "netv" / "static" / "vendor" / "VENDOR.md"
    md.write_text(re.sub(r"`sha384-[^`]+`", "pending", md.read_text(), count=1))
    with pytest.raises(vr.RefreshError, match="unexpected shape"):
        vr.refresh(tree, registry.get)


def test_package_without_a_row_fails(tree: Path, registry: Registry) -> None:
    md = tree / "netv" / "static" / "vendor" / "VENDOR.md"
    md.write_text("| File |\n")
    with pytest.raises(vr.RefreshError, match="no htmx row"):
        vr.refresh(tree, registry.get)


def test_run_writes_a_summary_and_reports_errors(
    tree: Path,
    registry: Registry,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(vr, "fetch", registry.get)
    summary = tree / "summary.md"
    assert vr.run(["--root", str(tree), "--summary-file", str(summary)]) == 0
    text = summary.read_text()
    assert text.startswith(
        "Updates vendored JavaScript within its current major version.\n\n"
        "- htmx 2.0.4 -> 2.0.9\n- hls.js 1.7.4 -> 1.8.0\n\n",
    )
    assert "dist.integrity" in text and "VENDOR.md" in text
    assert capsys.readouterr().out == "htmx 2.0.4 -> 2.0.9\nhls.js 1.7.4 -> 1.8.0\n"

    summary.unlink()
    assert vr.run(["--root", str(tree), "--summary-file", str(summary)]) == 0
    assert capsys.readouterr().out == "vendored JavaScript is current\n"
    assert not summary.exists()

    (tree / "netv" / "static" / "vendor" / "VENDOR.md").write_text("")
    assert vr.run(["--root", str(tree)]) == 1
    assert "error: no htmx row in VENDOR.md" in capsys.readouterr().err


def test_dry_run_flag_writes_no_summary_and_no_files(
    tree: Path,
    registry: Registry,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(vr, "fetch", registry.get)
    summary = tree / "summary.md"
    assert vr.run(["--root", str(tree), "--summary-file", str(summary), "--dry-run"]) == 0
    assert capsys.readouterr().out == "htmx 2.0.4 -> 2.0.9\nhls.js 1.7.4 -> 1.8.0\n"
    assert not summary.exists()
    assert (tree / "netv" / "static" / "vendor" / "htmx-2.0.4.min.js").exists()


def test_help_describes_the_script_and_each_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        vr.run(["--help"])
    assert stop.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "Refresh the vendored JavaScript from the npm registry." in text
    assert "--summary-file SUMMARY_FILE write a markdown summary here" in text
    assert "--dry-run report updates without writing" in text


def test_arguments_default_to_the_repository_root() -> None:
    parser = argparse.ArgumentParser()
    vr._add_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.root == Path(vr.__file__).resolve().parent.parent
    assert defaults.summary_file is None
    assert defaults.dry_run is False
    given = parser.parse_args(["--root", "a", "--summary-file", "b", "--dry-run"])
    assert (given.root, given.summary_file, given.dry_run) == (Path("a"), Path("b"), True)


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
