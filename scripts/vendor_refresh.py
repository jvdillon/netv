#!/usr/bin/env python3
"""Refresh the JavaScript vendored in netv/static/vendor from the npm registry.

For each package in PACKAGES, find the newest stable release within the pinned major
version. If it is newer than the vendored copy, download the package tarball from
registry.npmjs.org, check it against the registry's `dist.integrity` (sha512), extract
the one dist file, and rewrite the vendored file, its row in VENDOR.md, and the <script>
tag in base.html.

Only registry.npmjs.org is contacted. No CDN copy is trusted. Major upgrades (htmx 3,
hls.js 2) are deliberately out of reach: they change behavior and need a human.

    scripts/vendor_refresh.py [--root DIR] [--summary-file FILE] [--dry-run]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib import request

import argparse
import base64
import hashlib
import io
import json
import re
import sys
import tarfile


_CWD: Final[Path] = Path(__file__).resolve().parent

REGISTRY: Final = "https://registry.npmjs.org/"
VENDOR_DIR: Final = Path("netv/static/vendor")
VENDOR_MD: Final = VENDOR_DIR / "VENDOR.md"
BASE_TEMPLATE: Final = Path("netv/templates/base.html")
VERSION_RE: Final = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")  # Stable releases only.
MAX_DOWNLOAD: Final = 32 * 1024 * 1024
SHA512_PREFIX: Final = "sha512-"

Fetch = Callable[[str], bytes]


@dataclass(frozen=True, kw_only=True, slots=True)
class Package:
    """One vendored library."""

    npm: str  # Package name on npm.
    major: int  # Stay within this major version.
    member: str  # Path of the dist file inside the tarball.
    prefix: str  # Vendored file name is f"{prefix}-{version}.min.js".
    label: str  # Name shown in VENDOR.md.


PACKAGES: Final = (
    Package(
        npm="htmx.org",
        major=2,
        member="package/dist/htmx.min.js",
        prefix="htmx",
        label="htmx",
    ),
    Package(
        npm="hls.js",
        major=1,
        member="package/dist/hls.min.js",
        prefix="hls",
        label="hls.js",
    ),
)


class RefreshError(Exception):
    """The registry data or the tarball cannot be trusted; nothing is written."""


def fetch(url: str) -> bytes:
    """Downloads `url`, refusing anything outside the npm registry.

    Args:
      url: The address to download. Must start with the npm registry URL.

    Returns:
      body: The response body, at most MAX_DOWNLOAD bytes.

    Raises:
      RefreshError: `url` is outside the registry or the body is too large.

    """
    if not url.startswith(REGISTRY):
        raise RefreshError(f"refusing to fetch {url}: not {REGISTRY}")
    # The prefix check above pins the scheme and host, so the URL is not attacker-chosen.
    opened = request.urlopen(  # pragma: nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        url,
        timeout=60,
    )
    with opened as response:
        data = bytes(response.read(MAX_DOWNLOAD + 1))
    if len(data) > MAX_DOWNLOAD:
        raise RefreshError(f"{url} is larger than {MAX_DOWNLOAD} bytes")
    return data


def registry_url(npm: str, version: str | None = None) -> str:
    """The registry metadata URL of a package, or of one of its versions."""
    url = REGISTRY + npm.replace("/", "%2F")
    return url if version is None else f"{url}/{version}"


def parse_version(version: str) -> tuple[int, int, int]:
    match = VERSION_RE.match(version)
    if not match:
        raise RefreshError(f"not a stable version: {version!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def latest_version(versions: list[str], major: int) -> str:
    """The newest stable release with the given major version."""
    candidates = [v for v in versions if VERSION_RE.match(v) and parse_version(v)[0] == major]
    if not candidates:
        raise RefreshError(f"no stable {major}.x release found")
    return max(candidates, key=parse_version)


def verify_integrity(data: bytes, integrity: str) -> None:
    """Checks `data` against an npm `dist.integrity` string (sha512 only).

    Args:
      data: The downloaded bytes.
      integrity: The registry's `dist.integrity` value: space-separated `algo-digest` entries.

    Raises:
      RefreshError: No sha512 entry exists, or its digest does not match `data`.

    """
    for entry in integrity.split():
        if entry.startswith(SHA512_PREFIX):
            actual = base64.b64encode(hashlib.sha512(data).digest()).decode()
            if actual != entry.removeprefix(SHA512_PREFIX):
                raise RefreshError(
                    "tarball does not match the registry's sha512 integrity",  # pragma: no mutate[diagnostic-text]
                )
            return
    raise RefreshError(f"no sha512 entry in dist.integrity: {integrity!r}")


def extract_member(tarball: bytes, member: str) -> bytes:
    """Reads one regular file out of a .tgz in memory. Nothing touches the disk.

    Args:
      tarball: The gzip-compressed tar bytes.
      member: Path of the file inside the archive.

    Returns:
      content: The file's bytes.

    Raises:
      RefreshError: `member` is missing or is not a regular file.

    """
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
        try:
            info = archive.getmember(member)
        except KeyError:
            raise RefreshError(f"{member} is not in the tarball") from None
        stream = archive.extractfile(info) if info.isfile() else None
        if stream is None:
            raise RefreshError(f"{member} is not a regular file")
        return stream.read()


def sri384(data: bytes) -> str:
    return "sha384-" + base64.b64encode(hashlib.sha384(data).digest()).decode()


def current_version(package: Package, vendor_md: str) -> str:
    """The version recorded in VENDOR.md for this package.

    Args:
      package: The library to look up.
      vendor_md: The text of VENDOR.md.

    Returns:
      version: The `X.Y.Z` string in the package's row.

    Raises:
      RefreshError: VENDOR.md has no row for the package.

    """
    match = re.search(
        rf"^\| `{package.prefix}-(\d+\.\d+\.\d+)\.min\.js` \|",
        vendor_md,
        re.MULTILINE,
    )
    if not match:
        raise RefreshError(f"no {package.prefix} row in VENDOR.md")
    return str(match.group(1))


def fetch_release(package: Package, version: str, get: Fetch) -> tuple[bytes, str]:
    """Downloads and verifies one release.

    Args:
      package: The library to download.
      version: The release to download.
      get: Downloads a URL. Tests pass a fake registry.

    Returns:
      release: The dist file's bytes and the tarball URL they came from.

    Raises:
      RefreshError: The tarball is outside the registry or fails its integrity check.

    """
    meta = json.loads(get(registry_url(package.npm, version)))
    dist = meta["dist"]
    tarball_url = str(dist["tarball"])
    if not tarball_url.startswith(REGISTRY):
        raise RefreshError(f"{package.npm}@{version} tarball is outside {REGISTRY}")
    tarball = get(tarball_url)
    verify_integrity(tarball, dist["integrity"])
    return extract_member(tarball, package.member), tarball_url


def refresh_package(root: Path, package: Package, get: Fetch, dry_run: bool) -> str | None:
    """Updates one package.

    Args:
      root: The repository root.
      package: The library to update.
      get: Downloads a URL. Tests pass a fake registry.
      dry_run: Report the update without writing anything.

    Returns:
      summary: A one-line description of the update, or None when already current.

    Raises:
      RefreshError: A file that has to change does not have the expected shape.

    """
    md_path = root / VENDOR_MD
    md = md_path.read_text()
    old = current_version(package, md)
    packument = json.loads(get(registry_url(package.npm)))
    new = latest_version(list(packument["versions"]), package.major)
    if parse_version(new) <= parse_version(old):
        return None
    line = f"{package.label} {old} -> {new}"
    if dry_run:
        return line

    data, tarball_url = fetch_release(package, new, get)
    old_name = f"{package.prefix}-{old}.min.js"
    new_name = f"{package.prefix}-{new}.min.js"
    source = f"{tarball_url} (`{package.member.removeprefix('package/')}`)"

    template = root / BASE_TEMPLATE
    html = template.read_text()
    if f"/static/vendor/{old_name}" not in html:
        raise RefreshError(f"base.html does not reference {old_name}")
    row = re.compile(
        rf"^\| `{re.escape(old_name)}` \| .*? \| .*? \| `sha384-[^`]+` \| (.*) \|$",
        re.MULTILINE,
    )
    if not row.search(md):
        raise RefreshError(f"VENDOR.md row for {old_name} has an unexpected shape")

    (root / VENDOR_DIR / new_name).write_bytes(data)
    (root / VENDOR_DIR / old_name).unlink()
    template.write_text(html.replace(f"/static/vendor/{old_name}", f"/static/vendor/{new_name}"))
    md_path.write_text(
        row.sub(
            lambda m: (
                f"| `{new_name}` | {package.label} {new} | {source} | `{sri384(data)}` | {m.group(1)} |"
            ),
            md,
        ),
    )
    return line


def refresh(root: Path, get: Fetch | None = None, dry_run: bool = False) -> list[str]:
    """Refreshes every package. Returns one summary line per package that changed."""
    get = get or fetch
    lines = (refresh_package(root, package, get, dry_run) for package in PACKAGES)
    return [line for line in lines if line is not None]


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the vendored JavaScript from the npm registry.",  # pragma: no mutate[diagnostic-text]
    )
    _add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        changes = refresh(args.root, dry_run=args.dry_run)
    except RefreshError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for change in changes:
        print(change)
    if not changes:
        print("vendored JavaScript is current")
    elif args.summary_file and not args.dry_run:
        # pragma: no mutate[diagnostic-text] start
        args.summary_file.write_text(
            "Updates vendored JavaScript within its current major version.\n\n"
            + "".join(f"- {change}\n" for change in changes)
            + "\nEach tarball was checked against the npm registry's `dist.integrity` (sha512) "
            "before extraction. The SHA-384 hashes in `netv/static/vendor/VENDOR.md` are recomputed. "
            "Skim the upstream release notes, and check whether the license files beside the "
            "scripts still match.\n",
        )
        # pragma: no mutate end
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=_CWD.parent)
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="write a markdown summary here when something changed",  # pragma: no mutate[diagnostic-text]
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report updates without writing",  # pragma: no mutate[diagnostic-text]
    )


if __name__ == "__main__":
    sys.exit(run())
