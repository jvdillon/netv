#!/bin/bash
# Rebuild netv/static/css/tailwind.css with the Tailwind standalone CLI (no Node).
#
# The output is committed so a `pip install pynetv` works without a build step.
# Rerun this after adding or changing any Tailwind class in a template or in
# netv/static/js, and commit the result.
#
# The version is pinned to 3.4.17, the version cdn.tailwindcss.com served when
# the Play CDN was replaced. Do not move to v4 without checking the look: its
# defaults differ.
set -euo pipefail

VERSION="3.4.17"
SHA256="7d24f7fa191d2193b78cd5f5a42a6093e14409521908529f42d80b11fde1f1d4"  # tailwindcss-linux-x64
URL="https://github.com/tailwindlabs/tailwindcss/releases/download/v${VERSION}/tailwindcss-linux-x64"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/netv"
BIN="${TAILWIND_BIN:-$CACHE/tailwindcss-$VERSION}"

if [[ ! -x "$BIN" ]]; then
    mkdir -p "$(dirname "$BIN")"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    curl -fsSL -o "$tmp/tailwindcss" "$URL"
    echo "$SHA256  $tmp/tailwindcss" | sha256sum -c -
    chmod +x "$tmp/tailwindcss"
    mv "$tmp/tailwindcss" "$BIN"
fi

# Content globs in the config are relative to the working directory.
cd "$ROOT"
exec "$BIN" \
    --config scripts/tailwind/tailwind.config.js \
    --input scripts/tailwind/input.css \
    --output netv/static/css/tailwind.css \
    --minify
