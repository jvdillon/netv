#!/bin/bash
# Install prerequisites for netv
set -e

echo "=== Checking prerequisites ==="
for cmd in git curl; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd not found. Install it with your package manager."
        exit 1
    fi
done

echo "=== Installing uv ==="
if command -v uv &> /dev/null; then
    echo "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== Installing Python 3.12 via uv ==="
uv python install 3.12

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Run: ./scripts/install-letsencrypt.sh <your-domain>"
echo "  2. Run: ./scripts/install-ffmpeg.sh  (optional, for transcoding)"
echo "  3. Run: sudo ./scripts/install-netv.sh"
