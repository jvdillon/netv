#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/ffmpeg_build/models}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
ENGINE="$MODEL_DIR/2x-nomosuni-compact_1080p_fp16.engine"

mkdir -p "$MODEL_DIR"

exec "$PYTHON" "$ROOT/tools/export-tensorrt.py" \
    --min-height 1080 \
    --opt-height 1080 \
    --max-height 1080 \
    --precision fp16 \
    --opt-level 5 \
    --workspace 8 \
    --output="$ENGINE"
