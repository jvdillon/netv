#!/bin/bash
# Build TensorRT engines for AI Upscale
#
# Prerequisites: uv sync --group ai_upscale
#   Or: pip install torch onnx onnxconverter-common tensorrt safetensors
#
# Model sourced from https://openmodeldb.info/
#
set -e

# Capture script directory (with error handling)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || {
    echo "ERROR: Failed to determine script directory" >&2
    exit 1
}
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${MODEL_DIR:-$HOME/ffmpeg_build/models}"
MODEL="2x-nomosuni-compact"
PRECISION="fp16"

# Use uv run if in a uv project, otherwise plain python3
# Note: PYTHON_CMD is an array to handle paths with spaces correctly
if [ -f "$PROJECT_DIR/pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=("uv" "run" "--project" "$PROJECT_DIR" "python3")
else
    PYTHON_CMD=("python3")
fi

# Helper: run python command
run_python() {
    "${PYTHON_CMD[@]}" "$@"
}

# Validate export-tensorrt.py exists
EXPORT_SCRIPT="$SCRIPT_DIR/export-tensorrt.py"
if [ ! -f "$EXPORT_SCRIPT" ]; then
    echo "ERROR: export-tensorrt.py not found at $EXPORT_SCRIPT" >&2
    exit 1
fi

# Show help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0"
    echo ""
    echo "Build 2x NomosUni TensorRT engines for AI Upscale."
    echo ""
    echo "Environment:"
    echo "  MODEL_DIR   Output directory (default: \$HOME/ffmpeg_build/models)"
    run_python "$EXPORT_SCRIPT" --list
    exit 0
fi

if [ -n "$1" ]; then
    echo "ERROR: NomosUni is the only supported model; no model argument is accepted." >&2
    exit 2
fi

echo "========================================"
echo "AI Upscale: TensorRT Engine Builder"
echo "========================================"
echo "Model: $MODEL"
echo "Output: $MODEL_DIR/"
echo ""

# Check dependencies
if ! run_python -c "import onnxconverter_common, torch, onnx, safetensors, tensorrt" 2>/dev/null; then
    echo "ERROR: Missing dependencies. Install with:"
    echo "  uv sync --group ai_upscale"
    echo "Or:"
    echo "  pip install torch onnx onnxconverter-common tensorrt safetensors"
    exit 1
fi

# Create output directory with validation
mkdir -p "$MODEL_DIR" || {
    echo "ERROR: Cannot create directory: $MODEL_DIR" >&2
    exit 1
}
if [ ! -w "$MODEL_DIR" ]; then
    echo "ERROR: No write permission for: $MODEL_DIR" >&2
    exit 1
fi

# Check disk space (engines are ~100-500MB each, need at least 2GB free)
REQUIRED_SPACE_KB=$((2 * 1024 * 1024))  # 2GB in KB
AVAILABLE_KB=$(df "$MODEL_DIR" 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$AVAILABLE_KB" ] && [ "$AVAILABLE_KB" -lt "$REQUIRED_SPACE_KB" ] 2>/dev/null; then
    echo "WARNING: Low disk space in $MODEL_DIR ($(( AVAILABLE_KB / 1024 ))MB available, recommend 2GB+)" >&2
fi

# Input resolutions to build engines for (output can be downscaled as needed)
RESOLUTIONS="480 720 1080"

SAFE_MODEL="$MODEL"

# Remove engines from model families that are no longer supported.
find "$MODEL_DIR" -maxdepth 1 -type f -name '*_*p_*.engine' \
    ! -name "${SAFE_MODEL}_*p_*.engine" -delete

# Build engines for common resolutions (FFmpeg TensorRT backend needs fixed shapes)
echo "Building TensorRT engines for resolutions: $RESOLUTIONS"
echo ""

# Use word splitting intentionally here (RESOLUTIONS is space-separated)
# shellcheck disable=SC2086
for res in $RESOLUTIONS; do
    engine="$MODEL_DIR/${SAFE_MODEL}_${res}p_${PRECISION}.engine"

    if [ -f "$engine" ]; then
        echo "  ${res}p: already exists, skipping"
    else
        echo "  ${res}p: building..."
        # Capture output to show errors if build fails
        if ! OUTPUT=$(run_python "$EXPORT_SCRIPT" \
            --precision "$PRECISION" \
            --min-height "$res" --opt-height "$res" --max-height "$res" \
            -o "$engine" 2>&1); then
            echo "ERROR building ${res}p engine:" >&2
            echo "$OUTPUT" >&2
            exit 1
        fi
        # Show filtered progress on success
        echo "$OUTPUT" | grep -E "^(Downloading|Using cached|Loading|Using ONNX|Engine saved|  )" || true
        # Verify engine was created
        if [ ! -f "$engine" ]; then
            echo "ERROR: Engine file not created: $engine" >&2
            echo "Build output:" >&2
            echo "$OUTPUT" >&2
            exit 1
        fi
    fi
done

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Engines built:"
# Safe listing of engine files (handles filenames with special chars)
find "$MODEL_DIR" -maxdepth 1 -name "${SAFE_MODEL}_*.engine" -type f -exec ls -lh {} \; 2>/dev/null | \
    while IFS= read -r line; do
        size=$(echo "$line" | awk '{print $5}')
        file=$(echo "$line" | awk '{print $NF}')
        echo "  $(basename "$file") ($size)"
    done
echo ""
echo "Test with:"
echo "  ffmpeg -init_hw_device cuda=cu -filter_hw_device cu \\"
echo "    -f lavfi -i testsrc=duration=3:size=1920x1080:rate=30 \\"
echo "    -vf \"format=rgb24,hwupload,dnn_processing=dnn_backend=8:model=$MODEL_DIR/${SAFE_MODEL}_1080p_${PRECISION}.engine\" \\"
echo "    -c:v hevc_nvenc test.mp4"
