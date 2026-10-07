#!/bin/sh
set -e
# Entrypoint for AI Upscale image
#
# Same as base entrypoint, plus:
# - Auto-builds TensorRT engines on first start if missing

# Fix cache directory ownership
mkdir -p /app/cache
if [ "$(stat -c '%U' /app/cache)" != "netv" ]; then
    chown -R netv:netv /app/cache 2>/dev/null || true
fi
# Ensure writable even on filesystems that ignore chown (e.g., some NAS mounts)
if ! gosu netv sh -c "touch /app/cache/.perm_test && rm /app/cache/.perm_test" 2>/dev/null; then
    chmod -R u+rwX,g+rwX /app/cache 2>/dev/null || true
    chmod g+s /app/cache 2>/dev/null || true
fi
# Final verification - warn if still not writable
if ! gosu netv sh -c "touch /app/cache/.perm_test && rm /app/cache/.perm_test" 2>/dev/null; then
    echo "WARNING: /app/cache is not writable by netv user"
    echo "Cache operations may fail. Check volume permissions."
fi

# Fix models directory ownership
mkdir -p /models
if [ "$(stat -c '%U' /models)" != "netv" ]; then
    chown -R netv:netv /models 2>/dev/null || true
fi
if ! gosu netv sh -c "touch /models/.perm_test && rm /models/.perm_test" 2>/dev/null; then
    chmod -R u+rwX,g+rwX /models 2>/dev/null || true
fi
if ! gosu netv sh -c "touch /models/.perm_test && rm /models/.perm_test" 2>/dev/null; then
    echo "WARNING: /models is not writable by netv user"
    echo "TensorRT engine caching may fail. Check volume permissions."
fi

# Add netv user to render device group (for VAAPI hardware encoding)
if [ -e /dev/dri/renderD128 ]; then
    RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
    RENDER_ADDED=false
    if groupadd --gid "$RENDER_GID" hostrender 2>/dev/null; then
        :  # Created new group
    fi
    if usermod -aG hostrender netv 2>/dev/null; then
        RENDER_ADDED=true
    fi
    if [ "$RENDER_ADDED" = "false" ]; then
        echo "WARNING: Could not add netv to render group (GID $RENDER_GID)"
        if [ "$RENDER_GID" = "65534" ]; then
            echo "  GID 65534 (nogroup) indicates Docker user namespace mapping issue."
            echo "  This is usually harmless - VAAPI may still work if container has device access."
            echo "  To fix: ensure 'render' group exists on host and user is in it, or use --privileged"
        else
            echo "  VAAPI hardware encoding may not be available."
            echo "  To fix on host: sudo usermod -aG render \$USER (then restart Docker)"
        fi
    fi
fi

# Keep the model volume Nomos-only.
find /models -maxdepth 1 -type f -name '*_*p_*.engine' \
    ! -name '2x-nomosuni-compact_*p_*.engine' -delete

# Build fixed-shape Nomos engines if any supported input size is missing.
NEEDS_MODEL_BUILD=false
for HEIGHT in 480 720 1080; do
    if [ ! -f "/models/2x-nomosuni-compact_${HEIGHT}p_fp16.engine" ]; then
        NEEDS_MODEL_BUILD=true
    fi
done
if [ "$NEEDS_MODEL_BUILD" = "true" ]; then
    echo "========================================"
    echo "AI Upscale: First start detected"
    echo "========================================"
    echo "Building TensorRT engines for your GPU..."
    echo "Model: 2x-nomosuni-compact (480p/720p/1080p)"
    echo "This only happens once (cached in /models volume)."
    echo ""
    # Run as netv user so files have correct ownership
    if ! gosu netv env MODEL_DIR=/models /app/tools/install-ai_upscale.sh; then
        echo "ERROR: Failed to build TensorRT engines"
        echo "Check GPU compatibility and CUDA installation"
        exit 1
    fi

    for HEIGHT in 480 720 1080; do
        if [ ! -f "/models/2x-nomosuni-compact_${HEIGHT}p_fp16.engine" ]; then
            echo "ERROR: Nomos ${HEIGHT}p TensorRT engine not found after build"
            exit 1
        fi
    done
fi

# Drop to netv user and run the selected app
if [ "${NETV_MODE:-web}" = "gateway" ]; then
    exec gosu netv python3 gateway.py --port "${NETV_GATEWAY_PORT:-8100}"
fi
exec gosu netv python3 main.py --port "${NETV_PORT:-8000}" ${NETV_HTTPS:+--https}
