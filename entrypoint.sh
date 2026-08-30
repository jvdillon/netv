#!/bin/sh
set -e
# Entrypoint: fix permissions and drop to netv user
#
# Handles two common Docker issues:
# 1. Bind-mounted ./cache owned by host user (permission denied)
# 2. /dev/dri/renderD128 GID mismatch (VAAPI unavailable)

# Fix cache directory ownership (skip if already correct to avoid slow recursive chown)
# Build/runtime note: this only applies to bind-mounted cache (e.g., NAS),
# not to image layers, so it does not affect build reproducibility.
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
mkdir -p /app/cache/users
if [ "$(stat -c '%U' /app/cache/users)" != "netv" ]; then
    chown -R netv:netv /app/cache/users 2>/dev/null || true
fi
# Ensure writable even on filesystems that ignore chown (e.g., some NAS mounts)
if ! gosu netv sh -c "touch /app/cache/users/.perm_test && rm /app/cache/users/.perm_test" 2>/dev/null; then
    chmod -R u+rwX,g+rwX /app/cache/users 2>/dev/null || true
    chmod g+s /app/cache/users 2>/dev/null || true
fi
# Final verification - warn if still not writable
if ! gosu netv sh -c "touch /app/cache/users/.perm_test && rm /app/cache/users/.perm_test" 2>/dev/null; then
    echo "WARNING: /app/cache/users is not writable by netv user"
    echo "Cache operations may fail. Check volume permissions."
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

# Drop to netv user and run the selected app
if [ "${NETV_MODE:-web}" = "gateway" ]; then
    exec gosu netv python3 gateway.py --port "${NETV_GATEWAY_PORT:-8100}"
fi
exec gosu netv python3 main.py --port "${NETV_PORT:-8000}" ${NETV_HTTPS:+--https}
