#!/bin/sh
# Entrypoint: fix permissions and drop to netv user
#
# Handles two common Docker issues:
# 1. Bind-mounted ./cache owned by host user (permission denied)
# 2. /dev/dri/renderD128 GID mismatch (VAAPI unavailable)

# Fix cache directory ownership (skip if already correct to avoid slow recursive chown)
mkdir -p /app/cache
if [ "$(stat -c '%U' /app/cache)" != "netv" ]; then
    chown -R netv:netv /app/cache
fi

# Add netv user to render device group (for VAAPI hardware encoding)
if [ -e /dev/dri/renderD128 ]; then
    RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
    groupadd --gid "$RENDER_GID" hostrender 2>/dev/null || true
    useradd netv 2>/dev/null || true
    usermod -g hostrender netv 2>/dev/null || true
fi

# Drop to netv user and run the app
exec gosu netv python3 main.py --port "${NETV_PORT:-8000}" ${NETV_HTTPS:+--https}
