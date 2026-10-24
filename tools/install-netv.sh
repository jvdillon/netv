#!/bin/bash
# Install netv systemd service
# Prerequisites: uv (install time only), install-letsencrypt.sh
#
# Usage: sudo ./install-netv.sh [--port PORT]
#   --port PORT  Port to listen on (default: 8000)
set -e

IPTV_DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER="${SUDO_USER:-$USER}"
PORT=8000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sudo $0 [--port PORT]"
            exit 1
            ;;
    esac
done

# Validate
if [ "$USER" = "root" ]; then
    echo "Error: Run with sudo, not as root directly"
    echo "Usage: sudo $0 [--port PORT]"
    exit 1
fi

# Find uv in user's environment (only needed at install time)
UV_PATH=$(su - "$USER" -c "which uv" 2>/dev/null)
if [ -z "$UV_PATH" ]; then
    echo "Error: uv not found for user $USER. Install with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "See: https://docs.astral.sh/uv/"
    exit 1
fi

echo "=== Syncing dependencies ==="
su - "$USER" -c "cd '$IPTV_DIR' && '$UV_PATH' sync"

if [ ! -d /etc/letsencrypt/live ]; then
    echo "Warning: Let's Encrypt not configured. Run install-letsencrypt.sh first for HTTPS."
    echo "Continuing with HTTP-only setup..."
    HTTPS_FLAG=""
else
    HTTPS_FLAG="--https"
fi

echo "=== Installing netv for user: $USER (port $PORT) ==="

echo "=== Adding $USER to ssl-cert group ==="
sudo usermod -aG ssl-cert "$USER"

echo "=== Installing netv systemd service ==="

# Build PATH - prefer custom ffmpeg in ~/.local/bin if it exists
USER_LOCAL_BIN="/home/$USER/.local/bin"
if [ -x "$USER_LOCAL_BIN/ffmpeg" ]; then
    echo "  Found custom ffmpeg in $USER_LOCAL_BIN"
    ENV_PATH="$USER_LOCAL_BIN:/usr/local/bin:/usr/bin:/bin"
else
    ENV_PATH="/usr/local/bin:/usr/bin:/bin"
fi

# Build LIBVA env vars if custom libva exists (for VAAPI on hybrid GPU systems)
USER_LOCAL_LIB="/home/$USER/.local/lib"
LIBVA_ENVS=""
if [ -f "$USER_LOCAL_LIB/libva.so" ]; then
    echo "  Found custom libva in $USER_LOCAL_LIB"

    # Auto-detect LIBVA driver based on GPU vendor
    LIBVA_DRIVER=""
    if lspci -nn 2>/dev/null | grep -qE "VGA.*\[8086:"; then
        LIBVA_DRIVER="i965"  # Intel
    elif lspci -nn 2>/dev/null | grep -qE "VGA.*\[1002:"; then
        LIBVA_DRIVER="radeonsi"  # AMD
    fi

    # Auto-detect DRI path
    DRI_PATH=""
    for p in /usr/lib/x86_64-linux-gnu/dri /usr/lib64/dri /usr/lib/dri; do
        if [ -d "$p" ]; then
            DRI_PATH="$p"
            break
        fi
    done

    if [ -n "$LIBVA_DRIVER" ] && [ -n "$DRI_PATH" ]; then
        echo "  Detected VAAPI driver: $LIBVA_DRIVER, path: $DRI_PATH"
        LIBVA_ENVS="Environment=\"LIBVA_DRIVER_NAME=$LIBVA_DRIVER\"
Environment=\"LIBVA_DRIVERS_PATH=$DRI_PATH\""
    fi
fi

cat <<EOF | sudo tee /etc/systemd/system/netv.service
[Unit]
Description=NetV IPTV Server
After=network.target

[Service]
Type=simple
User=$USER
Group=ssl-cert
WorkingDirectory=$IPTV_DIR
Environment="PATH=$ENV_PATH"
$LIBVA_ENVS
ExecStart=$IPTV_DIR/.venv/bin/python ./main.py --port $PORT $HTTPS_FLAG
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable netv
sudo systemctl start netv

if [ -n "$HTTPS_FLAG" ]; then
    echo "=== Installing certbot deploy hook (restart netv on renewal) ==="
    cat <<'EOF' | sudo tee /etc/letsencrypt/renewal-hooks/deploy/netv
#!/bin/bash
# Fix cert permissions and restart netv after renewal
chgrp -R ssl-cert /etc/letsencrypt/archive/
chmod -R g+r /etc/letsencrypt/archive/
systemctl restart netv
EOF
    sudo chmod +x /etc/letsencrypt/renewal-hooks/deploy/netv
fi

echo ""
echo "=== Done ==="
echo ""
echo "Commands:"
echo "  sudo systemctl status netv     # Check status"
echo "  sudo systemctl restart netv    # Restart after code changes"
echo "  journalctl -u netv -f          # View logs"
