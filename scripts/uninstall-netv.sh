#!/bin/bash
# Uninstall netv systemd service
#
# Usage: sudo ./uninstall-netv.sh
set -e

if [ "$EUID" -ne 0 ]; then
    echo "Error: Run with sudo"
    echo "Usage: sudo $0"
    exit 1
fi

echo "=== Uninstalling netv ==="

if systemctl is-active --quiet netv 2>/dev/null; then
    echo "Stopping netv service..."
    systemctl stop netv
fi

if systemctl is-enabled --quiet netv 2>/dev/null; then
    echo "Disabling netv service..."
    systemctl disable netv
fi

if [ -f /etc/systemd/system/netv.service ]; then
    echo "Removing service file..."
    rm /etc/systemd/system/netv.service
    systemctl daemon-reload
fi

if [ -f /etc/letsencrypt/renewal-hooks/deploy/netv ]; then
    echo "Removing certbot hook..."
    rm /etc/letsencrypt/renewal-hooks/deploy/netv
fi

echo ""
echo "=== Done ==="
echo ""
echo "The netv service has been removed."
echo "Project files and cache remain in place - delete manually if desired."
