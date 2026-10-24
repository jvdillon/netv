#!/bin/bash
# Install and configure Let's Encrypt certificates
set -e

DOMAIN="${1:-}"

if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 <domain>"
    echo "Example: $0 yourdomain.com"
    exit 1
fi

echo "=== Installing certbot ==="
sudo apt update
sudo apt install -y certbot

# Detect web server and choose authenticator
if systemctl is-active --quiet apache2; then
    echo "=== Apache detected, using apache authenticator ==="
    sudo apt install -y python3-certbot-apache
    CERTBOT_MODE="--apache"
elif systemctl is-active --quiet nginx; then
    echo "=== Nginx detected, using nginx authenticator ==="
    sudo apt install -y python3-certbot-nginx
    CERTBOT_MODE="--nginx"
else
    echo "=== No web server detected, using standalone mode ==="
    echo "Note: Port 80 must be free for domain verification"
    CERTBOT_MODE="--standalone"
fi

echo "=== Obtaining certificate for $DOMAIN ==="
sudo certbot $CERTBOT_MODE -d "$DOMAIN"

echo "=== Setting up ssl-cert group permissions ==="
sudo chgrp -R ssl-cert /etc/letsencrypt/archive/
sudo chmod -R g+r /etc/letsencrypt/archive/

echo ""
echo "=== Done ==="
echo ""
echo "Certificate installed for $DOMAIN"
echo "Certbot timer will auto-renew (check: systemctl list-timers | grep certbot)"
echo ""
echo "To give a user access to certs:"
echo "  sudo usermod -aG ssl-cert <username>"
