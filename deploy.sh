#!/usr/bin/env bash
# deploy.sh — One-command VPS setup for Eatlytic
# Usage: ssh root@<hetzner-ip> "bash -s" < deploy.sh

set -e

echo "=== Eatlytic VPS Setup ==="

# 1. Install Docker
if ! command -v docker &>/dev/null; then
    echo "[1/5] Installing Docker..."
    apt-get update
    apt-get install -y docker.io docker-compose
    systemctl enable --now docker
else
    echo "[1/5] Docker already installed."
fi

# 2. Install nginx + certbot
if ! command -v nginx &>/dev/null; then
    echo "[2/5] Installing nginx + certbot..."
    apt-get install -y nginx certbot python3-certbot-nginx
else
    echo "[2/5] nginx already installed."
fi

# 3. Prompt for domain
read -p "Enter your domain (e.g. eatlytic.in): " DOMAIN
if [ -z "$DOMAIN" ]; then
    echo "No domain provided. Skipping SSL. App will run on HTTP."
    SSL=false
else
    SSL=true
fi

# 4. Deploy the app
echo "[3/5] Deploying Eatlytic..."
mkdir -p /opt/eatlytic/data
cd /opt/eatlytic

# Upload code via scp separately, then:
# docker compose up -d --build
echo "  -> Run: cd /opt/eatlytic && docker compose up -d --build"

# 5. Configure nginx + SSL
if [ "$SSL" = true ]; then
    echo "[4/5] Configuring nginx for $DOMAIN..."
    cat > /etc/nginx/sites-available/eatlytic <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
    ln -sf /etc/nginx/sites-available/eatlytic /etc/nginx/sites-enabled/
    nginx -t && systemctl reload nginx

    echo "[5/5] Getting SSL certificate..."
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --register-unsafely-without-email
else
    echo "[4/5] Skipping nginx/SSL (no domain)."
fi

echo ""
echo "=== Done ==="
echo "App: http://$DOMAIN (or http://<server-ip>)"
echo "Health: curl http://localhost:80/health"
echo "Logs: docker logs -f eatlytic"
