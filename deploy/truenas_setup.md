# TrueNAS Setup Guide — SFU Library MCP Server

One-time setup for deploying the SFU Library MCP server on TrueNAS SCALE 25.04.1.

## Prerequisites

- TrueNAS SCALE 25.04.1 at 192.168.1.142
- Docker v27.5.0 + Compose v2.32.3
- SSH user `gordoz` (key-based auth, ProxyJump via Windows host)
- `gordoz` sudo restricted to `/usr/bin/docker` only

## 1. ZFS Dataset Creation (Admin — TrueNAS Web Shell)

**These commands must be run by an admin user via TrueNAS Web Shell or SSH as `truenas_admin`.
The `gordoz` user cannot create ZFS datasets.**

```bash
zfs create MAIN/sfu-library-mcp
mkdir -p /mnt/MAIN/sfu-library-mcp/{app,secrets,logs,data,downloads}
chown -R gordoz:gordoz /mnt/MAIN/sfu-library-mcp
chmod 700 /mnt/MAIN/sfu-library-mcp/secrets
```

Resulting layout:
```
/mnt/MAIN/sfu-library-mcp/
├── app/          # Git clone or rsync target
├── secrets/      # chmod 700 credential files
├── logs/         # Persistent log volume
├── data/         # Token cache
└── downloads/    # Downloaded PDFs
```

## 2. Create Secret Files (Admin — TrueNAS Web Shell)

```bash
echo -n "YOUR_SFU_USERNAME" > /mnt/MAIN/sfu-library-mcp/secrets/sfu_username
echo -n "YOUR_SFU_PASSWORD" > /mnt/MAIN/sfu-library-mcp/secrets/sfu_password
echo -n "YOUR_MFA_SECRET" > /mnt/MAIN/sfu-library-mcp/secrets/sfu_mfa_secret
echo -n "YOUR_MFA_DEVICE_NAME" > /mnt/MAIN/sfu-library-mcp/secrets/sfu_mfa_device_name
echo -n "YOUR_ZOTERO_API_KEY" > /mnt/MAIN/sfu-library-mcp/secrets/zotero_api_key
echo -n "YOUR_ZOTERO_USER_ID" > /mnt/MAIN/sfu-library-mcp/secrets/zotero_user_id

chown gordoz:gordoz /mnt/MAIN/sfu-library-mcp/secrets/*
chmod 600 /mnt/MAIN/sfu-library-mcp/secrets/*
```

## 3. SSH Config (Dev Container)

Already configured in `~/.ssh/config`:
```
Host truenas
    HostName 192.168.1.142
    User gordoz
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump windows-host

Host windows-host
    HostName host.docker.internal
    User gordo
    IdentityFile ~/.ssh/id_ed25519
```

## 4. First Deployment

From the dev container:
```bash
bash deploy/deploy.sh
```

## 5. Docker Cleanup (Optional)

Free ~54GB of unused images on TrueNAS:
```bash
# Run from TrueNAS Web Shell
sudo docker image prune -a --filter "until=720h"
```

## 6. Verification

```bash
bash deploy/healthcheck.sh
```
