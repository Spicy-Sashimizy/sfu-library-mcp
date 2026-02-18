# TrueNAS CE Setup Guide — Vanse Data Stack

One-time setup for deploying the Vanse Data Stack on a TrueNAS CE home lab server.

## Prerequisites

- TrueNAS CE (Community Edition) installed and accessible on LAN
- Docker available on TrueNAS (via Apps or direct install)
- LAN IP address assigned (e.g., 192.168.1.100)
- ClaudeBox dev container running with SSH key generated

## 1. ZFS Dataset Creation

Create a dedicated ZFS dataset hierarchy for the Vanse stack:

```bash
# On TrueNAS shell or via SSH
zfs create tank/vanse
zfs create tank/vanse/postgres
zfs create tank/vanse/n8n
zfs create tank/vanse/metabase
zfs create tank/vanse/backups
zfs create tank/vanse/logs

# Set permissions
chown -R 1000:1000 /mnt/tank/vanse
chmod -R 755 /mnt/tank/vanse
```

The ZFS datasets provide:
- Per-dataset snapshots and compression
- `tank/vanse/postgres` — PostgreSQL data (most critical)
- `tank/vanse/backups` — pg_dump backups with 30-day retention

## 2. Docker Setup

Verify Docker is available:

```bash
docker --version
docker compose version
```

If Docker is not available, install it following the TrueNAS CE Docker guide or enable it via the TrueNAS Apps system.

## 3. User and SSH Setup

Create a dedicated `vanse` user for deployments:

```bash
# Create user (if not already existing)
useradd -m -s /bin/bash vanse
usermod -aG docker vanse

# Set up SSH directory
mkdir -p /home/vanse/.ssh
chmod 700 /home/vanse/.ssh
```

Copy the dev container's public key:

```bash
# From the dev container:
ssh-copy-id -i /workspaces/vansedataadstackshit/.ssh/id_ed25519.pub vanse@192.168.1.100

# Or manually on TrueNAS:
# Paste the contents of id_ed25519.pub into:
echo "ssh-ed25519 AAAA... claudebox" >> /home/vanse/.ssh/authorized_keys
chmod 600 /home/vanse/.ssh/authorized_keys
chown -R vanse:vanse /home/vanse/.ssh
```

Test connectivity from the dev container:

```bash
ssh -i /workspaces/vansedataadstackshit/.ssh/id_ed25519 vanse@192.168.1.100 "echo ok"
```

## 4. Firewall Rules

Only expose services on the LAN, not externally:

| Port | Service | Access |
|------|---------|--------|
| 22 | SSH | LAN only |
| 5678 | n8n | LAN only |
| 3000 | Metabase | LAN only |
| 5432 | PostgreSQL | Internal (Docker network only) |

PostgreSQL is NOT exposed to the host — it's internal to the Docker network. Only n8n and Metabase have ports mapped.

## 5. Environment File

Create the production `.env` on TrueNAS (this file is never synced via rsync):

```bash
mkdir -p /mnt/tank/vanse/app
cat > /mnt/tank/vanse/app/.env << 'ENVEOF'
# Production .env — fill in real credentials
DB_PASSWORD=<strong-password-here>
DATABASE_URL=postgresql://vanse:<db-password>@vanse_db:5432/vanse_leads

SERPAPI_KEY=
APOLLO_API_KEY=
HUNTER_API_KEY=
ANTHROPIC_API_KEY=

DATAIMPULSE_USERNAME=
DATAIMPULSE_PASSWORD=

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_SES_REGION=us-east-1
MAILGUN_API_KEY=
MAILGUN_DOMAIN=vanse.vancitylandscaper.com

SENDING_DOMAIN=vanse.vancitylandscaper.com
SENDING_FROM_NAME=Vanse Equipment Team
SENDING_FROM_EMAIL=team@vanse.vancitylandscaper.com
REPLY_TO_EMAIL=info@vanseindustry.com

N8N_PASSWORD=<n8n-password>
MB_ADMIN_EMAIL=admin@vanseindustry.com
MB_ADMIN_PASSWORD=<metabase-password>
ENVEOF

chmod 600 /mnt/tank/vanse/app/.env
```

## 6. First Deployment

From the ClaudeBox dev container:

```bash
# 1. Generate SSH key (if not already done)
mkdir -p /workspaces/vansedataadstackshit/.ssh
ssh-keygen -t ed25519 -f /workspaces/vansedataadstackshit/.ssh/id_ed25519 -N ""

# 2. Copy key to TrueNAS
ssh-copy-id -i /workspaces/vansedataadstackshit/.ssh/id_ed25519.pub vanse@192.168.1.100

# 3. Deploy
make deploy
```

The deploy script will:
1. Run unit tests locally
2. rsync code to TrueNAS
3. Build the Docker image
4. Start all 4 services
5. Run database migrations
6. Show health status

## 7. Cron Jobs

Set up automated tasks on TrueNAS:

```bash
# Edit vanse user's crontab
crontab -e -u vanse
```

Add these entries:

```cron
# Daily database backup at 2 AM
0 2 * * * /mnt/tank/vanse/app/deploy/backup.sh >> /mnt/tank/vanse/logs/backup.log 2>&1

# Health check every 6 hours
0 */6 * * * cd /mnt/tank/vanse/app && python -m deploy.health_check >> /mnt/tank/vanse/logs/health.log 2>&1
```

## 8. Monitoring

### Manual Health Check

From the dev container (remote):

```bash
make health-check-remote
```

On TrueNAS (local):

```bash
cd /mnt/tank/vanse/app
python -m deploy.health_check
```

### DNS Verification

Before enabling email outreach:

```bash
make verify-dns
```

### Log Access

```bash
# Scraper logs
docker logs vanse_scrapers --tail 100

# n8n logs
docker logs vanse_n8n --tail 100

# Database logs
docker logs vanse_db --tail 100

# All services
docker compose -f docker-compose.prod.yml logs --tail 50
```

## 9. Troubleshooting

### Containers won't start

```bash
# Check compose config
docker compose -f docker-compose.prod.yml config

# Check individual container logs
docker logs vanse_db 2>&1 | tail -20
docker logs vanse_scrapers 2>&1 | tail -20
```

### Database connection issues

```bash
# Check if DB is accepting connections
docker exec vanse_db pg_isready -U vanse -d vanse_leads

# Check DB logs
docker logs vanse_db --tail 50

# Connect manually
docker exec -it vanse_db psql -U vanse -d vanse_leads
```

### Disk space

```bash
# ZFS dataset usage
zfs list -r tank/vanse

# Docker disk usage
docker system df
```

### Migration failures

```bash
# Run migrations manually
docker exec vanse_scrapers python -m database.migrate

# Check applied migrations
docker exec vanse_db psql -U vanse -d vanse_leads -c \
    "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;"
```

## 10. Disaster Recovery

### Restore from backup

```bash
# List available backups
ls -la /mnt/tank/vanse/backups/db_*.sql.gz

# Restore (interactive — prompts for confirmation)
bash /mnt/tank/vanse/app/deploy/restore.sh /mnt/tank/vanse/backups/db_YYYYMMDD_HHMMSS.sql.gz
```

### Rebuild from scratch

If the server needs to be rebuilt:

1. Install TrueNAS CE
2. Follow this guide from step 1
3. Restore the latest database backup
4. Run `make deploy` from the dev container

### ZFS snapshots (recommended)

```bash
# Create manual snapshot before risky changes
zfs snapshot tank/vanse/postgres@pre-migration

# Roll back if needed
zfs rollback tank/vanse/postgres@pre-migration

# Auto-snapshots (if zfs-auto-snapshot is available)
zfs set com.sun:auto-snapshot=true tank/vanse/postgres
```
