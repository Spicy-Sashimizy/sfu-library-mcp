#!/usr/bin/env bash
# Run via cron on TrueNAS: 0 2 * * * /mnt/tank/vanse/app/deploy/backup.sh
set -euo pipefail

BACKUP_DIR="/mnt/tank/vanse/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

mkdir -p "${BACKUP_DIR}"

# Database backup
echo "[$(date)] Starting database backup..."
docker exec vanse_db pg_dump -U vanse vanse_leads | gzip > "${BACKUP_DIR}/db_${TIMESTAMP}.sql.gz"

# Config backup
cp /mnt/tank/vanse/app/.env "${BACKUP_DIR}/env_${TIMESTAMP}.bak"
cp /mnt/tank/vanse/app/config/config.yaml "${BACKUP_DIR}/config_${TIMESTAMP}.yaml"

# Cleanup old backups
find "${BACKUP_DIR}" -name "db_*.sql.gz" -mtime +${RETENTION_DAYS} -delete
find "${BACKUP_DIR}" -name "env_*.bak" -mtime +${RETENTION_DAYS} -delete
find "${BACKUP_DIR}" -name "config_*.yaml" -mtime +${RETENTION_DAYS} -delete

echo "[$(date)] Backup complete: db_${TIMESTAMP}.sql.gz"
