#!/usr/bin/env bash
# Restore database from a backup file
# Usage: ./deploy/restore.sh /mnt/tank/vanse/backups/db_YYYYMMDD_HHMMSS.sql.gz
set -euo pipefail

BACKUP_FILE="${1:-}"

if [ -z "${BACKUP_FILE}" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    echo ""
    echo "Available backups:"
    ls -la /mnt/tank/vanse/backups/db_*.sql.gz 2>/dev/null || echo "  No backups found"
    exit 1
fi

if [ ! -f "${BACKUP_FILE}" ]; then
    echo "ERROR: Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

echo "═══ Vanse Data Stack — Database Restore ═══"
echo "Backup file: ${BACKUP_FILE}"
echo ""
echo "WARNING: This will DROP the existing vanse_leads database and recreate it."
read -p "Continue? (y/N): " confirm
if [ "${confirm}" != "y" ] && [ "${confirm}" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

# Drop and recreate database
echo "→ Dropping existing database..."
docker exec vanse_db psql -U vanse -d postgres -c "DROP DATABASE IF EXISTS vanse_leads;"
docker exec vanse_db psql -U vanse -d postgres -c "CREATE DATABASE vanse_leads OWNER vanse;"

# Restore from backup
echo "→ Restoring from backup..."
gunzip -c "${BACKUP_FILE}" | docker exec -i vanse_db psql -U vanse -d vanse_leads

echo "→ Verifying restore..."
docker exec vanse_db psql -U vanse -d vanse_leads -c "SELECT COUNT(*) as companies FROM companies;"

echo "═══ Restore complete ═══"
