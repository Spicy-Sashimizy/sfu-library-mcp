# Cron Setup — Phase P Monthly Sync

OpenAlex publishes a fresh snapshot manifest on the **first Monday of each month**. `scripts/opensearch_sync.py` does a delta sync against the local OpenSearch index.

## Crontab line (host machine)

Run on the host, not inside the dev container, so the job survives container rebuilds. The host needs `docker exec` access to whatever runs the training venv.

```cron
# Sync new OpenAlex parts into local OpenSearch on the 2nd of each month at 03:15 local time.
# Runs the day after OpenAlex's first-Monday publish window to avoid catching a partial manifest.
15 3 2 * * cd /workspaces/sfu-library-mcp-training && /workspaces/sfu-library-mcp-training/.venv/bin/python3 scripts/opensearch_sync.py --resume >> logs/opensearch_sync.log 2>&1
```

Install:
```bash
crontab -e   # paste the line above
crontab -l   # verify
```

## Manual run

```bash
# Dry-run shows delta size without downloading
.venv/bin/python3 scripts/opensearch_sync.py --dry-run

# Resume after interruption
.venv/bin/python3 scripts/opensearch_sync.py --resume
```

## What this script does

1. Fetches the latest OpenAlex manifest.
2. Compares to `data/openalex_snapshot/last_sync.json` (manifest hash).
3. Downloads only new/updated JSONL parts.
4. Re-runs SPLADE indexing on the delta.
5. Updates `last_sync.json`.

Idempotent — safe to re-run. SIGINT graceful with checkpoint resume.

## Failure handling

- If the script fails, it leaves `last_sync.json` unchanged so the next run retries.
- Watch `logs/opensearch_sync.log` for "delta size" and "indexed N docs" lines. Empty delta is expected if you run mid-month.

## Disk budget reminder

Each monthly delta is typically 0.5-2 GB compressed. The full corpus is ~15-30 GB (filtered to `publication_year ≥ 2015 AND has_abstract`). Check disk before kicking off in the first quarter of the year (snapshots are larger after end-of-year publishing).
