#!/usr/bin/env python3
"""Check progress of snapshot download and SPLADE indexing.

Reads the status/checkpoint files written by snapshot_downloader.py
and splade_indexer.py and prints a human-readable summary.

Usage:
    python scripts/check_progress.py
"""

import json
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "openalex_snapshot"

FILES = {
    "Download status": DATA_DIR / "snapshot_status.json",
    "Download checkpoint": DATA_DIR / "download_checkpoint.json",
    "Indexer status": DATA_DIR / "indexer_status.json",
    "Indexer checkpoint": DATA_DIR / "indexer_checkpoint.json",
    "Indexer heartbeat": DATA_DIR / "indexer_heartbeat",
    "Sync status": DATA_DIR / "sync_status.json",
    "Last sync": DATA_DIR / "last_sync.json",
}


def format_time_ago(timestamp_str: str) -> str:
    try:
        t = time.mktime(time.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S"))
        ago = time.time() - t
        if ago < 60:
            return f"{int(ago)}s ago"
        if ago < 3600:
            return f"{int(ago/60)}m ago"
        return f"{ago/3600:.1f}h ago"
    except Exception:
        return timestamp_str


def print_section(title: str, data: dict):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    state = data.get("state", "unknown")
    icon = {"running": "▶", "completed": "✓", "interrupted": "⏸", "dry_run": "🔍"}.get(state, "?")
    print(f"  State: {icon} {state}")

    ts = data.get("timestamp", "")
    if ts:
        print(f"  Updated: {ts} ({format_time_ago(ts)})")

    if "completed_parts" in data and "total_parts" in data:
        pct = 100.0 * data["completed_parts"] / max(data["total_parts"], 1)
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  Progress: [{bar}] {pct:.1f}%")
        print(f"  Parts: {data['completed_parts']}/{data['total_parts']}")

    if "total_kept" in data:
        print(f"  Records kept: {data['total_kept']:,}")

    if "total_indexed" in data:
        print(f"  Docs indexed: {data['total_indexed']:,}")

    if "total_errors" in data:
        print(f"  Errors: {data['total_errors']:,}")

    if "total_files" in data and "file_index" in data:
        pct = 100.0 * data.get("file_index", 0) / max(data["total_files"], 1)
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  Progress: [{bar}] {pct:.1f}%")
        print(f"  Files: {data.get('file_index', 0)}/{data['total_files']}")

    if "docs_per_sec" in data:
        print(f"  Throughput: {data['docs_per_sec']:.1f} docs/sec")

    if "eta" in data:
        print(f"  ETA: {data['eta']}")

    if "progress_pct" in data and "completed_parts" not in data:
        print(f"  Progress: {data['progress_pct']:.1f}%")

    gpu = data.get("gpu", {})
    if gpu and gpu.get("device") == "cuda":
        print(f"  GPU: {gpu.get('allocated_gb', '?')} GB used ({gpu.get('utilization_pct', '?')}%)")

    if "pid" in data:
        pid = data["pid"]
        try:
            os.kill(pid, 0)
            print(f"  PID: {pid} (running)")
        except OSError:
            print(f"  PID: {pid} (not running)")

    if "total_time_seconds" in data:
        t = data["total_time_seconds"]
        print(f"  Total time: {int(t//3600)}h {int((t%3600)//60)}m {int(t%60)}s")


def main():
    print("=" * 50)
    print("  SPLADE Pipeline Progress Monitor")
    print("=" * 50)

    found = False
    for label, path in FILES.items():
        if path.exists():
            found = True
            if path.name == "indexer_heartbeat":
                try:
                    ts = float(path.read_text().strip())
                    ago = time.time() - ts
                    status = "alive" if ago < 120 else f"stale ({int(ago)}s ago)"
                    print(f"\n  Indexer heartbeat: {status}")
                except Exception:
                    pass
                continue

            try:
                data = json.loads(path.read_text())
                print_section(label, data)
            except Exception as e:
                print(f"\n  {label}: error reading ({e})")

    if not found:
        print("\n  No progress files found. Pipeline hasn't started yet.")
        print(f"  Looking in: {DATA_DIR}")

    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage("/")
    print(f"\n{'─' * 50}")
    print(f"  Disk: {free / (1024**3):.1f} GB free / {total / (1024**3):.1f} GB total")
    print(f"{'─' * 50}")

    # Check for output files
    chunks = sorted(DATA_DIR.glob("works_part_*.jsonl.gz"))
    if chunks:
        total_size = sum(f.stat().st_size for f in chunks)
        print(f"  Output chunks: {len(chunks)} files, {total_size / (1024**2):.1f} MB total")

    print()


if __name__ == "__main__":
    main()
