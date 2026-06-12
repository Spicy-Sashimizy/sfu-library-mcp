#!/usr/bin/env python3
"""Remove tantivy segment files not referenced by the index's own metadata.

Why this exists: on 2026-06-12 two orphaned build workers from a stopped run
kept their merge threads alive for ~4 min next to the restarted build and may
have dropped segment files (foreign UUIDs) into sections/other__recent and
sections/social_sciences__recent. Those files are invisible to the live index
(not in its meta.json) but inflate the on-disk/packed size. Foreign files are
NOT in .managed.json either, so tantivy's own GC never deletes them.

ONLY run against sections whose build has COMPLETED (writer idle) — deleting
under a live writer is unsafe.

Usage:
    .venv/bin/python3 scripts/sweep_stray_tantivy_files.py <index_root> <section> [--delete]

Default is a dry run (lists strays + bytes); pass --delete to remove.
"""

import json
import re
import sys
from pathlib import Path

KEEP = {"meta.json", ".managed.json", ".tantivy-meta.lock", ".tantivy-writer.lock"}
SEG_RE = re.compile(r"^([0-9a-f]{32})\.")


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    root, section = Path(sys.argv[1]), sys.argv[2]
    delete = "--delete" in sys.argv
    tdir = root / "sections" / section / "tantivy"
    meta = json.loads((tdir / "meta.json").read_text())
    live = {seg["segment_id"].replace("-", "") for seg in meta["segments"]}
    print(f"{section}: {len(live)} live segments in meta.json")

    strays, stray_bytes = [], 0
    for f in sorted(tdir.iterdir()):
        if f.name in KEEP:
            continue
        m = SEG_RE.match(f.name)
        if m and m.group(1) in live:
            continue
        strays.append(f)
        stray_bytes += f.stat().st_size
    if not strays:
        print("no strays — clean")
        return
    for f in strays:
        print(f"  stray: {f.name}  {f.stat().st_size:,} B")
    print(f"{len(strays)} stray files, {stray_bytes / 1e6:.1f} MB total")
    if delete:
        for f in strays:
            f.unlink()
        print("deleted")
    else:
        print("dry run — pass --delete to remove")


if __name__ == "__main__":
    main()
