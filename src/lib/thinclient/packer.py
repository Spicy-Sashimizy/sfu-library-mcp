"""Hot/cold section pack/unpack — operates on BUILT ARTIFACTS, not source docs.

Key improvement over the measured OpenSearch flow (eval_sectioned_index.py):
that design re-indexed JSONL on unpack (measured 2,016 docs/s => ~2.4 h for a
med_bio-sized section at 150M). Packing the engine artifacts themselves makes
unpack pure zstd decompression at disk speed (GB/min), and parity is exact by
construction — the same files come back (verified via content checksums).

pack:   sections/<name>/  ->  packed/<name>.tar.zst   (zstd-19, long-distance
        matching, multithreaded), then the live dir is removed.
unpack: the reverse. The archive is KEPT after unpack so re-packing a section
        that received no writes is just deleting the live dir.

CLI:
    .venv/bin/python3 -m lib.thinclient.packer pack    <index_root> <section>
    .venv/bin/python3 -m lib.thinclient.packer unpack  <index_root> <section>
    .venv/bin/python3 -m lib.thinclient.packer repack  <index_root> <section>
    .venv/bin/python3 -m lib.thinclient.packer status  <index_root>
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import time
from pathlib import Path

import zstandard

logger = logging.getLogger("sfu_library_mcp")

ZSTD_LEVEL = 19
ZSTD_WINDOW_LOG = 27  # 128 MB long-distance matching window


def _cctx() -> zstandard.ZstdCompressor:
    try:
        params = zstandard.ZstdCompressionParameters.from_level(
            ZSTD_LEVEL, enable_ldm=True, window_log=ZSTD_WINDOW_LOG,
            threads=os.cpu_count() or 8)
        return zstandard.ZstdCompressor(compression_params=params)
    except Exception:
        return zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=os.cpu_count() or 8)


def _dir_checksum(path: Path) -> str:
    """Order-stable checksum of file names + contents (for pack parity checks)."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if not f.is_file():
            continue
        h.update(str(f.relative_to(path)).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _load_manifest(index_root: Path) -> dict:
    p = Path(index_root) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {"sections": {}}


def _save_manifest(index_root: Path, manifest: dict) -> None:
    p = Path(index_root) / "manifest.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, p)


def pack_section(index_root: Path, section: str, verify: bool = True,
                 remove_live: bool = True) -> dict:
    """Compress a live section dir into packed/<section>.tar.zst."""
    index_root = Path(index_root)
    live = index_root / "sections" / section
    if not live.is_dir():
        raise FileNotFoundError(f"no live section at {live}")
    packed_dir = index_root / "packed"
    packed_dir.mkdir(exist_ok=True)
    archive = packed_dir / f"{section}.tar.zst"

    checksum = _dir_checksum(live) if verify else ""
    live_bytes = sum(f.stat().st_size for f in live.rglob("*") if f.is_file())

    t0 = time.perf_counter()
    tmp = archive.with_suffix(".zst.tmp")
    with open(tmp, "wb") as fh, _cctx().stream_writer(fh) as zw:
        with tarfile.open(fileobj=zw, mode="w|") as tar:
            tar.add(live, arcname=section)
    os.replace(tmp, archive)
    secs = time.perf_counter() - t0

    if remove_live:
        shutil.rmtree(live)

    manifest = _load_manifest(index_root)
    entry = manifest["sections"].setdefault(section, {})
    entry.update({"state": "packed", "archive": str(archive.relative_to(index_root)),
                  "live_bytes": live_bytes, "packed_bytes": archive.stat().st_size,
                  "checksum": checksum, "packed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    _save_manifest(index_root, manifest)

    info = {"section": section, "live_mb": round(live_bytes / 1e6, 1),
            "packed_mb": round(archive.stat().st_size / 1e6, 1),
            "ratio": round(live_bytes / max(archive.stat().st_size, 1), 2),
            "pack_secs": round(secs, 1)}
    logger.info("packed %s", info)
    return info


def unpack_section(index_root: Path, section: str, verify: bool = True) -> dict:
    """Restore a packed section to live (the off-domain 'library expansion')."""
    index_root = Path(index_root)
    archive = index_root / "packed" / f"{section}.tar.zst"
    if not archive.is_file():
        raise FileNotFoundError(f"no archive at {archive}")
    live_parent = index_root / "sections"
    live = live_parent / section
    if live.exists():
        shutil.rmtree(live)

    t0 = time.perf_counter()
    dctx = zstandard.ZstdDecompressor(max_window_size=2 ** ZSTD_WINDOW_LOG)
    with open(archive, "rb") as fh, dctx.stream_reader(fh) as zr:
        with tarfile.open(fileobj=io.BufferedReader(zr, buffer_size=8 << 20),
                          mode="r|") as tar:
            try:
                tar.extractall(live_parent, filter="data")
            except TypeError:  # Python < 3.11.4: no extraction filters;
                tar.extractall(live_parent)  # archives are self-produced
    secs = time.perf_counter() - t0

    manifest = _load_manifest(index_root)
    entry = manifest["sections"].setdefault(section, {})
    parity_ok = None
    if verify and entry.get("checksum"):
        parity_ok = _dir_checksum(live) == entry["checksum"]
        if not parity_ok:
            raise RuntimeError(
                f"unpack parity FAILED for {section}: artifact checksum mismatch")
    entry["state"] = "live"
    _save_manifest(index_root, manifest)

    info = {"section": section, "unpack_secs": round(secs, 1),
            "parity_checksum_ok": parity_ok}
    logger.info("unpacked %s", info)
    return info


def repack_section(index_root: Path, section: str) -> dict:
    """Drop a live section whose archive is still current (no writes since
    unpack) — 'repack' is just deleting the live dir."""
    index_root = Path(index_root)
    archive = index_root / "packed" / f"{section}.tar.zst"
    live = index_root / "sections" / section
    if not archive.is_file():
        raise FileNotFoundError(f"no archive for {section}; use pack_section")
    if live.exists():
        shutil.rmtree(live)
    manifest = _load_manifest(index_root)
    manifest["sections"].setdefault(section, {})["state"] = "packed"
    _save_manifest(index_root, manifest)
    return {"section": section, "state": "packed"}


def status(index_root: Path) -> dict:
    manifest = _load_manifest(index_root)
    return {name: {"state": e.get("state"),
                   "live_mb": round(e.get("live_bytes", 0) / 1e6, 1),
                   "packed_mb": round(e.get("packed_bytes", 0) / 1e6, 1)}
            for name, e in manifest.get("sections", {}).items()}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cmd, root = sys.argv[1], Path(sys.argv[2])
    if cmd == "status":
        print(json.dumps(status(root), indent=2))
    else:
        section = sys.argv[3]
        fn = {"pack": pack_section, "unpack": unpack_section,
              "repack": repack_section}[cmd]
        print(json.dumps(fn(root, section), indent=2))
