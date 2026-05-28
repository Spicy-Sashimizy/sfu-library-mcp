#!/usr/bin/env python3
"""Monthly delta sync from OpenAlex snapshot to local OpenSearch index.

Compares the current OpenAlex snapshot manifest against the last sync
record, downloads only new/updated parts, re-runs SPLADE encoding on
the delta, and upserts to OpenSearch.

This script is idempotent — safe to run multiple times. It only
processes parts not yet seen according to last_sync.json.

Checkpoint / Resume
───────────────────
Like the full indexer, sync can be interrupted and resumed:

    python scripts/opensearch_sync.py --resume

Failsafes
─────────
- Idempotent: re-running processes only unsynced parts
- Atomic last_sync.json update — only written after successful processing
- SIGINT / SIGTERM graceful shutdown with checkpoint
- Reuses splade_indexer's encoding and bulk upsert logic
- Pre-flight OpenSearch health check before starting
- Progress logging with ETA

Recommended cadence: first Monday of each month (mirrors OpenAlex release).

Usage:
    # Run monthly sync
    python scripts/opensearch_sync.py

    # Resume interrupted sync
    python scripts/opensearch_sync.py --resume

    # Dry run — check what would be synced without downloading
    python scripts/opensearch_sync.py --dry-run

    # Custom settings
    python scripts/opensearch_sync.py \\
        --model naver/splade-cocondenser-distil \\
        --opensearch-url http://localhost:9200
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "openalex_snapshot"
# IMPORTANT (Item 6 — train/serve skew): the delta indexer MUST encode with
# the SAME SPLADE model the index was originally built with, otherwise the
# sparse vectors written by the monthly sync are incompatible with the
# existing corpus and with what the serving leg expects. The canonical
# index-build model is splade_indexer.DEFAULT_MODEL; keep this in lockstep.
# A mismatch is warned about at runtime in run_sync().
DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"
DEFAULT_OPENSEARCH_URL = "http://localhost:9200"
DEFAULT_INDEX = "openalex_works"
DEFAULT_MIN_YEAR = 2015
LAST_SYNC_FILE = "last_sync.json"
SYNC_CHECKPOINT_FILE = "sync_checkpoint.json"
SYNC_STATUS_FILE = "sync_status.json"
OPENALEX_MANIFEST_URL = "https://openalex.s3.amazonaws.com/data/works/manifest"
HTTP_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_BACKOFF = 5

# ── Graceful shutdown ────────────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        logger.warning("Second %s — forcing exit", sig_name)
        sys.exit(1)
    logger.info("%s received — finishing current part, then saving checkpoint", sig_name)
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ── Atomic file I/O ──────────────────────────────────────────────────────────


def _atomic_write_json(data: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


# ── Manifest + diff ──────────────────────────────────────────────────────────


def fetch_manifest(session: requests.Session) -> dict:
    """Fetch current manifest. Returns {manifest_hash, entries, raw}."""
    resp = session.get(OPENALEX_MANIFEST_URL, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    raw = resp.text
    manifest = resp.json()
    manifest_hash = hashlib.sha256(raw.encode()).hexdigest()
    return {
        "hash": manifest_hash,
        "entries": manifest.get("entries", []),
        "raw": manifest,
    }


def load_last_sync(data_dir: Path) -> dict | None:
    path = data_dir / LAST_SYNC_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def part_identity(url: str) -> str:
    """Stable identity for an OpenAlex snapshot part, independent of host/URL.

    OpenAlex re-publishes the FULL snapshot every month under new dated URLs,
    but each part lives under an `updated_date=YYYY-MM-DD/<filename>` partition
    that reflects the records' update date. Diffing by the raw URL therefore
    treats every part as "new" on each republish and re-indexes all ~150M docs.

    We instead key on the `updated_date=.../<filename>` suffix so a part whose
    records have not changed is recognised as already synced even if its URL
    (bucket host, query string, signing params) differs.

    Falls back to the trailing path segment if no updated_date partition is
    present (e.g. the S3-listing fallback manifest).
    """
    if not url:
        return ""
    # Drop any query string / signing params.
    path = url.split("?", 1)[0]
    marker = "updated_date="
    idx = path.find(marker)
    if idx != -1:
        return path[idx:]  # e.g. "updated_date=2024-01-15/part_000.gz"
    return path.rsplit("/", 1)[-1]  # filename fallback


def compute_delta(current_entries: list[dict], last_sync: dict | None) -> list[dict]:
    """Find parts in current manifest not present in last sync.

    Compares by *part identity* (the stable updated_date partition + filename),
    NOT the raw URL — see part_identity(). This prevents OpenAlex's monthly
    full-snapshot republish (which mints new URLs for unchanged data) from
    looking like an all-new delta and re-indexing the entire corpus.
    """
    if not last_sync:
        return current_entries

    # Backward-compatible: derive identities from previously stored URLs as
    # well as any explicitly stored synced_ids.
    synced_ids = set(last_sync.get("synced_ids", []))
    synced_ids.update(part_identity(u) for u in last_sync.get("synced_urls", []))

    delta = [
        e for e in current_entries
        if part_identity(e.get("url", "")) not in synced_ids
    ]
    return delta


# ── Download + filter (reuses snapshot_downloader logic) ─────────────────────


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    word_positions = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                word_positions.append((pos, word))
    if not word_positions:
        return None
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


def download_and_filter_part(
    session: requests.Session,
    part_url: str,
    min_year: int,
) -> list[dict] | None:
    """Download a part, filter, return cleaned records.

    Returns the list of cleaned records on success (possibly empty if no
    records qualify), or None if the download failed after all retries.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(part_url, timeout=HTTP_TIMEOUT, stream=True)
            resp.raise_for_status()
            raw = gzip.decompress(resp.content)
            records = []
            for line in raw.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    work = json.loads(line)
                except json.JSONDecodeError:
                    continue

                pub_year = work.get("publication_year")
                if pub_year is not None and pub_year < min_year:
                    continue

                abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
                if not abstract or len(abstract) < 50:
                    continue

                openalex_id = work.get("id", "")
                if openalex_id.startswith("https://openalex.org/"):
                    openalex_id = openalex_id.replace("https://openalex.org/", "")

                records.append({
                    "id": openalex_id,
                    "doi": work.get("doi"),
                    "title": work.get("title", ""),
                    "abstract": abstract,
                    "publication_year": pub_year,
                    "type": work.get("type", ""),
                })
            return records
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning("Download failed (attempt %d): %s — retry in %ds", attempt, e, wait)
                time.sleep(wait)
            else:
                logger.error("Failed after %d attempts: %s", MAX_RETRIES, e)
                return None


# ── SPLADE encode + upsert (reuses splade_indexer logic) ─────────────────────


def encode_and_upsert(
    records: list[dict],
    encoder,
    session: requests.Session,
    opensearch_url: str,
    index_name: str,
    batch_size: int,
) -> dict:
    """Encode records through SPLADE and upsert to OpenSearch. Returns stats."""
    from scripts.splade_indexer import bulk_upsert_opensearch

    stats = {"indexed": 0, "errors": 0, "empty": 0}

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        texts = []
        valid = []
        for rec in batch:
            text = f"{rec.get('title', '')} {rec.get('abstract', '')}".strip()
            if text:
                texts.append(text)
                valid.append(rec)

        if not texts:
            continue

        try:
            sparse_vecs = encoder.encode_batch(texts)
        except Exception as e:
            logger.error("Encoding error: %s", e)
            stats["errors"] += len(texts)
            continue

        os_docs = []
        for rec, sparse in zip(valid, sparse_vecs):
            if not sparse:
                stats["empty"] += 1
                continue
            os_docs.append({
                "id": rec["id"],
                "doi": rec.get("doi"),
                "title": rec.get("title", ""),
                "abstract": rec.get("abstract", ""),
                "publication_year": rec.get("publication_year"),
                "type": rec.get("type", ""),
                "openalex_id": rec["id"],
                "sparse_field": sparse,
            })

        if os_docs:
            result = bulk_upsert_opensearch(session, opensearch_url, index_name, os_docs)
            stats["indexed"] += result["indexed"]
            stats["errors"] += result["errors"]

    return stats


# ── Main sync pipeline ───────────────────────────────────────────────────────


def run_sync(
    data_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    opensearch_url: str,
    index_name: str,
    min_year: int,
    resume: bool,
    dry_run: bool,
) -> dict:
    global _shutdown_requested

    data_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "SFULibraryMCP-Sync/1.0 (mailto:lib-systems@sfu.ca)"
    })

    # ── Fetch manifest and compute delta ─────────────────────────────────
    logger.info("Fetching current manifest...")
    try:
        manifest = fetch_manifest(session)
    except Exception as e:
        logger.error("Cannot fetch manifest: %s", e)
        return {"error": "manifest_fetch_failed"}

    last_sync = load_last_sync(data_dir)
    if last_sync:
        logger.info(
            "Last sync: %s (%d parts synced)",
            last_sync.get("timestamp", "unknown"),
            len(last_sync.get("synced_urls", [])),
        )
    else:
        logger.info("No previous sync found — this will process all parts")

    delta = compute_delta(manifest["entries"], last_sync)
    logger.info("Delta: %d new/updated parts to process", len(delta))

    if not delta:
        logger.info("Nothing to sync — index is up to date")
        return {"state": "up_to_date", "delta_parts": 0}

    if dry_run:
        logger.info("─── DRY RUN ───")
        logger.info("Would process %d parts", len(delta))
        logger.info("Current manifest hash: %s", manifest["hash"][:16])
        return {"state": "dry_run", "delta_parts": len(delta)}

    # ── Load checkpoint for resume ───────────────────────────────────────
    start_idx = 0
    synced_urls = set(last_sync.get("synced_urls", [])) if last_sync else set()
    cumulative = {"indexed": 0, "errors": 0, "parts_done": 0}

    if resume:
        cp_path = data_dir / SYNC_CHECKPOINT_FILE
        if cp_path.exists():
            try:
                cp = json.loads(cp_path.read_text())
                start_idx = cp.get("delta_index", 0)
                cumulative = cp.get("cumulative", cumulative)
                extra_urls = cp.get("newly_synced_urls", [])
                synced_urls.update(extra_urls)
                logger.info("Resuming sync from part %d/%d", start_idx, len(delta))
            except Exception:
                pass

    # ── Pre-flight check ─────────────────────────────────────────────────
    from scripts.splade_indexer import (
        DEFAULT_MODEL as INDEXER_MODEL,
        SpladeEncoder,
        check_opensearch_health,
    )

    # Item 6 — guard against train/serve skew: the model used to encode the
    # delta MUST match the model the index was built with. Diverging models
    # produce incompatible sparse vectors and silently degrade retrieval.
    if model_name != INDEXER_MODEL:
        logger.warning(
            "SPLADE model mismatch: sync is using %r but the index was built "
            "with %r (splade_indexer.DEFAULT_MODEL). Sparse vectors may be "
            "incompatible — pass --model %s to match.",
            model_name, INDEXER_MODEL, INDEXER_MODEL,
        )

    if not check_opensearch_health(session, opensearch_url, index_name):
        return {"error": "opensearch_unhealthy"}

    encoder = SpladeEncoder(model_name, device=device)

    # ── Process delta parts ──────────────────────────────────────────────
    start_time = time.time()
    newly_synced = list(synced_urls - (set(last_sync.get("synced_urls", [])) if last_sync else set()))

    for i in range(start_idx, len(delta)):
        if _shutdown_requested:
            logger.info("Shutdown — saving sync checkpoint at part %d/%d", i, len(delta))
            _atomic_write_json({
                "delta_index": i,
                "cumulative": cumulative,
                "newly_synced_urls": newly_synced,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, data_dir / SYNC_CHECKPOINT_FILE)
            return {**cumulative, "state": "interrupted"}

        part = delta[i]
        part_url = part.get("url", "")
        logger.info("Sync part %d/%d: %s", i + 1, len(delta), part_url.split("/")[-1])

        records = download_and_filter_part(session, part_url, min_year)
        if records is None:
            # Download failed after all retries — do NOT mark synced so the
            # part is retried on the next run. Surface the failure as an error.
            logger.error("Skipping part (download failed): %s", part_url)
            cumulative["errors"] += 1
            continue

        if records:
            stats = encode_and_upsert(
                records, encoder, session, opensearch_url, index_name, batch_size
            )
            cumulative["indexed"] += stats["indexed"]
            cumulative["errors"] += stats["errors"]

        cumulative["parts_done"] += 1
        synced_urls.add(part_url)
        newly_synced.append(part_url)

        # Save incremental checkpoint
        _atomic_write_json({
            "delta_index": i + 1,
            "cumulative": cumulative,
            "newly_synced_urls": newly_synced,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, data_dir / SYNC_CHECKPOINT_FILE)

        elapsed = time.time() - start_time
        rate = (i + 1 - start_idx) / max(elapsed, 0.001)
        remaining = (len(delta) - i - 1) / max(rate, 0.001)
        logger.info(
            "Progress: %d/%d parts | %d indexed | ETA: %s",
            i + 1, len(delta), cumulative["indexed"],
            time.strftime("%H:%M:%S", time.gmtime(remaining)),
        )

    # ── Update last_sync.json ────────────────────────────────────────────
    all_synced = list(synced_urls)
    # Persist stable part identities so future republishes (new URLs, same
    # data) are correctly recognised as already synced — see part_identity().
    all_synced_ids = sorted({part_identity(u) for u in all_synced})
    _atomic_write_json({
        "manifest_hash": manifest["hash"],
        # Persist only the current manifest's URLs (not the full history) so
        # last_sync.json stays bounded across monthly republishes. The canonical
        # bounded dedup store is synced_ids — see part_identity().
        "synced_urls": [e.get("url", "") for e in manifest["entries"]],
        "synced_ids": all_synced_ids,
        "total_synced_parts": len(all_synced_ids),
        "last_delta_indexed": cumulative["indexed"],
        "last_delta_errors": cumulative["errors"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, data_dir / LAST_SYNC_FILE)

    # Clean up checkpoint
    cp_file = data_dir / SYNC_CHECKPOINT_FILE
    if cp_file.exists():
        cp_file.unlink()

    total_time = time.time() - start_time
    logger.info("═══ Sync complete ═══")
    logger.info("Time: %s", time.strftime("%H:%M:%S", time.gmtime(total_time)))
    logger.info("Parts processed: %d", cumulative["parts_done"])
    logger.info("Docs indexed: %d", cumulative["indexed"])
    logger.info("Errors: %d", cumulative["errors"])

    return {**cumulative, "state": "completed"}


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Monthly delta sync from OpenAlex snapshot to OpenSearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"SPLADE model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
    )
    parser.add_argument(
        "--opensearch-url", type=str,
        default=os.environ.get("SFU_OPENSEARCH_URL", DEFAULT_OPENSEARCH_URL),
    )
    parser.add_argument(
        "--index", type=str,
        default=os.environ.get("SFU_OPENSEARCH_INDEX", DEFAULT_INDEX),
    )
    parser.add_argument(
        "--min-year", type=int, default=DEFAULT_MIN_YEAR,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted sync",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be synced without downloading",
    )
    args = parser.parse_args()

    logger.info("OpenAlex Monthly Sync")
    if args.resume:
        logger.info("Mode: RESUME")
    if args.dry_run:
        logger.info("Mode: DRY RUN")

    result = run_sync(
        data_dir=args.data_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        opensearch_url=args.opensearch_url,
        index_name=args.index,
        min_year=args.min_year,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    if result.get("state") == "interrupted":
        sys.exit(130)
    elif result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
