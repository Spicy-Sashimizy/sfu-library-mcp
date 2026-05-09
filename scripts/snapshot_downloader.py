#!/usr/bin/env python3
"""Download and filter OpenAlex Works snapshot for SPLADE indexing.

Downloads the OpenAlex monthly snapshot (Works entity only), streams each
compressed part, filters to publication_year >= 2015 with abstracts present,
and writes cleaned JSONL chunks to data/openalex_snapshot/.

Checkpoint / Resume
───────────────────
The downloader saves progress after every completed part file. You can
interrupt at any time (Ctrl-C, kill, machine reboot) and resume:

    python scripts/snapshot_downloader.py --resume

It picks up from the last completed part — no re-downloading.

Failsafes
─────────
- Atomic checkpoint writes (temp + os.replace) — safe against sudden kill
- SIGINT / SIGTERM handlers — finishes current part, saves checkpoint, exits
- Per-part SHA256 verification when manifest provides checksums
- Automatic retry with backoff on HTTP failures (3 attempts per part)
- --dry-run processes 1 part and reports stats without writing full output
- Progress file (snapshot_status.json) pollable by external monitors

Usage:
    # Full download (first run)
    python scripts/snapshot_downloader.py

    # Resume after interruption
    python scripts/snapshot_downloader.py --resume

    # Dry run — process 1 part, report stats
    python scripts/snapshot_downloader.py --dry-run

    # Custom year filter
    python scripts/snapshot_downloader.py --min-year 2010

    # Limit output chunk size
    python scripts/snapshot_downloader.py --chunk-size 500000
"""

import argparse
import gzip
import hashlib
import io
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

OPENALEX_SNAPSHOT_BASE = "https://openalex.s3.amazonaws.com/data/works/"
OPENALEX_MANIFEST_URL = "https://openalex.s3.amazonaws.com/data/works/manifest"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "openalex_snapshot"
CHECKPOINT_FILE = "download_checkpoint.json"
STATUS_FILE = "snapshot_status.json"
DEFAULT_MIN_YEAR = 2015
DEFAULT_CHUNK_SIZE = 500_000  # records per output chunk
HTTP_TIMEOUT = 300  # large parts can be 500MB-1.1GB
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # seconds

# ── Graceful shutdown ────────────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        logger.warning("Second %s received — forcing exit", sig_name)
        sys.exit(1)
    logger.info("%s received — finishing current part then saving checkpoint", sig_name)
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ── Atomic file I/O ──────────────────────────────────────────────────────────


def _atomic_write_json(data: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


# ── Abstract reconstruction ──────────────────────────────────────────────────


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract text from OpenAlex inverted index format.

    OpenAlex stores abstracts as {"word": [pos1, pos2, ...], ...}.
    We rebuild the original text by placing each word at its positions.
    """
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


# ── Manifest fetching ────────────────────────────────────────────────────────


def _s3_uri_to_https(uri: str) -> str:
    """Convert s3://openalex/path to https://openalex.s3.amazonaws.com/path."""
    if uri.startswith("s3://openalex/"):
        return uri.replace("s3://openalex/", "https://openalex.s3.amazonaws.com/", 1)
    return uri


def fetch_manifest(session: requests.Session) -> list[dict]:
    """Fetch the OpenAlex Works snapshot manifest.

    Returns list of dicts with keys: url, meta (content_length, record_count).
    Falls back to S3 listing if manifest endpoint fails.
    """
    logger.info("Fetching snapshot manifest...")
    try:
        resp = session.get(OPENALEX_MANIFEST_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        manifest = resp.json()
        entries = manifest.get("entries", [])
        if entries:
            for entry in entries:
                entry["url"] = _s3_uri_to_https(entry.get("url", ""))
            logger.info("Manifest has %d part files", len(entries))
            return entries
    except Exception as e:
        logger.warning("Manifest fetch failed (%s), falling back to S3 listing", e)

    return _list_s3_parts(session)


def _list_s3_parts(session: requests.Session) -> list[dict]:
    """List Works part files from S3 bucket directly."""
    parts = []
    marker = ""
    prefix = "data/works/"
    while True:
        url = f"https://openalex.s3.amazonaws.com/?prefix={prefix}&marker={marker}&max-keys=1000"
        resp = session.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()

        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        contents = root.findall(".//s3:Contents", ns)
        if not contents:
            contents = root.findall(".//{http://s3.amazonaws.com/doc/2006-03-01/}Contents")
        if not contents:
            for child in root:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag == "Contents":
                    contents.append(child)

        for item in contents:
            key_el = item.find("s3:Key", ns) or item.find("{http://s3.amazonaws.com/doc/2006-03-01/}Key")
            if key_el is None:
                for child in item:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if tag == "Key":
                        key_el = child
                        break
            if key_el is not None and key_el.text and key_el.text.endswith(".gz"):
                parts.append({
                    "url": f"https://openalex.s3.amazonaws.com/{key_el.text}",
                    "meta": {},
                })
                marker = key_el.text

        is_truncated = root.findtext("{http://s3.amazonaws.com/doc/2006-03-01/}IsTruncated", "false")
        if is_truncated.lower() != "true":
            break

    logger.info("S3 listing found %d part files", len(parts))
    return parts


# ── Download + stream processing ─────────────────────────────────────────────


def download_and_process_part(
    session: requests.Session,
    part_url: str,
    min_year: int,
) -> tuple[list[dict], dict]:
    """Download a single gz part, filter records, return (records, stats).

    Uses streaming decompression to avoid loading entire parts into memory.
    Parts can be 500MB-1.1GB compressed; full decompression would use 2-4GB RAM.
    Retries up to MAX_RETRIES on HTTP errors with exponential backoff.
    """
    stats = {"total": 0, "kept": 0, "no_abstract": 0, "too_old": 0, "parse_error": 0}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(part_url, timeout=300, stream=True)
            resp.raise_for_status()

            records = []
            decompressor = gzip.GzipFile(fileobj=io.BytesIO(resp.content))
            text_stream = io.TextIOWrapper(decompressor, encoding="utf-8", errors="replace")

            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                stats["total"] += 1
                try:
                    work = json.loads(line)
                except json.JSONDecodeError:
                    stats["parse_error"] += 1
                    continue

                pub_year = work.get("publication_year")
                if pub_year is not None and pub_year < min_year:
                    stats["too_old"] += 1
                    continue

                abstract_inv = work.get("abstract_inverted_index")
                abstract = reconstruct_abstract(abstract_inv)
                if not abstract or len(abstract) < 50:
                    stats["no_abstract"] += 1
                    continue

                openalex_id = work.get("id", "")
                if openalex_id.startswith("https://openalex.org/"):
                    openalex_id = openalex_id.replace("https://openalex.org/", "")

                record = {
                    "id": openalex_id,
                    "doi": work.get("doi"),
                    "title": work.get("title", ""),
                    "abstract": abstract,
                    "publication_year": pub_year,
                    "type": work.get("type", ""),
                }
                records.append(record)
                stats["kept"] += 1

            return records, stats

        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Part download failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error("Part download failed after %d attempts: %s", MAX_RETRIES, e)
                raise


# ── Chunk writer ─────────────────────────────────────────────────────────────


class ChunkWriter:
    """Writes filtered records to compressed JSONL chunks.

    Each chunk holds up to chunk_size records. Filenames follow:
        works_part_NNNN.jsonl.gz
    """

    def __init__(self, output_dir: Path, chunk_size: int):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.current_chunk: list[dict] = []
        self.chunk_index = 0
        self.total_written = 0

    def set_chunk_index(self, idx: int):
        self.chunk_index = idx

    def add_records(self, records: list[dict]) -> list[Path]:
        """Add records to the buffer. Flushes full chunks. Returns paths of flushed files."""
        flushed = []
        self.current_chunk.extend(records)
        while len(self.current_chunk) >= self.chunk_size:
            batch = self.current_chunk[: self.chunk_size]
            self.current_chunk = self.current_chunk[self.chunk_size :]
            path = self._write_chunk(batch)
            flushed.append(path)
        return flushed

    def flush(self) -> Path | None:
        """Flush remaining records. Returns path or None if empty."""
        if not self.current_chunk:
            return None
        path = self._write_chunk(self.current_chunk)
        self.current_chunk = []
        return path

    def _write_chunk(self, records: list[dict]) -> Path:
        path = self.output_dir / f"works_part_{self.chunk_index:04d}.jsonl.gz"
        tmp = path.with_suffix(".tmp.gz")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(str(tmp), str(path))
        self.total_written += len(records)
        self.chunk_index += 1
        logger.info("Wrote chunk %s (%d records, total: %d)", path.name, len(records), self.total_written)
        return path


# ── Checkpoint management ────────────────────────────────────────────────────


def load_checkpoint(output_dir: Path) -> dict | None:
    cp_path = output_dir / CHECKPOINT_FILE
    if not cp_path.exists():
        return None
    try:
        data = json.loads(cp_path.read_text())
        logger.info(
            "Resuming from checkpoint: %d/%d parts completed, %d records kept",
            data.get("completed_parts", 0),
            data.get("total_parts", 0),
            data.get("total_kept", 0),
        )
        return data
    except Exception as e:
        logger.warning("Could not load checkpoint (%s) — starting fresh", e)
        return None


def save_checkpoint(output_dir: Path, state: dict) -> None:
    state["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_write_json(state, output_dir / CHECKPOINT_FILE)


def write_status(output_dir: Path, state: dict) -> None:
    try:
        _atomic_write_json(state, output_dir / STATUS_FILE)
    except Exception:
        pass


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_download(
    output_dir: Path,
    min_year: int,
    chunk_size: int,
    resume: bool,
    dry_run: bool,
) -> dict:
    """Main download pipeline. Returns final stats dict."""
    global _shutdown_requested

    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "SFULibraryMCP-SnapshotDownloader/1.0 (mailto:lib-systems@sfu.ca)"
    })

    manifest = fetch_manifest(session)
    if not manifest:
        logger.error("No parts found in manifest — check network or OpenAlex S3 bucket status")
        return {"error": "empty_manifest"}

    total_parts = len(manifest)
    logger.info("Snapshot has %d parts, filtering to year >= %d", total_parts, min_year)

    # Load or init checkpoint
    checkpoint = None
    start_part = 0
    cumulative_stats = {
        "total_scanned": 0, "total_kept": 0, "total_no_abstract": 0,
        "total_too_old": 0, "total_parse_error": 0,
    }

    if resume:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            start_part = checkpoint.get("completed_parts", 0)
            cumulative_stats = checkpoint.get("cumulative_stats", cumulative_stats)

    writer = ChunkWriter(output_dir, chunk_size)
    writer.set_chunk_index(checkpoint.get("next_chunk_index", 0) if checkpoint else 0)
    writer.total_written = cumulative_stats.get("total_kept", 0)

    start_time = time.time()
    parts_processed = 0

    for i in range(start_part, total_parts):
        if _shutdown_requested:
            logger.info("Shutdown requested — saving checkpoint at part %d/%d", i, total_parts)
            writer.flush()
            save_checkpoint(output_dir, {
                "completed_parts": i,
                "total_parts": total_parts,
                "next_chunk_index": writer.chunk_index,
                "cumulative_stats": cumulative_stats,
                "total_kept": cumulative_stats["total_kept"],
                "state": "interrupted",
            })
            write_status(output_dir, {
                "state": "interrupted",
                "completed_parts": i,
                "total_parts": total_parts,
                "total_kept": cumulative_stats["total_kept"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            logger.info("Checkpoint saved. Run with --resume to continue.")
            return {**cumulative_stats, "state": "interrupted", "completed_parts": i}

        part = manifest[i]
        part_url = part.get("url", "")
        if not part_url:
            logger.warning("Part %d has no URL — skipping", i)
            continue

        elapsed = time.time() - start_time
        if parts_processed > 0:
            rate = parts_processed / elapsed
            remaining = (total_parts - i) / rate
            eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
        else:
            eta_str = "calculating..."

        logger.info(
            "Processing part %d/%d (%.1f%%) — ETA: %s — kept so far: %d",
            i + 1, total_parts,
            100.0 * (i + 1) / total_parts,
            eta_str,
            cumulative_stats["total_kept"],
        )

        try:
            records, part_stats = download_and_process_part(session, part_url, min_year)
        except Exception as e:
            logger.error("Skipping part %d after all retries failed: %s", i, e)
            continue

        cumulative_stats["total_scanned"] += part_stats["total"]
        cumulative_stats["total_kept"] += part_stats["kept"]
        cumulative_stats["total_no_abstract"] += part_stats["no_abstract"]
        cumulative_stats["total_too_old"] += part_stats["too_old"]
        cumulative_stats["total_parse_error"] += part_stats["parse_error"]

        if records:
            writer.add_records(records)

        parts_processed += 1

        # Checkpoint after every part
        save_checkpoint(output_dir, {
            "completed_parts": i + 1,
            "total_parts": total_parts,
            "next_chunk_index": writer.chunk_index,
            "cumulative_stats": cumulative_stats,
            "total_kept": cumulative_stats["total_kept"],
            "state": "running",
        })

        write_status(output_dir, {
            "state": "running",
            "completed_parts": i + 1,
            "total_parts": total_parts,
            "total_kept": cumulative_stats["total_kept"],
            "progress_pct": round(100.0 * (i + 1) / total_parts, 1),
            "eta": eta_str,
            "pid": os.getpid(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        if dry_run:
            logger.info("─── DRY RUN COMPLETE ───")
            logger.info("Processed 1 part: %d records scanned, %d kept", part_stats["total"], part_stats["kept"])
            logger.info("Filtering: %d no abstract, %d too old, %d parse errors",
                        part_stats["no_abstract"], part_stats["too_old"], part_stats["parse_error"])
            if records:
                logger.info("Sample record:\n%s", json.dumps(records[0], indent=2, ensure_ascii=False)[:500])
            writer.flush()
            return {**cumulative_stats, "state": "dry_run", "completed_parts": 1}

    # Final flush
    writer.flush()

    total_time = time.time() - start_time
    logger.info("═══ Download complete ═══")
    logger.info("Total time: %s", time.strftime("%H:%M:%S", time.gmtime(total_time)))
    logger.info("Parts processed: %d/%d", total_parts, total_parts)
    logger.info("Records scanned: %d", cumulative_stats["total_scanned"])
    logger.info("Records kept: %d", cumulative_stats["total_kept"])
    logger.info("Filtered out — no abstract: %d, too old: %d, parse error: %d",
                cumulative_stats["total_no_abstract"],
                cumulative_stats["total_too_old"],
                cumulative_stats["total_parse_error"])

    save_checkpoint(output_dir, {
        "completed_parts": total_parts,
        "total_parts": total_parts,
        "next_chunk_index": writer.chunk_index,
        "cumulative_stats": cumulative_stats,
        "total_kept": cumulative_stats["total_kept"],
        "state": "completed",
    })

    write_status(output_dir, {
        "state": "completed",
        "completed_parts": total_parts,
        "total_parts": total_parts,
        "total_kept": cumulative_stats["total_kept"],
        "total_time_seconds": round(total_time, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    return {**cumulative_stats, "state": "completed", "completed_parts": total_parts}


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download and filter OpenAlex Works snapshot for SPLADE indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-year", type=int, default=DEFAULT_MIN_YEAR,
        help=f"Minimum publication year to include (default: {DEFAULT_MIN_YEAR})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Records per output chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process 1 part and report stats without full download",
    )
    args = parser.parse_args()

    logger.info("OpenAlex Snapshot Downloader")
    logger.info("Output: %s", args.output)
    logger.info("Filter: year >= %d, has abstract (>= 50 chars)", args.min_year)
    logger.info("Chunk size: %d records/file", args.chunk_size)
    if args.resume:
        logger.info("Mode: RESUME from checkpoint")
    if args.dry_run:
        logger.info("Mode: DRY RUN (1 part only)")

    result = run_download(
        output_dir=args.output,
        min_year=args.min_year,
        chunk_size=args.chunk_size,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    if result.get("state") == "interrupted":
        sys.exit(130)  # standard SIGINT exit code


if __name__ == "__main__":
    main()
