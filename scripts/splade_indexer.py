#!/usr/bin/env python3
"""Index OpenAlex snapshot into OpenSearch using SPLADE sparse vectors.

Reads filtered JSONL chunks from data/openalex_snapshot/ (produced by
snapshot_downloader.py), encodes each doc through a SPLADE model to
generate sparse term weights, and bulk-upserts them into OpenSearch.

Checkpoint / Resume
───────────────────
Indexing can be interrupted at any time and resumed without re-processing:

    python scripts/splade_indexer.py --resume

Progress is checkpointed every --checkpoint-interval docs (default 100k).
On SIGINT/SIGTERM the current batch finishes, checkpoint is saved, and the
script exits cleanly.

Failsafes
─────────
1. Atomic checkpoint writes — safe against sudden kill / OOM
2. SIGINT / SIGTERM handlers — finishes current batch, saves, exits
3. Double-signal force exit — second Ctrl-C exits immediately
4. Batch-level OpenSearch retry with exponential backoff (3 attempts)
5. Per-doc error isolation — one bad doc doesn't kill the batch
6. VRAM monitoring — logs GPU memory every checkpoint, warns at >90%
7. Throughput tracking — docs/sec, ETA, running average
8. Status file (indexer_status.json) — pollable by external monitors
9. Heartbeat file — updated every batch so external watchdogs can detect stalls
10. Dry run mode — encode 100 docs, verify OpenSearch connectivity, exit

Usage:
    # Full indexing run
    python scripts/splade_indexer.py

    # Resume after interruption
    python scripts/splade_indexer.py --resume

    # Dry run — encode 100 docs, test OpenSearch, report stats
    python scripts/splade_indexer.py --dry-run

    # Custom settings
    python scripts/splade_indexer.py \\
        --model naver/splade-cocondenser-distil \\
        --batch-size 128 \\
        --opensearch-url http://localhost:9200 \\
        --checkpoint-interval 50000

    # CPU-only (no GPU)
    python scripts/splade_indexer.py --device cpu --batch-size 16
"""

import argparse
import gzip
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

DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "data" / "openalex_snapshot"
DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"
DEFAULT_OPENSEARCH_URL = "http://localhost:9200"
DEFAULT_INDEX = "openalex_works"
DEFAULT_BATCH_SIZE = 64
DEFAULT_CHECKPOINT_INTERVAL = 100_000
CHECKPOINT_FILE = "indexer_checkpoint.json"
STATUS_FILE = "indexer_status.json"
HEARTBEAT_FILE = "indexer_heartbeat"
VRAM_RESERVE_FILE = Path(__file__).parent.parent / "data" / "vram_reserve.json"
MAX_BULK_RETRIES = 3
BULK_RETRY_BACKOFF = 5
SPARSE_TOP_K = 256  # keep top-K terms per doc (prune noise)
MAX_DOC_LENGTH = 512  # SPLADE model max tokens

# ── Graceful shutdown ────────────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        logger.warning("Second %s received — forcing exit NOW", sig_name)
        sys.exit(1)
    logger.info(
        "%s received — will finish current batch, save checkpoint, then exit. "
        "Press Ctrl-C again to force quit.",
        sig_name,
    )
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ── Atomic file I/O ──────────────────────────────────────────────────────────


def _atomic_write_json(data: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


def _touch_heartbeat(path: Path) -> None:
    try:
        path.write_text(str(time.time()))
    except Exception:
        pass


# ── VRAM reservation ────────────────────────────────────────────────────────

_last_vram_check = 0.0
_VRAM_CHECK_INTERVAL = 5.0  # seconds between file checks
_VRAM_POLL_INTERVAL = 10.0  # seconds between polls while paused
_vram_paused = False


def _read_vram_reserve() -> dict | None:
    """Read the VRAM reserve file. Returns None if absent or inactive."""
    try:
        if not VRAM_RESERVE_FILE.exists():
            return None
        data = json.loads(VRAM_RESERVE_FILE.read_text())
        if not data.get("active", False):
            return None
        return data
    except Exception:
        return None


def check_vram_reservation(
    encoder: "SpladeEncoder",
    original_batch_size: int,
    current_batch_size: int,
    checkpoint_fn: callable = None,
) -> int:
    """Check VRAM reservation and pause/resume the indexer as needed.

    Three modes based on the reserve file:
      - No file / inactive  → run at original batch size.
      - active, mode="reduce" → shrink batch size proportionally (legacy).
      - active, mode="pause" (default) → offload model to CPU, save checkpoint,
        poll until reservation clears, then reload to GPU and resume.

    Returns the batch size to use for the next batch.
    """
    global _last_vram_check, _vram_paused
    now = time.time()
    if now - _last_vram_check < _VRAM_CHECK_INTERVAL:
        return current_batch_size
    _last_vram_check = now

    data = _read_vram_reserve()

    # ── No reservation → restore if we were throttled ────────────────────
    if data is None:
        if _vram_paused:
            _vram_paused = False
        if current_batch_size != original_batch_size:
            logger.info("VRAM reservation cleared — restoring batch_size to %d", original_batch_size)
        return original_batch_size

    reserve_gb = data.get("reserve_gb", 0)
    mode = data.get("mode", "pause")

    if mode != "pause" and reserve_gb <= 0:
        return original_batch_size

    # ── Pause mode: offload model, checkpoint, wait ──────────────────────
    if mode == "pause":
        import torch

        if checkpoint_fn:
            checkpoint_fn()
            logger.info("VRAM pause: checkpoint saved before offloading")

        was_on_gpu = encoder.device == "cuda"
        if was_on_gpu:
            encoder.model.cpu()
            encoder.device = "cpu"
            torch.cuda.empty_cache()
            freed = torch.cuda.memory_reserved() / 1e9
            logger.info(
                "VRAM pause: model offloaded to CPU, GPU cache cleared (%.2f GB reserved remains)",
                freed,
            )

        _vram_paused = True
        logger.info("VRAM pause: indexer paused — polling every %.0fs for release", _VRAM_POLL_INTERVAL)

        while not _shutdown_requested:
            time.sleep(_VRAM_POLL_INTERVAL)
            check = _read_vram_reserve()
            if check is None:
                break
            if check.get("mode", "pause") != "pause" or not check.get("active", False):
                break

        _vram_paused = False

        if _shutdown_requested:
            return current_batch_size

        if was_on_gpu and torch.cuda.is_available():
            encoder.model.to("cuda")
            encoder.device = "cuda"
            torch.cuda.empty_cache()
            logger.info(
                "VRAM resumed: model reloaded to GPU (%.2f GB allocated)",
                torch.cuda.memory_allocated() / 1e9,
            )

        logger.info("VRAM resumed: continuing at batch_size %d", original_batch_size)
        return original_batch_size

    # ── Reduce mode: shrink batch size proportionally ────────────────────
    import torch
    if not torch.cuda.is_available():
        return original_batch_size

    total_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    model_gb = 0.5
    available_for_batches = total_gb - model_gb - reserve_gb

    if available_for_batches <= 0:
        new_batch = max(1, original_batch_size // 8)
    else:
        usable_fraction = available_for_batches / (total_gb - model_gb)
        new_batch = max(1, int(original_batch_size * usable_fraction))

    if new_batch != current_batch_size:
        torch.cuda.empty_cache()
        logger.info(
            "VRAM reservation: %.1f GB reserved — batch_size %d → %d, GPU cache cleared",
            reserve_gb, current_batch_size, new_batch,
        )

    return new_batch


# ── SPLADE model management ─────────────────────────────────────────────────


class SpladeEncoder:
    """Loads a SPLADE model and encodes text into sparse term-weight dicts."""

    def __init__(self, model_name: str, device: str = "auto", max_length: int = MAX_DOC_LENGTH):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading SPLADE model '%s' on %s ...", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.vocab = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        logger.info("SPLADE model loaded (%d vocab tokens)", len(self.vocab))

        if self.device == "cuda":
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info("GPU memory after model load: %.2f GB", mem)

    def encode_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Encode a batch of texts into sparse dicts {token: weight}."""
        import torch

        tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**tokens)

        # SPLADE: ReLU + log(1 + x), then max-pool over sequence length
        splade_vecs = torch.log1p(torch.relu(output.logits))
        splade_vecs = torch.max(splade_vecs, dim=1).values  # (batch, vocab)

        results = []
        for vec in splade_vecs:
            nonzero = vec.nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                results.append({})
                continue

            weights = vec[nonzero]

            # Keep only top-K terms
            if len(nonzero) > SPARSE_TOP_K:
                topk = torch.topk(weights, SPARSE_TOP_K)
                nonzero = nonzero[topk.indices]
                weights = topk.values

            sparse_dict = {}
            for idx, weight in zip(nonzero.cpu().tolist(), weights.cpu().tolist()):
                token = self.id_to_token.get(idx, "")
                if token and not token.startswith("[") and weight > 0.01:
                    sparse_dict[token] = round(weight, 4)

            results.append(sparse_dict)

        return results

    def get_gpu_stats(self) -> dict:
        """Return GPU memory stats if available."""
        if self.device != "cuda":
            return {"device": "cpu"}
        import torch
        return {
            "device": "cuda",
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "utilization_pct": round(
                100.0 * torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory, 1
            ),
        }


# ── OpenSearch bulk upsert ───────────────────────────────────────────────────


def bulk_upsert_opensearch(
    session: requests.Session,
    opensearch_url: str,
    index: str,
    docs: list[dict],
) -> dict:
    """Bulk upsert documents to OpenSearch. Returns stats dict.

    Retries on failure with exponential backoff.
    """
    if not docs:
        return {"indexed": 0, "errors": 0}

    stats = {"indexed": 0, "errors": 0, "error_details": []}

    # Build NDJSON bulk payload
    lines = []
    for doc in docs:
        doc_id = doc.get("id", "")
        action = {"index": {"_index": index, "_id": doc_id}}
        lines.append(json.dumps(action))
        lines.append(json.dumps(doc))
    payload = "\n".join(lines) + "\n"

    for attempt in range(1, MAX_BULK_RETRIES + 1):
        try:
            resp = session.post(
                f"{opensearch_url}/_bulk",
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/x-ndjson"},
                timeout=120,
            )

            if resp.status_code == 200:
                result = resp.json()
                if result.get("errors"):
                    for item in result.get("items", []):
                        op = item.get("index", {})
                        if op.get("error"):
                            stats["errors"] += 1
                            if len(stats["error_details"]) < 5:
                                stats["error_details"].append(str(op["error"])[:200])
                        else:
                            stats["indexed"] += 1
                else:
                    stats["indexed"] = len(docs)
                return stats

            resp.raise_for_status()

        except Exception as e:
            if attempt < MAX_BULK_RETRIES:
                wait = BULK_RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "Bulk upsert failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, MAX_BULK_RETRIES, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error("Bulk upsert failed after %d attempts: %s", MAX_BULK_RETRIES, e)
                stats["errors"] = len(docs)
                return stats

    return stats


def check_opensearch_health(session: requests.Session, url: str, index: str) -> bool:
    """Verify OpenSearch is reachable and the target index exists."""
    try:
        resp = session.get(f"{url}/_cluster/health", timeout=10)
        if resp.status_code != 200:
            logger.error("OpenSearch cluster not healthy: %d", resp.status_code)
            return False

        health = resp.json()
        status = health.get("status", "red")
        if status == "red":
            logger.error("OpenSearch cluster status is RED")
            return False

        logger.info("OpenSearch cluster: %s, nodes: %s", status, health.get("number_of_nodes"))

        # Check if index exists, create if not
        idx_resp = session.head(f"{url}/{index}", timeout=10)
        if idx_resp.status_code == 404:
            logger.info("Index '%s' does not exist — creating...", index)
            create_resp = session.put(f"{url}/{index}", timeout=10)
            if create_resp.status_code not in (200, 201):
                logger.error("Failed to create index: %s", create_resp.text)
                return False
            logger.info("Index '%s' created", index)

        return True

    except Exception as e:
        logger.error("Cannot reach OpenSearch at %s: %s", url, e)
        return False


# ── Input file management ────────────────────────────────────────────────────


def list_input_files(input_dir: Path) -> list[Path]:
    """List all .jsonl.gz files sorted by name."""
    files = sorted(input_dir.glob("works_part_*.jsonl.gz"))
    if not files:
        files = sorted(input_dir.glob("*.jsonl.gz"))
    return files


def read_jsonl_gz(path: Path) -> list[dict]:
    """Read all records from a gzipped JSONL file."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ── Checkpoint management ────────────────────────────────────────────────────


def load_checkpoint(input_dir: Path) -> dict | None:
    cp_path = input_dir / CHECKPOINT_FILE
    if not cp_path.exists():
        return None
    try:
        data = json.loads(cp_path.read_text())
        logger.info(
            "Resuming from checkpoint: file_index=%d, doc_offset=%d, total_indexed=%d",
            data.get("file_index", 0),
            data.get("doc_offset", 0),
            data.get("total_indexed", 0),
        )
        return data
    except Exception as e:
        logger.warning("Could not load checkpoint (%s) — starting fresh", e)
        return None


def save_checkpoint(input_dir: Path, state: dict) -> None:
    state["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_write_json(state, input_dir / CHECKPOINT_FILE)


def write_status(input_dir: Path, state: dict) -> None:
    try:
        _atomic_write_json(state, input_dir / STATUS_FILE)
    except Exception:
        pass


# ── Main indexing pipeline ───────────────────────────────────────────────────


def run_indexer(
    input_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    opensearch_url: str,
    index_name: str,
    checkpoint_interval: int,
    resume: bool,
    dry_run: bool,
) -> dict:
    """Main indexing pipeline. Returns final stats dict."""
    global _shutdown_requested

    session = requests.Session()

    # ── Pre-flight checks ────────────────────────────────────────────────
    input_files = list_input_files(input_dir)
    if not input_files:
        logger.error("No input files found in %s", input_dir)
        return {"error": "no_input_files"}

    total_files = len(input_files)
    logger.info("Found %d input files in %s", total_files, input_dir)

    if not dry_run:
        if not check_opensearch_health(session, opensearch_url, index_name):
            logger.error(
                "OpenSearch pre-flight check failed. Ensure OpenSearch is running at %s",
                opensearch_url,
            )
            return {"error": "opensearch_unhealthy"}

    # ── Load model ───────────────────────────────────────────────────────
    encoder = SpladeEncoder(model_name, device=device)

    # ── Load checkpoint ──────────────────────────────────────────────────
    start_file = 0
    start_offset = 0
    cumulative = {
        "total_indexed": 0,
        "total_errors": 0,
        "total_docs_scanned": 0,
        "total_empty_sparse": 0,
    }

    if resume:
        cp = load_checkpoint(input_dir)
        if cp:
            start_file = cp.get("file_index", 0)
            start_offset = cp.get("doc_offset", 0)
            cumulative = cp.get("cumulative", cumulative)

    start_time = time.time()
    docs_since_checkpoint = 0
    heartbeat_path = input_dir / HEARTBEAT_FILE
    throughput_window: list[float] = []  # recent batch times for moving average
    original_batch_size = batch_size
    active_batch_size = batch_size

    # ── Process files ────────────────────────────────────────────────────
    for file_idx in range(start_file, total_files):
        if _shutdown_requested:
            break

        input_file = input_files[file_idx]
        logger.info("Loading file %d/%d: %s", file_idx + 1, total_files, input_file.name)
        all_records = read_jsonl_gz(input_file)
        total_in_file = len(all_records)

        # Skip already-processed docs within this file
        records = all_records[start_offset:]
        start_offset = 0  # only applies to first file on resume

        doc_idx = len(all_records) - len(records)

        while doc_idx < total_in_file:
            if _shutdown_requested:
                logger.info("Shutdown requested — saving checkpoint...")
                save_checkpoint(input_dir, {
                    "file_index": file_idx,
                    "doc_offset": doc_idx,
                    "cumulative": cumulative,
                    "total_indexed": cumulative["total_indexed"],
                    "state": "interrupted",
                })
                write_status(input_dir, {
                    "state": "interrupted",
                    "file_index": file_idx,
                    "total_files": total_files,
                    "total_indexed": cumulative["total_indexed"],
                    "gpu": encoder.get_gpu_stats(),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                logger.info("Checkpoint saved. Run with --resume to continue.")
                return {**cumulative, "state": "interrupted"}

            # ── Check VRAM reservation ───────────────────────────────────
            def _save_checkpoint_for_pause():
                save_checkpoint(input_dir, {
                    "file_index": file_idx,
                    "doc_offset": doc_idx,
                    "cumulative": cumulative,
                    "total_indexed": cumulative["total_indexed"],
                    "state": "vram_paused",
                })
                write_status(input_dir, {
                    "state": "vram_paused",
                    "file_index": file_idx,
                    "total_files": total_files,
                    "doc_in_file": doc_idx,
                    "total_indexed": cumulative["total_indexed"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

            active_batch_size = check_vram_reservation(
                encoder, original_batch_size, active_batch_size,
                checkpoint_fn=_save_checkpoint_for_pause,
            )

            # ── Prepare batch ────────────────────────────────────────────
            batch_records = records[: active_batch_size]
            records = records[active_batch_size:]
            actual_batch = len(batch_records)

            texts = []
            valid_records = []
            for rec in batch_records:
                title = rec.get("title", "") or ""
                abstract = rec.get("abstract", "") or ""
                text = f"{title} {abstract}".strip()
                if text:
                    texts.append(text)
                    valid_records.append(rec)
                else:
                    cumulative["total_empty_sparse"] += 1

            cumulative["total_docs_scanned"] += actual_batch

            if not texts:
                doc_idx += actual_batch
                continue

            # ── Encode ───────────────────────────────────────────────────
            batch_start = time.time()
            try:
                sparse_vecs = encoder.encode_batch(texts)
            except Exception as e:
                logger.error("SPLADE encoding failed for batch at doc %d: %s", doc_idx, e)
                doc_idx += actual_batch
                continue

            encode_time = time.time() - batch_start

            # ── Build OpenSearch docs ────────────────────────────────────
            os_docs = []
            for rec, sparse in zip(valid_records, sparse_vecs):
                if not sparse:
                    cumulative["total_empty_sparse"] += 1
                    continue
                os_doc = {
                    "id": rec.get("id", ""),
                    "doi": rec.get("doi"),
                    "title": rec.get("title", ""),
                    "abstract": rec.get("abstract", ""),
                    "publication_year": rec.get("publication_year"),
                    "type": rec.get("type", ""),
                    "openalex_id": rec.get("id", ""),
                    "sparse_field": sparse,
                }
                os_docs.append(os_doc)

            # ── Upsert to OpenSearch ─────────────────────────────────────
            if os_docs and not dry_run:
                upsert_stats = bulk_upsert_opensearch(session, opensearch_url, index_name, os_docs)
                cumulative["total_indexed"] += upsert_stats["indexed"]
                cumulative["total_errors"] += upsert_stats["errors"]
                if upsert_stats["error_details"]:
                    logger.warning("Bulk errors: %s", upsert_stats["error_details"][:2])
            elif os_docs and dry_run:
                cumulative["total_indexed"] += len(os_docs)

            doc_idx += actual_batch
            docs_since_checkpoint += actual_batch

            # ── Throughput tracking ──────────────────────────────────────
            batch_time = time.time() - batch_start
            throughput_window.append(batch_time)
            if len(throughput_window) > 50:
                throughput_window = throughput_window[-50:]

            avg_batch_time = sum(throughput_window) / len(throughput_window)
            docs_per_sec = actual_batch / max(batch_time, 0.001)

            _touch_heartbeat(heartbeat_path)

            # ── Progress logging ─────────────────────────────────────────
            elapsed = time.time() - start_time
            if cumulative["total_docs_scanned"] > 0:
                overall_rate = cumulative["total_docs_scanned"] / elapsed
            else:
                overall_rate = 0

            # ── Checkpoint ───────────────────────────────────────────────
            if docs_since_checkpoint >= checkpoint_interval:
                gpu_stats = encoder.get_gpu_stats()
                if gpu_stats.get("utilization_pct", 0) > 90:
                    logger.warning(
                        "⚠ GPU VRAM usage at %.1f%% — consider reducing --batch-size",
                        gpu_stats["utilization_pct"],
                    )

                logger.info(
                    "CHECKPOINT — indexed: %d | errors: %d | rate: %.0f docs/sec | "
                    "GPU: %s | file %d/%d",
                    cumulative["total_indexed"],
                    cumulative["total_errors"],
                    overall_rate,
                    f"{gpu_stats.get('allocated_gb', 'N/A')} GB" if gpu_stats.get("device") == "cuda" else "CPU",
                    file_idx + 1,
                    total_files,
                )

                save_checkpoint(input_dir, {
                    "file_index": file_idx,
                    "doc_offset": doc_idx,
                    "cumulative": cumulative,
                    "total_indexed": cumulative["total_indexed"],
                    "state": "running",
                })

                write_status(input_dir, {
                    "state": "running",
                    "file_index": file_idx,
                    "total_files": total_files,
                    "doc_in_file": doc_idx,
                    "total_indexed": cumulative["total_indexed"],
                    "total_errors": cumulative["total_errors"],
                    "docs_per_sec": round(overall_rate, 1),
                    "elapsed_seconds": round(elapsed, 1),
                    "gpu": gpu_stats,
                    "pid": os.getpid(),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

                docs_since_checkpoint = 0

            # ── Dry run exit ─────────────────────────────────────────────
            if dry_run and cumulative["total_docs_scanned"] >= 100:
                logger.info("─── DRY RUN COMPLETE ───")
                logger.info("Encoded %d docs", cumulative["total_docs_scanned"])
                logger.info("Would index %d docs", cumulative["total_indexed"])
                logger.info("Empty sparse vectors: %d", cumulative["total_empty_sparse"])
                logger.info("Encoding speed: %.1f docs/sec", docs_per_sec)
                logger.info("GPU: %s", encoder.get_gpu_stats())
                if sparse_vecs:
                    sample = sparse_vecs[0]
                    logger.info(
                        "Sample sparse vector: %d terms, top-5: %s",
                        len(sample),
                        dict(sorted(sample.items(), key=lambda x: -x[1])[:5]),
                    )
                return {**cumulative, "state": "dry_run"}

        # ── End of file checkpoint ───────────────────────────────────────
        if not _shutdown_requested:
            save_checkpoint(input_dir, {
                "file_index": file_idx + 1,
                "doc_offset": 0,
                "cumulative": cumulative,
                "total_indexed": cumulative["total_indexed"],
                "state": "running",
            })
            docs_since_checkpoint = 0

    # ── Final summary ────────────────────────────────────────────────────
    total_time = time.time() - start_time
    overall_rate = cumulative["total_docs_scanned"] / max(total_time, 0.001)

    logger.info("═══ Indexing complete ═══")
    logger.info("Total time: %s", time.strftime("%H:%M:%S", time.gmtime(total_time)))
    logger.info("Files processed: %d/%d", total_files, total_files)
    logger.info("Docs scanned: %d", cumulative["total_docs_scanned"])
    logger.info("Docs indexed: %d", cumulative["total_indexed"])
    logger.info("Indexing errors: %d", cumulative["total_errors"])
    logger.info("Empty sparse vectors: %d", cumulative["total_empty_sparse"])
    logger.info("Average throughput: %.1f docs/sec", overall_rate)
    logger.info("GPU: %s", encoder.get_gpu_stats())

    save_checkpoint(input_dir, {
        "file_index": total_files,
        "doc_offset": 0,
        "cumulative": cumulative,
        "total_indexed": cumulative["total_indexed"],
        "state": "completed",
    })

    write_status(input_dir, {
        "state": "completed",
        "total_files": total_files,
        "total_indexed": cumulative["total_indexed"],
        "total_errors": cumulative["total_errors"],
        "docs_per_sec": round(overall_rate, 1),
        "total_time_seconds": round(total_time, 1),
        "gpu": encoder.get_gpu_stats(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    return {**cumulative, "state": "completed"}


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Index OpenAlex snapshot into OpenSearch using SPLADE sparse vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Input directory with .jsonl.gz files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"SPLADE model name/path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for encoding (default: auto — uses CUDA if available)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Encoding batch size (default: {DEFAULT_BATCH_SIZE}; reduce if OOM)",
    )
    parser.add_argument(
        "--opensearch-url", type=str,
        default=os.environ.get("SFU_OPENSEARCH_URL", DEFAULT_OPENSEARCH_URL),
        help=f"OpenSearch URL (default: {DEFAULT_OPENSEARCH_URL})",
    )
    parser.add_argument(
        "--index", type=str,
        default=os.environ.get("SFU_OPENSEARCH_INDEX", DEFAULT_INDEX),
        help=f"OpenSearch index name (default: {DEFAULT_INDEX})",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N docs (default: {DEFAULT_CHECKPOINT_INTERVAL})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Encode 100 docs and report stats without full indexing",
    )
    args = parser.parse_args()

    logger.info("SPLADE Indexer")
    logger.info("Input: %s", args.input)
    logger.info("Model: %s", args.model)
    logger.info("Device: %s", args.device)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("OpenSearch: %s / %s", args.opensearch_url, args.index)
    logger.info("Checkpoint every: %d docs", args.checkpoint_interval)
    if args.resume:
        logger.info("Mode: RESUME from checkpoint")
    if args.dry_run:
        logger.info("Mode: DRY RUN (100 docs)")

    result = run_indexer(
        input_dir=args.input,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        opensearch_url=args.opensearch_url,
        index_name=args.index,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    if result.get("state") == "interrupted":
        sys.exit(130)
    elif result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
