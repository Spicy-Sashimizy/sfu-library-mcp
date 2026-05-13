#!/usr/bin/env python3
"""Watch the SPLADE indexer, run benchmark when done, then start quality training.

Workflow
────────
1. Poll indexer_status.json + heartbeat every POLL_INTERVAL seconds.
2. When the indexer finishes (state != "running" or PID is gone), run benchmark_splade.py.
3. Parse benchmark results and validate against thresholds.
4. If OK, launch train_embedding_model.py with high-quality hyperparameters.

Run in the background:
    nohup python scripts/post_index_watchdog.py > logs/watchdog.log 2>&1 &
    echo $! > logs/watchdog.pid

Or foreground (you'll see live output):
    python scripts/post_index_watchdog.py
"""

import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python3"
STATUS_FILE = ROOT / "data" / "openalex_snapshot" / "indexer_status.json"
HEARTBEAT_FILE = ROOT / "data" / "openalex_snapshot" / "indexer_heartbeat"
PID_FILE = ROOT / "data" / "openalex_snapshot" / "indexer.pid"
BENCHMARK_SCRIPT = ROOT / "scripts" / "benchmark_splade.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_embedding_model.py"
EVAL_QUERIES = ROOT / "data" / "sfu_eval_queries.json"
BENCHMARK_OUT = ROOT / "data" / "eval_results" / "post_index_benchmark.json"
LOG_DIR = ROOT / "logs"
TRAIN_LOG = LOG_DIR / "quality_training.log"
TRAIN_PID_FILE = LOG_DIR / "quality_training.pid"
WATCHDOG_STATUS = ROOT / "data" / "watchdog_status.json"

# ── Tuning ─────────────────────────────────────────────────────────────────────
POLL_INTERVAL = 60          # seconds between indexer polls
HEARTBEAT_STALE_S = 300     # consider dead if heartbeat is > 5 min old

# ── Benchmark thresholds (what "as expected" means) ───────────────────────────
# SPLADE must return at least this many results per query on average
MIN_AVG_SPLADE_HITS = 5.0
# BM25F must also be working
MIN_AVG_BM25_HITS = 5.0
# SPLADE encoding must not be pathologically slow
MAX_AVG_ENCODE_MS = 600.0
# SPLADE must contribute meaningfully to RRF (at least this overlap with RRF)
MIN_SPLADE_RRF_OVERLAP = 0.20

# ── Quality training parameters (different from run_training.sh defaults) ──────
# Defaults in run_training.sh:
#   base_model = sentence-transformers/all-MiniLM-L6-v2
#   epochs     = 6,  batch = 32, lr = 2e-5, max-seq = 512,
#   weight-decay = 0.01, warmup-ratio = 0.1
#
# Quality run uses a larger base model, lower LR, more epochs,
# more warmup, stronger regularisation, and more frequent eval.
QUALITY_TRAIN_ARGS = {
    "--base-model":            "BAAI/bge-base-en-v1.5",
    "--output":                str(ROOT / "models" / "sfu-academic-embed-v5-quality"),
    "--epochs":                "10",
    "--batch-size":            "16",
    "--gradient-accumulation": "4",       # effective batch = 64
    "--learning-rate":         "1e-5",    # conservative vs default 2e-5
    "--warmup-ratio":          "0.15",    # more warmup for large model
    "--weight-decay":          "0.05",    # stronger regularisation
    "--max-seq-length":        "512",
    "--save-steps":            "50",
    "--eval-steps":            "150",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_status(state: str, **extra):
    WATCHDOG_STATUS.parent.mkdir(parents=True, exist_ok=True)
    data = {"state": state, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    tmp = WATCHDOG_STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(WATCHDOG_STATUS))


def _indexer_alive() -> bool:
    """Return True if the indexer process is still running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)   # signal 0 = probe only
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def _heartbeat_fresh() -> bool:
    """Return True if heartbeat file was updated within HEARTBEAT_STALE_S seconds."""
    if not HEARTBEAT_FILE.exists():
        return False
    try:
        mtime = HEARTBEAT_FILE.stat().st_mtime
        return (time.time() - mtime) < HEARTBEAT_STALE_S
    except OSError:
        return False


def _read_status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text())
    except Exception:
        return {}


def _indexer_done() -> tuple[bool, str]:
    """Return (done, reason). done=True when indexing has finished."""
    status = _read_status()
    state = status.get("state", "")

    if state in ("complete", "done", "finished"):
        return True, f"state={state}"

    # If state is running but PID is gone and heartbeat is stale, assume crashed/done
    if state == "running":
        if not _indexer_alive() and not _heartbeat_fresh():
            return True, "pid-gone-heartbeat-stale"
        return False, "running"

    # No status file yet or unknown state
    if not STATUS_FILE.exists():
        return False, "no-status-file"

    return False, f"state={state}"


# ── Phase 1: Monitor ───────────────────────────────────────────────────────────

def wait_for_indexer():
    log.info("Watchdog started — polling indexer every %ds", POLL_INTERVAL)
    _write_status("monitoring")
    iterations = 0
    while True:
        done, reason = _indexer_done()
        status = _read_status()
        pct = status.get("total_indexed", 0)
        total_files = status.get("total_files", "?")
        file_idx = status.get("file_index", "?")
        dps = status.get("docs_per_sec", 0)

        if done:
            log.info("Indexer finished — reason: %s", reason)
            log.info("  Final: %d docs indexed, %s/%s files", pct, file_idx, total_files)
            _write_status("indexer_done", reason=reason, total_indexed=pct)
            return status
        else:
            log.info(
                "Indexer running — file %s/%s | %d docs indexed | %.0f docs/s",
                file_idx, total_files, pct, dps,
            )

        iterations += 1
        time.sleep(POLL_INTERVAL)


# ── Phase 2: Benchmark ─────────────────────────────────────────────────────────

def run_benchmark() -> dict:
    log.info("Running SPLADE benchmark …")
    _write_status("benchmarking")
    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PYTHON),
        str(BENCHMARK_SCRIPT),
        "--queries", str(EVAL_QUERIES),
        "--output", str(BENCHMARK_OUT),
        "--top-k", "10",
        "--device", "auto",
    ]
    log.info("CMD: %s", shlex.join(cmd))

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        log.error("Benchmark script exited with code %d", result.returncode)
        _write_status("benchmark_failed", returncode=result.returncode)
        return {}

    if not BENCHMARK_OUT.exists():
        log.error("Benchmark output file not found: %s", BENCHMARK_OUT)
        _write_status("benchmark_failed", reason="output-missing")
        return {}

    try:
        data = json.loads(BENCHMARK_OUT.read_text())
    except Exception as e:
        log.error("Could not parse benchmark output: %s", e)
        _write_status("benchmark_failed", reason=str(e))
        return {}

    log.info("Benchmark complete — saved to %s", BENCHMARK_OUT)
    return data


# ── Phase 3: Validate ──────────────────────────────────────────────────────────

def validate_benchmark(data: dict) -> tuple[bool, list[str]]:
    """Check benchmark results against expected thresholds."""
    if not data:
        return False, ["No benchmark data"]

    failures = []

    # Extract subject-level stats to compute avg hits
    subject_stats = data.get("subject_stats", {})
    if subject_stats:
        avg_splade = sum(s["avg_splade_hits"] for s in subject_stats.values()) / len(subject_stats)
        avg_bm25 = sum(s["avg_bm25_hits"] for s in subject_stats.values()) / len(subject_stats)
        avg_sr_overlap = sum(s["avg_overlap_sr"] for s in subject_stats.values()) / len(subject_stats)
    else:
        # Fall back to per_query if available
        per_query = data.get("per_query", [])
        if per_query:
            avg_splade = sum(q.get("splade_hits", 0) for q in per_query) / len(per_query)
            avg_bm25 = sum(q.get("bm25_hits", 0) for q in per_query) / len(per_query)
            avg_sr_overlap = data.get("avg_overlap_splade_rrf", 0.0)
        else:
            return False, ["No subject_stats or per_query in benchmark output"]

    avg_encode_ms = data.get("avg_encode_ms", 0.0)

    log.info("Benchmark validation:")
    log.info("  avg SPLADE hits@10 = %.1f  (need >= %.1f)", avg_splade, MIN_AVG_SPLADE_HITS)
    log.info("  avg BM25F  hits@10 = %.1f  (need >= %.1f)", avg_bm25, MIN_AVG_BM25_HITS)
    log.info("  avg encode time    = %.1f ms  (need <= %.0f ms)", avg_encode_ms, MAX_AVG_ENCODE_MS)
    log.info("  SPLADE↔RRF overlap = %.1%%  (need >= %.0f%%)", avg_sr_overlap * 100, MIN_SPLADE_RRF_OVERLAP * 100)

    if avg_splade < MIN_AVG_SPLADE_HITS:
        failures.append(f"SPLADE avg hits {avg_splade:.1f} < {MIN_AVG_SPLADE_HITS}")
    if avg_bm25 < MIN_AVG_BM25_HITS:
        failures.append(f"BM25F avg hits {avg_bm25:.1f} < {MIN_AVG_BM25_HITS}")
    if avg_encode_ms > MAX_AVG_ENCODE_MS:
        failures.append(f"avg encode {avg_encode_ms:.1f} ms > {MAX_AVG_ENCODE_MS} ms")
    if avg_sr_overlap < MIN_SPLADE_RRF_OVERLAP:
        failures.append(f"SPLADE↔RRF overlap {avg_sr_overlap:.1%} < {MIN_SPLADE_RRF_OVERLAP:.0%}")

    ok = len(failures) == 0
    _write_status(
        "benchmark_validated" if ok else "benchmark_below_threshold",
        avg_splade_hits=round(avg_splade, 2),
        avg_bm25_hits=round(avg_bm25, 2),
        avg_encode_ms=round(avg_encode_ms, 1),
        avg_splade_rrf_overlap=round(avg_sr_overlap, 4),
        failures=failures,
    )
    return ok, failures


# ── Phase 4: Quality training ──────────────────────────────────────────────────

def launch_training():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _write_status("launching_training", params=QUALITY_TRAIN_ARGS)

    data_train = ROOT / "data" / "splits" / "train.jsonl"
    data_val = ROOT / "data" / "splits" / "val.jsonl"
    output_dir = Path(QUALITY_TRAIN_ARGS["--output"])

    # Check if GPU is available to add --fp16
    fp16 = []
    try:
        gpu_check = subprocess.run(
            [str(PYTHON), "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True
        )
        if gpu_check.stdout.strip() == "True":
            fp16 = ["--fp16"]
            log.info("GPU detected — enabling fp16")
    except Exception:
        pass

    cmd = [
        str(PYTHON), str(TRAIN_SCRIPT),
        "--data", str(data_train),
        "--val-data", str(data_val),
    ]
    for flag, val in QUALITY_TRAIN_ARGS.items():
        cmd += [flag, val]
    cmd += fp16

    log.info("Launching quality training run:")
    log.info("  Output:     %s", output_dir)
    log.info("  Base model: %s", QUALITY_TRAIN_ARGS["--base-model"])
    log.info("  Epochs:     %s  (default was 6)", QUALITY_TRAIN_ARGS["--epochs"])
    log.info("  LR:         %s  (default was 2e-5)", QUALITY_TRAIN_ARGS["--learning-rate"])
    log.info("  Batch:      %s × grad_accum %s = eff. %d  (default was 32)",
             QUALITY_TRAIN_ARGS["--batch-size"], QUALITY_TRAIN_ARGS["--gradient-accumulation"],
             int(QUALITY_TRAIN_ARGS["--batch-size"]) * int(QUALITY_TRAIN_ARGS["--gradient-accumulation"]))
    log.info("  Log:        %s", TRAIN_LOG)
    log.info("CMD: %s", shlex.join(cmd))

    with open(TRAIN_LOG, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            start_new_session=True,   # detach so training outlives this script
        )

    TRAIN_PID_FILE.write_text(str(proc.pid))
    log.info("Training started — PID %d", proc.pid)
    log.info("  Follow logs:  tail -f %s", TRAIN_LOG)
    log.info("  Check status: cat %s", output_dir / "checkpoints" / "training_status.json")

    _write_status("training_launched", pid=proc.pid, log=str(TRAIN_LOG))
    return proc.pid


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("post_index_watchdog  —  %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 70)

    # Phase 1
    indexer_final = wait_for_indexer()
    log.info("")

    # Phase 2
    benchmark_data = run_benchmark()
    log.info("")

    # Phase 3
    ok, failures = validate_benchmark(benchmark_data)
    if not ok:
        log.warning("Benchmark below thresholds — NOT starting training:")
        for f in failures:
            log.warning("  ✗ %s", f)
        log.warning("Inspect %s and re-run manually if the index looks correct.", BENCHMARK_OUT)
        sys.exit(1)

    log.info("Benchmark passed all thresholds — proceeding to quality training.")
    log.info("")

    # Phase 4
    pid = launch_training()
    log.info("")
    log.info("Watchdog done. Training PID %d is running independently.", pid)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
