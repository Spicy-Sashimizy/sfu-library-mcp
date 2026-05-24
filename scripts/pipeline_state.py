#!/usr/bin/env python3
"""Atomic stage-level state + verification helper for run_splade_pipeline_local.sh.

The orchestrator shells out to the subcommands below to manage a single JSON
state file (written temp -> fsync -> os.replace, so a crash never leaves a
half-written state) and to run the per-stage verification gates that must pass
before the pipeline advances.

State file schema (data/training/splade_pipeline_state.json):
    {
      "started":        <unix ts of first run>,
      "updated":        <unix ts of last write>,
      "model_path":     "models/sfu-splade-v1",   # the fine-tuned model
      "stages": {
        "mine":      {"status": "done", "ts": ...},
        "triplets":  {"status": "done", "ts": ...},
        "finetune":  {"status": "done", "ts": ..., "config": {...}},
        "reindex":   {"status": "done", "ts": ...},
        "benchmark": {"status": "done", "ts": ...}
      }
    }

Subcommands
───────────
    init            Create the state file if absent (idempotent).
    get   STAGE     Print the stage status ("done" / "pending" / "running").
                    Exits 0 always; prints "missing" if no such entry.
    is-done STAGE   Exit 0 if the stage is "done", else exit 1 (for `if`).
    set   STAGE STATUS [--config JSON]   Mark a stage; atomic write.
    set-model PATH  Record the fine-tuned model path.
    show            Pretty-print the whole state.

    verify-finetune  MODEL_DIR
        Gate after stage 3: load the model offline, encode a doc, assert a
        valid non-empty SPLADE sparse vector. Exit 0 ok / non-zero fail-loud.
    verify-reindex   [--url URL] [--index IDX] [--min-count N] [--doc-id ID]
                     [--baseline-file FILE]
        Gate after stage 4: doc count is sane (>= min-count) and a spot-checked
        doc's sparse_field differs from the pre-reindex baseline (proves the
        re-encode actually changed stored vectors). Exit 0 ok / non-zero fail.
    snapshot-doc     [--url URL] [--index IDX] [--doc-id ID] --out FILE
        Capture a doc's sparse_field BEFORE re-indexing, so verify-reindex can
        prove it changed. Atomic write.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_STATE = REPO_ROOT / "data/training/splade_pipeline_state.json"
STAGES = ["mine", "triplets", "finetune", "reindex", "benchmark"]
DEFAULT_URL = os.environ.get(
    "SFU_OPENSEARCH_URL",
    "http://claudebox-sfu-library-mcp-training-opensearch:9200",
)
DEFAULT_INDEX = os.environ.get("SFU_OPENSEARCH_INDEX", "openalex_works")


# ── Atomic JSON I/O ─────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write obj as JSON to path atomically (temp -> fsync -> os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _load(path: Path) -> dict:
    if not path.exists():
        return {"started": None, "updated": None, "model_path": None, "stages": {}}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        # Corrupt/partial state: treat as fresh (a .tmp would never be read here
        # since os.replace is atomic, but be defensive).
        return {"started": None, "updated": None, "model_path": None, "stages": {}}


# ── State commands ──────────────────────────────────────────────────────────

def cmd_init(args) -> int:
    path = Path(args.state)
    st = _load(path)
    now = time.time()
    if st.get("started") is None:
        st["started"] = now
    st["updated"] = now
    _atomic_write_json(path, st)
    print(f"state initialized: {path}")
    return 0


def cmd_get(args) -> int:
    st = _load(Path(args.state))
    entry = st.get("stages", {}).get(args.stage)
    print(entry["status"] if entry else "missing")
    return 0


def cmd_is_done(args) -> int:
    st = _load(Path(args.state))
    entry = st.get("stages", {}).get(args.stage)
    return 0 if (entry and entry.get("status") == "done") else 1


def cmd_set(args) -> int:
    path = Path(args.state)
    st = _load(path)
    now = time.time()
    if st.get("started") is None:
        st["started"] = now
    entry = {"status": args.status, "ts": now}
    if args.config:
        try:
            entry["config"] = json.loads(args.config)
        except json.JSONDecodeError:
            entry["config"] = args.config
    st.setdefault("stages", {})[args.stage] = entry
    st["updated"] = now
    _atomic_write_json(path, st)
    print(f"{args.stage} -> {args.status}")
    return 0


def cmd_set_model(args) -> int:
    path = Path(args.state)
    st = _load(path)
    st["model_path"] = args.path
    st["updated"] = time.time()
    _atomic_write_json(path, st)
    print(f"model_path -> {args.path}")
    return 0


def cmd_show(args) -> int:
    print(json.dumps(_load(Path(args.state)), indent=2))
    return 0


# ── Verification gates ──────────────────────────────────────────────────────

def cmd_verify_finetune(args) -> int:
    """Load the fine-tuned model offline and prove it emits a sparse vector."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"FAIL: model dir does not exist: {model_dir}", file=sys.stderr)
        return 2
    # Must look like an HF model the indexer can from_pretrained().
    has_weights = any((model_dir / f).exists()
                      for f in ("model.safetensors", "pytorch_model.bin"))
    if not (model_dir / "config.json").exists() or not has_weights:
        print(f"FAIL: {model_dir} missing config.json or weights "
              f"(not a loadable HF model)", file=sys.stderr)
        return 3
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: torch/transformers import failed: {exc}", file=sys.stderr)
        return 4
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForMaskedLM.from_pretrained(str(model_dir))
        model.eval()
        enc = tok(["Musqueam land rights and UNDRIP implementation in BC"],
                  return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**enc).logits
            weighted = torch.log1p(torch.relu(logits))
            mask = enc["attention_mask"].unsqueeze(-1)
            rep, _ = torch.max(weighted * mask, dim=1)
        nnz = int((rep > 0).sum().item())
        vocab = int(rep.shape[-1])
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: model load / encode raised: {exc}", file=sys.stderr)
        return 5
    if nnz <= 0:
        print(f"FAIL: sparse rep has {nnz} active terms (expected > 0)",
              file=sys.stderr)
        return 6
    if nnz >= vocab:
        print(f"FAIL: sparse rep not sparse ({nnz}/{vocab} active)",
              file=sys.stderr)
        return 7
    print(f"OK: model loads; sparse vector has {nnz} active terms "
          f"(vocab={vocab}) — valid SPLADE rep.")
    return 0


def _fetch_doc_sparse(url: str, index: str, doc_id: str) -> dict | None:
    import requests
    r = requests.get(f"{url}/{index}/_doc/{doc_id}",
                     params={"_source": "sparse_field"}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    body = r.json()
    if not body.get("found"):
        return None
    return body.get("_source", {}).get("sparse_field") or {}


def _first_doc_id(url: str, index: str) -> str | None:
    import requests
    r = requests.post(f"{url}/{index}/_search",
                      json={"size": 1, "_source": False,
                            "query": {"match_all": {}}}, timeout=30)
    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])
    return hits[0]["_id"] if hits else None


def cmd_snapshot_doc(args) -> int:
    """Capture a doc's sparse_field BEFORE re-index so the gate can prove change."""
    import requests  # noqa: F401  (import-time check)
    url, index = args.url, args.index
    doc_id = args.doc_id or _first_doc_id(url, index)
    if not doc_id:
        print("FAIL: could not pick a doc id to snapshot", file=sys.stderr)
        return 2
    sparse = _fetch_doc_sparse(url, index, doc_id)
    if sparse is None:
        print(f"FAIL: doc {doc_id} not found / no sparse_field", file=sys.stderr)
        return 3
    _atomic_write_json(Path(args.out),
                       {"doc_id": doc_id, "sparse_field": sparse,
                        "captured": time.time()})
    print(f"OK: snapshot of {doc_id} ({len(sparse)} terms) -> {args.out}")
    return 0


def cmd_verify_reindex(args) -> int:
    """Gate after re-index: count sane + a doc's sparse_field changed."""
    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: requests import failed: {exc}", file=sys.stderr)
        return 4
    url, index = args.url, args.index
    # 1) Doc count sane.
    try:
        r = requests.get(f"{url}/{index}/_count", timeout=30)
        r.raise_for_status()
        count = int(r.json().get("count", 0))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: _count query raised: {exc}", file=sys.stderr)
        return 5
    if count < args.min_count:
        print(f"FAIL: doc count {count:,} < min {args.min_count:,} "
              f"(index looks truncated)", file=sys.stderr)
        return 6
    print(f"OK: doc count = {count:,} (>= {args.min_count:,})")

    # 2) Spot-check: a doc's sparse_field changed vs the pre-reindex baseline.
    baseline = None
    doc_id = args.doc_id
    if args.baseline_file and Path(args.baseline_file).exists():
        try:
            b = json.loads(Path(args.baseline_file).read_text())
            baseline = b.get("sparse_field")
            doc_id = doc_id or b.get("doc_id")
        except (json.JSONDecodeError, OSError):
            baseline = None
    if baseline is None:
        print("WARN: no baseline snapshot available — skipping the "
              "'sparse_field changed' spot-check (count gate still enforced).",
              file=sys.stderr)
        return 0
    if not doc_id:
        print("FAIL: baseline present but no doc_id to re-fetch", file=sys.stderr)
        return 7
    try:
        now_sparse = _fetch_doc_sparse(url, index, doc_id)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: re-fetch of {doc_id} raised: {exc}", file=sys.stderr)
        return 8
    if now_sparse is None:
        print(f"FAIL: doc {doc_id} missing after re-index", file=sys.stderr)
        return 9
    if now_sparse == baseline:
        print(f"FAIL: doc {doc_id} sparse_field IDENTICAL to pre-reindex "
              f"baseline — re-encode did not change stored vectors.",
              file=sys.stderr)
        return 10
    # Quantify the change for the log.
    keys_b, keys_n = set(baseline), set(now_sparse)
    changed = len(keys_b ^ keys_n)
    print(f"OK: doc {doc_id} sparse_field CHANGED "
          f"(was {len(keys_b)} terms, now {len(keys_n)}, "
          f"{changed} term keys differ) — re-encode confirmed.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Atomic pipeline state + gates")
    p.add_argument("--state", default=str(DEFAULT_STATE),
                   help="State JSON path")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)

    g = sub.add_parser("get"); g.add_argument("stage"); g.set_defaults(func=cmd_get)
    d = sub.add_parser("is-done"); d.add_argument("stage"); d.set_defaults(func=cmd_is_done)

    s = sub.add_parser("set")
    s.add_argument("stage"); s.add_argument("status")
    s.add_argument("--config", default=None)
    s.set_defaults(func=cmd_set)

    m = sub.add_parser("set-model"); m.add_argument("path"); m.set_defaults(func=cmd_set_model)
    sub.add_parser("show").set_defaults(func=cmd_show)

    vf = sub.add_parser("verify-finetune")
    vf.add_argument("model_dir")
    vf.set_defaults(func=cmd_verify_finetune)

    sd = sub.add_parser("snapshot-doc")
    sd.add_argument("--url", default=DEFAULT_URL)
    sd.add_argument("--index", default=DEFAULT_INDEX)
    sd.add_argument("--doc-id", default=None)
    sd.add_argument("--out", required=True)
    sd.set_defaults(func=cmd_snapshot_doc)

    vr = sub.add_parser("verify-reindex")
    vr.add_argument("--url", default=DEFAULT_URL)
    vr.add_argument("--index", default=DEFAULT_INDEX)
    vr.add_argument("--min-count", type=int, default=150_000_000)
    vr.add_argument("--doc-id", default=None)
    vr.add_argument("--baseline-file", default=None)
    vr.set_defaults(func=cmd_verify_reindex)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
