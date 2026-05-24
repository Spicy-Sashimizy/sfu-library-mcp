#!/usr/bin/env python3
"""Q3.3 — Fine-tune a SPLADE sparse encoder on the SFU corpus.

Fine-tuning adjusts the SPLADE encoder so it expands SFU-specific vocabulary in
the sparse representation: "Indigenous" -> "First Nations", "Musqueam", "treaty
rights" — terms the pretrained MS-MARCO model underweights because they are rare
in generic web search.

Base model:   naver/splade-cocondenser-ensembledistil  (the Q2.3 swap target)
Output:       models/sfu-splade-v1  (default)

Training objective
───────────────────
Two terms, combining the standard SPLADE recipe with the house-style
MultipleNegativesRankingLoss:

  1. MultipleNegativesRankingLoss (contrastive)
       In-batch softmax over sparse dot-product similarity. For a batch of
       (anchor, positive[, hard_negative]) triplets, every other positive (and
       any supplied hard negatives) acts as a negative for a given anchor.
       This is the same in-batch-negatives loss used in train_embedding_model.py,
       applied to SPLADE sparse vectors instead of dense embeddings.

  2. FLOPS regularization (sparsity)
       lambda_q * FLOPS(query_reps) + lambda_d * FLOPS(doc_reps), where
       FLOPS(R) = sum_j (mean_i R[i, j])^2  (Paria et al. 2020 / SPLADEv2).
       Keeps the learned representations sparse so OpenSearch rank_features
       queries stay fast. lambda_q > lambda_d because query sparsity matters more
       for latency.

SPLADE representation (matches scripts/splade_indexer.py)
─────────────────────────────────────────────────────────
    logits = model(**enc).logits          # (B, L, V)
    weighted = log1p(relu(logits))         # SPLADE term weighting
    rep = max over sequence dim -> (B, V)  # max-pool with attention mask

Cloud cost / runtime  (target: DigitalOcean gpu-l40sx1-48gb @ $1.57/hr)
───────────────────────────────────────────────────────────────────────
The 48 GB L40S comfortably runs a full fine-tune at a large batch. With bf16
autocast + a tuned batch + grad-accum, the 9,135-triplet set converges well
inside the roadmap's 8-10 GPU-hour budget; in practice early-stopping usually
trips earlier.

  Optimized cloud defaults (--batch-size 48 --grad-accum 1, bf16):
      ~3-5 GPU-hours for 3 epochs of 9,135 triplets   →  ~$5-8
      Hard wall-clock budget --max-runtime-min 600     →  ≤ $15.70 worst case
      LoRA path (--lora) is faster/cheaper still (~2-3 GPU-hr, ~$3-5) but trades
      a little quality — it only adapts low-rank deltas, not full MLM weights.

  On the local RTX 4070 Ti SUPER (16 GB) a full fine-tune at the cloud batch
  WILL OOM. Local fallbacks, cheapest first:
    * --batch-size 2 --grad-accum 16   (keeps effective batch 32, fits ~16 GB)
    * --max-seq-length 128             (halves activation memory)
    * --grad-checkpoint                 (trades compute for memory)
    * --lora                            (smallest footprint; needs `pip install peft`)
  Use --smoke (CPU/GPU, 1-2 steps, 8 synthetic triplets) to validate the
  pipeline locally before paying for a GPU droplet.

Cost-safety wiring (guards real money)
───────────────────────────────────────
This script enforces an INNER wall-clock budget; the OUTER orchestrator
(scripts/cloud/run_splade_finetune.sh) enforces a HARD deadline and always
destroys the droplet. The two are independent layers:

  --max-runtime-min N    Soft budget. On hitting it the trainer checkpoints and
                         exits 0 (the orchestrator then pulls the checkpoint and
                         tears the droplet down). Default 600 min.
  --checkpoint-every-min Durable resumable checkpoint cadence (default 15 min).
  --resume-from DIR      Resume model+optimizer+scheduler+step from a checkpoint.

Checkpoints are resumable (model+optimizer+scheduler+global_step+rng) and are
written atomically with a checksum manifest so a partial transfer can't be
mistaken for a good checkpoint (see CHECKPOINT_MANIFEST / verify_checkpoint).

Data format
───────────
JSONL matching data/training/hard_negatives_triplets.jsonl:
    {"anchor": "...", "positive": "...", "negative": "...", ...}
"negative" is optional. The Q3.1 hard-negative triplets
(data/training/hard_negatives_triplets.jsonl, 9,135 rows) are the intended input.

Usage
─────
    # Cloud (DO L40S 48GB) full run — optimized defaults
    HF_HUB_OFFLINE=1 python scripts/finetune_splade.py \\
        --base-model naver/splade-cocondenser-ensembledistil \\
        --train-data data/training/hard_negatives_triplets.jsonl \\
        --output models/sfu-splade-v1 \\
        --epochs 3 --batch-size 48 --grad-accum 1 \\
        --max-runtime-min 600 --checkpoint-every-min 15

    # Cheaper LoRA variant (needs `pip install peft`)
    HF_HUB_OFFLINE=1 python scripts/finetune_splade.py --lora --batch-size 64

    # Resume after an interrupted/destroyed run
    HF_HUB_OFFLINE=1 python scripts/finetune_splade.py \\
        --resume-from models/sfu-splade-v1/checkpoints/latest

    # Local 16GB fallback (will be slow; mostly for sanity, not full quality)
    python scripts/finetune_splade.py \\
        --train-data data/training/hard_negatives_triplets.jsonl \\
        --batch-size 2 --grad-accum 16 --max-seq-length 128

    # Smoke test — 1-2 steps on synthetic triplets, no data file needed
    python scripts/finetune_splade.py --smoke

After training: re-index all docs with the new model (scripts/splade_indexer.py,
~11 min) then run scripts/benchmark_llm_judge.py to confirm the gain.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN_DATA = REPO_ROOT / "data/training/hard_negatives_triplets.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "models/sfu-splade-v1"
DEFAULT_BASE_MODEL = "naver/splade-cocondenser-ensembledistil"

CHECKPOINT_MANIFEST = "checkpoint_manifest.json"   # checksum + step metadata
TRAIN_STATE_FILE = "training_state.pt"             # optim/sched/step/rng

# Synthetic SFU triplets so --smoke runs with no data files / no OpenSearch.
_SMOKE_TRIPLETS = [
    {
        "anchor": "Musqueam land rights and UNDRIP implementation in British Columbia",
        "positive": "Indigenous title and the UN Declaration on the Rights of "
                    "Indigenous Peoples: a Coast Salish case study.",
        "negative": "SciPy 1.0: fundamental algorithms for scientific computing in Python.",
    },
    {
        "anchor": "Site C dam environmental impact on Treaty 8 First Nations",
        "positive": "Hydroelectric development and treaty rights in the Peace River watershed.",
        "negative": "A survey of convolutional neural network architectures.",
    },
    {
        "anchor": "Secwepemc language revitalization pedagogy",
        "positive": "Immersion programs and intergenerational transmission of Salish languages.",
        "negative": "Quantum error correction thresholds for surface codes.",
    },
    {
        "anchor": "Metis identity and the Daniels decision in Canadian law",
        "positive": "Constitutional status of Metis peoples after Daniels v. Canada.",
        "negative": "Thermodynamic limits of reversible computation.",
    },
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_triplets(data_path: Path, max_samples: int | None = None) -> list[dict]:
    """Load triplet/hard-negative rows from JSONL."""
    rows: list[dict] = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if max_samples:
        rows = rows[:max_samples]
    logger.info("Loaded %d triplets from %s", len(rows), data_path)
    return rows


def clean_triplets(rows: list[dict]) -> list[dict]:
    """Keep rows with a usable anchor + positive."""
    out = []
    for r in rows:
        a = (r.get("anchor") or "").strip()
        p = (r.get("positive") or "").strip()
        if a and p:
            out.append({"anchor": a, "positive": p, "negative": (r.get("negative") or "").strip()})
    logger.info("Usable triplets: %d / %d", len(out), len(rows))
    return out


# ── SPLADE representation + losses ──────────────────────────────────────────────

def splade_rep(logits, attention_mask):
    """SPLADE sparse representation, matching scripts/splade_indexer.py.

    log1p(relu(logits)) then max-pool over the sequence dimension, masking out
    padding tokens. Returns (B, V).
    """
    import torch

    weighted = torch.log1p(torch.relu(logits))                      # (B, L, V)
    mask = attention_mask.unsqueeze(-1)                             # (B, L, 1)
    weighted = weighted * mask                                     # zero padding
    rep, _ = torch.max(weighted, dim=1)                            # (B, V)
    return rep


def flops_loss(reps):
    """FLOPS sparsity regularizer (Paria et al. 2020).

    FLOPS(R) = sum_j (mean_i |R[i, j]|)^2  over the batch. Penalizes tokens that
    are active across many examples, pushing the representation toward sparsity.
    """
    import torch

    return torch.sum(torch.mean(reps, dim=0) ** 2)


def mnrl_loss(anchor_reps, doc_reps, scale: float = 1.0):
    """MultipleNegativesRankingLoss over sparse reps (in-batch negatives).

    doc_reps holds all positives followed by any hard negatives. For row i the
    target is the i-th positive; every other column is a negative. Identical in
    spirit to losses.MultipleNegativesRankingLoss in train_embedding_model.py,
    but on SPLADE dot-product similarity rather than cosine of dense vectors.
    """
    import torch
    import torch.nn.functional as F

    scores = (anchor_reps @ doc_reps.T) * scale                    # (B, D)
    labels = torch.arange(anchor_reps.size(0), device=anchor_reps.device)
    return F.cross_entropy(scores, labels)


# ── Checkpoint round-trip (verify-on-arrival) ───────────────────────────────────

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(ckpt_dir: Path, global_step: int, epoch: int) -> None:
    """Write a checksum+size manifest so a partial transfer can't be mistaken
    for a good checkpoint. verify_checkpoint() re-checks these on arrival."""
    files = {}
    for p in sorted(ckpt_dir.rglob("*")):
        if p.is_file() and p.name != CHECKPOINT_MANIFEST:
            rel = str(p.relative_to(ckpt_dir))
            files[rel] = {"size": p.stat().st_size, "sha256": _sha256(p)}
    manifest = {
        "global_step": global_step,
        "epoch": epoch,
        "created": time.time(),
        "files": files,
    }
    tmp = ckpt_dir / (CHECKPOINT_MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(ckpt_dir / CHECKPOINT_MANIFEST)


def verify_checkpoint(ckpt_dir: Path) -> bool:
    """Verify every file in the manifest matches by size + sha256.

    Used both locally (after writing) and conceptually by the orchestrator after
    an rsync pull/push — a half-transferred file fails here and is not resumed.
    """
    mpath = ckpt_dir / CHECKPOINT_MANIFEST
    if not mpath.exists():
        logger.error("No manifest in %s — refusing to trust checkpoint", ckpt_dir)
        return False
    manifest = json.loads(mpath.read_text())
    for rel, meta in manifest.get("files", {}).items():
        p = ckpt_dir / rel
        if not p.exists():
            logger.error("Checkpoint missing file: %s", rel)
            return False
        if p.stat().st_size != meta["size"]:
            logger.error("Checkpoint size mismatch (%s): %d != %d",
                         rel, p.stat().st_size, meta["size"])
            return False
        if _sha256(p) != meta["sha256"]:
            logger.error("Checkpoint checksum mismatch: %s", rel)
            return False
    logger.info("Checkpoint verified OK: %s (step %s)",
                ckpt_dir, manifest.get("global_step"))
    return True


def save_checkpoint(ckpt_root: Path, tag: str, model, tokenizer, optimizer,
                    scheduler, scaler, global_step: int, epoch: int,
                    is_lora: bool) -> Path:
    """Write a resumable checkpoint atomically, then update `latest` symlink.

    Layout:
        <ckpt_root>/<tag>/                       (HF model + tokenizer + state)
            config.json, model.safetensors, ...  (or adapter_* for LoRA)
            training_state.pt                    (optim/sched/scaler/step/rng)
            checkpoint_manifest.json             (size+sha256 of every file)
        <ckpt_root>/latest -> <tag>
    """
    import torch

    ckpt_root.mkdir(parents=True, exist_ok=True)
    final_dir = ckpt_root / tag
    staging = ckpt_root / (tag + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # HF weights (full model, or LoRA adapter only)
    model.save_pretrained(str(staging))
    tokenizer.save_pretrained(str(staging))

    state = {
        "global_step": global_step,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "is_lora": is_lora,
    }
    torch.save(state, staging / TRAIN_STATE_FILE)

    write_manifest(staging, global_step, epoch)
    if not verify_checkpoint(staging):
        raise RuntimeError(f"Checkpoint failed self-verification: {staging}")

    # Atomic swap into place
    if final_dir.exists():
        shutil.rmtree(final_dir)
    staging.replace(final_dir)

    latest = ckpt_root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_dir() and not latest.is_symlink():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(tag, target_is_directory=True)
    except OSError:
        # Filesystems without symlink support: write a pointer file instead.
        (ckpt_root / "latest.txt").write_text(tag)

    logger.info("Checkpoint saved: %s (step %d, epoch %d)", final_dir, global_step, epoch)
    return final_dir


def resolve_resume_dir(resume_from: str) -> Path:
    """Resolve --resume-from, following `latest` symlink / latest.txt pointer."""
    p = Path(resume_from)
    if (p / "latest.txt").exists() and not (p / TRAIN_STATE_FILE).exists():
        p = p / (p / "latest.txt").read_text().strip()
    return p


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    base_model: str,
    train_data: Path,
    output_dir: Path,
    epochs: int,
    lambda_q: float,
    lambda_d: float,
    batch_size: int,
    grad_accum: int,
    learning_rate: float = 2e-5,
    max_seq_length: int = 256,
    max_samples: int | None = None,
    grad_checkpoint: bool = False,
    smoke: bool = False,
    *,
    num_workers: int = 4,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    compile_model: bool = False,
    max_steps: int | None = None,
    max_runtime_min: float | None = 600.0,
    checkpoint_every_min: float = 15.0,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 1e-4,
    resume_from: str | None = None,
    dry_run: bool = False,
) -> None:
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    vram_gb = 0.0
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        logger.info("GPU: %s (%.1f GB VRAM)", props.name, vram_gb)
        if not smoke and vram_gb < 24 and batch_size * grad_accum >= 32 and batch_size > 2:
            logger.warning(
                "VRAM is %.0f GB but batch-size=%d — SPLADE full fine-tune likely "
                "to OOM. See the fallback note in this script's docstring "
                "(--batch-size 2 --grad-accum 16, --max-seq-length 128, "
                "--grad-checkpoint, or --lora).", vram_gb, batch_size,
            )
    else:
        logger.warning("No CUDA device — running on CPU (very slow; fine for --smoke).")

    # ── Mixed precision: prefer bf16 on Ampere+ (no GradScaler needed); fall
    #    back to fp16 + GradScaler on older GPUs; plain fp32 on CPU.
    use_amp = device.type == "cuda"
    amp_dtype = None
    scaler = None
    if use_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            logger.info("AMP: bf16 autocast (no GradScaler)")
        else:
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
            logger.info("AMP: fp16 autocast + GradScaler")

    # ── Data ──
    if smoke:
        rows = clean_triplets(_SMOKE_TRIPLETS)
        logger.info("SMOKE: using %d synthetic triplets (no data file required)", len(rows))
    else:
        if not train_data.exists():
            logger.error("Training data not found: %s", train_data)
            logger.error("Mine hard negatives first (roadmap Q3.1) or pass "
                         "--train-data data/training/hard_negatives_triplets.jsonl")
            sys.exit(1)
        rows = clean_triplets(load_triplets(train_data, max_samples))
    if not rows:
        logger.error("No usable triplets. Aborting.")
        sys.exit(1)

    eff_batch = min(batch_size, len(rows)) if smoke else batch_size
    steps_per_epoch = max(1, (len(rows) + eff_batch - 1) // eff_batch)
    optim_steps_total = max(1, (steps_per_epoch * epochs) // max(1, grad_accum))
    if max_steps:
        optim_steps_total = min(optim_steps_total, max_steps)
    warmup_steps = min(int(0.1 * optim_steps_total), max(0, optim_steps_total - 1))

    # ── DRY RUN: print the plan and exit before loading the model / paying. ──
    if dry_run:
        print("─" * 64)
        print("finetune_splade.py --dry-run plan")
        print("─" * 64)
        print(f"  device              : {device} ({vram_gb:.0f} GB VRAM)" if device.type == "cuda"
              else f"  device              : {device}")
        print(f"  base_model          : {base_model}")
        print(f"  train_data          : {train_data}")
        print(f"  usable_triplets     : {len(rows)}")
        print(f"  output              : {output_dir}")
        print(f"  epochs              : {epochs}")
        print(f"  batch_size          : {eff_batch}  grad_accum: {grad_accum}  "
              f"eff_batch: {eff_batch * grad_accum}")
        print(f"  max_seq_length      : {max_seq_length}")
        print(f"  optim_steps (planned): {optim_steps_total}  (warmup {warmup_steps})")
        print(f"  amp                 : {amp_dtype}")
        print(f"  lora                : {use_lora} (r={lora_r}, alpha={lora_alpha})")
        print(f"  torch.compile       : {compile_model}")
        print(f"  num_workers         : {num_workers}")
        print(f"  max_steps           : {max_steps}")
        print(f"  max_runtime_min     : {max_runtime_min}  (soft budget; checkpoint+exit)")
        print(f"  checkpoint_every_min: {checkpoint_every_min}")
        print(f"  early_stop_patience : {early_stop_patience} (min_delta {early_stop_min_delta})")
        print(f"  resume_from         : {resume_from}")
        print("─" * 64)
        print("DRY RUN — no model loaded, no GPU work, no files written.")
        return

    # ── Model + tokenizer ──
    logger.info("Loading base SPLADE model: %s", base_model)
    resume_dir = None
    load_src = base_model
    if resume_from:
        resume_dir = resolve_resume_dir(resume_from)
        if not verify_checkpoint(resume_dir):
            logger.error("Refusing to resume from unverified checkpoint: %s", resume_dir)
            sys.exit(1)
        # For a full (non-LoRA) checkpoint, the HF weights live in resume_dir.
        state_meta = json.loads((resume_dir / CHECKPOINT_MANIFEST).read_text())
        if not use_lora:
            load_src = str(resume_dir)
        logger.info("Resuming from %s (step %s)", resume_dir, state_meta.get("global_step"))

    tokenizer = AutoTokenizer.from_pretrained(load_src)
    model = AutoModelForMaskedLM.from_pretrained(load_src)

    is_lora = False
    if use_lora:
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError:
            logger.error("--lora requires peft. Install with: "
                         "sudo %s/.venv/bin/pip install peft", REPO_ROOT)
            sys.exit(1)
        if resume_dir is not None:
            model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
        else:
            # Target the transformer attention + MLM head projections.
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                target_modules=["query", "key", "value", "dense"],
                bias="none", task_type="FEATURE_EXTRACTION",
            )
            model = get_peft_model(model, lora_cfg)
        is_lora = True
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    model.to(device)
    if grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    if compile_model and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile (first step will be slow)")
        model = torch.compile(model)
    model.train()

    def encode(texts: list[str]):
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_seq_length, return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        return splade_rep(logits, enc["attention_mask"])

    # ── Optimizer / schedule ──
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, optim_steps_total)

    global_step = 0
    start_epoch = 0
    if resume_dir is not None:
        state = torch.load(resume_dir / TRAIN_STATE_FILE, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"].cpu() if hasattr(state["torch_rng"], "cpu")
                                 else state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            try:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            except Exception:  # noqa: BLE001 — RNG restore is best-effort
                pass
        global_step = int(state.get("global_step", 0))
        start_epoch = int(state.get("epoch", 0))
        logger.info("Resumed optimizer/scheduler at global_step=%d, epoch=%d",
                    global_step, start_epoch)

    logger.info(
        "Training: epochs=%d, batch=%d, grad_accum=%d, eff_batch=%d, lr=%s, "
        "lambda_q=%s, lambda_d=%s, max_seq=%d, optim_steps=%d, amp=%s, lora=%s, "
        "max_runtime_min=%s, ckpt_every_min=%s, max_steps=%s",
        epochs, eff_batch, grad_accum, eff_batch * grad_accum, learning_rate,
        lambda_q, lambda_d, max_seq_length, optim_steps_total, amp_dtype, is_lora,
        max_runtime_min, checkpoint_every_min, max_steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = output_dir / "checkpoints"

    optimizer.zero_grad()

    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(x, **kw):
            return x

    # ── Time budgets + early stopping bookkeeping ──
    t_start = time.monotonic()
    last_ckpt = t_start
    runtime_budget_s = (max_runtime_min * 60.0) if max_runtime_min else None
    ckpt_interval_s = checkpoint_every_min * 60.0
    best_loss = float("inf")
    no_improve = 0
    stop_reason = "completed"

    def _checkpoint(tag: str, epoch: int):
        nonlocal last_ckpt
        if smoke:
            return
        save_checkpoint(ckpt_root, tag, model, tokenizer, optimizer, scheduler,
                        scaler, global_step, epoch, is_lora)
        last_ckpt = time.monotonic()

    stop = False
    for epoch in range(start_epoch, epochs):
        import random
        order = list(range(len(rows)))
        if not smoke:
            random.Random(42 + epoch).shuffle(order)

        batches = [order[i:i + eff_batch] for i in range(0, len(order), eff_batch)]
        epoch_loss = 0.0
        n = 0

        pbar = _tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}")
        for bi, idx in enumerate(pbar):
            batch = [rows[i] for i in idx]
            anchors = [b["anchor"] for b in batch]
            docs = [b["positive"] for b in batch]
            hard_negs = [b["negative"] for b in batch if b["negative"]]
            doc_texts = docs + hard_negs

            def _step():
                a_rep = encode(anchors)
                d_rep = encode(doc_texts)
                rank = mnrl_loss(a_rep, d_rep)
                reg = lambda_q * flops_loss(a_rep) + lambda_d * flops_loss(d_rep)
                return rank + reg, rank, reg

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss, rank, reg = _step()
                if scaler is not None:
                    scaler.scale(loss / grad_accum).backward()
                else:
                    (loss / grad_accum).backward()
            else:
                loss, rank, reg = _step()
                (loss / grad_accum).backward()

            epoch_loss += loss.item()
            n += 1

            if (bi + 1) % grad_accum == 0 or (bi + 1) == len(batches):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "rank": f"{rank.item():.4f}",
                    "reg": f"{reg.item():.4g}",
                    "step": global_step,
                })

            if smoke and global_step >= 2:
                logger.info("SMOKE: reached 2 optimizer steps, stopping early.")
                stop = True
                break

            # ── Hard caps checked every batch ──
            now = time.monotonic()
            if max_steps and global_step >= max_steps:
                logger.info("Reached --max-steps=%d; stopping.", max_steps)
                stop_reason = "max_steps"
                stop = True
                break
            if runtime_budget_s is not None and (now - t_start) >= runtime_budget_s:
                logger.warning("Hit --max-runtime-min=%.0f (%.1f min elapsed); "
                               "checkpointing and exiting cleanly.",
                               max_runtime_min, (now - t_start) / 60.0)
                stop_reason = "runtime_budget"
                stop = True
                break
            if not smoke and (now - last_ckpt) >= ckpt_interval_s:
                logger.info("Periodic checkpoint (%.1f min since last).",
                            (now - last_ckpt) / 60.0)
                _checkpoint("latest_ckpt", epoch)

        mean_loss = epoch_loss / max(1, n)
        logger.info("Epoch %d/%d — mean loss=%.4f", epoch + 1, epochs, mean_loss)

        if smoke:
            break

        # Save an epoch checkpoint so an interrupted next epoch can resume here.
        _checkpoint("latest_ckpt", epoch + 1)

        # ── Early stopping ──
        if early_stop_patience > 0:
            if best_loss - mean_loss > early_stop_min_delta:
                best_loss = mean_loss
                no_improve = 0
            else:
                no_improve += 1
                logger.info("Early-stop: no improvement (%d/%d, best=%.4f)",
                            no_improve, early_stop_patience, best_loss)
                if no_improve >= early_stop_patience:
                    logger.info("Early stopping: converged.")
                    stop_reason = "early_stop"
                    stop = True

        if stop:
            break

    # ── Save final model (HF format so splade_indexer.py can load via from_pretrained) ──
    if is_lora:
        logger.info("Merging LoRA adapters into base weights for inference export")
        if hasattr(model, "merge_and_unload"):
            export_model = model.merge_and_unload()
        else:
            export_model = model
        export_model.save_pretrained(str(output_dir))
    else:
        # torch.compile wraps the module; save the original.
        export_model = getattr(model, "_orig_mod", model)
        export_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model saved to %s (stop_reason=%s, global_step=%d)",
                output_dir, stop_reason, global_step)

    # Final resumable checkpoint too (so resume-from a budget exit still works).
    if not smoke and stop_reason in ("runtime_budget", "max_steps"):
        _checkpoint("latest_ckpt", epoch)

    # ── Reload + sanity check sparsity on one synthetic doc ──
    model.eval()
    with torch.no_grad():
        rep = encode([_SMOKE_TRIPLETS[0]["positive"]])
        nnz = int((rep > 0).sum().item())
    logger.info("Reload sanity — sample doc has %d active terms (sparse rep OK)", nnz)

    elapsed_min = (time.monotonic() - t_start) / 60.0
    peak_vram_gb = None
    if device.type == "cuda":
        try:
            peak_vram_gb = torch.cuda.max_memory_reserved() / 1e9
            logger.info("Peak VRAM (max_memory_reserved): %.2f GB / %.1f GB total",
                        peak_vram_gb, vram_gb)
        except Exception:  # noqa: BLE001
            pass
    print(f"\nSPLADE model saved: {output_dir}")
    print(f"  stop_reason : {stop_reason}")
    print(f"  global_step : {global_step}")
    print(f"  wall_time   : {elapsed_min:.1f} min")
    if peak_vram_gb is not None:
        print(f"  peak_vram   : {peak_vram_gb:.2f} GB / {vram_gb:.1f} GB total")
    print("Next steps:")
    print(f"  1. Set SPLADE_MODEL in scripts/splade_indexer.py to {output_dir}")
    print("  2. Re-index: python scripts/splade_indexer.py   (~11 min)")
    print("  3. Run: python scripts/benchmark_llm_judge.py   (confirm NDCG gain)")


def _is_cuda_oom(exc: BaseException) -> bool:
    """True if exc is a CUDA out-of-memory error (class or message match).

    torch.cuda.OutOfMemoryError exists on modern torch, but some code paths
    raise a plain RuntimeError whose message contains 'out of memory'. Catch
    both so the auto-backoff is robust across torch versions.
    """
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001 — torch may not expose the class
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _cuda_empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass


def train_with_oom_backoff(base_kwargs: dict) -> dict:
    """Run train() with CUDA-OOM auto-backoff.

    On torch.cuda.OutOfMemoryError (or an 'out of memory' RuntimeError) we
    empty_cache() and retry with a progressively smaller config. The ladder,
    starting from the 16 GB-safe config the orchestrator passes in:

        1. requested config (e.g. batch 2 / grad_accum 16 / seq 128 / grad-ckpt)
        2. batch 1 / grad_accum 32  (keep effective batch, halve activations)
        3. batch 1 / grad_accum 32 / max_seq_length 96 + grad_checkpoint
        4. LoRA (smallest footprint; needs peft) at batch 1 / grad_accum 32

    Each rung preserves the effective batch where possible so the optimization
    dynamics barely change. The config that finally succeeds is written to
    <output>/oom_backoff_result.json and returned so the orchestrator can
    record exactly what ran.
    """
    base_batch = base_kwargs["batch_size"]
    base_accum = base_kwargs["grad_accum"]
    base_seq = base_kwargs["max_seq_length"]
    eff = max(1, base_batch * base_accum)

    ladder: list[dict] = [dict(base_kwargs)]
    # Rung 2: batch 1, keep effective batch.
    if base_batch > 1:
        ladder.append({**base_kwargs, "batch_size": 1,
                       "grad_accum": eff, "grad_checkpoint": True})
    # Rung 3: also shrink the sequence length.
    ladder.append({**base_kwargs, "batch_size": 1, "grad_accum": eff,
                   "max_seq_length": min(base_seq, 96), "grad_checkpoint": True})
    # Rung 4: LoRA — smallest footprint.
    ladder.append({**base_kwargs, "batch_size": 1, "grad_accum": eff,
                   "max_seq_length": min(base_seq, 96),
                   "grad_checkpoint": True, "use_lora": True})

    output_dir = base_kwargs["output_dir"]
    last_exc: BaseException | None = None
    for i, cfg in enumerate(ladder):
        attempt = {
            "rung": i + 1,
            "batch_size": cfg["batch_size"],
            "grad_accum": cfg["grad_accum"],
            "max_seq_length": cfg["max_seq_length"],
            "grad_checkpoint": cfg.get("grad_checkpoint", False),
            "use_lora": cfg.get("use_lora", False),
        }
        logger.info("OOM-backoff rung %d/%d: %s", i + 1, len(ladder), attempt)
        _cuda_empty_cache()
        try:
            train(**cfg)
            result = {"success": True, **attempt}
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                (Path(output_dir) / "oom_backoff_result.json").write_text(
                    json.dumps(result, indent=2))
            except OSError:
                pass
            logger.info("OOM-backoff: rung %d succeeded — config recorded.", i + 1)
            return result
        except BaseException as exc:  # noqa: BLE001 — inspect, re-raise non-OOM
            if _is_cuda_oom(exc):
                last_exc = exc
                logger.warning("OOM-backoff: rung %d OOM'd (%s). "
                               "empty_cache() + retry at a smaller config.",
                               i + 1, type(exc).__name__)
                _cuda_empty_cache()
                continue
            # Not an OOM — a real error; don't mask it.
            raise
    logger.error("OOM-backoff: exhausted all %d rungs and still OOM.", len(ladder))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OOM-backoff exhausted with no recorded exception")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q3.3 — Fine-tune a SPLADE sparse encoder on SFU triplets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="Base SPLADE (MLM) model to fine-tune")
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA),
                        help="Triplet/hard-negative JSONL")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lambda-q", type=float, default=0.0008,
                        help="FLOPS regularization weight for query reps")
    parser.add_argument("--lambda-d", type=float, default=0.0006,
                        help="FLOPS regularization weight for doc reps")
    parser.add_argument("--batch-size", type=int, default=48,
                        help="Per-step batch (48 tuned for the 48GB L40S; "
                             "use 2 + --grad-accum 16 on a local 16GB GPU)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch*accum)")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training to first N triplets (for testing)")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM, slower)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader/tokenization worker hint (reserved; "
                             "tokenization runs inline but kept for parity)")
    # ── Speed / cost knobs ──
    parser.add_argument("--lora", action="store_true",
                        help="Train low-rank PEFT adapters instead of full weights "
                             "(faster/cheaper ~$3-5, slightly lower quality; needs peft)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--compile", dest="compile_model", action="store_true",
                        help="torch.compile the model (first step slow, then faster)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Hard cap on optimizer steps (stops as soon as hit)")
    parser.add_argument("--early-stop-patience", type=int, default=2,
                        help="Stop after N epochs without mean-loss improvement "
                             "(0 disables)")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    # ── Cost-safety: wall-clock budget + checkpoint cadence + resume ──
    parser.add_argument("--max-runtime-min", type=float, default=600.0,
                        help="Soft wall-clock budget: checkpoint and exit cleanly "
                             "when reached (the outer orchestrator enforces a "
                             "separate HARD deadline). 0 disables.")
    parser.add_argument("--checkpoint-every-min", type=float, default=15.0,
                        help="Write a resumable checkpoint at least this often")
    parser.add_argument("--resume-from", default=None,
                        help="Resume from a checkpoint dir (or its 'latest' link)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved training plan and exit without "
                             "loading the model or touching the GPU")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 1-2 steps on synthetic triplets to validate the "
                             "pipeline without a full (paid) training run")
    parser.add_argument("--oom-backoff", action="store_true",
                        help="On CUDA OutOfMemoryError, empty_cache() and retry at a "
                             "progressively smaller config (batch 2->1, shorter seq, "
                             "then LoRA). The winning config is written to "
                             "<output>/oom_backoff_result.json. Recommended on 16 GB GPUs.")
    args = parser.parse_args()

    max_runtime = args.max_runtime_min if args.max_runtime_min and args.max_runtime_min > 0 else None

    if args.smoke:
        logger.info("=== SMOKE MODE: validating import + SPLADE training path only ===")
        train(
            base_model=args.base_model,
            train_data=Path(args.train_data),
            output_dir=Path(REPO_ROOT / "models/_smoke_splade"),
            epochs=1,
            lambda_q=args.lambda_q,
            lambda_d=args.lambda_d,
            batch_size=2,
            grad_accum=1,
            max_seq_length=64,
            max_samples=8,
            grad_checkpoint=args.grad_checkpoint,
            smoke=True,
            num_workers=0,
            use_lora=False,
            compile_model=False,
            max_steps=None,
            max_runtime_min=None,
            checkpoint_every_min=args.checkpoint_every_min,
            early_stop_patience=0,
            resume_from=None,
            dry_run=False,
        )
        logger.info("=== SMOKE OK — pipeline validated. ===")
        return

    train_kwargs = dict(
        base_model=args.base_model,
        train_data=Path(args.train_data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        lambda_q=args.lambda_q,
        lambda_d=args.lambda_d,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
        grad_checkpoint=args.grad_checkpoint,
        smoke=False,
        num_workers=args.num_workers,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        compile_model=args.compile_model,
        max_steps=args.max_steps,
        max_runtime_min=max_runtime,
        checkpoint_every_min=args.checkpoint_every_min,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        resume_from=args.resume_from,
        dry_run=args.dry_run,
    )

    # --oom-backoff retries at a smaller config on CUDA OOM (no-op for --dry-run,
    # which never touches the GPU).
    if args.oom_backoff and not args.dry_run:
        train_with_oom_backoff(train_kwargs)
    else:
        train(**train_kwargs)


if __name__ == "__main__":
    main()
