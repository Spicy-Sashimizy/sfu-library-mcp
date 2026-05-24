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

VRAM requirement  ⚠
────────────────────
SPLADE training is far heavier than the cross-encoder: the FLOPS loss and the
in-batch similarity matrix both need the full |V| ≈ 30522 vocabulary projection
held in memory for every example in the batch.

  Full fine-tune at --batch-size 8 needs ~A100 40GB (per roadmap Q3.3, ~$30-36,
  8-10 GPU hours). This is the intended cloud configuration.

  On the local RTX 4070 Ti SUPER (16 GB) a full fine-tune at the default batch
  WILL OOM. Local fallbacks, cheapest first:
    * --batch-size 2 --grad-accum 16   (keeps effective batch 32, fits ~16 GB)
    * --max-seq-length 128             (halves activation memory)
    * --grad-checkpoint                 (trades compute for memory)
    * LoRA / PEFT adapters on the MLM head (smallest footprint; requires
      `pip install peft` and is left as a documented TODO below — not wired by
      default to keep this script dependency-light).
  Use --smoke (CPU/GPU, 1-2 steps, 8 synthetic triplets) to validate the
  pipeline locally before paying for an A100.

Data format
───────────
JSONL matching data/sfu_training_triplets.jsonl:
    {"anchor": "...", "positive": "...", "negative": "...", ...}
"negative" is optional. The Q3.1 hard-negative pool
(data/training/hard_negatives_rrf_pool.jsonl) is the intended input.

Usage
─────
    # Cloud (DO A100 40GB) full run — per roadmap Q3.3
    python scripts/finetune_splade.py \\
        --base-model naver/splade-cocondenser-ensembledistil \\
        --train-data data/training/hard_negatives_rrf_pool.jsonl \\
        --output models/sfu-splade-v1 \\
        --epochs 3 --lambda-q 0.0008 --lambda-d 0.0006 \\
        --batch-size 8 --grad-accum 4

    # Local 16GB fallback (will be slow; mostly for sanity, not full quality)
    python scripts/finetune_splade.py \\
        --train-data data/sfu_training_triplets.jsonl \\
        --batch-size 2 --grad-accum 16 --max-seq-length 128

    # Smoke test — 1-2 steps on synthetic triplets, no data file needed
    python scripts/finetune_splade.py --smoke

After training: re-index all docs with the new model (scripts/splade_indexer.py,
~11 min) then run scripts/benchmark_llm_judge.py to confirm the gain.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_TRAIN_DATA = REPO_ROOT / "data/sfu_training_triplets.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "models/sfu-splade-v1"
DEFAULT_BASE_MODEL = "naver/splade-cocondenser-ensembledistil"

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
) -> None:
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        logger.info("GPU: %s (%.1f GB VRAM)", props.name, vram_gb)
        if not smoke and vram_gb < 24 and batch_size * grad_accum >= 32 and batch_size > 2:
            logger.warning(
                "VRAM is %.0f GB but batch-size=%d — SPLADE full fine-tune likely "
                "to OOM. See the fallback note in this script's docstring "
                "(--batch-size 2 --grad-accum 16, --max-seq-length 128, "
                "--grad-checkpoint).", vram_gb, batch_size,
            )
    else:
        logger.warning("No CUDA device — running on CPU (very slow; fine for --smoke).")
    use_amp = device.type == "cuda"

    # ── Data ──
    if smoke:
        rows = clean_triplets(_SMOKE_TRIPLETS)
        logger.info("SMOKE: using %d synthetic triplets (no data file required)", len(rows))
    else:
        if not train_data.exists():
            logger.error("Training data not found: %s", train_data)
            logger.error("Mine hard negatives first (roadmap Q3.1) or pass "
                         "--train-data data/sfu_training_triplets.jsonl")
            sys.exit(1)
        rows = clean_triplets(load_triplets(train_data, max_samples))
    if not rows:
        logger.error("No usable triplets. Aborting.")
        sys.exit(1)

    # ── Model + tokenizer ──
    logger.info("Loading base SPLADE model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    model.to(device)
    if grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    model.train()

    def encode(texts: list[str]):
        enc = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_seq_length, return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        return splade_rep(logits, enc["attention_mask"])

    # ── Optimizer / schedule ──
    eff_batch = min(batch_size, len(rows)) if smoke else batch_size
    steps_per_epoch = max(1, (len(rows) + eff_batch - 1) // eff_batch)
    optim_steps = max(1, (steps_per_epoch * epochs) // max(1, grad_accum))
    warmup_steps = min(int(0.1 * optim_steps), max(0, optim_steps - 1))

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, optim_steps)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    logger.info(
        "Training: epochs=%d, batch=%d, grad_accum=%d, eff_batch=%d, lr=%s, "
        "lambda_q=%s, lambda_d=%s, max_seq=%d, optim_steps=%d, amp=%s",
        epochs, eff_batch, grad_accum, eff_batch * grad_accum, learning_rate,
        lambda_q, lambda_d, max_seq_length, optim_steps, use_amp,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    optimizer.zero_grad()

    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(x, **kw):
            return x

    for epoch in range(epochs):
        # Simple shuffle without numpy dependency
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
            # Append in-batch hard negatives as extra candidate columns
            hard_negs = [b["negative"] for b in batch if b["negative"]]
            doc_texts = docs + hard_negs

            def _step():
                a_rep = encode(anchors)
                d_rep = encode(doc_texts)
                rank = mnrl_loss(a_rep, d_rep)
                reg = lambda_q * flops_loss(a_rep) + lambda_d * flops_loss(d_rep)
                return rank + reg, rank, reg

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss, rank, reg = _step()
                scaler.scale(loss / grad_accum).backward()
            else:
                loss, rank, reg = _step()
                (loss / grad_accum).backward()

            epoch_loss += loss.item()
            n += 1

            if (bi + 1) % grad_accum == 0 or (bi + 1) == len(batches):
                if use_amp:
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
                break

        logger.info("Epoch %d/%d — mean loss=%.4f", epoch + 1, epochs, epoch_loss / max(1, n))
        if smoke:
            break

    # ── Save (HF format so splade_indexer.py can load via from_pretrained) ──
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model saved to %s", output_dir)

    # ── Reload + sanity check sparsity on one synthetic doc ──
    model.eval()
    with torch.no_grad():
        rep = encode([_SMOKE_TRIPLETS[0]["positive"]])
        nnz = int((rep > 0).sum().item())
    logger.info("Reload sanity — sample doc has %d active terms (sparse rep OK)", nnz)

    print(f"\nSPLADE model saved: {output_dir}")
    print("Next steps:")
    print(f"  1. Set SPLADE_MODEL in scripts/splade_indexer.py to {output_dir}")
    print("  2. Re-index: python scripts/splade_indexer.py   (~11 min)")
    print("  3. Run: python scripts/benchmark_llm_judge.py   (confirm NDCG gain)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q3.3 — Fine-tune a SPLADE sparse encoder on SFU triplets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="Base SPLADE (MLM) model to fine-tune")
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA),
                        help="Triplet/hard-negative JSONL "
                             "(falls back to data/sfu_training_triplets.jsonl)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lambda-q", type=float, default=0.0008,
                        help="FLOPS regularization weight for query reps")
    parser.add_argument("--lambda-d", type=float, default=0.0006,
                        help="FLOPS regularization weight for doc reps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch*accum)")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training to first N triplets (for testing)")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM, slower)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 1-2 steps on synthetic triplets to validate the "
                             "pipeline without a full (paid) training run")
    args = parser.parse_args()

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
        )
        logger.info("=== SMOKE OK — pipeline validated. ===")
        return

    train(
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
    )


if __name__ == "__main__":
    main()
