#!/usr/bin/env python3
"""Q3.2 — Fine-tune a CrossEncoder reranker on SFU academic-search triplets.

The cross-encoder re-scores the merged top-N RRF candidates by joint
query-document relevance (see src/lib/reranker.py). Fine-tuning on SFU triplets
teaches the model SFU-specific vocabulary ("Musqueam", "Secwépemc", "Métis",
"UNDRIP", "Site C", "BC Hydro", ...) that is rare in the base MS MARCO training
data but common in SFU queries.

Base model:   cross-encoder/ms-marco-MiniLM-L-6-v2  (already vetted in Q1.4)
Output:       models/sfu-cross-encoder-v1  (default)
Loss:         BinaryCrossEntropy over (query, doc) pairs labelled 1.0 (positive)
              / 0.0 (hard negative). Each triplet expands into one positive and
              one negative pair.

VRAM / hardware
───────────────
This model is tiny (MiniLM-L6, ~22M params). It fits comfortably on the local
RTX 4070 Ti SUPER (16 GB) at the default batch size, and on any DigitalOcean
L40S (48 GB). CUDA is auto-detected; falls back to CPU (slow but correct).

  Local RTX 4070 Ti SUPER (16 GB):  ~batch 16-32, AMP on, ~1-2 GPU-hours for 9K triplets
  DO L40S (48 GB, ~$2.49/hr):       per roadmap Q3.2, ~4-5h / ~$12-15 for the full pool

Data format
───────────
JSONL, one object per line, matching data/sfu_training_triplets.jsonl:
    {"anchor": "...", "positive": "...", "negative": "...", "strategy": "...",
     "subject": "...", "metadata": {...}}
"negative" is optional; rows without it contribute only a positive pair.
The hard-negative pool from Q3.1 (data/training/hard_negatives_rrf_pool.jsonl)
uses the same shape and is the intended production input.

Usage
─────
    # Local-GPU full run (RTX 4070 Ti SUPER)
    python scripts/train_cross_encoder.py \\
        --base-model cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --train-data data/sfu_training_triplets.jsonl \\
        --output models/sfu-cross-encoder-v1 \\
        --epochs 3 --batch-size 16 --warmup-steps 200

    # Cloud (DO L40S) full run — per roadmap Q3.2
    python scripts/train_cross_encoder.py \\
        --base-model cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --train-data data/training/hard_negatives_rrf_pool.jsonl \\
        --output models/sfu-cross-encoder-v1 \\
        --epochs 3 --batch-size 16 --warmup-steps 200

    # Smoke test — 1-2 steps on a handful of triplets, no real training
    python scripts/train_cross_encoder.py --smoke

After training: point src/lib/reranker.py at models/sfu-cross-encoder-v1 and
re-run scripts/benchmark_llm_judge.py to confirm the gain.
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
DEFAULT_OUTPUT = REPO_ROOT / "models/sfu-cross-encoder-v1"
DEFAULT_BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# A handful of synthetic SFU-flavoured triplets so --smoke runs with zero data
# files present (OpenSearch / data/training/ may not exist yet).
_SMOKE_TRIPLETS = [
    {
        "anchor": "Musqueam land rights and UNDRIP implementation in British Columbia",
        "positive": "Indigenous title and the UN Declaration on the Rights of "
                    "Indigenous Peoples: a Coast Salish case study.",
        "negative": "SciPy 1.0: fundamental algorithms for scientific computing in Python.",
    },
    {
        "anchor": "Site C dam environmental impact on Treaty 8 First Nations",
        "positive": "Hydroelectric development and treaty rights in the Peace River "
                    "watershed: cumulative effects assessment.",
        "negative": "A survey of convolutional neural network architectures for image "
                    "classification.",
    },
    {
        "anchor": "Secwepemc language revitalization pedagogy",
        "positive": "Immersion programs and intergenerational transmission of "
                    "Interior Salish languages.",
        "negative": "Quantum error correction thresholds for surface codes.",
    },
    {
        "anchor": "Metis identity and the Daniels decision in Canadian law",
        "positive": "Section 91(24) and the constitutional status of Metis peoples "
                    "after Daniels v. Canada.",
        "negative": "Thermodynamic limits of reversible computation.",
    },
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_triplets(data_path: Path, max_samples: int | None = None) -> list[dict]:
    """Load triplet/hard-negative rows from a JSONL file."""
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


def build_input_examples(rows: list[dict]):
    """Expand triplets into CrossEncoder (query, doc) InputExamples.

    Each row yields a positive pair (label 1.0) and, when a negative is present,
    a hard-negative pair (label 0.0).
    """
    from sentence_transformers import InputExample

    examples = []
    n_pos = n_neg = 0
    for item in rows:
        anchor = (item.get("anchor") or "").strip()
        positive = (item.get("positive") or "").strip()
        negative = (item.get("negative") or "").strip()
        if not anchor or not positive:
            continue
        examples.append(InputExample(texts=[anchor, positive], label=1.0))
        n_pos += 1
        if negative:
            examples.append(InputExample(texts=[anchor, negative], label=0.0))
            n_neg += 1
    logger.info("Built %d pairs (%d positive, %d hard-negative)",
                len(examples), n_pos, n_neg)
    return examples


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    base_model: str,
    train_data: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    warmup_steps: int,
    max_seq_length: int = 256,
    max_samples: int | None = None,
    smoke: bool = False,
) -> None:
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers.cross_encoder import CrossEncoder

    # ── Device ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s (%.1f GB VRAM)", props.name, props.total_memory / 1e9)
    else:
        logger.warning("No CUDA device — running on CPU (slow; fine for --smoke).")
    use_amp = device == "cuda"

    # ── Data ──
    if smoke:
        rows = _SMOKE_TRIPLETS
        logger.info("SMOKE: using %d synthetic triplets (no data file required)",
                    len(rows))
    else:
        if not train_data.exists():
            logger.error("Training data not found: %s", train_data)
            logger.error("Mine hard negatives first (roadmap Q3.1) or pass "
                         "--train-data data/sfu_training_triplets.jsonl")
            sys.exit(1)
        rows = load_triplets(train_data, max_samples)

    train_examples = build_input_examples(rows)
    if not train_examples:
        logger.error("No usable (query, doc) pairs built from input. Aborting.")
        sys.exit(1)

    # ── Model ──
    logger.info("Loading base CrossEncoder: %s", base_model)
    model = CrossEncoder(base_model, num_labels=1, max_length=max_seq_length, device=device)

    # ── DataLoader ──
    eff_batch = min(batch_size, len(train_examples)) if smoke else batch_size
    train_dataloader = DataLoader(
        train_examples,
        shuffle=not smoke,
        batch_size=max(1, eff_batch),
    )

    steps_per_epoch = max(1, len(train_dataloader))
    total_steps = steps_per_epoch * epochs
    eff_warmup = min(warmup_steps, max(0, total_steps - 1))
    logger.info(
        "Training: epochs=%d, batch=%d, pairs=%d, steps/epoch=%d, total_steps=%d, "
        "warmup=%d, amp=%s",
        epochs, eff_batch, len(train_examples), steps_per_epoch, total_steps,
        eff_warmup, use_amp,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fit ──
    # old_fit is the stable legacy CrossEncoder training loop (BCEWithLogitsLoss
    # by default for num_labels=1). It takes a DataLoader of (text-pair, label)
    # InputExamples — matching the house style in train_embedding_model.py.
    model.old_fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=eff_warmup,
        output_path=str(output_dir),
        use_amp=use_amp,
        show_progress_bar=True,
    )

    # old_fit only writes output_path when save_best_model + evaluator are set;
    # save explicitly so the final weights always land on disk.
    model.save(str(output_dir))
    logger.info("Model saved to %s", output_dir)

    # ── Sanity check: reload and score one synthetic pair ──
    reloaded = CrossEncoder(str(output_dir), max_length=max_seq_length, device=device)
    sample = [(
        _SMOKE_TRIPLETS[0]["anchor"], _SMOKE_TRIPLETS[0]["positive"],
    )]
    score = reloaded.predict(sample)
    logger.info("Reload OK — sample relevance score: %s", score)

    print(f"\nCross-encoder saved: {output_dir}")
    print("Next steps:")
    print(f"  1. Point src/lib/reranker.py CrossEncoder path at {output_dir}")
    print("  2. Run: python scripts/benchmark_llm_judge.py  (confirm NDCG gain)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q3.2 — Fine-tune a CrossEncoder reranker on SFU triplets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="Base CrossEncoder to fine-tune")
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA),
                        help="Triplet/hard-negative JSONL "
                             "(falls back to data/sfu_training_triplets.jsonl)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=256,
                        help="Max combined query+doc token length")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training to first N triplets (for testing)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 1-2 steps on synthetic triplets to validate the "
                             "pipeline without a full (paid) training run")
    args = parser.parse_args()

    if args.smoke:
        # Tiny, fast, deterministic-ish path. Override the heavy knobs.
        logger.info("=== SMOKE MODE: validating import + training path only ===")
        train(
            base_model=args.base_model,
            train_data=Path(args.train_data),
            output_dir=Path(REPO_ROOT / "models/_smoke_cross_encoder"),
            epochs=1,
            batch_size=2,
            warmup_steps=0,
            max_seq_length=128,
            max_samples=8,
            smoke=True,
        )
        logger.info("=== SMOKE OK — pipeline validated. ===")
        return

    train(
        base_model=args.base_model,
        train_data=Path(args.train_data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
        smoke=False,
    )


if __name__ == "__main__":
    main()
