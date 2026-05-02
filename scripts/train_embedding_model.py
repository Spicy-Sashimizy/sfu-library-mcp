#!/usr/bin/env python3
"""Fine-tune a sentence-transformer model for academic search reranking.

Takes training triplets (from generate_training_data.py) and fine-tunes
all-MiniLM-L6-v2 using MultipleNegativesRankingLoss.

Designed to run on a consumer GPU (RTX 3060/3070/4070, 8-12GB VRAM).
Training time: 2-6 hours for full dataset, minutes for small test runs.

Usage:
    python scripts/train_embedding_model.py --data data/training_triplets.jsonl
    python scripts/train_embedding_model.py --data data/training_triplets.jsonl --epochs 5 --batch-size 128
    python scripts/train_embedding_model.py --data data/training_triplets.jsonl --base-model allenai/specter2
"""

import argparse
import json
import logging
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path) -> list[dict]:
    """Load training triplets from JSONL file."""
    pairs = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info("Loaded %d training pairs from %s", len(pairs), data_path)
    return pairs


def train(
    data_path: str,
    output_dir: str = "models/sfu-academic-embed-v1",
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_samples: int | None = None,
    eval_split: float = 0.05,
    use_negatives: bool = True,
):
    """Fine-tune a sentence-transformer on academic triplets.

    Uses MultipleNegativesRankingLoss which treats other in-batch positives
    as negatives, making it very effective for contrastive learning.
    """
    from sentence_transformers import (
        SentenceTransformer,
        InputExample,
        losses,
        evaluation,
    )
    from torch.utils.data import DataLoader

    data = load_training_data(Path(data_path))
    if max_samples:
        data = data[:max_samples]
        logger.info("Truncated to %d samples", len(data))

    # Split into train/eval
    split_idx = max(1, int(len(data) * (1 - eval_split)))
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    logger.info("Train: %d, Eval: %d", len(train_data), len(eval_data))

    # Build training examples
    train_examples = []
    for item in train_data:
        anchor = item["anchor"]
        positive = item["positive"]
        if use_negatives and "negative" in item:
            train_examples.append(InputExample(texts=[anchor, positive, item["negative"]]))
        else:
            train_examples.append(InputExample(texts=[anchor, positive]))

    logger.info("Built %d training examples", len(train_examples))

    # Load base model
    logger.info("Loading base model: %s", base_model)
    model = SentenceTransformer(base_model)
    logger.info(
        "Model: %d params, %d-dim embeddings",
        sum(p.numel() for p in model.parameters()),
        model.get_sentence_embedding_dimension(),
    )

    # Training dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # Loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Warmup steps
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    # Build evaluator from eval data
    evaluator = None
    if eval_data:
        eval_sentences1 = []
        eval_sentences2 = []
        eval_scores = []
        for item in eval_data:
            eval_sentences1.append(item["anchor"])
            eval_sentences2.append(item["positive"])
            eval_scores.append(1.0)  # positive pairs have similarity 1.0
            if "negative" in item:
                eval_sentences1.append(item["anchor"])
                eval_sentences2.append(item["negative"])
                eval_scores.append(0.0)  # negative pairs have similarity 0.0

        if eval_sentences1:
            evaluator = evaluation.EmbeddingSimilarityEvaluator(
                eval_sentences1, eval_sentences2, eval_scores,
                name="academic-eval",
            )

    logger.info(
        "Starting training: %d epochs, batch_size=%d, lr=%s, warmup=%d steps",
        epochs, batch_size, learning_rate, warmup_steps,
    )

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        evaluator=evaluator,
        evaluation_steps=max(100, len(train_dataloader) // 5),
        output_path=output_dir,
        show_progress_bar=True,
        save_best_model=True if evaluator else False,
    )

    logger.info("Model saved to %s", output_dir)

    # Print model info
    model = SentenceTransformer(output_dir)
    dim = model.get_sentence_embedding_dimension()
    params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print(f"\nTraining Complete!")
    print(f"  Output: {output_dir}")
    print(f"  Parameters: {params:,}")
    print(f"  Embedding dim: {dim}")
    print(f"  Model size (FP32): ~{size_mb:.1f} MB")
    print(f"\nNext steps:")
    print(f"  1. Benchmark: python scripts/benchmark_embeddings.py --custom-model {output_dir}")
    print(f"  2. Quantize: python scripts/quantize_model.py --model {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune embedding model for academic search")
    parser.add_argument("--data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--output", type=str, default="models/sfu-academic-embed-v1")
    parser.add_argument("--base-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--no-negatives", action="store_true", help="Ignore negative examples")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_samples=args.max_samples,
        eval_split=args.eval_split,
        use_negatives=not args.no_negatives,
    )


if __name__ == "__main__":
    main()
