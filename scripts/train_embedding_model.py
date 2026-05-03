#!/usr/bin/env python3
"""Fine-tune a sentence-transformer model for SFU academic search reranking.

Designed to run on a consumer GPU (RTX 3060/3070/4070, 8-12GB VRAM) or CPU.
Training time: 2-6 hours GPU, 15-20 hours CPU for full 15K dataset.

Checkpoint/Resume
─────────────────
Training can be interrupted at any time (Ctrl-C or kill) and resumed without
losing progress. Checkpoints are saved:
  - After every epoch (epoch checkpoint)
  - Every --save-steps global steps (step checkpoint)
  - On SIGINT/SIGTERM (emergency checkpoint before exit)

To resume a stopped run:
    python scripts/train_embedding_model.py \\
        --data data/splits/train.jsonl \\
        --resume

The script finds the latest checkpoint automatically.

Usage:
    # First run
    python scripts/train_embedding_model.py \\
        --data data/splits/train.jsonl \\
        --val-data data/splits/val.jsonl \\
        --output models/sfu-academic-embed-v1 \\
        --epochs 3 --batch-size 64 --learning-rate 2e-5 --fp16

    # Resume after interruption
    python scripts/train_embedding_model.py \\
        --data data/splits/train.jsonl \\
        --val-data data/splits/val.jsonl \\
        --output models/sfu-academic-embed-v1 \\
        --resume

    # CPU-only run (container without GPU)
    python scripts/train_embedding_model.py \\
        --data data/splits/train.jsonl \\
        --output models/sfu-academic-embed-v1 \\
        --batch-size 16 --gradient-accumulation 4
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Checkpoint management ─────────────────────────────────────────────────────

CHECKPOINT_STATE_FILE = "training_state.json"
CHECKPOINT_OPT_FILE = "optimizer_state.pt"


def _checkpoint_path(checkpoint_dir: Path, label: str) -> Path:
    return checkpoint_dir / label


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_score: float,
    loss_history: list[float],
    checkpoint_dir: Path,
    label: str = "latest",
) -> Path:
    """Save model + optimizer state + training state to checkpoint_dir/label."""
    import torch

    ckpt_path = checkpoint_dir / label
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Save model in sentence-transformers format (always loadable)
    model.save(str(ckpt_path))

    # Save optimizer and scheduler states (for exact resume)
    opt_data = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
    }
    torch.save(opt_data, ckpt_path / CHECKPOINT_OPT_FILE)

    # Human-readable state file (used by resume logic)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "loss_history": loss_history[-50:],  # Keep last 50 loss values
        "checkpoint_label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (checkpoint_dir / CHECKPOINT_STATE_FILE).write_text(json.dumps(state, indent=2))

    logger.info("Checkpoint saved: %s (epoch=%d, step=%d, score=%.4f)",
                ckpt_path, epoch, global_step, best_score)
    return ckpt_path


def load_checkpoint(checkpoint_dir: Path) -> dict | None:
    """Load training state from checkpoint_dir. Returns state dict or None."""
    import torch

    state_file = checkpoint_dir / CHECKPOINT_STATE_FILE
    if not state_file.exists():
        return None

    state = json.loads(state_file.read_text())
    label = state.get("checkpoint_label", "latest")
    ckpt_path = checkpoint_dir / label

    if not ckpt_path.exists():
        logger.warning("Checkpoint dir %s not found — starting fresh", ckpt_path)
        return None

    opt_file = ckpt_path / CHECKPOINT_OPT_FILE
    if opt_file.exists():
        state["optimizer_data"] = torch.load(opt_file, map_location="cpu", weights_only=False)
    state["model_path"] = str(ckpt_path)

    logger.info("Found checkpoint: epoch=%d, step=%d, best_score=%.4f",
                state["epoch"], state["global_step"], state["best_score"])
    return state


# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_data(data_path: Path, max_samples: int | None = None) -> list[dict]:
    """Load training triplets from JSONL file."""
    pairs = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if max_samples:
        pairs = pairs[:max_samples]
    logger.info("Loaded %d training pairs from %s", len(pairs), data_path)
    return pairs


def build_input_examples(data: list[dict], use_negatives: bool = True):
    """Convert JSONL data to sentence-transformers InputExample objects."""
    from sentence_transformers import InputExample
    examples = []
    for item in data:
        anchor = item.get("anchor", "")
        positive = item.get("positive", "")
        if not anchor or not positive:
            continue
        if use_negatives and "negative" in item:
            examples.append(InputExample(texts=[anchor, positive, item["negative"]]))
        else:
            examples.append(InputExample(texts=[anchor, positive]))
    return examples


# ── Training setup ────────────────────────────────────────────────────────────

def build_evaluator(val_data: list[dict] | None, name: str = "sfu-eval"):
    """Build EmbeddingSimilarityEvaluator from validation data."""
    from sentence_transformers import evaluation
    if not val_data:
        return None
    s1, s2, scores = [], [], []
    for item in val_data:
        s1.append(item.get("anchor", ""))
        s2.append(item.get("positive", ""))
        scores.append(1.0)
        if "negative" in item:
            s1.append(item.get("anchor", ""))
            s2.append(item["negative"])
            scores.append(0.0)
    if not s1:
        return None
    return evaluation.EmbeddingSimilarityEvaluator(s1, s2, scores, name=name)


def run_evaluation(model, evaluator, checkpoint_dir: Path, global_step: int) -> float:
    """Run evaluator and return the score."""
    if evaluator is None:
        return 0.0
    model.eval()
    score = evaluator(model, output_path=str(checkpoint_dir), epoch=0, steps=global_step)
    model.train()
    return float(score) if score is not None else 0.0


# ── Manual training loop ──────────────────────────────────────────────────────

def train(
    data_path: str,
    output_dir: str = "models/sfu-academic-embed-v1",
    val_data_path: str | None = None,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_seq_length: int = 256,
    fp16: bool = False,
    gradient_accumulation: int = 1,
    save_steps: int = 500,
    eval_steps: int = 500,
    max_samples: int | None = None,
    use_negatives: bool = True,
    checkpoint_dir: str | None = None,
    resume: bool = False,
    log_dir: str = "logs/sfu-embed-training",
):
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, losses
    from transformers import get_linear_schedule_with_warmup

    # ── Device setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB VRAM)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Use fp16 only on CUDA
    use_fp16 = fp16 and device.type == "cuda"
    if fp16 and device.type != "cuda":
        logger.warning("--fp16 requested but no GPU found — disabling FP16")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else output_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    train_data = load_training_data(Path(data_path), max_samples)
    val_data = load_training_data(Path(val_data_path)) if val_data_path else None

    train_examples = build_input_examples(train_data, use_negatives)
    logger.info("Training examples: %d", len(train_examples))

    # ── Load model ──
    # If resuming, load from checkpoint; else load base model
    checkpoint_state = None
    if resume:
        checkpoint_state = load_checkpoint(ckpt_dir)
        if checkpoint_state:
            logger.info("Resuming from checkpoint: epoch %d, step %d",
                        checkpoint_state["epoch"], checkpoint_state["global_step"])
            model = SentenceTransformer(checkpoint_state["model_path"])
        else:
            logger.info("No checkpoint found — starting fresh from base model")
            model = SentenceTransformer(base_model)
    else:
        model = SentenceTransformer(base_model)

    model.max_seq_length = max_seq_length
    model.to(device)

    logger.info("Model: %d params, %d-dim embeddings, max_seq=%d",
                sum(p.numel() for p in model.parameters()),
                model.get_sentence_embedding_dimension(),
                max_seq_length)

    # ── Build DataLoader ──
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )
    train_dataloader.collate_fn = model.smart_batching_collate

    # ── Loss function ──
    train_loss = losses.MultipleNegativesRankingLoss(model)
    train_loss.to(device)

    # ── Optimizer and scheduler ──
    steps_per_epoch = len(train_dataloader) // gradient_accumulation
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(
        list(model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-6,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # FP16 scaler
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # ── Restore optimizer state if resuming ──
    start_epoch = 0
    global_step = 0
    best_score = 0.0
    loss_history: list[float] = []

    if checkpoint_state and checkpoint_state.get("optimizer_data"):
        opt_data = checkpoint_state["optimizer_data"]
        optimizer.load_state_dict(opt_data["optimizer_state_dict"])
        if opt_data.get("scheduler_state_dict"):
            scheduler.load_state_dict(opt_data["scheduler_state_dict"])
        if scaler and opt_data.get("scaler_state_dict"):
            scaler.load_state_dict(opt_data["scaler_state_dict"])
        start_epoch = checkpoint_state["epoch"] + 1
        global_step = checkpoint_state["global_step"]
        best_score = checkpoint_state.get("best_score", 0.0)
        loss_history = checkpoint_state.get("loss_history", [])
        logger.info("Optimizer state restored. Resuming from epoch %d, step %d",
                    start_epoch, global_step)

    # ── Evaluator ──
    evaluator = build_evaluator(val_data)
    if evaluator:
        logger.info("Validation evaluator ready (%d sentence pairs)", len(val_data) * 2)
    else:
        logger.info("No validation data — save-best-model disabled")

    # ── SIGINT/SIGTERM handler — save checkpoint before exit ──
    _interrupt_requested = {"flag": False}

    def _handle_signal(sig, frame):
        if _interrupt_requested["flag"]:
            logger.warning("Second interrupt — forcing exit")
            sys.exit(1)
        _interrupt_requested["flag"] = True
        logger.warning("Interrupt received (signal %d). Will save checkpoint after current batch...", sig)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Training log file ──
    log_file = Path(log_dir) / "training_log.jsonl"

    def _log_step(epoch: int, step: int, loss: float, lr: float, score: float | None = None):
        entry = {"epoch": epoch, "step": step, "loss": loss, "lr": lr}
        if score is not None:
            entry["val_score"] = score
        with log_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    logger.info("Starting training: epochs=%d, batch=%d, grad_accum=%d, lr=%s, warmup=%d",
                epochs, batch_size, gradient_accumulation, learning_rate, warmup_steps)
    logger.info("Checkpoint every %d steps and after each epoch → %s", save_steps, ckpt_dir)
    if checkpoint_state:
        logger.info("Skipping epochs 0-%d (already completed)", start_epoch - 1)

    # ── Epoch loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        accum_loss = 0.0
        accum_count = 0

        logger.info("\n── Epoch %d/%d ─────────────────────────────────────────", epoch + 1, epochs)

        try:
            from tqdm import tqdm
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        except ImportError:
            pbar = train_dataloader

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Check for interrupt
            if _interrupt_requested["flag"]:
                logger.info("Saving emergency checkpoint before exit...")
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch=epoch, global_step=global_step,
                    best_score=best_score, loss_history=loss_history,
                    checkpoint_dir=ckpt_dir, label="emergency",
                )
                logger.info("Emergency checkpoint saved. Exiting.")
                sys.exit(0)

            # Unpack batch and move to device
            features, labels = batch
            features = [
                {k: v.to(device) if hasattr(v, "to") else v for k, v in f.items()}
                for f in features
            ]
            if hasattr(labels, "to"):
                labels = labels.to(device)

            # Forward pass
            if use_fp16:
                with torch.cuda.amp.autocast():
                    loss_value = train_loss(features, labels)
            else:
                loss_value = train_loss(features, labels)

            # Scale loss for gradient accumulation
            loss_scaled = loss_value / gradient_accumulation

            if use_fp16:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accum_loss += loss_value.item()
            accum_count += 1

            # Optimizer step (every gradient_accumulation batches)
            if (batch_idx + 1) % gradient_accumulation == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                avg_batch_loss = accum_loss / accum_count
                accum_loss = 0.0
                accum_count = 0
                global_step += 1
                epoch_loss += avg_batch_loss
                epoch_steps += 1
                loss_history.append(avg_batch_loss)

                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else learning_rate

                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({
                        "loss": f"{avg_batch_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": global_step,
                    })

                _log_step(epoch, global_step, avg_batch_loss, current_lr)

                # Step-level checkpoint
                if global_step > 0 and global_step % save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        epoch=epoch, global_step=global_step,
                        best_score=best_score, loss_history=loss_history,
                        checkpoint_dir=ckpt_dir, label=f"step_{global_step}",
                    )
                    # Update "latest" pointer
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        epoch=epoch, global_step=global_step,
                        best_score=best_score, loss_history=loss_history,
                        checkpoint_dir=ckpt_dir, label="latest",
                    )

                # Mid-epoch evaluation
                if eval_steps > 0 and global_step > 0 and global_step % eval_steps == 0:
                    if evaluator:
                        val_score = run_evaluation(model, evaluator, ckpt_dir, global_step)
                        logger.info("Step %d eval: val_score=%.4f (best=%.4f)", global_step, val_score, best_score)
                        _log_step(epoch, global_step, avg_batch_loss, current_lr, val_score)
                        if val_score > best_score:
                            best_score = val_score
                            model.save(str(output_path))
                            logger.info("New best model saved to %s", output_path)

        # ── End of epoch ──
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)

        # Epoch-level evaluation
        val_score = 0.0
        if evaluator:
            val_score = run_evaluation(model, evaluator, ckpt_dir, global_step)
            if val_score > best_score:
                best_score = val_score
                model.save(str(output_path))
                logger.info("New best model saved to %s (score=%.4f)", output_path, best_score)

        logger.info("Epoch %d/%d complete — loss=%.4f, val_score=%.4f, best=%.4f",
                    epoch + 1, epochs, avg_epoch_loss, val_score, best_score)

        # Save epoch checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch=epoch, global_step=global_step,
            best_score=best_score, loss_history=loss_history,
            checkpoint_dir=ckpt_dir, label=f"epoch_{epoch}",
        )
        # Always update "latest" after each epoch
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch=epoch, global_step=global_step,
            best_score=best_score, loss_history=loss_history,
            checkpoint_dir=ckpt_dir, label="latest",
        )

        _log_step(epoch, global_step, avg_epoch_loss, learning_rate, val_score)

    # ── Training complete ──
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("  Epochs trained:      %d", epochs - start_epoch)
    logger.info("  Total steps:         %d", global_step)
    logger.info("  Best val score:      %.4f", best_score)
    logger.info("  Best model at:       %s", output_path)
    logger.info("  Checkpoints at:      %s", ckpt_dir)

    # If no evaluator, save final model as "best"
    if not evaluator:
        model.save(str(output_path))
        logger.info("  Saved final model to %s", output_path)

    # Print model stats
    reload = SentenceTransformer(str(output_path))
    dim = reload.get_sentence_embedding_dimension()
    params = sum(p.numel() for p in reload.parameters())
    size_mb = sum(p.nelement() * p.element_size() for p in reload.parameters()) / 1e6

    print(f"\nModel saved: {output_path}")
    print(f"  Parameters:    {params:,}")
    print(f"  Embedding dim: {dim}")
    print(f"  Size (FP32):   ~{size_mb:.1f} MB")
    print(f"\nNext steps:")
    print(f"  Benchmark: python scripts/evaluate_sfu_queries.py --custom-model {output_path}")
    print(f"  Quantize:  python scripts/quantize_model.py --model {output_path} --validate")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model for SFU academic search (with checkpoint/resume)"
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training JSONL file (e.g. data/splits/train.jsonl)")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Path to validation JSONL file (e.g. data/splits/val.jsonl)")
    parser.add_argument("--output", type=str, default="models/sfu-academic-embed-v1",
                        help="Output directory for the best model")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (default: {output}/checkpoints)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in --checkpoint-dir")
    parser.add_argument("--base-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Base model to fine-tune (ignored if --resume)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed-precision training (GPU only)")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="Accumulate gradients over N batches (effective batch = batch_size * N)")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save a checkpoint every N optimizer steps")
    parser.add_argument("--eval-steps", type=int, default=500,
                        help="Evaluate on validation set every N steps (0 to disable)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training to first N samples (for testing)")
    parser.add_argument("--no-negatives", action="store_true",
                        help="Ignore negative examples in training data")
    parser.add_argument("--log-dir", type=str, default="logs/sfu-embed-training",
                        help="TensorBoard / JSONL log directory")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        val_data_path=args.val_data,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        fp16=args.fp16,
        gradient_accumulation=args.gradient_accumulation,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        max_samples=args.max_samples,
        use_negatives=not args.no_negatives,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
