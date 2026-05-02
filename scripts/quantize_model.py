#!/usr/bin/env python3
"""Export and quantize a sentence-transformer model to ONNX INT8.

Takes a fine-tuned (or off-the-shelf) sentence-transformer model and:
1. Exports to ONNX format
2. Quantizes to INT8 (dynamic quantization)
3. Validates the quantized model produces similar embeddings

Final INT8 model is ~22MB (vs ~86MB FP32), uses ~50MB RAM at inference.

Usage:
    python scripts/quantize_model.py --model models/sfu-academic-embed-v1
    python scripts/quantize_model.py --model sentence-transformers/all-MiniLM-L6-v2
    python scripts/quantize_model.py --model models/sfu-academic-embed-v1 --validate
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def export_to_onnx(model_path: str, output_dir: str) -> Path:
    """Export a sentence-transformer model to ONNX format."""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting %s to ONNX...", model_path)

    # For sentence-transformers, the underlying transformer is in a subdirectory
    # Try loading directly first
    try:
        model = ORTModelForFeatureExtraction.from_pretrained(model_path, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        # If it's a sentence-transformers model, look for the transformer subdir
        st_model_path = Path(model_path)
        if (st_model_path / "1_Pooling").exists():
            # Copy pooling config alongside ONNX model
            model = ORTModelForFeatureExtraction.from_pretrained(model_path, export=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            pooling_src = st_model_path / "1_Pooling"
            pooling_dst = output_path / "1_Pooling"
            if pooling_src.exists() and not pooling_dst.exists():
                shutil.copytree(str(pooling_src), str(pooling_dst))
        else:
            raise

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Copy sentence-transformers config files if they exist
    for config_file in ["sentence_bert_config.json", "config_sentence_transformers.json", "modules.json"]:
        src = Path(model_path) / config_file
        if src.exists():
            shutil.copy2(str(src), str(output_path / config_file))

    logger.info("ONNX model exported to %s", output_path)
    return output_path


def quantize_to_int8(onnx_dir: str, output_dir: str) -> Path:
    """Quantize an ONNX model to INT8 using dynamic quantization."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Quantizing to INT8...")

    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    quantizer.quantize(
        save_dir=output_path,
        quantization_config=qconfig,
    )

    # Copy tokenizer and config files
    onnx_path = Path(onnx_dir)
    for f in onnx_path.iterdir():
        if f.suffix in (".json", ".txt") and not (output_path / f.name).exists():
            shutil.copy2(str(f), str(output_path / f.name))

    # Copy pooling config
    pooling_src = onnx_path / "1_Pooling"
    pooling_dst = output_path / "1_Pooling"
    if pooling_src.exists() and not pooling_dst.exists():
        shutil.copytree(str(pooling_src), str(pooling_dst))

    logger.info("INT8 model saved to %s", output_path)
    return output_path


def validate_quantized_model(original_path: str, quantized_path: str, tolerance: float = 0.02):
    """Validate that quantized model produces similar embeddings to original."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    logger.info("Validating quantized model...")

    test_texts = [
        "CRISPR gene editing in Atlantic salmon embryos",
        "Machine learning for protein structure prediction",
        "Climate change effects on boreal forest ecosystems",
        "Quantum error correction using surface codes",
        "Deep learning approaches for drug discovery",
    ]

    # Encode with original model
    original_model = SentenceTransformer(original_path)
    original_embs = original_model.encode(test_texts, normalize_embeddings=True)

    # Try to load quantized model via sentence-transformers
    try:
        quantized_model = SentenceTransformer(quantized_path)
        quantized_embs = quantized_model.encode(test_texts, normalize_embeddings=True)
    except Exception as e:
        logger.warning("Could not load quantized model via sentence-transformers: %s", e)
        logger.info("Quantized model may need ONNX Runtime for inference")
        return False

    # Compare embeddings
    cosine_sims = []
    for i in range(len(test_texts)):
        sim = np.dot(original_embs[i], quantized_embs[i])
        cosine_sims.append(sim)

    mean_sim = np.mean(cosine_sims)
    min_sim = np.min(cosine_sims)

    print(f"\nValidation Results:")
    print(f"  Mean cosine similarity (original vs quantized): {mean_sim:.6f}")
    print(f"  Min cosine similarity: {min_sim:.6f}")
    print(f"  Tolerance: {tolerance}")

    if min_sim >= (1.0 - tolerance):
        print(f"  PASS: Quantized model embeddings are within tolerance")
        return True
    else:
        print(f"  WARN: Some embeddings differ more than tolerance")
        return False


def print_size_comparison(original_path: str, onnx_path: str, int8_path: str):
    """Print model size comparison."""
    def dir_size(path: str) -> float:
        total = 0
        for f in Path(path).rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total / (1024 * 1024)

    sizes = {}
    for name, path in [("Original (FP32)", original_path), ("ONNX (FP32)", onnx_path), ("ONNX (INT8)", int8_path)]:
        p = Path(path)
        if p.exists():
            sizes[name] = dir_size(path)

    if sizes:
        print(f"\nModel Size Comparison:")
        for name, size in sizes.items():
            print(f"  {name}: {size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export and quantize embedding model to ONNX INT8")
    parser.add_argument("--model", type=str, required=True, help="Path to sentence-transformer model")
    parser.add_argument("--onnx-output", type=str, default=None, help="ONNX output dir (default: {model}-onnx)")
    parser.add_argument("--int8-output", type=str, default=None, help="INT8 output dir (default: {model}-int8)")
    parser.add_argument("--validate", action="store_true", help="Validate quantized model")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export (already done)")
    args = parser.parse_args()

    model_name = args.model.rstrip("/")
    onnx_dir = args.onnx_output or f"{model_name}-onnx"
    int8_dir = args.int8_output or f"{model_name}-int8"

    if not args.skip_onnx:
        export_to_onnx(args.model, onnx_dir)

    quantize_to_int8(onnx_dir, int8_dir)
    print_size_comparison(args.model, onnx_dir, int8_dir)

    if args.validate:
        validate_quantized_model(args.model, int8_dir)

    print(f"\nDone! INT8 model ready at: {int8_dir}")
    print(f"Set SFU_EMBEDDING_MODEL_PATH={int8_dir} to use it.")


if __name__ == "__main__":
    main()
