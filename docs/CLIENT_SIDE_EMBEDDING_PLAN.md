# Client-Side Embedding Model — Feasibility Plan

## Goal

Train or fine-tune a small embedding model for academic paper search that:
1. Runs locally on any hardware (8GB RAM, no GPU required at inference)
2. Replaces dependence on the Semantic Scholar API for SPECTER2 embeddings
3. Approaches SPECTER2 quality for academic retrieval/reranking
4. Can be trained on a consumer gaming PC (e.g., RTX 3070/4070 with 8-12GB VRAM)

---

## Current State

### What Already Exists (from docs)

The thin client architecture (THIN_CLIENT_NEW_ARCHITECTURE.md) currently relies on:
- **SPECTER2 embeddings from Semantic Scholar API** — free, 768-dim vectors, zero local RAM
- **Stage 2 reranking** — cosine similarity between query and paper embeddings
- **No local embedding compute** — all semantic understanding comes from the API

### The Problem with API Dependence

| Issue | Impact |
|-------|--------|
| Semantic Scholar rate limit (1 req/s) | Bottleneck for batch reranking |
| API goes down → no semantic reranking | Single point of failure |
| No offline capability | Can't work without internet |
| 768-dim vectors require API round-trip | Adds 300-800ms latency per search |
| Can't customize for SFU-specific domains | Generic academic model, no institutional tuning |

---

## Feasibility Analysis

### Is Training Worth It? (The "1% more accuracy" Question)

Before building anything, we need to benchmark the gap:

| Scenario | Model | NDCG@10 (estimated) | RAM | Latency |
|----------|-------|---------------------|-----|---------|
| **No embedding rerank** | None (BM25 only) | ~0.66 | 0 | 0ms |
| **General small model (off-the-shelf)** | all-MiniLM-L6-v2 (43MB) | ~0.70 | 80MB | <5ms |
| **Academic small model (fine-tuned)** | MiniLM fine-tuned on academic pairs | ~0.73-0.76 | 80MB | <5ms |
| **SPECTER2 via API** | SPECTER2 (768-dim) | ~0.71 | 0 (API) | 300-800ms |
| **SPECTER2 locally** | allenai/specter2 (110M params) | ~0.71 | ~450MB | ~15ms |

**Key insight:** A fine-tuned small model (22M params, 43MB) could potentially *match or exceed* SPECTER2 on the specific task of SFU query-to-paper matching, because:
1. SPECTER2 is optimized for paper-to-paper similarity (citation prediction), not query-to-paper retrieval
2. A model fine-tuned on actual search queries → relevant papers will be better at the actual task
3. The 384-dim output of MiniLM is sufficient for reranking 20-50 results

### Decision Point

**Run this benchmark FIRST before committing to training:**
```bash
# Compare: off-the-shelf vs SPECTER2 API on 50 real queries
# If gap < 5% NDCG → don't fine-tune, use off-the-shelf
# If gap > 5% NDCG → fine-tuning is worth the effort
```

---

## Proposed Architecture

### Option A: Small Transformer (Recommended Starting Point)

```
User Query → Qwen3-1.7B (parse) → Structured Query → OpenAlex BM25 (50 results)
                                                          │
                                                          ▼
                                              ┌─────────────────────────┐
                                              │  LOCAL EMBEDDING MODEL   │
                                              │  all-MiniLM-L6-v2       │
                                              │  (or fine-tuned variant) │
                                              │  22M params, 43-66MB    │
                                              │  384-dim embeddings     │
                                              │  Inference: <5ms/doc    │
                                              └────────────┬────────────┘
                                                           │
                                              Cosine sim (query ↔ papers)
                                                           │
                                                           ▼
                                              Top 20 → Multi-Signal Scoring → Top 10
```

**RAM budget:**
- Qwen3-1.7B (query parsing): ~1.5GB
- Embedding model (reranking): ~80-130MB
- App + Ollama overhead: ~0.5GB
- **Total: ~2.1-2.2GB** — same as current plan

### Option B: Non-Transformer / Ensemble (Explored but Not Recommended)

As discussed in the Discord conversation, non-transformer methods could work:

| Method | Pros | Cons |
|--------|------|------|
| TF-IDF + SVD (LSA) | Tiny (<5MB), fast, no deps | Low semantic quality, basically fancy BM25 |
| Word2Vec averaging | Small (~100MB vocab), fast | No contextual understanding, poor for phrases |
| Stacked ensemble (BM25 + TF-IDF + metadata features) | No model to train, interpretable | Can't capture semantic similarity |
| ONNX MiniLM (quantized) | 11MB INT8, runs on anything | Slightly lower quality than FP16 |

**Verdict:** Transformer-based models at the 22-110M param range are already small enough to run locally on anything. Going smaller (non-transformer) sacrifices too much quality for marginal resource savings. The embedding model is NOT the bottleneck — it uses 80-130MB vs. 1.5GB for the query LLM.

### Option C: Run SPECTER2 Locally (Middle Ground)

```python
# SPECTER2 is a SciBERT-based model (110M params)
# Can be run locally with sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("allenai/specter2")
embeddings = model.encode(["CRISPR gene editing in salmon"])
```

| Property | Value |
|----------|-------|
| Model size | ~440MB (FP16) or ~220MB (INT8 quantized) |
| RAM usage | ~450MB (FP16) or ~300MB (INT8) |
| Inference speed | ~15ms per document (CPU) |
| Quality | Same as API (it IS the same model) |
| Offline? | Yes |

**This eliminates the API dependency with zero quality loss.** The question becomes: is 450MB acceptable?

---

## Recommended Path: Phased Approach

### Phase 1: Baseline (1-2 days)

**Goal:** Establish whether fine-tuning is worth it.

1. Pick 50 real academic queries relevant to SFU users
2. Run each through OpenAlex → get top 50 results
3. Score results using:
   - (a) No embedding (BM25 rank only)
   - (b) all-MiniLM-L6-v2 (off-the-shelf, 43MB)
   - (c) SPECTER2 via API
   - (d) SPECTER2 locally (sentence-transformers)
4. Manually judge relevance (or use citation-based proxy: if paper A cites paper B, they're related)
5. Compare NDCG@10 across all four

**If (b) is within 3% of (c/d):** Use MiniLM off-the-shelf. Done. No training needed.
**If (b) is 5%+ below (c/d):** Proceed to Phase 2.
**If (d) runs fine locally:** Just use SPECTER2 locally. Done. No training needed.

### Phase 2: Fine-Tuning (1-2 weeks)

**Goal:** Train a small model that matches SPECTER2 on academic query-to-paper retrieval.

#### Training Data Generation

You need (query, positive_paper, negative_paper) triplets:

```python
# Strategy 1: Use OpenAlex search logs
# For each query → top-clicked result = positive
# Random result from page 2+ = hard negative

# Strategy 2: Synthetic from paper metadata
# Take a paper title/abstract → generate a "query" that would find it
# Use Qwen3 or Claude to generate natural language queries:
#   Paper: "CRISPR-Cas9 knockout in Atlantic salmon embryos"
#   Query: "gene editing techniques for salmon aquaculture"

# Strategy 3: Citation-based (self-supervised, no labeling)
# Paper A cites Paper B → (A_abstract, B_abstract) = positive pair
# Paper A doesn't cite Paper C → (A_abstract, C_abstract) = negative pair
# This is how SPECTER2 was trained — replicate at smaller scale
```

**Target dataset size:** 5,000-20,000 triplets (sufficient for LoRA fine-tuning of a small model)

#### Training Setup

```bash
# On gaming PC (RTX 3070/4070, 8-12GB VRAM)
# Training time: 2-6 hours for a 22M param model

pip install sentence-transformers unsloth

# Base model: all-MiniLM-L6-v2 (22M params, 384-dim output)
# Fine-tuning method: Full fine-tune (model is small enough)
# Loss: MultipleNegativesRankingLoss (standard for retrieval)
# Epochs: 3-5
# Batch size: 64-128 (fits easily in 8GB VRAM)
```

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Training data: (query, positive_passage) pairs
train_examples = [
    InputExample(texts=["CRISPR salmon aquaculture", 
                        "CRISPR-Cas9 knockout efficiency in Atlantic salmon embryos..."]),
    # ... 5000+ examples
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./sfu-academic-embed-v1"
)
```

#### Why NOT Unsloth Here

Unsloth is designed for fine-tuning **generative LLMs** (Qwen, Llama, etc.) with LoRA. Embedding models use a different training paradigm (contrastive learning, not next-token prediction). Use `sentence-transformers` library directly — it's purpose-built for this.

**Unsloth IS useful for:** Fine-tuning Qwen3-1.7B for better query parsing (Phase 8 in existing plan).

### Phase 3: Quantization & Deployment (1-2 days)

```bash
# Convert to ONNX for fastest CPU inference
optimum-cli export onnx --model ./sfu-academic-embed-v1 ./sfu-academic-embed-v1-onnx

# Quantize to INT8 (halves model size, minimal quality loss)
optimum-cli onnxruntime quantize --onnx_model ./sfu-academic-embed-v1-onnx \
    --output ./sfu-academic-embed-v1-int8 --quantization_approach dynamic
```

**Final model sizes:**
| Format | Size | RAM | Quality vs FP32 |
|--------|------|-----|-----------------|
| FP32 (original) | 86MB | ~130MB | Baseline |
| FP16 | 43MB | ~80MB | ~99.9% |
| INT8 (ONNX) | 22MB | ~50MB | ~99.5% |
| INT4 (experimental) | 11MB | ~30MB | ~98% |

**The INT8 model at 22MB + 50MB RAM is the sweet spot for "any hardware."**

### Phase 4: Integration (1-2 days)

Replace the Semantic Scholar API call in Stage 2 with local inference:

```python
# In reranker.py (upgraded)
from sentence_transformers import SentenceTransformer

# Load once on startup (~50MB RAM, INT8 ONNX)
_embed_model = SentenceTransformer("./models/sfu-academic-embed-v1-int8")

def _compute_semantic_scores(query: str, papers: list[dict]) -> list[float]:
    """Local embedding-based reranking. No API call needed."""
    texts = [query] + [p.get("title", "") + " " + p.get("abstract", "") for p in papers]
    embeddings = _embed_model.encode(texts, normalize_embeddings=True)
    query_emb = embeddings[0]
    paper_embs = embeddings[1:]
    # Cosine similarity (embeddings are normalized → dot product)
    return (paper_embs @ query_emb).tolist()
```

**Latency comparison:**
| Method | Latency (20 papers) | Offline? |
|--------|---------------------|----------|
| Semantic Scholar API | 300-800ms | No |
| SPECTER2 local (FP16) | ~15ms | Yes |
| MiniLM fine-tuned (INT8) | ~3ms | Yes |

---

## Hardware Requirements

### For Inference (End Users)

| Component | RAM | Notes |
|-----------|-----|-------|
| Qwen3-1.7B (query parsing) | ~1.5GB | Via Ollama |
| Embedding model (INT8 ONNX) | ~50MB | Loaded in Python |
| App + overhead | ~0.5GB | Next.js + Python |
| **Total** | **~2.1GB** | Runs on 4GB+ machines |

### For Training (Your Gaming PC)

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | RTX 3060 (12GB VRAM) | RTX 3070/4070 (8-12GB) |
| RAM | 16GB | 32GB |
| Disk | 20GB free | 50GB free |
| Training time | 2-6 hours | 1-3 hours (better GPU) |
| Dataset prep | 1-3 days | — |

---

## Comparison: All Options Side-by-Side

| Option | Quality (est.) | Model Size | RAM | Latency | Offline | Training Needed | Complexity |
|--------|---------------|------------|-----|---------|---------|-----------------|------------|
| No embedding (BM25 only) | Baseline | 0 | 0 | 0 | Yes | No | None |
| all-MiniLM-L6-v2 (off-the-shelf) | +5-8% | 43MB | 80MB | <5ms | Yes | No | Low |
| **MiniLM fine-tuned (recommended)** | **+10-15%** | **22MB (INT8)** | **50MB** | **<3ms** | **Yes** | **Yes (6h)** | **Medium** |
| SPECTER2 local | +8-10% | 440MB | 450MB | ~15ms | Yes | No | Low |
| SPECTER2 API (current plan) | +8-10% | 0 | 0 | 300-800ms | No | No | None |
| Custom large model (overkill) | +12-18% | 1-4GB | 1-4GB | 50-200ms | Yes | Yes (days) | High |

---

## Recommendation

### Start Here (Zero Training)

1. **Use all-MiniLM-L6-v2 off-the-shelf** (43MB, 80MB RAM, <5ms)
2. Run the Phase 1 benchmark against SPECTER2
3. If the gap is small → ship it, you're done
4. Keep SPECTER2 API as optional enhancement (user setting)

### If Gap Is Significant (Fine-Tune)

1. Generate 5,000+ training triplets using citation data + synthetic queries
2. Fine-tune MiniLM on your gaming PC (2-6 hours)
3. Quantize to INT8 ONNX (22MB, 50MB RAM)
4. Benchmark: if it beats SPECTER2 on your test set → ship it
5. Set up periodic re-tuning (Phase 8 from THIN_CLIENT_PLAN) using accumulated user queries

### What NOT to Do

- Don't train a model from scratch — fine-tuning existing models is vastly more efficient
- Don't target >100M params — diminishing returns for reranking 20-50 results
- Don't use Unsloth for the embedding model — it's for generative LLMs, not encoders
- Don't skip the benchmark — you might spend weeks for 1% improvement
- Don't build a custom training pipeline — sentence-transformers handles everything

---

## Integration with Existing Architecture

```
THIN_CLIENT_NEW_ARCHITECTURE.md (updated Stage 2):

Stage 2: Local Semantic Rerank (REPLACES Semantic Scholar dependency)
  ├── Model: sfu-academic-embed-v1 (fine-tuned MiniLM, 22MB INT8)
  ├── Input: query + 50 paper titles/abstracts from OpenAlex
  ├── Output: cosine similarity scores → reorder → keep top 20
  ├── Latency: <3ms (vs 300-800ms API)
  ├── Offline: Yes (fully local)
  └── Fallback: all-MiniLM-L6-v2 off-the-shelf if custom model unavailable

Semantic Scholar API (OPTIONAL, no longer required):
  ├── Still useful for: TLDR summaries, citation counts, citation graph
  ├── NOT needed for: embedding-based reranking
  └── Feature flag: SFU_FEATURE_SEMANTIC_SCHOLAR_ENABLED=true/false
```

---

## Training Data Sources (Free, No Labeling)

| Source | What You Get | Volume | Method |
|--------|-------------|--------|--------|
| OpenAlex citation graph | (paper_A, cited_paper_B) positive pairs | Millions available | Self-supervised |
| OpenAlex related_works | (paper, related_paper) positive pairs | Built-in to API | Self-supervised |
| Semantic Scholar TLDR | (TLDR_as_query, paper_abstract) pairs | 80M+ | Synthetic queries |
| Your own query logs | (user_query, clicked_result) pairs | Grows over time | Implicit feedback |
| Claude/Qwen synthetic | Generate search queries for papers | As many as needed | LLM-generated |

**Best strategy:** Start with 10,000 OpenAlex citation pairs (free, automated, no human labeling) + 1,000 synthetic queries generated by Claude. Fine-tune. Evaluate. Iterate.

---

## Timeline

| Week | Task | Output |
|------|------|--------|
| 1 | Benchmark off-the-shelf vs SPECTER2 API on 50 queries | Decision: fine-tune or not |
| 2 | If fine-tuning: generate training data (citations + synthetic) | 10K+ triplets |
| 3 | Train on gaming PC + quantize to ONNX INT8 | sfu-academic-embed-v1 (22MB) |
| 4 | Integrate into thin client reranker + benchmark | Latency + quality numbers |
| 5 | Ship as default, SPECTER2 API becomes optional | Client-side reranking live |

**Total effort:** ~3-5 weeks part-time, mostly automated data generation.
**Cost:** $0 (all free tools + your own GPU).
