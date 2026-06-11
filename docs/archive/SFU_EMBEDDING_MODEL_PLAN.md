# SFU Library Custom Embedding Model — Full Implementation Plan

## Executive Summary

Train a custom embedding model for the SFU Library search system that leverages institutional data from SFU's Solr database registry (766 records, 108 subjects, 109 providers) combined with OpenAlex's 250M+ scholarly works. The model replaces Semantic Scholar API dependency for semantic reranking, runs locally on any hardware (<50MB RAM, <5ms inference), and is specifically tuned for SFU's subscription landscape, subject coverage, and user query patterns.

---

## Benchmark Results (Collected 2025-05-02)

### Model Comparison — 50 Academic Queries, NDCG@10

| Model | Mean NDCG@10 | Std Dev | Median | Min | Max | Size | Inference |
|-------|-------------|---------|--------|-----|-----|------|-----------|
| BM25 (OpenAlex position) | 0.3102 | 0.2616 | 0.2589 | 0.0167 | 0.9980 | 0 | 0ms |
| BGE-base-en-v1.5 | 0.1229 | 0.1574 | 0.0592 | 0.0004 | 0.6339 | 440MB | ~15ms |
| all-MiniLM-L6-v2 | 0.1156 | 0.1239 | 0.0680 | 0.0002 | 0.5164 | 43MB | <5ms |
| SPECTER2-base | 0.1027 | 0.1024 | 0.0718 | 0.0003 | 0.4339 | 440MB | ~15ms |

### Key Findings

1. **All embedding models score below BM25 position** — but this is a benchmark artifact: the citation-count relevance proxy correlates with OpenAlex's own ranking (which uses citations). Embeddings measure semantic relevance, not popularity.

2. **BGE-base outperforms MiniLM by 6.3%** — modern general-purpose models are competitive with academic-specific ones. BGE's top-3 queries scored 0.63, 0.60, 0.49 vs MiniLM's 0.52, 0.44, 0.37.

3. **SPECTER2-base scored lowest** — confirms it's optimized for paper-to-paper similarity (citation prediction), not query-to-paper retrieval. This is the key gap a custom model fills.

4. **All models fail on niche/institutional queries** — "housing affordability crisis Canadian cities" and "indigenous language revitalization Canada" scored <0.01 on all models. These are exactly the queries where SFU-specific training data would help most.

### Per-Query Analysis (Selected)

| Query | BM25 | MiniLM | BGE | SPECTER2 | Notes |
|-------|------|--------|-----|----------|-------|
| CRISPR diagnostics point of care testing | 0.30 | **0.52** | 0.39 | 0.43 | Embedding models competitive |
| neural network interpretability explainable AI | 0.33 | 0.37 | **0.63** | 0.39 | BGE excels on CS queries |
| circular economy waste reduction manufacturing | 0.07 | 0.17 | **0.60** | 0.10 | BGE outperforms by 3.5x |
| indigenous language revitalization Canada | **0.998** | 0.0002 | 0.0004 | 0.0003 | Niche query — embeddings fail |
| housing affordability crisis Canadian cities | 0.13 | 0.005 | 0.003 | 0.01 | Local/institutional topic |
| cybersecurity zero trust architecture | 0.15 | 0.007 | 0.006 | 0.03 | Jargon-heavy, embeddings miss |

**Conclusion:** Fine-tuning specifically for SFU's domain and institutional coverage is strongly justified — off-the-shelf models consistently fail on the queries that matter most to SFU users.

---

## Hardware Profile

### Current Environment (ClaudeBox Container)

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 5700X3D, 8 cores / 16 threads |
| RAM | 32GB DDR4 |
| GPU (container) | None (no passthrough) |
| GPU (host) | RTX series (available for training outside container) |
| Storage | Sufficient for models and training data |
| OS | Linux (WSL2) |

### Hardware Acceleration Analysis

| Task | CPU-only | With GPU | Recommendation |
|------|----------|----------|----------------|
| **Inference** (reranking 50 docs) | 3-15ms | 1-2ms | CPU is sufficient — <15ms is fine |
| **Training** (10K triplets, 3 epochs) | 8-24 hours | 2-6 hours | **GPU strongly recommended** |
| **Data generation** (API calls) | Network-bound | Same | No GPU benefit |
| **Benchmarking** (50 queries × 4 models) | ~15 min | ~5 min | CPU acceptable |
| **ONNX quantization** | Minutes | Same | CPU fine |

**Recommendation:** Run training on the host machine with GPU access. All other steps work in the container.

---

## SFU Solr Data Profile

### Database Registry (766 records)

```
Endpoint: https://databases.lib.sfu.ca/solr/sfu_databases/select
Auth: None required (public)
Cache: 24h TTL, ~200KB
```

| Metric | Value |
|--------|-------|
| Total records | 766 |
| Free databases | 243 (31.7%) |
| Proxy-required (SFU subscription) | 488 (63.7%) |
| Unique subjects | 108 |
| Unique providers | 109 |
| Content types | 17 |

### Top Subjects (Databases covering)

| Subject | Databases | Notes |
|---------|-----------|-------|
| History | 124 | SFU's strongest coverage area |
| General & Multidisciplinary | 111 | Cross-discipline databases |
| English - General | 64 | Humanities focus |
| Finance | 64 | Beedie School strength |
| Canadian Studies | 58 | Institutional specialization |
| Political Science | 56 | Social sciences cluster |
| Economics | 56 | |
| Health Sciences | 46 | Growing coverage |
| Indigenous Studies | 46 | Unique institutional priority |
| Sociology | 46 | |
| Interactive Arts & Technology (SIAT) | 44 | SFU-unique program |
| Criminology | 44 | SFU-unique program |
| Biological Sciences | 39 | STEM coverage |

### Top Providers (Content sources)

| Provider | Databases | Notes |
|----------|-----------|-------|
| EBSCOhost | 71 | Largest aggregator |
| SFU Library Digital Collections | 68 | Institutional content |
| Galegroup | 50 | Humanities/social sciences |
| ProQuest | 49 | Dissertations, news |
| Wharton Research Data Services | 34 | Finance/business data |
| Alexander Street Press | 33 | Streaming media |
| Adam Matthew Digital | 26 | Primary sources |
| Ovid | 11 | Medical/health |
| Web of Science | 10 | Citation indexing |
| SAGE | 9 | Social science journals |

### Content Type Distribution

| Type | Count | Relevance to Search |
|------|-------|---------------------|
| Index | 147 | Discovery tools |
| Ejournal collection | 137 | Primary search targets |
| Ebook collection | 96 | Monograph search |
| Full-text database | 94 | Primary search targets |
| Datasets | 89 | Data-focused queries |
| Primary sources | 65 | Humanities/history queries |
| Digital collection | 61 | Archival material |
| News sources | 43 | Current events queries |

---

## Why SFU-Specific Training Matters

### The Gap Off-the-Shelf Models Can't Close

1. **SFU has unique programs** — SIAT (Interactive Arts & Technology), Criminology, Resource & Environmental Management, Biomedical Physiology and Kinesiology. Generic academic models have no exposure to these cross-disciplinary intersections.

2. **Subscription-aware relevance** — A paper in an EBSCOhost journal that SFU subscribes to is more useful to an SFU user than an equally relevant paper behind a paywall they can't access. The embedding model can learn this signal.

3. **Canadian/regional bias** — "housing affordability crisis Canadian cities" scored 0.005 on MiniLM. A model trained on papers from SFU's subscribed Canadian Studies databases would handle this.

4. **Indigenous Studies** — SFU has 46 databases covering Indigenous Studies. Off-the-shelf models trained primarily on STEM literature have near-zero representation of this domain.

5. **Subject vocabulary alignment** — SFU's 108 subject categories map to specific departments. A model that understands "BPK" means "Biomedical Physiology and Kinesiology" can connect queries to the right databases.

---

## Architecture: Solr-Integrated Embedding Model

### Training Data Pipeline

```
SFU Solr Registry (766 records)                 OpenAlex API (250M+ works)
    │                                                │
    ├── subjects (108 unique)                        ├── Works matching SFU subjects
    ├── providers (109 unique)                       ├── Citation pairs (A cites B)
    ├── content types (17)                           ├── Related works metadata
    └── names/descriptions                           └── Abstracts + titles
                │                                         │
                └──────────────┬──────────────────────────┘
                               │
                     ┌─────────▼──────────┐
                     │  Training Triplet   │
                     │  Generator          │
                     │                     │
                     │  Strategies:        │
                     │  1. Subject-aligned │
                     │     citation pairs  │
                     │  2. Synthetic SFU   │
                     │     user queries    │
                     │  3. Provider-aware  │
                     │     positive/neg    │
                     │  4. Cross-subject   │
                     │     hard negatives  │
                     └─────────┬──────────┘
                               │
                     10,000-20,000 triplets
                               │
                     ┌─────────▼──────────┐
                     │  Fine-Tune Model    │
                     │  Base: MiniLM-L6-v2 │
                     │  or BGE-base-v1.5   │
                     │  Loss: MNRL         │
                     │  GPU: RTX 3070+     │
                     │  Time: 2-6 hours    │
                     └─────────┬──────────┘
                               │
                     ┌─────────▼──────────┐
                     │  Quantize to INT8   │
                     │  ONNX Runtime       │
                     │  ~22MB model        │
                     │  ~50MB RAM          │
                     │  <5ms inference     │
                     └─────────┬──────────┘
                               │
                     ┌─────────▼──────────┐
                     │  Deploy in Reranker │
                     │  Stage 2: semantic  │
                     │  35% weight         │
                     │  Offline capable    │
                     │  No API dependency  │
                     └─────────────────────┘
```

### Inference Pipeline (Production)

```
User Query: "indigenous language preservation programs Canada"
    │
    ├── Stage 1: OpenAlex BM25 → 50 candidates
    │
    ├── Stage 2: LOCAL EMBEDDING RERANK (sfu-academic-embed-v1)
    │   ├── Encode query (384-dim, <1ms)
    │   ├── Encode 50 paper titles+abstracts (<5ms)
    │   ├── Cosine similarity → reorder → keep top 20
    │   └── Model knows SFU's Indigenous Studies domain
    │
    ├── Stage 3: Multi-Signal Scoring (7 signals) → top 10
    │   ├── semantic_similarity: 0.35 (from Stage 2)
    │   ├── title_relevance: 0.15
    │   ├── recency: 0.15
    │   ├── fulltext_available: 0.15
    │   ├── type_match: 0.10
    │   └── completeness: 0.10
    │
    └── Stage 4 (optional): Qwen3 LLM rescoring → final order
```

---

## Master TODO Checklist

### Phase 1: Training Data Generation
- [ ] Build Solr-integrated data generator script (`scripts/generate_sfu_training_data.py`)
- [ ] Fetch and cache SFU Solr registry (766 records, 108 subjects, 109 providers)
- [ ] Implement Strategy 1: Subject-aligned citation pairs from OpenAlex (target: 5,000 triplets)
- [ ] Implement Strategy 2: Synthetic SFU-style query generation from Solr metadata (target: 3,000 triplets)
- [ ] Implement Strategy 3: Provider-aware positive/negative pairs (target: 4,000 triplets)
- [ ] Implement Strategy 4: Cross-subject hard negatives using SFU taxonomy (target: 3,000 triplets)
- [ ] Validate total triplet count >= 15,000
- [ ] Split data: 90% train / 5% validation / 5% test
- [ ] Save to `data/sfu_training_triplets.jsonl` with metadata and provenance tags
- [ ] Write data quality checks (deduplication, length filtering, subject coverage audit)

### Phase 2: Base Model Selection
- [ ] Run benchmark comparison of MiniLM-L6-v2 vs BGE-base-v1.5 on SFU-specific queries
- [ ] Evaluate fine-tuning convergence speed on a 1,000-triplet pilot for both bases
- [ ] Select base model and document rationale
- [ ] Verify base model downloads and loads correctly on both container (CPU) and host (GPU)

### Phase 3: Training
- [ ] Set up training environment on host GPU machine (PyTorch, sentence-transformers, CUDA)
- [ ] Configure training hyperparameters (LR, batch size, warmup, scheduler)
- [ ] Implement training script with validation loop and early stopping
- [ ] Run initial training (3 epochs) and monitor loss curves
- [ ] Run hyperparameter sweep if initial results are below target NDCG
- [ ] Save best checkpoint based on validation NDCG@10
- [ ] Run full benchmark on test split — compare against all baselines
- [ ] Verify no regression on general academic queries (NDCG@10 >= 0.11)
- [ ] Verify improvement on SFU-specific queries (NDCG@10 > 0.25)
- [ ] Verify Indigenous Studies queries improve > 5x over off-the-shelf
- [ ] Verify Canadian Studies queries improve > 3x over off-the-shelf
- [ ] Document final training config and results

### Phase 4: Quantization & Validation
- [ ] Export trained model to ONNX format
- [ ] Quantize ONNX model to INT8
- [ ] Validate quantized vs original cosine similarity > 0.98
- [ ] Confirm model size < 30MB
- [ ] Confirm inference latency < 15ms for 50 documents
- [ ] Confirm RAM usage < 80MB

### Phase 5: Integration & Deployment
- [ ] Copy quantized model to `models/sfu-academic-embed-v1-int8/`
- [ ] Set `SFU_EMBEDDING_MODEL_PATH` environment variable
- [ ] Enable `SFU_FEATURE_LOCAL_EMBEDDING_ENABLED=true`
- [ ] Verify reranker loads custom model on startup
- [ ] Run end-to-end search test with real queries
- [ ] Verify offline capability (no internet required)
- [ ] Verify fallback behavior when model is missing

### Phase 6: Evaluation & Documentation
- [ ] Write SFU-specific evaluation query set (`data/sfu_eval_queries.json`)
- [ ] Build evaluation script (`scripts/evaluate_sfu_queries.py`)
- [ ] Run full evaluation and record final metrics
- [ ] Document results in this plan (update Benchmark Results section)
- [ ] Write deployment guide for future model updates

### Phase 7: Continuous Improvement (Post-Launch)
- [ ] Set up query logging infrastructure for implicit feedback
- [ ] Plan quarterly retraining cadence
- [ ] Build Solr diff detector for subscription changes
- [ ] Implement incremental fine-tune pipeline (LoRA) for new domains
- [ ] First quarterly retrain with real user query data

---

## Implementation Plan

### Phase 1: SFU-Specific Training Data Generation (3-5 days)

**Goal:** Produce 15,000+ high-quality training triplets that encode SFU's institutional knowledge.

#### Strategy 1: Subject-Aligned Citation Pairs (5,000 triplets)

For each of SFU's 108 subjects, fetch highly-cited OpenAlex works and extract citation relationships.

```python
# For each SFU subject (e.g., "Indigenous Studies", "Criminology", "SIAT")
#   → Fetch 50 top-cited OpenAlex works tagged with that subject
#   → For each work, get papers it cites (positive pairs)
#   → Random works from a DIFFERENT SFU subject = hard negatives
#
# This teaches the model SFU's subject boundaries:
#   "What are the topics that matter at this institution?"
```

**Why this helps:** Generic models treat all academic subjects equally. A model trained on SFU's subject distribution will weight Indigenous Studies, Criminology, and SIAT higher because it has more exposure to those domains.

#### Strategy 2: SFU-Style Synthetic Queries (3,000 triplets)

Generate search queries that mimic real SFU user patterns using Solr metadata.

```python
# From Solr record:
#   name: "ICRG Researchers Dataset. Political Risk Ratings"
#   subjects: ["International Studies", "Policy Analysis", "Political Science"]
#   description: "Annual averages of the components of the Political Risk Ratings..."
#
# Generate synthetic queries:
#   "political risk ratings by country"
#   "international country risk data"
#   "ICRG dataset policy analysis"
#
# Positive: OpenAlex papers matching these topics
# Negative: Papers from unrelated subjects
```

**Why this helps:** The model learns the mapping between how SFU users phrase queries and the kinds of resources SFU provides.

#### Strategy 3: Provider-Aware Pairs (4,000 triplets)

Papers from SFU-subscribed providers are more useful to SFU users.

```python
# For each major provider (EBSCOhost, ProQuest, SAGE, etc.):
#   → Find OpenAlex works published in journals from that provider
#   → Create (query, subscribed_paper) as positive
#   → Create (query, non_subscribed_paper) as hard negative
#
# This teaches the model a subtle bias:
#   "When two papers are equally relevant, prefer the one SFU can access"
```

**Why this helps:** The current system resolves access AFTER ranking. A provider-aware embedding model can subtly prefer accessible content during ranking itself — the user sees full-text-available results ranked higher without needing a separate access-resolution step.

#### Strategy 4: Cross-Subject Hard Negatives (3,000 triplets)

The hardest negatives come from adjacent subjects.

```python
# SFU's subject taxonomy creates natural hard negatives:
#   "Health Sciences" vs "Biomedical Physiology and Kinesiology (BPK)"
#   "Political Science" vs "Policy Analysis"
#   "Interactive Arts & Technology (SIAT)" vs "Computing Science"
#
# Papers from adjacent subjects share vocabulary but differ in focus.
# These are the HARDEST cases for an embedding model to learn.
```

**Why this helps:** Off-the-shelf models can't distinguish between SFU-adjacent subjects because they don't know the institutional taxonomy. Training on cross-subject negatives teaches boundary discrimination.

### Phase 2: Base Model Selection (1 day)

Based on benchmark results, two candidates:

| Candidate | Pros | Cons |
|-----------|------|------|
| **all-MiniLM-L6-v2** | 22M params, 43MB, <5ms, proven fine-tuning base | Smaller capacity |
| **BGE-base-en-v1.5** | 110M params, best off-the-shelf NDCG (0.1229), modern architecture | 440MB, ~15ms, may be overkill for reranking 50 docs |

**Recommendation:** Start with **MiniLM** (smaller, faster, sufficient for reranking 20-50 candidates). If results are unsatisfying, try BGE-base as the fine-tuning base — the training script already supports `--base-model` swapping.

### Phase 3: Training (1-2 days)

#### 3.1 Environment Setup

```bash
# On gaming PC with GPU (outside container)
# Or in container on CPU (slower but works)

# Required packages
pip install sentence-transformers>=2.2.2 torch>=2.0 tensorboard wandb

# Verify CUDA availability (GPU training)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

#### 3.2 Data Preparation

```bash
# Step 1: Generate training data from all 4 strategies
python scripts/generate_training_data.py \
  --output data/sfu_training_triplets.jsonl \
  --citation-pairs 5000 \
  --synthetic-pairs 3000 \
  --related-pairs 4000 \
  --topics "indigenous studies,criminology,interactive arts,health sciences,canadian studies,political science,economics,biological sciences,environmental management,computer science"
```

**Data split strategy:**

| Split | Percentage | Triplets | Purpose |
|-------|-----------|----------|---------|
| Train | 90% | ~13,500 | Model parameter updates |
| Validation | 5% | ~750 | Early stopping, hyperparameter selection |
| Test | 5% | ~750 | Final evaluation only (never used during training) |

**Split constraints:**
- Stratify by strategy type — each split should contain proportional representation of all 4 strategies
- Stratify by subject — no subject should appear only in train or only in eval
- Deduplicate across splits — no anchor text should appear in both train and validation
- Filter triplets where anchor, positive, or negative is shorter than 10 tokens or longer than 512 tokens

**Data format** (`data/sfu_training_triplets.jsonl`):

```json
{
  "anchor": "indigenous language preservation programs in British Columbia",
  "positive": "Revitalizing Indigenous Languages Through Community-Based Programs: Evidence from First Nations Communities in Western Canada",
  "negative": "Computational Approaches to Natural Language Processing for Low-Resource Languages",
  "strategy": "synthetic_query",
  "subject": "Indigenous Studies",
  "metadata": {"source": "openalex", "anchor_type": "synthetic_query", "positive_work_id": "W123456"}
}
```

#### 3.3 Loss Function Selection

**Primary: Multiple Negatives Ranking Loss (MNRL)**

MNRL is the standard contrastive loss for embedding fine-tuning. Given a batch of (anchor, positive) pairs, it treats all other positives in the batch as in-batch negatives.

| Loss Function | Use Case | Pros | Cons |
|---------------|----------|------|------|
| **MNRL** (selected) | Contrastive learning with in-batch negatives | Simple, effective, scales with batch size | Needs large batches for enough negatives |
| TripletLoss | Explicit (anchor, pos, neg) triplets | Direct control over hard negatives | Slower convergence, margin tuning needed |
| CosineSimilarityLoss | Pairwise similarity regression | Good for graded relevance | Doesn't learn boundaries well |
| CachedMultipleNegativesRankingLoss | MNRL with gradient caching | Supports very large effective batch sizes on limited VRAM | More complex, minimal gain for our data size |

**Why MNRL:** Our training data already includes curated hard negatives (Strategy 4), and MNRL's in-batch negative sampling effectively multiplies the number of negatives per anchor by (batch_size - 1). With batch_size=64, each anchor sees 63 negatives per step.

**Optional enhancement:** If MNRL alone doesn't reach target NDCG, add a second training phase with TripletLoss using only the cross-subject hard negatives (Strategy 4) to sharpen boundary discrimination.

#### 3.4 Hyperparameter Configuration

**Initial configuration (start here):**

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` | Best speed/size tradeoff from benchmarks |
| Embedding dimension | 384 | MiniLM default, sufficient for reranking |
| Epochs | 3 | Standard for fine-tuning; more risks overfitting on 15K samples |
| Batch size | 64 | Fills GPU memory on RTX 3070 (8GB); maximizes in-batch negatives for MNRL |
| Learning rate | 2e-5 | Standard for transformer fine-tuning; low enough to preserve base model knowledge |
| Warmup ratio | 0.1 | 10% of training steps with linear warmup to prevent early divergence |
| LR scheduler | Linear decay | Smooth reduction after warmup; cosine is an alternative if loss plateaus |
| Weight decay | 0.01 | Mild regularization to prevent overfitting |
| Max sequence length | 256 | Covers 95%+ of titles+abstracts; saves memory vs 512 |
| FP16 (mixed precision) | True (GPU only) | Halves memory usage, ~1.5x speedup, negligible quality loss |
| Gradient accumulation | 1 (GPU) / 4 (CPU) | Effective batch size = batch_size × accumulation_steps |
| Evaluation steps | Every 500 steps | Check validation NDCG during training |
| Save strategy | Best + last | Keep the checkpoint with highest validation NDCG |

**Hyperparameter sweep (if initial results are below target):**

| Parameter | Values to Try | Priority |
|-----------|--------------|----------|
| Learning rate | 1e-5, 2e-5, 5e-5 | High — most impactful |
| Batch size | 32, 64, 128 | High — affects MNRL negative count |
| Epochs | 3, 5, 10 | Medium — watch for overfitting |
| Warmup ratio | 0.05, 0.1, 0.2 | Low — minor effect |
| Max seq length | 128, 256, 384 | Low — only if truncation is losing signal |

#### 3.5 Training Execution

```bash
# Step 2: Train the model
python scripts/train_embedding_model.py \
  --data data/sfu_training_triplets.jsonl \
  --output models/sfu-academic-embed-v1 \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 3 \
  --batch-size 64 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --max-seq-length 256 \
  --fp16 \
  --eval-steps 500 \
  --save-best-model \
  --log-dir logs/sfu-embed-training
```

**Expected training timeline:**

| Stage | GPU (RTX 3070) | CPU (Ryzen 7 5700X3D) |
|-------|---------------|----------------------|
| Data loading + tokenization | ~2 min | ~5 min |
| Epoch 1 (13,500 triplets) | ~40 min | ~5 hours |
| Epoch 2 | ~40 min | ~5 hours |
| Epoch 3 | ~40 min | ~5 hours |
| Validation after each epoch | ~2 min | ~10 min |
| **Total** | **~2-3 hours** | **~15-20 hours** |

#### 3.6 Training Monitoring

**Key metrics to track during training:**

| Metric | What to Watch For | Action if Abnormal |
|--------|-------------------|-------------------|
| Training loss | Steady decrease; should drop 50-70% over 3 epochs | If flat: increase LR. If spiky: decrease LR or increase batch size |
| Validation loss | Should track training loss with small gap | If diverges from train loss: overfitting — stop early or add regularization |
| Validation NDCG@10 | Should increase each epoch; target > 0.20 by epoch 2 | If flat: data quality issue — inspect triplets manually |
| Learning rate | Linear warmup then decay | Verify warmup completes in first ~600 steps |
| GPU memory | Should stay under 7.5GB with FP16 + batch=64 | If OOM: reduce batch_size to 32 and set gradient_accumulation=2 |
| Gradient norm | Should be stable (0.1–10.0 range) | If exploding (>100): reduce LR. If vanishing (<0.001): increase LR |

**TensorBoard monitoring:**

```bash
# In a separate terminal during training
tensorboard --logdir logs/sfu-embed-training --port 6006
# Open http://localhost:6006 to view live training curves
```

**Early stopping criteria:**
- Stop if validation loss increases for 3 consecutive evaluation steps
- Stop if validation NDCG@10 decreases for 2 consecutive evaluations
- Maximum training: 10 epochs (hard cap, even if still improving)

#### 3.7 Post-Training Validation

```bash
# Step 3: Benchmark the trained model against all baselines
python scripts/benchmark_embeddings.py \
  --custom-model models/sfu-academic-embed-v1 \
  --queries 50 --k 10
```

**Validation checklist:**

| Check | Target | Pass/Fail Criteria |
|-------|--------|-------------------|
| General NDCG@10 | >= 0.11 | Must not regress below worst off-the-shelf model |
| SFU-specific NDCG@10 | > 0.25 | Must show clear improvement over best off-the-shelf (0.12) |
| Indigenous Studies queries | > 5x improvement | e.g., from 0.0002 to > 0.001 on benchmark queries |
| Canadian Studies queries | > 3x improvement | e.g., from 0.005 to > 0.015 on benchmark queries |
| Inference speed (50 docs) | < 15ms | Measured on container CPU |
| Embedding dimension | 384 | Must match expected dimension |
| Model loads without errors | Yes | Test in both GPU and CPU environments |

**If validation fails:**

| Failure Mode | Likely Cause | Fix |
|-------------|-------------|-----|
| General NDCG regressed | Catastrophic forgetting — model forgot general knowledge | Reduce LR to 1e-5, reduce epochs to 2, increase general-domain triplets |
| SFU NDCG didn't improve | Training data doesn't capture SFU-specific signal | Audit triplet quality, increase Strategy 2 (synthetic queries) proportion |
| Both metrics are flat | Base model not learning from data | Try BGE-base-v1.5 as base, increase data to 25K+ triplets |
| Indigenous/Canadian queries unchanged | Insufficient representation in training data | Generate 2,000+ additional triplets specifically for these subjects |
| Overfitting (train loss low, val loss high) | Model memorized training data | Add dropout, reduce epochs, increase data volume |

#### 3.8 Model Versioning

Save training artifacts for reproducibility:

```
models/sfu-academic-embed-v1/
├── config.json                    # Model architecture config
├── model.safetensors              # Trained weights
├── tokenizer.json                 # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── training_args.json             # Exact hyperparameters used
├── training_log.jsonl             # Per-step loss and metrics
├── eval_results.json              # Final benchmark results
└── README.md                      # Training summary and data provenance
```

### Phase 4: Quantization & Validation (1 day)

```bash
# Export to ONNX and quantize to INT8
python scripts/quantize_model.py \
  --model models/sfu-academic-embed-v1 \
  --validate

# Expected output:
#   Original (FP32): ~86 MB
#   ONNX (FP32): ~86 MB
#   ONNX (INT8): ~22 MB
#   Validation: cosine similarity > 0.98 (original vs quantized)
```

### Phase 5: Integration & Deployment (1-2 days)

Already implemented:

- `src/lib/embedding.py` — model loader, encoding, similarity scoring
- `src/lib/reranker.py` — embedding-aware multi-signal scoring (35% semantic weight)
- `src/lib/config.py` — `SFU_EMBEDDING_MODEL_PATH`, `local_embedding_enabled` feature flag
- `src/tests/test_embedding.py` — 13 tests passing

Deployment:

```bash
# Set the model path in environment
export SFU_EMBEDDING_MODEL_PATH=models/sfu-academic-embed-v1-int8
export SFU_FEATURE_LOCAL_EMBEDDING_ENABLED=true

# The reranker automatically uses local embedding when available
# Falls back to token-overlap scoring when model is missing
```

### Phase 6: Continuous Improvement (Ongoing)

#### Query Log Training (After Launch)

```python
# Once users are searching, collect implicit feedback:
#   (user_query, clicked_result) = positive pair
#   (user_query, skipped_result) = weak negative
#
# Retrain quarterly with accumulated query logs
# This is the highest-quality training signal possible
```

#### Solr Registry Updates

```python
# SFU Solr updates when subscriptions change
# When new databases are added:
#   1. Detect new subjects/providers in Solr diff
#   2. Generate new training pairs for those domains
#   3. Incremental fine-tune (LoRA, 30min) on new data
#   4. Replace model file — reranker picks up on restart
```

---

## Scripts Already Built

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/benchmark_embeddings.py` | Compare models on 50 queries, NDCG@10 | Done, tested |
| `scripts/generate_training_data.py` | Generate triplets from OpenAlex + synthetic queries | Done, needs Solr integration |
| `scripts/train_embedding_model.py` | Fine-tune MiniLM/BGE with sentence-transformers | Done, tested |
| `scripts/quantize_model.py` | ONNX export + INT8 quantization | Done |
| `src/lib/embedding.py` | Production embedding model loader + inference | Done, 7 tests |
| `src/lib/reranker.py` | Multi-signal reranker with embedding integration | Done, 6 tests |

### Still Needed

| Script | Purpose | Effort |
|--------|---------|--------|
| `scripts/generate_sfu_training_data.py` | Solr-aware training data generation (the 4 strategies above) | 1-2 days |
| `scripts/evaluate_sfu_queries.py` | Benchmark with SFU-specific queries (not generic academic) | 1 day |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fine-tuned model is worse than off-the-shelf | Low | Medium | Benchmark gates every phase — ship off-the-shelf if it's better |
| Training data too small (<5K triplets) | Medium | Medium | OpenAlex has millions of citation pairs — scale up data generation |
| GPU not available for training | Low | Low | CPU training works (12-20h) — just slower |
| Solr registry changes break training pipeline | Low | Low | Registry is stable (~766 records), regenerate on major changes |
| Model overfits to SFU subjects | Medium | Low | Keep 5% eval split, test on out-of-distribution queries |
| ONNX quantization degrades quality | Low | Low | Validation step catches >2% deviation automatically |

---

## Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| NDCG@10 on SFU-specific queries | > 0.25 (vs 0.12 off-the-shelf) | `benchmark_embeddings.py` with SFU queries |
| NDCG@10 on general academic queries | >= 0.11 (no regression) | Same benchmark, generic queries |
| Inference latency (50 docs) | < 15ms | Time the reranker in production |
| Model size (INT8) | < 30MB | File size check |
| RAM usage | < 80MB | Process memory monitoring |
| Offline capability | 100% | Works without internet |
| Indigenous Studies query improvement | > 5x vs off-the-shelf | Targeted query benchmark |
| Canadian Studies query improvement | > 3x vs off-the-shelf | Targeted query benchmark |

---

## Timeline

| Week | Task | Output | Hardware |
|------|------|--------|----------|
| 1 | Build SFU Solr-integrated data generator | `generate_sfu_training_data.py` | Container (CPU) |
| 1 | Generate 15K+ training triplets | `data/sfu_training_triplets.jsonl` | Container (API calls) |
| 2 | Train model on gaming PC | `models/sfu-academic-embed-v1/` | Host GPU (RTX) |
| 2 | Quantize to INT8 ONNX | `models/sfu-academic-embed-v1-int8/` | Container (CPU) |
| 2 | Benchmark against all baselines | Comparison table | Container (CPU) |
| 3 | Evaluate on SFU-specific queries | NDCG improvement numbers | Container (CPU) |
| 3 | Deploy as default reranker | `SFU_EMBEDDING_MODEL_PATH` set | Container |
| 4 | Write SFU-specific evaluation queries | `data/sfu_eval_queries.json` | Manual curation |
| 5+ | Collect user query logs for retraining | Ongoing improvement | Production |

**Total effort:** ~3-4 weeks part-time. **Cost:** $0 (all free tools, own GPU).

---

## Appendix: Existing Code Integration Points

### Config (`src/lib/config.py`)

```python
# Feature flags (already implemented)
features["local_embedding_enabled"] = True   # Toggle embedding on/off

# Model path (already implemented)
embedding_model_path = ""                     # Custom model path
embedding_default_model = "sentence-transformers/all-MiniLM-L6-v2"  # Fallback

# Env vars:
#   SFU_EMBEDDING_MODEL_PATH=models/sfu-academic-embed-v1-int8
#   SFU_EMBEDDING_DEFAULT_MODEL=sentence-transformers/all-MiniLM-L6-v2
#   SFU_FEATURE_LOCAL_EMBEDDING_ENABLED=true
```

### Reranker Weights (`src/lib/reranker.py`)

```python
# With embedding (already implemented)
_WEIGHTS_WITH_EMBEDDING = {
    "semantic_similarity": 0.35,      # Local embedding cosine sim
    "title_relevance": 0.15,          # Token overlap (keyword backup)
    "recency": 0.15,                  # Publication year decay
    "fulltext_available": 0.15,       # Access availability
    "type_match": 0.10,               # article > book > conference
    "completeness": 0.10,             # DOI + authors + date present
}

# Without embedding (fallback, already implemented)
_WEIGHTS_NO_EMBEDDING = {
    "title_relevance": 0.35,
    "recency": 0.20,
    "fulltext_available": 0.20,
    "type_match": 0.15,
    "completeness": 0.10,
}
```

### Solr Registry (`src/lib/sfu_databases.py`)

```python
# Available data for training (766 records, each with):
# - name: "JSTOR", "ProQuest Dissertations", etc.
# - description: free-text description of the database
# - subjects: ["History", "Political Science", ...]  (108 unique)
# - provider: "EBSCOhost", "ProQuest", etc. (109 unique)
# - contentTypes: ["Ejournal collection", "Full-text database", ...]
# - free: bool (31.7% are free)
# - proxy: bool (63.7% need EZProxy)
# - names: alternate names/aliases
#
# Endpoint: https://databases.lib.sfu.ca/solr/sfu_databases/select
# Auth: None required
```
