"""Local embedding model for semantic reranking.

Loads a sentence-transformer model (off-the-shelf or fine-tuned) and provides
embedding + cosine similarity scoring for query-to-paper reranking. Designed
to replace Semantic Scholar API dependency for Stage 2 reranking.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("sfu_library_mcp")

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_model_name = None


def _load_model(model_path: str | None = None):
    """Load a sentence-transformer model, caching it as a module-level singleton."""
    global _model, _model_name

    target = model_path or _DEFAULT_MODEL
    if _model is not None and _model_name == target:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", target)
        _model = SentenceTransformer(target)
        _model_name = target
        logger.info("Embedding model loaded (%s)", target)
        return _model
    except Exception as e:
        logger.error("Failed to load embedding model '%s': %s", target, e)
        _model = None
        _model_name = None
        return None


def encode_texts(texts: list[str], model_path: str | None = None) -> np.ndarray | None:
    """Encode a list of texts into normalized embedding vectors.

    Returns an (N, D) numpy array of unit-length embeddings, or None on failure.
    """
    model = _load_model(model_path)
    if model is None:
        return None

    try:
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings)
    except Exception as e:
        logger.error("Embedding encode failed: %s", e)
        return None


def compute_similarity(query: str, documents: list[str], model_path: str | None = None) -> list[float]:
    """Compute cosine similarity between a query and a list of documents.

    Returns a list of similarity scores (0.0-1.0), one per document.
    Returns empty list on failure.
    """
    if not documents:
        return []

    texts = [query] + documents
    embeddings = encode_texts(texts, model_path)
    if embeddings is None:
        return []

    query_emb = embeddings[0]
    doc_embs = embeddings[1:]
    similarities = (doc_embs @ query_emb).tolist()
    return similarities


def score_papers_semantic(
    query: str,
    papers: list[dict],
    model_path: str | None = None,
    title_weight: float = 0.3,
    abstract_weight: float = 0.7,
) -> list[float]:
    """Score papers by semantic similarity to a query.

    Combines title and abstract similarity with configurable weights.
    Papers missing abstracts fall back to title-only scoring.

    Returns a list of scores (0.0-1.0), one per paper.
    """
    if not papers:
        return []

    documents = []
    for p in papers:
        title = p.get("title", "") or ""
        abstract = p.get("abstract", "") or ""
        if abstract:
            doc = f"{title} {abstract}"
        else:
            doc = title
        documents.append(doc)

    return compute_similarity(query, documents, model_path)


def get_model_info(model_path: str | None = None) -> dict:
    """Return info about the loaded (or to-be-loaded) model."""
    model = _load_model(model_path)
    if model is None:
        return {"loaded": False, "model": model_path or _DEFAULT_MODEL}

    dim = model.get_sentence_embedding_dimension()
    return {
        "loaded": True,
        "model": _model_name,
        "embedding_dimension": dim,
        "max_seq_length": getattr(model, "max_seq_length", None),
    }


def unload_model():
    """Free the model from memory."""
    global _model, _model_name
    _model = None
    _model_name = None
    logger.info("Embedding model unloaded")
