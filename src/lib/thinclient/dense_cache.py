"""Query-driven dense warm cache — the index that builds itself from queries.

Two-tier delta over the static usearch seed (design + lineage in
docs/DENSE_WARMCACHE_RESEARCH.md): SQLite is the sole durability point; the
in-RAM delta is searched by brute-force Hamming (popcount LUT) + exact int8
rescore, which sidesteps HNSW insert-order recall degradation entirely at
delta scale and merges with seed scores exactly (corpus-independent dot
products). A single background worker dedups, CPU-encodes surfaced docs in
small batches, and appends. Everything is bounded: queue, doc cap, one thread.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("sfu_library_mcp")

DELTA_KEY_BASE = 1 << 40        # disjoint from seed keys (0..len(seed)-1)
ENCODE_BATCH = 32               # measured ~87 docs/s CPU at this batch size
QUEUE_MAX_DOCS = 2_000          # enqueue is drop-newest beyond this
DEFAULT_MAX_DOCS = 500_000      # brute-force stays cheap (~24 MB codes)
_POPCNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None],
                        axis=1).sum(1).astype(np.uint8)

_SCHEMA = (
    "PRAGMA journal_mode=WAL;"
    "CREATE TABLE IF NOT EXISTS delta ("
    "  id TEXT PRIMARY KEY, key INTEGER UNIQUE, vec BLOB,"
    "  title TEXT, doi TEXT, year INTEGER, type TEXT, is_oa INTEGER,"
    "  added_at REAL);"
)


class DenseWarmCache:
    """Usage-driven dense delta alongside the static seed index."""

    def __init__(self, dense_dir: Path, seed_ids: list[str],
                 model_path: str, dim: int = 384,
                 max_docs: int = DEFAULT_MAX_DOCS, encoder=None):
        self.dir = Path(dense_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.dim = dim
        self.max_docs = max_docs
        self._encoder = encoder          # injectable for tests
        self._db = sqlite3.connect(str(self.dir / "delta.sqlite"),
                                   check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db_lock = threading.Lock()

        self._lock = threading.Lock()    # guards the RAM arrays below
        self._n = 0
        cap = 4096
        self._codes = np.zeros((cap, dim // 8), dtype=np.uint8)
        self._int8 = np.zeros((cap, dim), dtype=np.int8)
        self._ids: list[str] = []
        self._known: set[str] = set(seed_ids)
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_DOCS)
        self._worker: threading.Thread | None = None
        self._closed = False
        self.stats = {"enqueued": 0, "encoded": 0, "dropped": 0, "errors": 0}

        # Rebuild the RAM tier from SQLite (durable truth) at startup.
        rows = self._db.execute(
            "SELECT id, vec FROM delta ORDER BY key").fetchall()
        for doc_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.int8)
            if vec.shape[0] != dim:
                continue
            self._append_row(doc_id, vec)
        if rows:
            logger.info("dense warm cache: restored %d docs from delta.sqlite",
                        self._n)

    # ── ingest ───────────────────────────────────────────────────────────────

    def enqueue(self, docs: list[dict]) -> None:
        """Fire-and-forget: queue surfaced docs for background encoding.
        Never blocks a query — drops on overflow."""
        if self._closed or self._n >= self.max_docs:
            return
        fresh = [d for d in docs
                 if d.get("openalex_id") and d.get("title")
                 and d["openalex_id"] not in self._known]
        if not fresh:
            return
        self._ensure_worker()
        for d in fresh:
            try:
                self._queue.put_nowait(
                    {k: d.get(k) for k in ("openalex_id", "title", "abstract",
                                           "doi", "year", "type", "is_oa")})
                self.stats["enqueued"] += 1
            except queue.Full:
                self.stats["dropped"] += 1
                return

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            with self._lock:
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(
                        target=self._run, daemon=True, name="dense-warmcache")
                    self._worker.start()

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is not None:
            return np.asarray(self._encoder(texts), dtype=np.float32)
        from lib.opensearch_retriever import _get_dense_model
        model = _get_dense_model(self.model_path)
        return np.asarray(
            model.encode(texts, batch_size=ENCODE_BATCH,
                         normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32)

    def _run(self) -> None:
        while not self._closed:
            batch: list[dict] = []
            try:
                batch.append(self._queue.get(timeout=5))
            except queue.Empty:
                continue
            while len(batch) < ENCODE_BATCH:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            batch = list({d["openalex_id"]: d for d in batch
                          if d["openalex_id"] not in self._known}.values())
            if not batch or self._n >= self.max_docs:
                continue
            try:
                texts = [f"{d['title']} {d.get('abstract') or ''}"[:2000]
                         for d in batch]
                X = self._encode(texts)
                # Byte-identical quantization to builder.build_dense_leg.
                int8 = np.clip(np.round(X * 127.0), -127, 127).astype(np.int8)
                now = time.time()
                with self._db_lock:
                    for i, d in enumerate(batch):
                        self._db.execute(
                            "INSERT OR IGNORE INTO delta VALUES (?,?,?,?,?,?,?,?,?)",
                            (d["openalex_id"], DELTA_KEY_BASE + self._n + i,
                             int8[i].tobytes(), d["title"], d.get("doi") or "",
                             d.get("year"), d.get("type") or "",
                             int(bool(d.get("is_oa"))), now))
                    self._db.commit()
                with self._lock:
                    for i, d in enumerate(batch):
                        self._append_row_locked(d["openalex_id"], int8[i])
                self.stats["encoded"] += len(batch)
            except Exception as exc:
                self.stats["errors"] += 1
                logger.warning("dense warm cache encode batch failed: %s", exc)
                if self.stats["errors"] >= 5 and self.stats["encoded"] == 0:
                    logger.error("dense warm cache disabled (encoder unusable)")
                    self._closed = True

    # ── RAM tier ─────────────────────────────────────────────────────────────

    def _append_row(self, doc_id: str, int8_vec: np.ndarray) -> None:
        with self._lock:
            self._append_row_locked(doc_id, int8_vec)

    def _append_row_locked(self, doc_id: str, int8_vec: np.ndarray) -> None:
        if self._n >= self._codes.shape[0]:        # capacity doubling
            self._codes = np.vstack([self._codes, np.zeros_like(self._codes)])
            self._int8 = np.vstack([self._int8, np.zeros_like(self._int8)])
        self._codes[self._n] = np.packbits(int8_vec > 0)
        self._int8[self._n] = int8_vec
        self._ids.append(doc_id)
        self._known.add(doc_id)
        self._n += 1

    # ── search / hydration ───────────────────────────────────────────────────

    def search(self, qbits: np.ndarray, q: np.ndarray,
               fetch: int) -> list[tuple[str, float]]:
        """Brute-force Hamming over the delta + exact int8 rescore.
        Scores are directly comparable with the seed leg's."""
        with self._lock:
            n = self._n
            if n == 0:
                return []
            codes = self._codes[:n]
            int8 = self._int8[:n]
            ids = list(self._ids)
        dists = _POPCNT[np.bitwise_xor(codes, qbits.ravel())].sum(axis=1)
        take = min(fetch, n)
        cand = np.argpartition(dists, take - 1)[:take]
        scores = (int8[cand].astype(np.float32) / 127.0) @ q
        order = np.argsort(-scores)
        return [(ids[int(cand[i])], float(scores[i])) for i in order]

    def metadata_rows(self, ids: list[str]) -> dict[str, tuple]:
        """meta.sqlite-shaped rows for warm docs missing from the main meta DB:
        (id, title, doi, year, type, is_oa, section)."""
        out: dict[str, tuple] = {}
        want = [i for i in ids if i in self._known]
        if not want:
            return out
        with self._db_lock:
            for start in range(0, len(want), 500):
                chunk = want[start:start + 500]
                marks = ",".join("?" * len(chunk))
                for r in self._db.execute(
                        f"SELECT id, title, doi, year, type, is_oa FROM delta "
                        f"WHERE id IN ({marks})", chunk):
                    out[r[0]] = (r[0], r[1], r[2], r[3], r[4], r[5], "warmcache")
        return out

    def info(self) -> dict:
        return {"docs": self._n, "max_docs": self.max_docs,
                "queue": self._queue.qsize(), **self.stats}

    def close(self) -> None:
        self._closed = True
