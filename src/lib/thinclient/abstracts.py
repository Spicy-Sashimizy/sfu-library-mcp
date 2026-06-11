"""Hot-section abstract sidecar — zstd-dictionary-compressed SQLite blobs.

Policy (docs/THIN_CLIENT_STACK_RESEARCH.md, decided 2026-06-11):
  - HOT sections store abstracts locally (~0.5 KB/doc compressed) so the
    reranker's top ~50-100 candidates are fetchable offline at full quality.
  - COLD sections store nothing; the retriever fetches missing abstracts from
    the OpenAlex API before rerank (one mget of ~100 docs ≈ 150 KB).
  - Never rerank on titles alone — that's the one variant that measurably
    loses quality (~+0.12 NDCG@10 comes from title+abstract rerank).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import zstandard

logger = logging.getLogger("sfu_library_mcp")

DICT_SAMPLE_TARGET = 20_000     # abstracts sampled for dictionary training
DICT_SIZE = 112 * 1024          # zstd recommends ~110 KB dictionaries
COMPRESSION_LEVEL = 19          # write-once read-many sidecar


class AbstractStoreWriter:
    """Build-time writer. Buffers a training sample, trains the dict on first
    flush, then compresses everything with it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.executescript(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
            "CREATE TABLE IF NOT EXISTS abs (id TEXT PRIMARY KEY, z BLOB);"
            "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v BLOB);"
        )
        self._pending: list[tuple[str, str]] = []
        self._cctx: zstandard.ZstdCompressor | None = None
        self._n = 0

    def add(self, doc_id: str, abstract: str) -> None:
        if not abstract:
            return
        if self._cctx is None:
            self._pending.append((doc_id, abstract))
            if len(self._pending) >= DICT_SAMPLE_TARGET:
                self._train_and_flush()
            return
        self._insert(doc_id, abstract)

    def _train_and_flush(self) -> None:
        samples = [a.encode("utf-8") for _, a in self._pending]
        try:
            zdict = zstandard.train_dictionary(DICT_SIZE, samples)
            self._db.execute("INSERT OR REPLACE INTO meta VALUES ('zdict', ?)",
                             (zdict.as_bytes(),))
            self._cctx = zstandard.ZstdCompressor(level=COMPRESSION_LEVEL, dict_data=zdict)
            logger.info("abstract store %s: trained %d-byte dict on %d samples",
                        self.path.name, len(zdict.as_bytes()), len(samples))
        except zstandard.ZstdError as exc:
            # Tiny corpora can fail dict training — fall back to no dictionary.
            logger.warning("dict training failed (%s) — storing without dict", exc)
            self._cctx = zstandard.ZstdCompressor(level=COMPRESSION_LEVEL)
        for doc_id, abstract in self._pending:
            self._insert(doc_id, abstract)
        self._pending.clear()

    def _insert(self, doc_id: str, abstract: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO abs VALUES (?, ?)",
                         (doc_id, self._cctx.compress(abstract.encode("utf-8"))))
        self._n += 1
        if self._n % 200_000 == 0:
            self._db.commit()

    def finish(self) -> int:
        if self._cctx is None and self._pending:
            self._train_and_flush()
        self._db.commit()
        self._db.execute("VACUUM")
        self._db.close()
        return self._n


class AbstractStore:
    """Query-time reader (thread-safe; SQLite point lookups are ~µs)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        row = self._db.execute("SELECT v FROM meta WHERE k='zdict'").fetchone()
        zdict = zstandard.ZstdCompressionDict(row[0]) if row else None
        self._dctx = (zstandard.ZstdDecompressor(dict_data=zdict)
                      if zdict else zstandard.ZstdDecompressor())

    def fetch(self, ids: list[str]) -> dict[str, str]:
        """Return {id: abstract} for ids present in this store."""
        out: dict[str, str] = {}
        with self._lock:
            for chunk_start in range(0, len(ids), 500):
                chunk = ids[chunk_start:chunk_start + 500]
                marks = ",".join("?" * len(chunk))
                for doc_id, z in self._db.execute(
                        f"SELECT id, z FROM abs WHERE id IN ({marks})", chunk):
                    out[doc_id] = self._dctx.decompress(z).decode("utf-8")
        return out
