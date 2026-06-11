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
# Per-language dictionary for Cyrillic abstracts: ru measured the largest
# held-out win of any real language (own dict −40.0% vs the English dict,
# −42.5% vs no dict; machine-code control lost to every real language) —
# data/eval_results/per_language_dicts_20260611.json.
RU_DICT_ID = 1
RU_DICT_MIN = 300               # below the eval's qualifying bar -> main dict


def _is_cyrillic(text: str) -> bool:
    head = text[:300]
    if not head:
        return False
    cyr = sum(1 for c in head if 0x0400 <= ord(c) < 0x0500)
    return cyr > len(head) * 0.25


class AbstractStoreWriter:
    """Build-time writer. Buffers a training sample, trains the dict on first
    flush, then compresses everything with it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.executescript(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
            "CREATE TABLE IF NOT EXISTS abs "
            "  (id TEXT PRIMARY KEY, z BLOB, d INTEGER NOT NULL DEFAULT 0);"
            "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v BLOB);"
        )
        self._pending: list[tuple[str, str]] = []
        self._pending_ru: list[tuple[str, str]] = []
        self._cctx: zstandard.ZstdCompressor | None = None
        self._cctx_ru: zstandard.ZstdCompressor | None = None
        self._n = 0

    def add(self, doc_id: str, abstract: str) -> None:
        if not abstract:
            return
        if _is_cyrillic(abstract):
            if self._cctx_ru is None:
                self._pending_ru.append((doc_id, abstract))
                if len(self._pending_ru) >= DICT_SAMPLE_TARGET:
                    self._train_ru()
                return
            self._insert(doc_id, abstract, RU_DICT_ID, self._cctx_ru)
            return
        if self._cctx is None:
            self._pending.append((doc_id, abstract))
            if len(self._pending) >= DICT_SAMPLE_TARGET:
                self._train_and_flush()
            return
        self._insert(doc_id, abstract, 0, self._cctx)

    def _train(self, key: str, pending: list[tuple[str, str]]) -> zstandard.ZstdCompressor | None:
        samples = [a.encode("utf-8") for _, a in pending]
        try:
            zdict = zstandard.train_dictionary(DICT_SIZE, samples)
            self._db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                             (key, zdict.as_bytes()))
            logger.info("abstract store %s: trained %d-byte %s on %d samples",
                        self.path.name, len(zdict.as_bytes()), key, len(samples))
            return zstandard.ZstdCompressor(level=COMPRESSION_LEVEL, dict_data=zdict)
        except zstandard.ZstdError as exc:
            # Tiny corpora can fail dict training — caller falls back.
            logger.warning("%s training failed (%s)", key, exc)
            return None

    def _train_and_flush(self) -> None:
        self._cctx = (self._train("zdict", self._pending)
                      or zstandard.ZstdCompressor(level=COMPRESSION_LEVEL))
        for doc_id, abstract in self._pending:
            self._insert(doc_id, abstract, 0, self._cctx)
        self._pending.clear()

    def _train_ru(self) -> None:
        """Train the Cyrillic dict, or fall back to the main dict below the
        qualifying sample bar (main cctx is forced to exist first)."""
        if self._cctx is None:
            self._train_and_flush()
        cctx = (self._train("zdict_ru", self._pending_ru)
                if len(self._pending_ru) >= RU_DICT_MIN else None)
        if cctx is not None:
            self._cctx_ru = cctx
            for doc_id, abstract in self._pending_ru:
                self._insert(doc_id, abstract, RU_DICT_ID, self._cctx_ru)
        else:
            self._cctx_ru = self._cctx
            for doc_id, abstract in self._pending_ru:
                self._insert(doc_id, abstract, 0, self._cctx)
        self._pending_ru.clear()

    def _insert(self, doc_id: str, abstract: str, dict_id: int,
                cctx: zstandard.ZstdCompressor) -> None:
        self._db.execute("INSERT OR REPLACE INTO abs VALUES (?, ?, ?)",
                         (doc_id, cctx.compress(abstract.encode("utf-8")), dict_id))
        self._n += 1
        if self._n % 200_000 == 0:
            self._db.commit()

    def finish(self) -> int:
        if self._cctx is None and self._pending:
            self._train_and_flush()
        if self._pending_ru:
            self._train_ru()
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

        def _dctx(key: str, fallback=None):
            row = self._db.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
            if not row:
                return fallback
            return zstandard.ZstdDecompressor(
                dict_data=zstandard.ZstdCompressionDict(row[0]))

        main = _dctx("zdict") or zstandard.ZstdDecompressor()
        # d=1 rows without a stored ru dict were compressed with the main dict
        # (writer's below-bar fallback) — decode them the same way.
        self._dctxs = {0: main, RU_DICT_ID: _dctx("zdict_ru", fallback=main)}
        cols = [r[1] for r in self._db.execute("PRAGMA table_info(abs)")]
        self._has_dict_col = "d" in cols   # pre-v2 stores: two columns, one dict

    def fetch(self, ids: list[str]) -> dict[str, str]:
        """Return {id: abstract} for ids present in this store."""
        out: dict[str, str] = {}
        sel = ("SELECT id, z, d FROM abs" if self._has_dict_col
               else "SELECT id, z, 0 FROM abs")
        with self._lock:
            for chunk_start in range(0, len(ids), 500):
                chunk = ids[chunk_start:chunk_start + 500]
                marks = ",".join("?" * len(chunk))
                for doc_id, z, d in self._db.execute(
                        f"{sel} WHERE id IN ({marks})", chunk):
                    dctx = self._dctxs.get(d, self._dctxs[0])
                    out[doc_id] = dctx.decompress(z).decode("utf-8")
        return out
