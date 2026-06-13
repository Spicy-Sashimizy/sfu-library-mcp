"""Hot-section abstract sidecar — clustered zstd blocks with per-script dicts.

Policy (docs/THIN_CLIENT_STACK_RESEARCH.md, decided 2026-06-11):
  - HOT sections store abstracts locally so the reranker's top ~50-100
    candidates are fetchable offline at full quality.
  - COLD sections store nothing; the retriever fetches missing abstracts from
    the OpenAlex API before rerank (one mget of ~100 docs ≈ 150 KB).
  - Never rerank on titles alone — that's the one variant that measurably
    loses quality (~+0.12 NDCG@10 comes from title+abstract rerank).

Format v3 (the measured 64.7 -> 52.9 GB config at 150M):
  - abstracts are grouped by Unicode-script bucket and packed into ~32 KB
    uncompressed blocks, each compressed whole with the bucket's trained
    zstd-19 dictionary. Script bucketing captures the big per-language wins
    for free (held-out eval per_language_dicts_20260611: ru −40% / fa −38% /
    ko −37% / zh −32% vs the English dict — all non-Latin scripts; full
    langid measured too slow for the build path at ~95 docs/s).
  - a doc fetch decompresses one 32 KB block (sub-ms) and slices.
Readers transparently handle v1 (per-doc, single dict), v2 (per-doc, +ru
dict column) and v3 stores.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path

import zstandard

logger = logging.getLogger("sfu_library_mcp")

DICT_SIZE = 112 * 1024          # zstd recommends ~110 KB dictionaries
COMPRESSION_LEVEL = 19          # write-once read-many sidecar
BLOCK_TARGET = 32 * 1024        # uncompressed bytes per block (measured config)
TRAIN_TARGET_DOCS = 20_000      # dict-training sample per bucket
DICT_MIN_DOCS = 300             # below the eval's qualifying bar -> fallback
FLUSH_BYTES = 4 << 20           # post-training per-bucket buffer cap
BLOCK_CACHE = 64                # reader: decompressed blocks kept hot

RU_DICT_ID = 1                  # v2 compatibility (reader only)

_SCRIPT_RANGES = (
    ("cyrillic", ((0x0400, 0x0500),)),
    ("han", ((0x4E00, 0xA000), (0x3400, 0x4DC0))),
    ("hangul", ((0xAC00, 0xD7B0), (0x1100, 0x1200))),
    ("arabic", ((0x0600, 0x0700), (0x0750, 0x0780), (0xFB50, 0xFE00))),
    ("kana", ((0x3040, 0x3100),)),
    ("greek", ((0x0370, 0x0400),)),
)


def script_bucket(text: str) -> str:
    """Dominant non-Latin script of the head of `text`, else 'latin'.
    ~µs per doc — the build-path-affordable stand-in for langid."""
    head = text[:300]
    if not head:
        return "latin"
    counts = dict.fromkeys((n for n, _ in _SCRIPT_RANGES), 0)
    for c in head:
        o = ord(c)
        for name, ranges in _SCRIPT_RANGES:
            if any(lo <= o < hi for lo, hi in ranges):
                counts[name] += 1
                break
    name, n = max(counts.items(), key=lambda kv: kv[1])
    return name if n > len(head) * 0.15 else "latin"


def _is_cyrillic(text: str) -> bool:    # kept for v2-era callers/tests
    return script_bucket(text) == "cyrillic"


class AbstractStoreWriter:
    """Build-time writer (v3). Buffers per script bucket, trains each bucket's
    dict at TRAIN_TARGET_DOCS (or finish), then packs ~32 KB blocks."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        # WAL (not OFF): the store must survive a SIGKILL so the per-slice
        # build resume can keep appending instead of rebuilding the section.
        self._db.executescript(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=OFF;"
            "CREATE TABLE IF NOT EXISTS blocks (b INTEGER PRIMARY KEY, d INTEGER, z BLOB);"
            "CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, b INTEGER,"
            "  off INTEGER, n INTEGER);"
            "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v BLOB);"
        )
        self._buffers: dict[str, list[tuple[str, bytes]]] = {}
        self._buffer_bytes: dict[str, int] = {}
        self._cctx: dict[str, zstandard.ZstdCompressor] = {}
        self._dict_names: list[str] = []      # index = dict id in blocks.d
        self._next_block = 0
        self._n = 0
        self._restore()

    def _restore(self) -> None:
        """Resume from a checkpoint()ed store: dict ids are positional, so the
        bucket order, the trained dicts, and the not-yet-flushed pending
        buffers must all come back exactly as persisted."""
        row = self._db.execute("SELECT v FROM meta WHERE k='dicts'").fetchone()
        if row is None:
            return
        self._dict_names = json.loads(row[0])
        latin_row = self._db.execute(
            "SELECT v FROM meta WHERE k='zdict_latin'").fetchone()
        latin = (zstandard.ZstdCompressionDict(latin_row[0])
                 if latin_row else None)
        for name in self._dict_names:
            r = self._db.execute("SELECT v FROM meta WHERE k=?",
                                 (f"zdict_{name}",)).fetchone()
            zd = zstandard.ZstdCompressionDict(r[0]) if r else latin
            self._cctx[name] = (zstandard.ZstdCompressor(
                level=COMPRESSION_LEVEL, dict_data=zd) if zd
                else zstandard.ZstdCompressor(level=COMPRESSION_LEVEL))
        for (k, v) in self._db.execute(
                "SELECT k, v FROM meta WHERE k LIKE 'pending_%'").fetchall():
            bucket = k[len("pending_"):]
            docs = json.loads(zstandard.ZstdDecompressor().decompress(v))
            self._buffers[bucket] = [(i, a.encode("utf-8")) for i, a in docs]
            self._buffer_bytes[bucket] = sum(len(d) for _, d in
                                             self._buffers[bucket])
        self._next_block = self._db.execute(
            "SELECT COALESCE(MAX(b)+1, 0) FROM blocks").fetchone()[0]
        self._n = self._db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        logger.info("abstract store %s: resumed at block %d, %d docs, "
                    "%d pending buffers", self.path.name, self._next_block,
                    self._n, len(self._buffers))

    def add(self, doc_id: str, abstract: str) -> None:
        if not abstract:
            return
        bucket = script_bucket(abstract)
        buf = self._buffers.setdefault(bucket, [])
        data = abstract.encode("utf-8")
        buf.append((doc_id, data))
        self._buffer_bytes[bucket] = self._buffer_bytes.get(bucket, 0) + len(data)
        if bucket in self._cctx:
            if self._buffer_bytes[bucket] >= FLUSH_BYTES:
                self._flush(bucket)
        elif len(buf) >= TRAIN_TARGET_DOCS:
            self._train(bucket)
            self._flush(bucket)

    def _train(self, bucket: str) -> None:
        samples = [d for _, d in self._buffers.get(bucket, [])]
        cctx = None
        if len(samples) >= DICT_MIN_DOCS:
            try:
                zdict = zstandard.train_dictionary(DICT_SIZE, samples)
                self._db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                                 (f"zdict_{bucket}", zdict.as_bytes()))
                cctx = zstandard.ZstdCompressor(level=COMPRESSION_LEVEL,
                                                dict_data=zdict)
                logger.info("abstract store %s: trained %s dict on %d samples",
                            self.path.name, bucket, len(samples))
            except zstandard.ZstdError as exc:
                logger.warning("%s dict training failed (%s)", bucket, exc)
        if cctx is None:
            # Below the bar / failed: share the latin dict when it exists.
            cctx = self._cctx.get("latin") or zstandard.ZstdCompressor(
                level=COMPRESSION_LEVEL)
        self._cctx[bucket] = cctx
        if bucket not in self._dict_names:
            self._dict_names.append(bucket)

    def _dict_id(self, bucket: str) -> int:
        return self._dict_names.index(bucket)

    def _flush(self, bucket: str) -> None:
        """Pack the bucket's buffer into ~BLOCK_TARGET uncompressed blocks."""
        buf = self._buffers.get(bucket, [])
        if not buf:
            return
        # A resume that restored pending buffers AND re-fed the crashed slice
        # can hold the same doc twice — keep the last copy.
        dedup = dict(buf)
        if len(dedup) < len(buf):
            buf[:] = list(dedup.items())
        cctx, did = self._cctx[bucket], self._dict_id(bucket)
        i = 0
        while i < len(buf):
            payload = bytearray()
            rows = []
            while i < len(buf) and len(payload) < BLOCK_TARGET:
                doc_id, data = buf[i]
                rows.append((doc_id, self._next_block, len(payload), len(data)))
                payload += data
                i += 1
            self._db.execute("INSERT INTO blocks VALUES (?,?,?)",
                             (self._next_block, did, cctx.compress(bytes(payload))))
            self._db.executemany("INSERT OR REPLACE INTO docs VALUES (?,?,?,?)",
                                 rows)
            self._next_block += 1
            self._n += len(rows)
        buf.clear()
        self._buffer_bytes[bucket] = 0
        if self._n % 200_000 < TRAIN_TARGET_DOCS:
            self._db.commit()

    def checkpoint(self) -> None:
        """Durability point for the per-slice build resume: flush trained
        buckets, persist untrained buckets' raw buffers (training quality
        needs the full TRAIN_TARGET_DOCS sample, so don't train early), and
        commit. _restore() is the inverse."""
        for bucket in list(self._buffers):
            if bucket in self._cctx:
                self._flush(bucket)
        self._db.execute("DELETE FROM meta WHERE k LIKE 'pending_%'")
        for bucket, buf in self._buffers.items():
            if not buf:
                continue
            payload = json.dumps([(i, d.decode("utf-8"))
                                  for i, d in dict(buf).items()])
            self._db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                (f"pending_{bucket}",
                 zstandard.ZstdCompressor(level=3).compress(payload.encode())))
        self._db.execute("INSERT OR REPLACE INTO meta VALUES ('dicts', ?)",
                         (json.dumps(self._dict_names),))
        self._db.commit()
        # Keep the -wal from growing unbounded across slices (see builder.py
        # slice_checkpoint): TRUNCATE checkpoints all frames into the db and
        # resets the file; data is durable from the commit above.
        self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def finish(self) -> int:
        # Train latin first so below-bar buckets can fall back to its dict.
        order = sorted(self._buffers, key=lambda b: b != "latin")
        for bucket in order:
            if bucket not in self._cctx:
                self._train(bucket)
            self._flush(bucket)
        self._db.execute("INSERT OR REPLACE INTO meta VALUES ('format', 'v3')")
        self._db.execute("INSERT OR REPLACE INTO meta VALUES ('dicts', ?)",
                         (json.dumps(self._dict_names),))
        self._db.execute("DELETE FROM meta WHERE k LIKE 'pending_%'")
        # Re-fed slices repoint docs rows at fresh blocks; drop orphans.
        self._db.execute("DELETE FROM blocks WHERE b NOT IN "
                         "(SELECT DISTINCT b FROM docs)")
        self._db.commit()
        self._db.execute("PRAGMA journal_mode=DELETE")  # ship without -wal/-shm
        self._db.execute("VACUUM")
        self._db.close()
        return self._n


class AbstractStore:
    """Query-time reader for v1/v2/v3 stores (thread-safe)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        tables = {r[0] for r in self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self._v3 = "blocks" in tables
        self._block_cache: OrderedDict[int, bytes] = OrderedDict()

        def _zdict(key: str):
            row = self._db.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
            return zstandard.ZstdCompressionDict(row[0]) if row else None

        if self._v3:
            names = json.loads(self._db.execute(
                "SELECT v FROM meta WHERE k='dicts'").fetchone()[0])
            plain = zstandard.ZstdDecompressor()
            latin = _zdict("zdict_latin")
            self._dctxs = {}
            for i, name in enumerate(names):
                zd = _zdict(f"zdict_{name}") or latin
                self._dctxs[i] = (zstandard.ZstdDecompressor(dict_data=zd)
                                  if zd else plain)
        else:
            zd = _zdict("zdict")
            main = (zstandard.ZstdDecompressor(dict_data=zd) if zd
                    else zstandard.ZstdDecompressor())
            zru = _zdict("zdict_ru")
            self._dctxs = {0: main,
                           RU_DICT_ID: (zstandard.ZstdDecompressor(dict_data=zru)
                                        if zru else main)}
            cols = [r[1] for r in self._db.execute("PRAGMA table_info(abs)")]
            self._has_dict_col = "d" in cols   # v1: two columns, one dict

    def _block(self, b: int) -> bytes:
        cached = self._block_cache.get(b)
        if cached is not None:
            self._block_cache.move_to_end(b)
            return cached
        d, z = self._db.execute(
            "SELECT d, z FROM blocks WHERE b=?", (b,)).fetchone()
        payload = self._dctxs.get(d, self._dctxs[0]).decompress(z)
        self._block_cache[b] = payload
        if len(self._block_cache) > BLOCK_CACHE:
            self._block_cache.popitem(last=False)
        return payload

    def fetch(self, ids: list[str]) -> dict[str, str]:
        """Return {id: abstract} for ids present in this store."""
        out: dict[str, str] = {}
        with self._lock:
            for chunk_start in range(0, len(ids), 500):
                chunk = ids[chunk_start:chunk_start + 500]
                marks = ",".join("?" * len(chunk))
                if self._v3:
                    for doc_id, b, off, n in self._db.execute(
                            f"SELECT id, b, off, n FROM docs "
                            f"WHERE id IN ({marks})", chunk):
                        out[doc_id] = self._block(b)[off:off + n].decode("utf-8")
                else:
                    sel = ("SELECT id, z, d FROM abs" if self._has_dict_col
                           else "SELECT id, z, 0 FROM abs")
                    for doc_id, z, d in self._db.execute(
                            f"{sel} WHERE id IN ({marks})", chunk):
                        dctx = self._dctxs.get(d, self._dctxs[0])
                        out[doc_id] = dctx.decompress(z).decode("utf-8")
        return out
