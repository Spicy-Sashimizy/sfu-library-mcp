"""Per-section artifact builder: tantivy (BM25F) + BMP (SPLADE) + sidecars.

One pass over a section's doc stream produces:
  sections/<name>/tantivy/        lexical index (title/abstract indexed-only,
                                  freq index option = the validated positions-off
                                  lossless lever; year/type/is_oa fast fields)
  sections/<name>/splade_NNN.bmp  sparse shards (BMP 8-bit block-max impacts;
                                  SPLADE dot products are corpus-independent so
                                  shard/section results merge exactly by score)
  sections/<name>/abstracts.sqlite  hot sections only (zstd-dict sidecar)
  rows into the global meta.sqlite  (id -> title/doi/year/type/is_oa/section)

BMP impact quantization: weights are floats (0..~3.5, 4-decimals); BMP takes
integer impacts, so we scale by QUANT_SCALE=100 on both doc and query sides —
ranking is scale-invariant for dot products; BMP further quantizes to 8-bit
block maxima internally (measured-negligible loss in the BMP paper / SIGIR'24).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("sfu_library_mcp")

QUANT_SCALE = 100
DEFAULT_BMP_SHARD_DOCS = 2_000_000   # RAM ceiling per BMP build shard
BMP_BLOCK_SIZE = 32                  # bsize from the BMP paper's SPLADE config
TANTIVY_WRITER_HEAP = 1_000_000_000


def open_meta_db(index_root: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(Path(index_root) / "meta.sqlite"))
    db.executescript(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
        "CREATE TABLE IF NOT EXISTS docs ("
        "  id TEXT PRIMARY KEY, title TEXT, doi TEXT, year INTEGER,"
        "  type TEXT, is_oa INTEGER, section TEXT);"
    )
    return db


class SectionBuilder:
    """Streaming builder for one section. Call add(doc) repeatedly, then finish()."""

    def __init__(self, index_root: Path, section: str, hot: bool,
                 meta_db: sqlite3.Connection,
                 bmp_shard_docs: int = DEFAULT_BMP_SHARD_DOCS):
        import tantivy

        from lib.thinclient.abstracts import AbstractStoreWriter

        self.section = section
        self.hot = hot
        self.dir = Path(index_root) / "sections" / section
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta = meta_db
        self._meta_batch: list[tuple] = []
        self._bmp_shard_docs = bmp_shard_docs

        tantivy_dir = self.dir / "tantivy"
        tantivy_dir.mkdir(exist_ok=True)
        schema = (tantivy.SchemaBuilder()
                  .add_text_field("id", stored=True, tokenizer_name="raw")
                  .add_text_field("title", stored=False, index_option="freq")
                  .add_text_field("abstract", stored=False, index_option="freq")
                  .add_integer_field("year", stored=False, indexed=True, fast=True)
                  .add_text_field("doctype", stored=False, tokenizer_name="raw")
                  .add_boolean_field("is_oa", stored=False, indexed=True)
                  .build())
        self._tantivy = tantivy.Index(schema, path=str(tantivy_dir))
        self._writer = self._tantivy.writer(heap_size=TANTIVY_WRITER_HEAP)
        self._tantivy_mod = tantivy

        self._bmp = None
        self._bmp_shard_idx = 0
        self._bmp_in_shard = 0

        self._abstracts = (AbstractStoreWriter(self.dir / "abstracts.sqlite")
                           if hot else None)
        self.count = 0

    def _bmp_indexer(self):
        import bmp
        if self._bmp is None:
            path = self.dir / f"splade_{self._bmp_shard_idx:03d}.bmp"
            self._bmp = bmp.Indexer(str(path), bsize=BMP_BLOCK_SIZE, compress_range=True)
            self._bmp_in_shard = 0
        return self._bmp

    def _rotate_bmp(self) -> None:
        if self._bmp is not None:
            self._bmp.finish()
            self._bmp = None
            self._bmp_shard_idx += 1

    def add(self, doc: dict) -> None:
        """doc keys: id (openalex W-id), title, abstract, publication_year,
        type, is_oa, doi, sparse_field ({term: float weight})."""
        doc_id = doc["id"]
        title = doc.get("title") or ""
        abstract = doc.get("abstract") or ""
        year = int(doc.get("publication_year") or 0)
        dtype = doc.get("type") or ""
        is_oa = bool(doc.get("is_oa"))

        self._writer.add_document(self._tantivy_mod.Document(
            id=doc_id, title=title, abstract=abstract,
            year=year, doctype=dtype, is_oa=is_oa,
        ))

        sparse = doc.get("sparse_field") or {}
        if sparse:
            vec = {t: max(1, int(round(w * QUANT_SCALE))) for t, w in sparse.items()
                   if w > 0}
            if vec:
                self._bmp_indexer().add_document(doc_id, vec)
                self._bmp_in_shard += 1
                if self._bmp_in_shard >= self._bmp_shard_docs:
                    self._rotate_bmp()

        if self._abstracts is not None:
            self._abstracts.add(doc_id, abstract)

        self._meta_batch.append((doc_id, title, doc.get("doi") or "", year,
                                 dtype, int(is_oa), self.section))
        if len(self._meta_batch) >= 5_000:
            self._flush_meta()
        self.count += 1

    def _flush_meta(self) -> None:
        self._meta.executemany(
            "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?)", self._meta_batch)
        self._meta_batch.clear()

    def finish(self) -> dict:
        self._flush_meta()
        self._meta.commit()
        self._writer.commit()
        self._writer.wait_merging_threads()
        self._rotate_bmp()
        n_abs = self._abstracts.finish() if self._abstracts else 0
        size = sum(f.stat().st_size for f in self.dir.rglob("*") if f.is_file())
        info = {
            "docs": self.count,
            "bmp_shards": self._bmp_shard_idx,
            "abstracts_stored": n_abs,
            "live_bytes": size,
            "hot": self.hot,
        }
        logger.info("section %s built: %s", self.section, info)
        return info


def build_dense_leg(index_root: Path, vectors_npy: Path, ids_json: Path) -> dict:
    """Build the usearch dense leg from precomputed fp32 vectors.

    Validated recipe (docs/COMPRESSION_EVAL_RESULTS.md): binary sign codes in
    a usearch b1 Hamming index (32x), int8 rescore matrix memmapped from disk,
    4-20x over-fetch + exact rescore => R@10 0.975-0.996 vs exact.
    """
    import json

    import numpy as np
    from usearch.index import Index

    dense_dir = Path(index_root) / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(vectors_npy, mmap_mode="r").astype(np.float32)
    ids = json.loads(Path(ids_json).read_text())
    assert len(ids) == len(X), f"ids ({len(ids)}) != vectors ({len(X)})"

    bits = np.packbits((X > 0).astype(np.uint8), axis=1)
    index = Index(ndim=X.shape[1], dtype="b1", metric="hamming")
    index.add(np.arange(len(X), dtype=np.uint64), bits)
    index.save(str(dense_dir / "b1.usearch"))

    # int8 rescore matrix: symmetric scale, vectors are L2-normalized (|x|<=1).
    np.save(dense_dir / "rescore_int8.npy",
            np.clip(np.round(X * 127.0), -127, 127).astype(np.int8))
    (dense_dir / "ids.json").write_text(json.dumps(ids))
    info = {"vectors": len(X), "dim": int(X.shape[1]),
            "b1_bytes": (dense_dir / "b1.usearch").stat().st_size,
            "rescore_bytes": (dense_dir / "rescore_int8.npy").stat().st_size}
    logger.info("dense leg built: %s", info)
    return info
