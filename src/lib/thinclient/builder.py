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
integer impacts, so we scale by QUANT_SCALE (70, see below) on both doc and
query sides —
ranking is scale-invariant for dot products; BMP further quantizes to 8-bit
block maxima internally (measured-negligible loss in the BMP paper / SIGIR'24).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("sfu_library_mcp")

# BMP stores impacts as u8 and SATURATES above 255 (measured: scale 1000
# collapses recall to 0.28; scale 100 clips weights > 2.55). Scale 70 keeps
# every SPLADE log1p weight <= 3.64 saturation-free: measured recall@50 vs
# float-exact 0.968-0.970, identical index size. Scores are scale-invariant
# for ranking, so retriever/index scale mismatch only rescales scores.
QUANT_SCALE = 70
BMP_IMPACT_MAX = 255
DEFAULT_BMP_SHARD_DOCS = 2_000_000   # RAM ceiling per BMP build shard
# BMP returns empty results below ~500 docs (and panics on heavy score ties);
# never leave a tail shard smaller than this — absorb it into the previous.
MIN_BMP_TAIL_DOCS = 5_000
# bsize=256 + clustered insertion: measured −54.2% size and −61% latency at
# EQUAL recall vs the paper's bsize=32 (eval_storage_levers.py, §9 of
# docs/LEXICAL_STORAGE_RESEARCH.md).
BMP_BLOCK_SIZE = 256
# Docs are buffered and sorted by top SPLADE term in chunks before BMP
# insertion (bounded-RAM approximation of global clustering; the win comes
# from term locality within 256-doc blocks, which chunk-local sorting keeps).
BMP_CLUSTER_CHUNK_DOCS = 500_000
TANTIVY_WRITER_HEAP = 1_000_000_000


# v2 meta schema: W-ids stored as INTEGER PRIMARY KEY (rowid alias, varint on
# disk) — measured 166.8 -> 99.4 B/doc vs TEXT ids (storage_levers_eval, ~10 GB
# at 150M). Ids that don't round-trip W<digits> go to the docs_other TEXT table.
META_SCHEMA = (
    "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
    "CREATE TABLE IF NOT EXISTS docs ("
    "  id INTEGER PRIMARY KEY, title TEXT, doi TEXT, year INTEGER,"
    "  type TEXT, is_oa INTEGER, section TEXT);"
    "CREATE TABLE IF NOT EXISTS docs_other ("
    "  id TEXT PRIMARY KEY, title TEXT, doi TEXT, year INTEGER,"
    "  type TEXT, is_oa INTEGER, section TEXT);"
)


def encode_meta_id(doc_id: str) -> int | None:
    """'W2031234567' -> 2031234567, or None when the id wouldn't round-trip
    (no W prefix, non-digits, leading zero) and must stay TEXT."""
    if len(doc_id) > 1 and doc_id[0] == "W":
        digits = doc_id[1:]
        if digits.isdigit() and digits[0] != "0":
            return int(digits)
    return None


def open_meta_db(index_root: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(Path(index_root) / "meta.sqlite"))
    db.executescript(META_SCHEMA)
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
        self._meta_batch_other: list[tuple] = []
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
        self._bmp_vocab: set[str] = set()
        self._bmp_chunk: list[tuple[str, str, dict]] = []  # (top_term, id, vec)

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
            # Per-shard term vocabulary sidecar: BMP panics on queries whose
            # terms are ALL absent from a shard — the retriever uses this to
            # skip non-overlapping shards.
            import zstandard
            vocab_path = self.dir / f"splade_{self._bmp_shard_idx:03d}.vocab.zst"
            vocab_path.write_bytes(zstandard.ZstdCompressor(level=9).compress(
                "\n".join(sorted(self._bmp_vocab)).encode()))
            self._bmp_vocab.clear()
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
            vec = {t: min(BMP_IMPACT_MAX, max(1, int(round(w * QUANT_SCALE))))
                   for t, w in sparse.items() if w > 0}
            if vec:
                self._bmp_chunk.append((max(vec, key=vec.get), doc_id, vec))
                if len(self._bmp_chunk) >= BMP_CLUSTER_CHUNK_DOCS:
                    self._flush_bmp_chunk()

        if self._abstracts is not None:
            self._abstracts.add(doc_id, abstract)

        nid = encode_meta_id(doc_id)
        row = (nid if nid is not None else doc_id, title, doc.get("doi") or "",
               year, dtype, int(is_oa), self.section)
        (self._meta_batch if nid is not None
         else self._meta_batch_other).append(row)
        if len(self._meta_batch) + len(self._meta_batch_other) >= 5_000:
            self._flush_meta()
        self.count += 1

    def _flush_bmp_chunk(self, final: bool = False) -> None:
        """Sort the buffered chunk by top SPLADE term, then stream into BMP
        (mid-chunk shard rotation preserves clustered order across shards).
        On the final flush, a too-small tail is absorbed into the current
        shard instead of opening a broken sub-minimum shard."""
        self._bmp_chunk.sort(key=lambda t: t[0])
        n = len(self._bmp_chunk)
        for pos, (_, doc_id, vec) in enumerate(self._bmp_chunk):
            self._bmp_indexer().add_document(doc_id, vec)
            self._bmp_vocab.update(vec)
            self._bmp_in_shard += 1
            if self._bmp_in_shard >= self._bmp_shard_docs:
                remaining = n - pos - 1
                if final and remaining < MIN_BMP_TAIL_DOCS:
                    continue  # absorb the tail into this shard
                self._rotate_bmp()
        self._bmp_chunk.clear()

    def _flush_meta(self) -> None:
        self._meta.executemany(
            "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?)", self._meta_batch)
        self._meta_batch.clear()
        if self._meta_batch_other:
            self._meta.executemany(
                "INSERT OR REPLACE INTO docs_other VALUES (?,?,?,?,?,?,?)",
                self._meta_batch_other)
            self._meta_batch_other.clear()

    def finish(self) -> dict:
        self._flush_meta()
        self._meta.commit()
        self._writer.commit()
        self._writer.wait_merging_threads()
        self._flush_bmp_chunk(final=True)
        if self._bmp is not None and self._bmp_in_shard < 500 and self._bmp_shard_idx == 0:
            logger.warning(
                "section %s: only %d sparse docs — below BMP's working minimum; "
                "sparse leg will be empty here (tantivy still covers it)",
                self.section, self._bmp_in_shard)
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

    Validated recipe (docs/archive/COMPRESSION_EVAL_RESULTS.md): binary sign codes in
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
