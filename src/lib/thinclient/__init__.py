"""Thin-client search stack — no-JVM replacement for OpenSearch serving.

Engines (validated in docs/THIN_CLIENT_STACK_RESEARCH.md):
  - tantivy  : BM25F lexical leg (title^3 + abstract, freq-only, fast-field filters)
  - bmp      : SPLADE sparse leg (block-max pruning, 8-bit impacts)
  - usearch  : dense leg (binary Hamming + int8/fp32 rescore from mmap)

Fusion stays app-side RRF (k=60), same as the OpenSearch path.

Layout of an index root (data/thinclient_index by default):
  manifest.json                 build provenance + section states (live/packed)
  meta.sqlite                   id -> title/doi/year/type/is_oa/section (global)
  sections/<name>/tantivy/      live lexical index
  sections/<name>/splade_*.bmp  live sparse shards
  sections/<name>/abstracts.sqlite   zstd-dict abstract sidecar (HOT sections only)
  packed/<name>.tar.zst         cold sections (zstd-19 LDM archive of artifacts)
  dense/                        usearch b1 index + int8 rescore memmap + ids
"""

from lib.thinclient.retriever import ThinClientRetriever  # noqa: F401
from lib.thinclient.sections import SECTIONS, PERSONAS, classify_doc  # noqa: F401
