"""Unit tests for the thin-client stack: builder -> retriever -> packer.

Self-contained (synthetic 400-doc corpus, no network, no models): SPLADE query
encoding is bypassed by querying BMP directly through the retriever's internals
where needed; the BM25F leg and filters run the real tantivy path.

Run:  .venv/bin/python3 -m pytest src/tests/test_thinclient.py -q
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.thinclient.abstracts import AbstractStore, AbstractStoreWriter  # noqa: E402
from lib.thinclient.builder import SectionBuilder, open_meta_db  # noqa: E402
from lib.thinclient.packer import pack_section, unpack_section  # noqa: E402
from lib.thinclient.retriever import ThinClientRetriever, _normalize_filters  # noqa: E402
from lib.thinclient.sections import classify_doc  # noqa: E402

TOPICS = {
    "med_bio": ("cancer therapy clinical patient gene", "tumor immunotherapy"),
    "cs_math": ("algorithm optimization graph network", "machine learning model"),
}


def make_docs(n: int = 400) -> list[dict]:
    docs = []
    for i in range(n):
        sec, (words, extra) = list(TOPICS.items())[i % 2]
        docs.append({
            "id": f"W{i:06d}",
            "title": f"{extra} study {i}",
            "abstract": f"A paper about {words}. Sample {i} with details on {extra}.",
            "publication_year": 2015 + (i % 10),
            "type": "article" if i % 3 else "review",
            "is_oa": bool((i // 2) % 2),  # varies WITHIN each section
            "doi": f"10.1/{i}",
            "sparse_field": {w: 1.0 + (i % 5) * 0.1 for w in words.split()[:4]},
        })
    return docs


@pytest.fixture(scope="module")
def index_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("tc_index")
    meta = open_meta_db(root)
    builders = {}
    counts = {}
    for d in make_docs():
        sec = classify_doc(d["title"], d["abstract"])
        if sec not in builders:
            builders[sec] = SectionBuilder(root, sec, hot=True, meta_db=meta,
                                           bmp_shard_docs=150)  # force shard rotation
        builders[sec].add(d)
        counts[sec] = counts.get(sec, 0) + 1
    infos = {s: b.finish() for s, b in builders.items()}
    meta.close()
    manifest = {"sections": {s: {"state": "live", "docs": i["docs"]}
                             for s, i in infos.items()}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_sections_disjoint_and_complete(index_root):
    import sqlite3
    db = sqlite3.connect(str(index_root / "meta.sqlite"))
    total = db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    assert total == 400
    per = dict(db.execute("SELECT section, COUNT(*) FROM docs GROUP BY section"))
    assert sum(per.values()) == 400 and len(per) >= 2


def test_bmp_shards_rotated(index_root):
    shards = list((index_root / "sections").glob("*/splade_*.bmp"))
    assert len(shards) > 2  # 150-doc shard cap on ~200-doc sections rotates


def test_bm25f_search_and_filters(index_root):
    r = ThinClientRetriever(index_root=str(index_root), remote_abstracts=False)
    assert r.is_available()
    hits = r.search("cancer immunotherapy tumor", top_k=10, mode="bm25f")
    assert hits and all(h["title"] for h in hits)
    assert all(set(h) >= {"doi", "openalex_id", "title", "abstract", "score",
                          "publication_year", "year", "date", "type", "is_oa",
                          "source"} for h in hits)
    filtered = r.search("cancer immunotherapy tumor", top_k=10, mode="bm25f",
                        filters={"publication_year": "2020-2024"})
    assert filtered
    assert all(2020 <= h["publication_year"] <= 2024 for h in filtered)
    oa = r.search("cancer therapy", top_k=10, mode="bm25f",
                  filters={"is_oa": True})
    assert oa and all(h["is_oa"] for h in oa)


def test_splade_leg_via_bmp(index_root):
    # Query BMP through the leg internals with a pre-encoded sparse vector
    # (avoids the ONNX model dependency in unit tests).
    r = ThinClientRetriever(index_root=str(index_root), remote_abstracts=False)
    r._load()
    # vocab sidecars must exist (BMP panics on zero-overlap queries without them)
    assert list((index_root / "sections").glob("*/splade_*.vocab.zst"))
    qvec = {"cancer": 200, "therapy": 100}
    out = []
    for sec in r._sections.values():
        for shard in sec["bmp"]:
            q_here = qvec
            if shard["vocab"] is not None:
                q_here = {t: w for t, w in qvec.items() if t in shard["vocab"]}
                if not q_here:
                    continue  # the skip path that prevents the panic
            ids, scores = shard["searcher"].search(q_here, k=10, alpha=1.0, beta=1.0)
            out.extend(zip(ids, scores))
    assert out
    top = max(out, key=lambda t: t[1])
    assert top[0].startswith("W")


def test_abstract_sidecar_roundtrip(index_root):
    stores = list((index_root / "sections").glob("*/abstracts.sqlite"))
    assert stores
    store = AbstractStore(stores[0])
    import sqlite3
    db = sqlite3.connect(str(stores[0]))
    some_ids = [row[0] for row in db.execute("SELECT id FROM abs LIMIT 5")]
    out = store.fetch(some_ids)
    assert len(out) == 5 and all(v.startswith("A paper about") for v in out.values())


def test_hydrate_fills_abstracts_locally(index_root):
    r = ThinClientRetriever(index_root=str(index_root), remote_abstracts=False)
    hits = r.search("graph optimization algorithm", top_k=5, mode="bm25f")
    assert hits and all(h["abstract"] for h in hits)  # hot section => local


def test_rrf_fusion_shape(index_root):
    r = ThinClientRetriever(index_root=str(index_root), remote_abstracts=False)
    r._load()
    # monkeypatch the splade leg to avoid the ONNX dependency
    r._splade_leg = lambda q, k, f: [(h["openalex_id"], 1.0) for h in
                                     r.search(q, k, mode="bm25f")][::-1]
    fused = r.search_rrf("machine learning network model", top_k=10)
    assert fused and fused[0]["source"] == "thinclient_rrf"
    assert fused == sorted(fused, key=lambda d: -d["score"])


def test_pack_unpack_parity(index_root):
    r = ThinClientRetriever(index_root=str(index_root), remote_abstracts=False)
    section = r.live_sections()[0]
    before = [(h["openalex_id"], round(h["score"], 6))
              for h in r.search("cancer therapy clinical", top_k=10, mode="bm25f")]

    info = pack_section(index_root, section)
    assert info["ratio"] > 1.0
    assert not (index_root / "sections" / section).exists()

    out = unpack_section(index_root, section)
    assert out["parity_checksum_ok"] is True  # bit-identical artifacts

    r2 = ThinClientRetriever(index_root=str(index_root), remote_abstracts=False)
    after = [(h["openalex_id"], round(h["score"], 6))
             for h in r2.search("cancer therapy clinical", top_k=10, mode="bm25f")]
    assert before == after  # exact result parity, scores included


def test_filter_normalization():
    f = _normalize_filters({"publication_year": "2018-2022", "type": "article",
                            "open_access.is_oa": "true"})
    assert f == {"year_range": (2018, 2022), "type": "article", "is_oa": True}
    assert _normalize_filters({"from_publication_date": "2019-05-01"}) == {
        "year_range": (2019, 3000)}
    assert _normalize_filters(None) is None
    assert _normalize_filters({}) is None
