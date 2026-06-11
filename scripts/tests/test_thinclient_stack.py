"""Regression suite for the thin-client stack (2026-06-11 batch):
meta v1/v2 schema, era sub-sections + pruning, abstracts v1/v2/v3,
dense warm cache, MCP index-management tools.

Run: .venv/bin/python3 -m pytest scripts/tests/test_thinclient_stack.py -q
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import zstandard

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from lib.thinclient.builder import (SectionBuilder, encode_meta_id,  # noqa: E402
                                    open_meta_db)
from lib.thinclient.retriever import ThinClientRetriever, _era_skip  # noqa: E402
from lib.thinclient.sections import (PERSONAS, SUBSECTION_NAMES, era_of,  # noqa: E402
                                     subsection_name)
import lib.thinclient.abstracts as A  # noqa: E402

EN = ("This study examines the political economy of climate adaptation in "
      "coastal regions, drawing on comparative survey data and case studies. ")
RU = ("В данной работе рассматривается политическая экономия адаптации к "
      "изменению климата в прибрежных регионах. ")
ZH = "本文研究沿海地区气候适应的政治经济学，基于比较调查数据和案例研究。"


def _mkdoc(i: int, year: int = 2015, section_word: str = "politics") -> dict:
    return {"id": f"W{i}", "title": f"{section_word} paper {i}",
            "abstract": f"{section_word} text {i}", "publication_year": year,
            "type": "article", "is_oa": i % 2 == 0, "doi": f"10.1/{i}",
            "sparse_field": {section_word: 1.2, f"t{i % 7}": 0.8}}


def _build_index(root: Path, n: int = 600, hot: bool = True) -> None:
    meta = open_meta_db(root)
    b = SectionBuilder(root, "social_sciences__recent", hot=hot, meta_db=meta)
    for i in range(1, n + 1):
        b.add(_mkdoc(i))
    b.finish()
    meta.commit()
    meta.close()


# ── meta schema ───────────────────────────────────────────────────────────────

def test_encode_meta_id():
    assert encode_meta_id("W2031234567") == 2031234567
    for bad in ("W0123", "X123", "W12a3", "W"):
        assert encode_meta_id(bad) is None


def test_meta_v2_roundtrip_and_overflow(tmp_path):
    meta = open_meta_db(tmp_path)
    b = SectionBuilder(tmp_path, "social_sciences__recent", hot=False, meta_db=meta)
    for i in range(1, 601):
        b.add(_mkdoc(i))
    odd = _mkdoc(9999)
    odd["id"] = "ODDID-42"
    b.add(odd)
    b.finish()
    meta.commit(); meta.close()

    db = sqlite3.connect(tmp_path / "meta.sqlite")
    assert db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 600
    assert db.execute("SELECT COUNT(*) FROM docs_other").fetchone()[0] == 1
    assert db.execute("SELECT typeof(id) FROM docs LIMIT 1").fetchone()[0] == "integer"

    r = ThinClientRetriever(index_root=str(tmp_path), remote_abstracts=False)
    r._load()
    rows = r._meta_rows(["ODDID-42", "W5", "Wnope"])
    assert rows["W5"][1] == "politics paper 5"
    assert rows["ODDID-42"][1] == "politics paper 9999"
    assert "Wnope" not in rows
    assert r._meta_numeric


def test_meta_v1_text_schema_still_served(tmp_path):
    db = sqlite3.connect(tmp_path / "meta.sqlite")
    db.executescript(
        "CREATE TABLE docs (id TEXT PRIMARY KEY, title TEXT, doi TEXT,"
        " year INTEGER, type TEXT, is_oa INTEGER, section TEXT);")
    db.execute("INSERT INTO docs VALUES ('W1','t','d',2020,'article',1,'s')")
    db.commit(); db.close()
    r = ThinClientRetriever(index_root=str(tmp_path), remote_abstracts=False)
    r._load()
    assert not r._meta_numeric
    assert r._meta_rows(["W1"])["W1"][1] == "t"


# ── era sub-sections ─────────────────────────────────────────────────────────

def test_era_helpers_and_personas():
    assert era_of(2015) == "recent" and era_of(1995) == "archive"
    assert era_of(None) == "archive"
    assert subsection_name("cs_math", 2020) == "cs_math__recent"
    assert PERSONAS["political_science"] == ["social_sciences__recent"]
    assert "social_sciences__archive" in PERSONAS["historical"]
    assert all(s.endswith("__recent") for s in PERSONAS["contemporary"])
    assert PERSONAS["all_hot"] == SUBSECTION_NAMES


def test_era_skip():
    assert _era_skip("x__archive", {"year_range": (2015, 2020)})
    assert _era_skip("x__recent", {"year_range": (1980, 1999)})
    assert not _era_skip("x__recent", {"year_range": (2000, 2020)})
    assert not _era_skip("x__recent", None)
    assert not _era_skip("legacy_name", {"year_range": (2015, 2020)})


def test_build_worker_era_routing(tmp_path):
    import build_thinclient_index as B
    spool = tmp_path / "spool" / "social_sciences"
    spool.mkdir(parents=True)
    with open(spool / "slice_000.jsonl.zst", "wb") as fh:
        zw = zstandard.ZstdCompressor(level=3).stream_writer(fh)
        for i in range(400):
            zw.write((json.dumps(_mkdoc(i + 1, year=2015 if i % 2 else 1990))
                      + "\n").encode())
        zw.close()
    infos = B.build_section_worker(str(tmp_path), "social_sciences",
                                   ["social_sciences__recent"], True, 2_000_000)
    assert infos["social_sciences__recent"]["docs"] == 200
    assert infos["social_sciences__recent"]["hot"]
    assert not infos["social_sciences__archive"]["hot"]
    assert (tmp_path / "sections/social_sciences__recent/abstracts.sqlite").exists()
    assert not (tmp_path / "sections/social_sciences__archive/abstracts.sqlite").exists()
    B.merge_section_meta(tmp_path, ["social_sciences"])

    r = ThinClientRetriever(index_root=str(tmp_path), remote_abstracts=False)
    res = r.search("politics", top_k=50, filters={"publication_year": "2012-2024"})
    assert res and all(d["year"] == 2015 for d in res)
    res_old = r.search("politics", top_k=50, filters={"publication_year": "1980-1999"})
    assert res_old and all(d["year"] == 1990 for d in res_old)
    assert r.cold_section_hint("quantum semiconductor physics") == "phys_eng__recent"
    assert r.cold_section_hint("quantum semiconductor physics",
                               filters={"publication_year": "1950-1990"}
                               ) == "phys_eng__archive"
    assert r.cold_section_hint("political governance") is None


# ── abstracts v3 (+ v1/v2 read compat) ───────────────────────────────────────

def test_script_bucket():
    assert A.script_bucket(EN) == "latin"
    assert A.script_bucket(RU) == "cyrillic"
    assert A.script_bucket(ZH) == "han"
    assert A.script_bucket("") == "latin"


def test_abstracts_v3_roundtrip(tmp_path):
    w = A.AbstractStoreWriter(tmp_path / "a.sqlite")
    docs = {f"W{i}": EN + f"variant {i}" for i in range(2000)}
    docs.update({f"WR{i}": RU + f"вариант {i}" for i in range(400)})
    docs.update({f"WZ{i}": ZH + f"变体 {i}" for i in range(40)})  # below bar
    for k, v in docs.items():
        w.add(k, v)
    w.add("Wempty", "")
    assert w.finish() == len(docs)

    r = A.AbstractStore(tmp_path / "a.sqlite")
    assert r._v3
    fetched = r.fetch(list(docs))
    assert fetched == docs
    db = sqlite3.connect(tmp_path / "a.sqlite")
    n_dicts = db.execute(
        "SELECT COUNT(*) FROM meta WHERE k LIKE 'zdict_%'").fetchone()[0]
    assert n_dicts == 2   # latin + cyrillic; han below bar -> latin fallback


def test_abstracts_v1_v2_read_compat(tmp_path):
    p1 = tmp_path / "v1.sqlite"
    db = sqlite3.connect(p1)
    db.executescript("CREATE TABLE abs (id TEXT PRIMARY KEY, z BLOB);"
                     "CREATE TABLE meta (k TEXT PRIMARY KEY, v BLOB);")
    db.execute("INSERT INTO abs VALUES (?,?)",
               ("W1", zstandard.ZstdCompressor(level=19).compress(b"old v1")))
    db.commit(); db.close()
    assert A.AbstractStore(p1).fetch(["W1"]) == {"W1": "old v1"}

    p2 = tmp_path / "v2.sqlite"
    db = sqlite3.connect(p2)
    db.executescript(
        "CREATE TABLE abs (id TEXT PRIMARY KEY, z BLOB, d INTEGER NOT NULL DEFAULT 0);"
        "CREATE TABLE meta (k TEXT PRIMARY KEY, v BLOB);")
    zd = zstandard.train_dictionary(
        112 * 1024, [(RU + str(i)).encode() for i in range(400)])
    db.execute("INSERT INTO meta VALUES ('zdict_ru', ?)", (zd.as_bytes(),))
    cc = zstandard.ZstdCompressor(level=19, dict_data=zd)
    db.execute("INSERT INTO abs VALUES (?,?,1)", ("WR1", cc.compress(RU.encode())))
    db.commit(); db.close()
    assert A.AbstractStore(p2).fetch(["WR1"]) == {"WR1": RU}


# ── dense warm cache ─────────────────────────────────────────────────────────

@pytest.fixture
def stub_encoder():
    rng = np.random.default_rng(7)
    cache: dict[str, np.ndarray] = {}

    def enc(texts):
        out = []
        for t in texts:
            if t not in cache:
                v = rng.standard_normal(384).astype(np.float32)
                cache[t] = v / np.linalg.norm(v)
            out.append(cache[t])
        return np.array(out)
    return enc


def _wait(pred, timeout=30):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.1)
    assert pred()


def test_dense_cache_lifecycle(tmp_path, stub_encoder):
    from lib.thinclient.dense_cache import DenseWarmCache
    c = DenseWarmCache(tmp_path, seed_ids=["W1"], model_path="",
                       encoder=stub_encoder)
    docs = [{"openalex_id": f"W{i}", "title": f"doc {i}",
             "abstract": f"about {i}", "doi": "", "year": 2020,
             "type": "article", "is_oa": True} for i in range(1, 101)]
    c.enqueue(docs)
    c.enqueue(docs)                      # dedup
    _wait(lambda: c.info()["docs"] == 99)   # seed id W1 skipped
    q = stub_encoder(["doc 42 about 42"])[0]
    hits = c.search(np.packbits(q > 0), q, fetch=10)
    assert hits[0][0] == "W42" and hits[0][1] > 0.9
    rows = c.metadata_rows(["W42", "Wmissing"])
    assert rows["W42"][1] == "doc 42" and "Wmissing" not in rows
    c.close()
    # crash-restore from sqlite
    c2 = DenseWarmCache(tmp_path, seed_ids=["W1"], model_path="",
                        encoder=stub_encoder)
    assert c2.info()["docs"] == 99
    assert c2.search(np.packbits(q > 0), q, fetch=5)[0][0] == "W42"
    c2.close()


def test_dense_cache_only_leg(tmp_path, stub_encoder, monkeypatch):
    _build_index(tmp_path)
    r = ThinClientRetriever(index_root=str(tmp_path), remote_abstracts=False)
    r._load()
    assert r._dense is None and r._dense_cache is not None
    r._dense_cache._encoder = stub_encoder
    res = r.search("politics", top_k=10)
    assert res
    _wait(lambda: r._dense_cache.info()["docs"] >= len(res))
    import lib.opensearch_retriever as osr
    monkeypatch.setattr(osr, "encode_dense",
                        lambda text, mp: stub_encoder([text])[0])
    d0 = res[0]
    out = r.dense_search(f"{d0['title']} {d0['abstract']}", top_k=5)
    assert out and out[0]["openalex_id"] == d0["openalex_id"]


# ── retriever serving extras ─────────────────────────────────────────────────

def test_metrics_query_log_and_reload(tmp_path):
    _build_index(tmp_path)
    r = ThinClientRetriever(index_root=str(tmp_path), remote_abstracts=False,
                            query_log_path=str(tmp_path / "qlog.jsonl"))
    res = r.search("politics", top_k=5)
    assert res and res[0]["abstract"]            # hot-section v3 sidecar
    m = r.metrics()
    assert m["legs"]["bm25f"]["count"] >= 1
    entry = json.loads((tmp_path / "qlog.jsonl").read_text().strip().split("\n")[-1])
    assert entry["leg"] == "bm25f" and entry["results"] == 5
    info = r.reload()
    assert "social_sciences__recent" in info["live_sections"]
    assert r.search("politics", top_k=3)


# ── MCP tools (integration; needs no network) ────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_index_tools(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    _build_index(tmp_path)
    monkeypatch.setenv("SFU_SEARCH_BACKEND", "thinclient")
    monkeypatch.setenv("SFU_THINCLIENT_INDEX_ROOT", str(tmp_path))
    monkeypatch.setenv("SFU_DENSE_WARMCACHE", "0")
    from lib import tools
    monkeypatch.setattr(tools, "_federated_router", None)
    monkeypatch.setattr(tools, "_thinclient_retriever", None)
    monkeypatch.setattr(tools, "_config", None, raising=False)

    names = {t.name for t in tools.TOOL_DEFINITIONS}
    assert {"get_index_status", "list_personas", "request_section_unpack",
            "reload_index"} <= names

    out = await tools._dispatch_tool("list_personas", {})
    data = json.loads(out[0].text)
    assert "contemporary" in data["personas"]
    assert data["live_sections"] == ["social_sciences__recent"]
    out = await tools._dispatch_tool("request_section_unpack", {"section": "bogus"})
    assert "Unknown section" in out[0].text
    out = await tools._dispatch_tool("request_section_unpack",
                                     {"section": "social_sciences__recent"})
    assert "already live" in out[0].text
    out = await tools._dispatch_tool("reload_index", {})
    assert json.loads(out[0].text)["reloaded"]["live_sections"]
