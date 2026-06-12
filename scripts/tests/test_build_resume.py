#!/usr/bin/env python3
"""Kill/resume parity test for the per-slice checkpointed section build.

Builds the same small real-data spool twice:
  clean: one uninterrupted build_section_worker run per section
  chaos: the worker subprocess is SIGKILLed every 8-25 s and restarted until
         it completes — exercising resume (partial-shard cleanup, tantivy
         dup-probe, abstracts pending-buffer restore) at random crash points.

Parity asserted between the two roots:
  meta rows exact-equal, tantivy num_docs + id probes, BMP merged search
  results exact-equal (shard layout is deterministic), abstract fetch parity.

Usage:
    .venv/bin/python3 scripts/tests/test_build_resume.py
Needs data/thinclient_index/spool_backup/ (hardlinked 150M spool).
"""

import json
import os
import random
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import zstandard

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

SOURCE_SPOOL = REPO_ROOT / "data/thinclient_index/spool_backup"
TEST_ROOT = REPO_ROOT / "data/thinclient_test_resume"
SECTIONS = ["social_sciences", "med_bio"]   # one hot (abstracts), one cold
HOT_SUBS = ["social_sciences__recent"]
DOCS_PER_SLICE = 12_000
N_SLICES = 5
BMP_SHARD_DOCS = 6_000          # forces mid-slice + boundary rotations
KILL_FIRST = 8.0                # first kill lands early (tight crash windows)
KILL_GROWTH = 1.3               # then escalate so the run must converge:
KILL_CAP = 120.0                # a fixed window < cycle time livelocks
MAX_CHAOS_RUNS = 60


def prep_spool(root: Path) -> None:
    for section in SECTIONS:
        out_dir = root / "spool" / section
        out_dir.mkdir(parents=True, exist_ok=True)
        src = SOURCE_SPOOL / section / "slice_000.jsonl.zst"
        dctx = zstandard.ZstdDecompressor()
        with open(src, "rb") as fh, dctx.stream_reader(fh) as reader:
            buf = b""
            lines: list[bytes] = []
            while len(lines) < DOCS_PER_SLICE * N_SLICES:
                chunk = reader.read(8 << 20)
                if not chunk:
                    break
                buf += chunk
                *new, buf = buf.split(b"\n")
                lines.extend(ln for ln in new if ln)
        lines = lines[:DOCS_PER_SLICE * N_SLICES]
        for s in range(N_SLICES):
            part = lines[s * DOCS_PER_SLICE:(s + 1) * DOCS_PER_SLICE]
            with open(out_dir / f"slice_{s:03d}.jsonl.zst", "wb") as fh:
                with zstandard.ZstdCompressor(level=3).stream_writer(fh) as zw:
                    zw.write(b"\n".join(part) + b"\n")


def worker_main(root: str, section: str) -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_thinclient_index import build_section_worker
    infos = build_section_worker(root, section, HOT_SUBS,
                                 keep_spool=True, bmp_shard_docs=BMP_SHARD_DOCS)
    print(json.dumps({k: v["docs"] for k, v in infos.items()}))


def run_clean(root: Path) -> None:
    for section in SECTIONS:
        subprocess.run([sys.executable, __file__, "--worker", str(root),
                        section], check=True, timeout=1200)


def run_chaos(root: Path) -> int:
    kills = 0
    for section in SECTIONS:
        for attempt in range(MAX_CHAOS_RUNS):
            window = min(KILL_FIRST * KILL_GROWTH ** attempt, KILL_CAP)
            proc = subprocess.Popen([sys.executable, __file__, "--worker",
                                     str(root), section])
            try:
                proc.wait(timeout=window * random.uniform(0.8, 1.2))
                if proc.returncode == 0:
                    break
                raise AssertionError(
                    f"{section}: worker exited {proc.returncode} (not a kill)")
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGKILL)
                proc.wait()
                kills += 1
        else:
            raise AssertionError(f"{section}: no completion in "
                                 f"{MAX_CHAOS_RUNS} chaos runs")
    return kills


def load_meta(root: Path, section: str) -> dict:
    rows = {}
    db = sqlite3.connect(str(root / f"meta_{section}.sqlite"))
    for table in ("docs", "docs_other"):
        for r in db.execute(f"SELECT * FROM {table}"):
            rows[(table, r[0])] = r[1:]
    db.close()
    return rows


def tantivy_probe(sub_dir: Path, ids: list[str]) -> tuple[int, int]:
    import tantivy
    idx = tantivy.Index.open(str(sub_dir / "tantivy"))
    idx.reload()
    searcher = idx.searcher()
    found = sum(
        bool(searcher.search(idx.parse_query(f'id:"{i}"', ["id"]), 1).hits)
        for i in ids)
    return searcher.num_docs, found


def bmp_results(sub_dir: Path, qvec: dict) -> list[tuple[str, float]]:
    import bmp
    merged = []
    for p in sorted(sub_dir.glob("splade_*.bmp")):
        vocab = set(zstandard.ZstdDecompressor().decompress(
            p.with_suffix(".vocab.zst").read_bytes()).decode().split("\n"))
        q = {t: w for t, w in qvec.items() if t in vocab}
        if not q:
            continue
        ids, scores = bmp.Searcher(str(p)).search(q, k=20, alpha=0.8, beta=1.0)
        merged.extend(zip(ids, map(float, scores)))
    merged.sort(key=lambda t: (-t[1], t[0]))
    return merged[:15]


def sample_docs(root: Path, section: str, n: int) -> list[dict]:
    """Every k-th doc of the test spool (parsed) for probes/queries."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_thinclient_index import iter_slice
    docs = []
    for s in range(N_SLICES):
        for i, doc in enumerate(
                iter_slice(root / "spool" / section / f"slice_{s:03d}.jsonl.zst")):
            if i % (DOCS_PER_SLICE // (n // N_SLICES + 1)) == 0:
                docs.append(doc)
    return docs


def verify(clean: Path, chaos: Path) -> None:
    from lib.thinclient.abstracts import AbstractStore
    from lib.thinclient.builder import QUANT_SCALE
    from lib.thinclient.sections import era_of

    for section in SECTIONS:
        m_clean, m_chaos = load_meta(clean, section), load_meta(chaos, section)
        assert m_clean == m_chaos, (
            f"{section}: meta mismatch — {len(m_clean)} vs {len(m_chaos)} rows; "
            f"sym-diff sample: {list(set(m_clean) ^ set(m_chaos))[:5]}")
        print(f"  {section}: meta parity OK ({len(m_clean)} rows)")

        docs = sample_docs(clean, section, 100)
        by_era: dict[str, list[str]] = {}
        for d in docs:
            by_era.setdefault(
                era_of(int(d.get("publication_year") or 0)), []).append(d["id"])
        for era, ids in by_era.items():
            sub = f"{section}__{era}"
            n1, f1 = tantivy_probe(clean / "sections" / sub, ids)
            n2, f2 = tantivy_probe(chaos / "sections" / sub, ids)
            assert n1 == n2, f"{sub}: tantivy num_docs {n1} != {n2}"
            assert f1 == f2 == len(ids), (
                f"{sub}: tantivy probes clean {f1}/{len(ids)} "
                f"chaos {f2}/{len(ids)}")
            print(f"  {sub}: tantivy parity OK ({n1} docs, "
                  f"{len(ids)}/{len(ids)} probes)")

            queries = [d for d in docs if d.get("sparse_field")
                       and era_of(int(d.get("publication_year") or 0)) == era][:5]
            for d in queries:
                qvec = {t: max(1, int(round(w * QUANT_SCALE)))
                        for t, w in sorted(d["sparse_field"].items(),
                                           key=lambda x: -x[1])[:64] if w > 0}
                r1 = bmp_results(clean / "sections" / sub, qvec)
                r2 = bmp_results(chaos / "sections" / sub, qvec)
                assert r1 == r2, (f"{sub}: BMP results diverge for query from "
                                  f"{d['id']}:\n{r1}\nvs\n{r2}")
            print(f"  {sub}: BMP search parity OK ({len(queries)} queries)")

    sub = "social_sciences__recent"
    s1 = AbstractStore(clean / "sections" / sub / "abstracts.sqlite")
    s2 = AbstractStore(chaos / "sections" / sub / "abstracts.sqlite")
    with_abs = [d["id"] for d in sample_docs(clean, "social_sciences", 400)
                if d.get("abstract")
                and era_of(int(d.get("publication_year") or 0)) == "recent"]
    a1, a2 = s1.fetch(with_abs), s2.fetch(with_abs)
    assert a1 == a2, (f"abstract fetch mismatch: clean {len(a1)} chaos "
                      f"{len(a2)}, diff ids {list(set(a1) ^ set(a2))[:5]}")
    assert len(a1) == len(with_abs), (
        f"abstracts missing: {len(a1)}/{len(with_abs)} fetched")
    print(f"  {sub}: abstracts parity OK ({len(a1)}/{len(with_abs)} fetched)")


def main() -> None:
    random.seed(20260612)
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    clean, chaos = TEST_ROOT / "clean", TEST_ROOT / "chaos"
    print("prepping test spool ...")
    prep_spool(clean)
    prep_spool(chaos)

    t0 = time.time()
    print("clean build ...")
    run_clean(clean)
    print(f"clean build done in {time.time() - t0:.0f}s")

    t0 = time.time()
    print(f"chaos build (SIGKILL escalating from {KILL_FIRST}s) ...")
    kills = run_chaos(chaos)
    print(f"chaos build done in {time.time() - t0:.0f}s after {kills} kills")
    assert kills >= 3, f"only {kills} kills — test too weak, lower KILL_EVERY"

    print("verifying parity ...")
    verify(clean, chaos)
    print(f"PASS: clean == chaos across {len(SECTIONS)} sections "
          f"({kills} mid-build kills survived)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2], sys.argv[3])
    else:
        main()
