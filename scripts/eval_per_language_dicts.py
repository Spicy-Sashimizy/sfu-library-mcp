#!/usr/bin/env python3
"""Per-language zstd dictionary eval with HELD-OUT methodology + machine-code control.

Extends the language analysis in scripts/eval_text_compression.py (§7 of
docs/LEXICAL_STORAGE_RESEARCH.md). The prior numbers trained and measured the
per-language dictionaries on the SAME docs (overfit). This eval:

  * reuses the same 20k-abstract sample (data/text_compression_sample.jsonl.gz)
    and the same dict params (112 KB dictionary, zstd level 19),
  * detects language with py3langid (same as the prior eval),
  * for every language with >= MIN_DOCS docs: fixed-seed 80/20 train/held-out
    split, trains a dict on the TRAIN split only, and measures bytes/doc on
    the HELD-OUT split under four codecs:
        own-language dict / English-trained dict / shared all-language dict /
        no dict (plain zstd-19),
  * adds a "machine_code" pseudo-language control: ~2000 chunks of real ELF
    binary bytes from /usr/bin + /usr/lib, sized like abstracts (800-1500 B),
    same split and dict params.

All dictionaries are trained ONLY on train splits, so held-out docs are never
seen by any dictionary (including the shared one).

Run (single process, CPU-light — a migration export shares this machine):
    /workspaces/sfu-library-thinclient/.venv/bin/python3 \
        scripts/eval_per_language_dicts.py

Output: data/eval_results/per_language_dicts_20260611.json
"""

import json
import gzip
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import zstandard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_per_language_dicts")

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_PATH = REPO_ROOT / "data/text_compression_sample.jsonl.gz"
OUTPUT = REPO_ROOT / "data/eval_results/per_language_dicts_20260611.json"

DICT_SIZE = 112 * 1024      # same as eval_text_compression.py
ZSTD_LEVEL = 19             # same as production sidecar + prior eval
MIN_DOCS = 300              # languages below this are skipped (too few to split)
SEED = 42
TRAIN_FRAC = 0.8
MC_CHUNKS = 2000            # machine-code control corpus size
MC_CHUNK_MIN, MC_CHUNK_MAX = 800, 1500   # sized like abstracts (~1200 B avg)


# ── corpus ───────────────────────────────────────────────────────────────────

def load_sample() -> list[dict]:
    docs = [json.loads(l) for l in gzip.open(SAMPLE_PATH, "rt")]
    logger.info("loaded %d docs from %s", len(docs), SAMPLE_PATH)
    return docs


def detect_languages(docs: list[dict]) -> dict[str, list[bytes]]:
    import py3langid
    by_lang: dict[str, list[bytes]] = defaultdict(list)
    for d in docs:
        lang, _ = py3langid.classify(d["abstract"][:600])
        by_lang[lang].append(d["abstract"].encode("utf-8"))
    return by_lang


def collect_machine_code(n_chunks: int, rng: random.Random) -> list[bytes]:
    """Raw chunks from real ELF binaries (.text-ish: offsets past the 4 KB
    header region, within the first 60% of the file where .text usually
    lives). Chunk sizes drawn uniformly from [MC_CHUNK_MIN, MC_CHUNK_MAX] to
    match abstract payload sizes."""
    elf_files = []
    for root in ("/usr/bin", "/usr/lib"):
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.islink(p) or os.path.getsize(p) < 64 * 1024:
                        continue
                    with open(p, "rb") as fh:
                        if fh.read(4) == b"\x7fELF":
                            elf_files.append(p)
                except OSError:
                    continue
            if len(elf_files) >= 600:
                break
        if len(elf_files) >= 600:
            break
    rng.shuffle(elf_files)
    logger.info("machine_code: %d ELF files available", len(elf_files))

    chunks: list[bytes] = []
    per_file = max(1, n_chunks // max(1, min(len(elf_files), 400)) + 1)
    for p in elf_files:
        try:
            data = open(p, "rb").read()
        except OSError:
            continue
        lo, hi = 0x1000, int(len(data) * 0.6)
        if hi - lo < MC_CHUNK_MAX:
            continue
        for _ in range(per_file):
            size = rng.randint(MC_CHUNK_MIN, MC_CHUNK_MAX)
            off = rng.randint(lo, hi - size)
            chunks.append(data[off:off + size])
            if len(chunks) >= n_chunks:
                return chunks
    return chunks


# ── measurement ──────────────────────────────────────────────────────────────

def split(payloads: list[bytes], rng: random.Random) -> tuple[list[bytes], list[bytes]]:
    idx = list(range(len(payloads)))
    rng.shuffle(idx)
    cut = int(len(idx) * TRAIN_FRAC)
    return [payloads[i] for i in idx[:cut]], [payloads[i] for i in idx[cut:]]


def train_dict(samples: list[bytes]):
    try:
        return zstandard.train_dictionary(DICT_SIZE, samples)
    except zstandard.ZstdError as e:
        logger.warning("dict training failed (%d samples): %s", len(samples), e)
        return None


def compressor(zdict=None) -> zstandard.ZstdCompressor:
    if zdict is None:
        return zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL, dict_data=zdict)


def total_compressed(cctx: zstandard.ZstdCompressor, payloads: list[bytes]) -> int:
    return sum(len(cctx.compress(p)) for p in payloads)


def main() -> None:
    docs = load_sample()
    by_lang = detect_languages(docs)
    lang_counts = {k: len(v) for k, v in
                   sorted(by_lang.items(), key=lambda kv: -len(kv[1]))}
    logger.info("languages >= %d docs: %s", MIN_DOCS,
                {k: v for k, v in lang_counts.items() if v >= MIN_DOCS})

    # fixed-seed splits per language (every language, so the shared dict can be
    # trained on the union of ALL train splits and never sees any held-out doc)
    rng = random.Random(SEED)
    splits: dict[str, tuple[list[bytes], list[bytes]]] = {}
    for lang, payloads in sorted(by_lang.items()):
        splits[lang] = split(payloads, random.Random(SEED))

    # machine-code control corpus
    mc_chunks = collect_machine_code(MC_CHUNKS, random.Random(SEED))
    logger.info("machine_code: %d chunks, avg %d B", len(mc_chunks),
                sum(map(len, mc_chunks)) // max(1, len(mc_chunks)))
    splits["machine_code"] = split(mc_chunks, random.Random(SEED))

    # dictionaries — all trained on TRAIN splits only
    shared_train = [p for lang, (tr, _) in splits.items()
                    if lang != "machine_code" for p in tr]
    rng.shuffle(shared_train)
    zd_shared = train_dict(shared_train[:20000])
    zd_english = train_dict(splits["en"][0][:20000])
    c_shared = compressor(zd_shared)
    c_english = compressor(zd_english)
    c_nodict = compressor(None)
    logger.info("shared dict: %d train docs; english dict: %d train docs",
                min(len(shared_train), 20000), len(splits["en"][0]))

    eligible = [lang for lang, (tr, ho) in splits.items()
                if len(tr) + len(ho) >= MIN_DOCS]
    rows = []
    for lang in eligible:
        tr, ho = splits[lang]
        zd_own = train_dict(tr[:20000])
        c_own = compressor(zd_own)
        raw = sum(len(p) for p in ho)
        n = len(ho)
        own = total_compressed(c_own, ho)
        eng = total_compressed(c_english, ho)
        shr = total_compressed(c_shared, ho)
        nod = total_compressed(c_nodict, ho)
        rows.append({
            "language": lang,
            "is_control": lang == "machine_code",
            "docs_total": len(tr) + n,
            "docs_train": len(tr),
            "docs_heldout": n,
            "raw_bytes_per_doc": round(raw / n, 1),
            "own_dict_bytes_per_doc": round(own / n, 1),
            "english_dict_bytes_per_doc": round(eng / n, 1),
            "shared_dict_bytes_per_doc": round(shr / n, 1),
            "no_dict_bytes_per_doc": round(nod / n, 1),
            "own_dict_ratio": round(raw / own, 3),
            "english_dict_ratio": round(raw / eng, 3),
            "shared_dict_ratio": round(raw / shr, 3),
            "no_dict_ratio": round(raw / nod, 3),
            "own_vs_english_win_pct": round((eng - own) / eng * 100, 1),
            "own_vs_shared_win_pct": round((shr - own) / shr * 100, 1),
            "own_vs_nodict_win_pct": round((nod - own) / nod * 100, 1),
            "own_dict_trained": zd_own is not None,
        })
        logger.info("%-12s done (%d held-out docs)", lang, n)

    rows.sort(key=lambda r: -r["own_vs_english_win_pct"])
    for rank, r in enumerate(rows, 1):
        r["rank_by_own_vs_english"] = rank
    by_shared = sorted(rows, key=lambda r: -r["own_vs_shared_win_pct"])
    for rank, r in enumerate(by_shared, 1):
        r["rank_by_own_vs_shared"] = rank

    out = {
        "config": {
            "sample": str(SAMPLE_PATH.relative_to(REPO_ROOT)),
            "docs": len(docs),
            "dict_size_bytes": DICT_SIZE,
            "zstd_level": ZSTD_LEVEL,
            "min_docs": MIN_DOCS,
            "train_frac": TRAIN_FRAC,
            "seed": SEED,
            "language_detector": "py3langid (abstract[:600])",
            "machine_code_control": {
                "chunks": len(mc_chunks),
                "chunk_bytes": [MC_CHUNK_MIN, MC_CHUNK_MAX],
                "source": "/usr/bin + /usr/lib ELF binaries, offsets in "
                          "[0x1000, 60% of file]",
            },
            "methodology": "dicts trained on 80% train split only; all "
                           "metrics measured on the 20% held-out split "
                           "(prior eval trained+measured on same docs)",
        },
        "language_counts": lang_counts,
        "ranking": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", OUTPUT)

    print("\n" + "=" * 118)
    print(f"PER-LANGUAGE DICT EVAL (held-out) — dict {DICT_SIZE // 1024} KB, "
          f"zstd-{ZSTD_LEVEL}, seed {SEED}, {int(TRAIN_FRAC * 100)}/"
          f"{100 - int(TRAIN_FRAC * 100)} split")
    print("=" * 118)
    print(f"{'lang':<13} {'docs':>6} {'raw B/d':>8} {'own B/d':>8} "
          f"{'eng B/d':>8} {'shared B/d':>10} {'nodict B/d':>10} "
          f"{'vs eng':>8} {'vs shared':>9} {'vs nodict':>9}")
    print("-" * 118)
    for r in rows:
        tag = r["language"] + (" *" if r["is_control"] else "")
        print(f"{tag:<13} {r['docs_total']:>6} {r['raw_bytes_per_doc']:>8} "
              f"{r['own_dict_bytes_per_doc']:>8} "
              f"{r['english_dict_bytes_per_doc']:>8} "
              f"{r['shared_dict_bytes_per_doc']:>10} "
              f"{r['no_dict_bytes_per_doc']:>10} "
              f"{r['own_vs_english_win_pct']:>7}% "
              f"{r['own_vs_shared_win_pct']:>8}% "
              f"{r['own_vs_nodict_win_pct']:>8}%")
    print("-" * 118)
    print("* = machine_code control (ELF chunks). "
          "Ranked by own-dict win vs English dict.")


if __name__ == "__main__":
    main()
