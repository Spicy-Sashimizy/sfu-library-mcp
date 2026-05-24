#!/usr/bin/env python3
"""Generate natural-language / conceptual paraphrases of the 120 eval queries.

The 120 queries in data/sfu_eval_queries.json are keyword-style (e.g.
"indigenous language revitalization First Nations British Columbia"). Keyword
queries play to lexical retrieval's strengths and would UNDERSTATE the value of a
dense semantic leg. To expose dense's lift fairly we generate, for each query,
1-2 natural-language / conceptual paraphrases (full questions, no keyword
stuffing) via the `claude` CLI subprocess — the same no-API-key pattern as
scripts/benchmark_llm_judge.py::_call_haiku.

GROUND-TRUTH INHERITANCE (critical)
───────────────────────────────────
A paraphrase of query Q is treated as having the SAME relevant docs as Q. The
judge cache keys are `"<query[:80]>||<doc_id>"`, so each output record stores
`original_query` AND `judge_key` = original_query[:80] (the exact cache-key
prefix benchmark_llm_judge.py::_cache_key uses) so eval_dense_poc.py can map a
paraphrase back to Q's graded docs without re-judging.

Output: data/eval_results/diverse_queries.json — a list of records:
    {"paraphrase", "original_query", "judge_key", "query_type": "keyword"|"natural", "subject"}
The original keyword query is itself emitted as a query_type="keyword" record so
the eval can score the keyword slice from the same file.

Usage
─────
    python scripts/make_diverse_queries.py \
        --eval-queries data/sfu_eval_queries.json \
        --output data/eval_results/diverse_queries.json \
        [--paraphrases 2] [--limit N]
"""
import argparse
import json
import logging
import re
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_diverse_queries")

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_EVAL = REPO_ROOT / "data/sfu_eval_queries.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/eval_results/diverse_queries.json"

# Cache keys in llm_judge_cache.json are query[:80] (see _cache_key in
# scripts/benchmark_llm_judge.py). Use the SAME truncation so paraphrases inherit
# the right ground truth.
JUDGE_KEY_LEN = 80

PARAPHRASE_SYSTEM = (
    "You rewrite academic library search queries as natural-language research "
    "questions. You output ONLY a JSON array of strings, nothing else."
)


def judge_key(query: str) -> str:
    return query[:JUDGE_KEY_LEN]


def _call_haiku(prompt: str, timeout: int = 90) -> str:
    """Call claude CLI subprocess (mirrors benchmark_llm_judge.py::_call_haiku)."""
    result = subprocess.run(
        ["claude", "-p",
         "--model", "claude-haiku-4-5-20251001",
         "--output-format", "text",
         "--system-prompt", PARAPHRASE_SYSTEM,
         "--no-session-persistence"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def _parse_array(text: str) -> list[str]:
    """Extract a JSON array of strings from a (possibly fenced) response."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group())
    except Exception:
        return []
    out = []
    for item in arr:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def paraphrase_query(query: str, subject: str, n: int) -> list[str]:
    """Generate n natural-language paraphrases of a keyword query."""
    prompt = (
        f"Rewrite this keyword search query as {n} distinct natural-language "
        f"research question(s) a graduate student might type into a library "
        f"search box. Preserve the EXACT topic and intent — do not broaden, "
        f"narrow, or add new concepts. Make them conversational/conceptual "
        f"(full sentences or questions), NOT keyword lists.\n\n"
        f"Subject area: {subject}\n"
        f'Keyword query: "{query}"\n\n'
        f"Reply ONLY with a JSON array of {n} strings."
    )
    for attempt in range(3):
        try:
            raw = _call_haiku(prompt)
            arr = _parse_array(raw)
            if arr:
                return arr[:n]
        except subprocess.TimeoutExpired:
            log.warning("paraphrase timed out (attempt %d) for %r", attempt + 1, query[:50])
        except Exception as e:
            log.warning("paraphrase failed (attempt %d): %s", attempt + 1, e)
        time.sleep(1)
    log.warning("No paraphrase generated for %r — keyword slice only", query[:50])
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NL paraphrases of eval queries")
    parser.add_argument("--eval-queries", default=str(DEFAULT_EVAL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--paraphrases", type=int, default=2,
                        help="paraphrases to request per query")
    parser.add_argument("--limit", type=int, default=0, help="limit #queries (0=all, for testing)")
    args = parser.parse_args()

    queries = json.loads(Path(args.eval_queries).read_text())
    if args.limit:
        queries = queries[: args.limit]

    records: list[dict] = []
    n_natural = 0
    for i, q in enumerate(queries):
        original = q["query"]
        subject = q.get("subject", "")
        key = judge_key(original)
        # Emit the original keyword query as its own record.
        records.append({
            "paraphrase": original,
            "original_query": original,
            "judge_key": key,
            "query_type": "keyword",
            "subject": subject,
        })
        paras = paraphrase_query(original, subject, args.paraphrases)
        for p in paras:
            if p.strip().lower() == original.strip().lower():
                continue
            records.append({
                "paraphrase": p,
                "original_query": original,
                "judge_key": key,
                "query_type": "natural",
                "subject": subject,
            })
            n_natural += 1
        if (i + 1) % 10 == 0:
            log.info("  %d/%d queries done (%d natural so far)", i + 1, len(queries), n_natural)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(records, indent=2))
    log.info("Wrote %d records (%d keyword + %d natural) -> %s",
             len(records), len(queries), n_natural, args.output)


if __name__ == "__main__":
    main()
