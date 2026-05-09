#!/usr/bin/env python3
"""Phase I-bis C: Generate synthetic search queries from cached paper texts via LLM.

For each unique paper text in the clean corpus, calls a local (Ollama) or cloud
(Claude Haiku / OpenAI) LLM to generate 3 short search queries a student would
type to find that paper. Responses are cached in data/synthetic_queries_cache.json.

Each generated query becomes an anchor paired with:
  positive  = the source paper
  negative  = a random paper from a different subject

Usage:
    # Local Ollama (zero cost, requires `ollama serve` + a pulled model):
    python -m scripts.generate_synthetic_queries --llm ollama

    # Claude Haiku (cheap, high quality, requires ANTHROPIC_API_KEY):
    python -m scripts.generate_synthetic_queries --llm claude

    # Dry-run to estimate counts without calling LLM:
    python -m scripts.generate_synthetic_queries --dry-run
"""
import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
CLEAN_TRIPLETS = REPO_ROOT / "data/sfu_training_triplets.clean.jsonl"
CACHE_FILE = REPO_ROOT / "data/synthetic_queries_cache.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/synthetic_query_triplets.jsonl"

SYSTEM_PROMPT = (
    "You are a university library search expert. "
    "Given an academic paper's title and abstract, write exactly 3 short search queries "
    "(5-12 words each) that a student in the given subject area would type into a library "
    "database to find this paper. Output ONLY a JSON array of 3 strings, nothing else."
)

_CACHE: dict = {}


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _is_valid(text: str, min_tokens: int = 5) -> bool:
    return bool(text) and len(text.split()) >= min_tokens


def _load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def load_corpus(path: Path) -> list[dict]:
    triplets = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    return triplets


def extract_unique_papers(triplets: list[dict]) -> list[tuple[str, str]]:
    """Return list of (paper_text, subject) for unique positives in the corpus."""
    seen: set[str] = set()
    results: list[tuple[str, str]] = []
    for t in triplets:
        pos = t.get("positive", "")
        subj = t.get("subject", "Unknown")
        if _is_valid(pos, min_tokens=10):
            h = _sha1(pos)
            if h not in seen:
                seen.add(h)
                results.append((pos, subj))
    return results


def _build_prompt(paper_text: str, subject: str) -> str:
    excerpt = paper_text[:800]
    return (
        f"Subject area: {subject}\n\n"
        f"Paper text:\n{excerpt}\n\n"
        f"Generate 3 search queries a {subject} student would use to find this paper."
    )


def call_ollama(paper_text: str, subject: str, model: str = "llama3", base_url: str = "http://localhost:11434") -> list[str]:
    import urllib.request

    prompt = _build_prompt(paper_text, subject)
    payload = json.dumps({
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            text = data.get("response", "").strip()
    except Exception as e:
        logger.warning("Ollama call failed: %s", e)
        return []

    return _parse_queries(text)


def call_claude(paper_text: str, subject: str, client, model: str = "claude-haiku-4-5-20251001") -> list[str]:
    prompt = _build_prompt(paper_text, subject)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
    except Exception as e:
        logger.warning("Claude call failed: %s", e)
        return []
    return _parse_queries(text)


def _parse_queries(text: str) -> list[str]:
    """Parse LLM output as a JSON array of query strings."""
    text = text.strip()
    # Try to find a JSON array in the response
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            queries = json.loads(text[start:end + 1])
            if isinstance(queries, list):
                return [str(q).strip() for q in queries if isinstance(q, str) and len(q.strip()) >= 5][:3]
        except json.JSONDecodeError:
            pass
    # Fallback: split by newlines and strip numbering
    lines = [l.strip().lstrip("0123456789.-) ") for l in text.split("\n") if l.strip()]
    queries = [l for l in lines if 5 <= len(l.split()) <= 20]
    return queries[:3]


def generate_queries_for_paper(
    paper_text: str,
    subject: str,
    llm: str,
    cache: dict,
    **llm_kwargs,
) -> list[str]:
    cache_key = _sha1(paper_text)
    if cache_key in cache:
        return cache[cache_key]

    if llm == "ollama":
        queries = call_ollama(paper_text, subject, **llm_kwargs)
    elif llm == "claude":
        queries = call_claude(paper_text, subject, **llm_kwargs)
    else:
        raise ValueError(f"Unknown LLM backend: {llm}")

    if queries:
        cache[cache_key] = queries
    return queries


def build_subject_index(papers: list[tuple[str, str]]) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for i, (_, subj) in enumerate(papers):
        idx.setdefault(subj, []).append(i)
    return idx


def run(
    papers: list[tuple[str, str]],
    output_path: Path,
    llm: str,
    cache: dict,
    max_total: int,
    seed: int,
    dry_run: bool,
    append: bool,
    llm_kwargs: dict,
    save_every: int = 100,
) -> int:
    rng = random.Random(seed)
    subject_index = build_subject_index(papers)
    seen_pairs: set[tuple[str, str]] = set()
    count = 0

    if dry_run:
        estimated = min(len(papers) * 3, max_total)
        logger.info("Dry-run: would process %d papers, estimated ~%d triplets", len(papers), estimated)
        return 0

    mode = "a" if append else "w"

    with output_path.open(mode) as out:
        for i, (paper_text, subject) in enumerate(papers):
            if count >= max_total:
                break

            if (i + 1) % 100 == 0:
                logger.info("  [%d/%d] count=%d", i + 1, len(papers), count)
                _save_cache(cache, CACHE_FILE)

            queries = generate_queries_for_paper(
                paper_text, subject, llm=llm, cache=cache, **llm_kwargs
            )
            if not queries:
                continue

            # Negative: random paper from a different subject
            other_subjects = [s for s in subject_index if s != subject]
            if not other_subjects:
                continue
            neg_subj = rng.choice(other_subjects)
            neg_idx = rng.choice(subject_index[neg_subj])
            neg_text = papers[neg_idx][0]

            for query in queries:
                if count >= max_total:
                    break
                if len(query.split()) < 3:
                    continue

                pair_key = (_sha1(query), _sha1(paper_text))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                triplet = {
                    "anchor": query,
                    "positive": paper_text[:4000],
                    "negative": neg_text[:4000],
                    "strategy": "llm_synthetic",
                    "subject": subject,
                    "metadata": {
                        "source": f"llm_{llm}",
                        "positive_sha1": _sha1(paper_text)[:12],
                    },
                }
                out.write(json.dumps(triplet) + "\n")
                count += 1

            # Rate limiting for cloud APIs
            if llm == "claude":
                time.sleep(0.1)

    _save_cache(cache, CACHE_FILE)
    return count


def main() -> None:
    global CACHE_FILE
    parser = argparse.ArgumentParser(
        description="Phase I-bis C: LLM-generated synthetic query triplets"
    )
    parser.add_argument("--input", default=str(CLEAN_TRIPLETS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--append-to", default=None,
                        help="Append to existing JSONL instead of --output")
    parser.add_argument("--llm", choices=["ollama", "claude"], default="ollama",
                        help="LLM backend")
    parser.add_argument("--ollama-model", default="llama3",
                        help="Ollama model name (default: llama3)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama base URL")
    parser.add_argument("--claude-model", default="claude-haiku-4-5-20251001",
                        help="Claude model for query generation")
    parser.add_argument("--max-total", type=int, default=26000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-file", default=str(CACHE_FILE))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    output_path = Path(args.append_to) if args.append_to else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    CACHE_FILE = Path(args.cache_file)
    cache = _load_cache(CACHE_FILE)
    logger.info("Cache: %d entries loaded from %s", len(cache), CACHE_FILE)

    triplets = load_corpus(input_path)
    papers = extract_unique_papers(triplets)
    logger.info("Found %d unique papers to generate queries for", len(papers))

    llm_kwargs: dict = {}
    if args.llm == "ollama":
        llm_kwargs = {"model": args.ollama_model, "base_url": args.ollama_url}
    elif args.llm == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set")
            sys.exit(1)
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            logger.error("anthropic package not installed: pip install anthropic")
            sys.exit(1)
        llm_kwargs = {"client": client, "model": args.claude_model}

    count = run(
        papers=papers,
        output_path=output_path,
        llm=args.llm,
        cache=cache,
        max_total=args.max_total,
        seed=args.seed,
        dry_run=args.dry_run,
        append=bool(args.append_to),
        llm_kwargs=llm_kwargs,
    )

    logger.info("Done. Wrote %d synthetic query triplets to %s", count, output_path)


if __name__ == "__main__":
    main()
