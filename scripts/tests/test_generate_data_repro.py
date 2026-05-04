"""Phase J.4 — Reproducibility regression test for generate_sfu_training_data.

Two runs of strategy 5 on the Solr cache with the same seed must produce
byte-identical output, confirming J.1 (sha1 dedup) and J.3 (seeded query gen)
work together.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts/ to path so we can import helpers directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def _run_strategy5(output_path: Path, seed: int = 42, max_total: int = 200) -> None:
    from generate_sfu_training_data import (
        GenerationState,
        SFUDatabaseRegistry,
        run_strategy5,
        SOLR_CACHE_FILE,
    )

    registry = SFUDatabaseRegistry(cache_file=SOLR_CACHE_FILE)
    registry.ensure_loaded()
    solr_docs = registry.get_all()

    state = GenerationState(output_file=str(output_path), started_at="test")
    state_file = output_path.with_suffix(".state.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_strategy5(
        solr_docs=solr_docs,
        max_total=max_total,
        output_file=output_path,
        state=state,
        state_file=state_file,
        seed=seed,
    )


class TestReproducibility:
    def test_strategy5_same_seed_byte_identical(self):
        """Two runs with the same seed produce the same JSONL output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = Path(tmpdir) / "run1.jsonl"
            out2 = Path(tmpdir) / "run2.jsonl"
            _run_strategy5(out1, seed=42, max_total=100)
            _run_strategy5(out2, seed=42, max_total=100)

            lines1 = out1.read_text().splitlines()
            lines2 = out2.read_text().splitlines()

            assert len(lines1) == len(lines2), "Different number of triplets"
            for i, (l1, l2) in enumerate(zip(lines1, lines2)):
                assert l1 == l2, f"Line {i} differs:\n  run1: {l1[:100]}\n  run2: {l2[:100]}"

    def test_strategy5_different_seeds_differ(self):
        """Different seeds should produce different outputs (sanity check)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = Path(tmpdir) / "seed42.jsonl"
            out2 = Path(tmpdir) / "seed99.jsonl"
            _run_strategy5(out1, seed=42, max_total=100)
            _run_strategy5(out2, seed=99, max_total=100)

            lines1 = out1.read_text().splitlines()
            lines2 = out2.read_text().splitlines()
            # Outputs should differ for different seeds
            assert lines1 != lines2

    def test_strategy5_subject_coverage(self):
        """Strategy 5 should cover at least 30 distinct subjects in 200 triplets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cov.jsonl"
            _run_strategy5(out, seed=42, max_total=200)
            subjects = set()
            for line in out.read_text().splitlines():
                row = json.loads(line)
                subjects.add(row["subject"])
            assert len(subjects) >= 30, f"Only {len(subjects)} subjects covered"

    def test_strategy5_no_api_calls(self, monkeypatch):
        """Strategy 5 must never call OpenAlex (zero API)."""
        import generate_sfu_training_data as gsd

        call_count = {"n": 0}

        def _mock_openalex_get(path, params, retries=3):
            call_count["n"] += 1
            raise AssertionError("Strategy 5 made an API call!")

        monkeypatch.setattr(gsd, "_openalex_get", _mock_openalex_get)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "noapi.jsonl"
            _run_strategy5(out, seed=42, max_total=50)

        assert call_count["n"] == 0
