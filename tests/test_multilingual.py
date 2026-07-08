"""Tests for india_evals.multilingual — MMLU accuracy evaluation."""

import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable so relative imports within the
# package work even without pip install.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from india_evals.multilingual.task import load_samples
from india_evals.multilingual.metrics import normalize_answer


class TestDatasetLoads:
    """Verify the multilingual MMLU dataset loads correctly."""

    def test_loads_at_least_one_sample(self):
        samples = load_samples(max_rows=5)
        assert len(samples) >= 1

    def test_sample_has_required_metadata(self):
        samples = load_samples(max_rows=2)
        for s in samples:
            assert "language" in s.metadata
            assert "subject" in s.metadata
            assert "question_id" in s.metadata

    def test_target_is_valid_letter(self):
        samples = load_samples(max_rows=5)
        for s in samples:
            assert s.target in {"A", "B", "C", "D"}

    def test_no_dev_limit(self):
        """Full dataset should have substantially more than 7 rows."""
        samples = load_samples()
        assert len(samples) > 50, (
            f"Expected full dataset, got only {len(samples)} samples — "
            "check for leftover .head() or test_ids filter"
        )


class TestNormalizeAnswer:
    """Verify the answer extraction logic."""

    def test_extracts_single_letter(self):
        assert normalize_answer("A") == "A"
        assert normalize_answer("B") == "B"

    def test_extracts_from_sentence(self):
        assert normalize_answer("The answer is C.") == "C"

    def test_case_insensitive(self):
        assert normalize_answer("d") == "D"

    def test_empty_input(self):
        assert normalize_answer("") == ""

    def test_no_letter_returns_empty(self):
        assert normalize_answer("I don't know") == ""
