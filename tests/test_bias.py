"""Tests for india_evals.bias — BharatBBQ bias evaluation."""

import sys
from pathlib import Path

import pytest
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from india_evals.bias.task import load_bharatbbq
from india_evals.bias.metrics import (
    bias_score_amb,
    bias_score_disamb,
    compute_all_bias_metrics,
)


class TestDatasetLoads:
    """Verify the BharatBBQ dataset loads correctly."""

    def test_loads_at_least_one_sample(self):
        samples = load_bharatbbq(max_rows_per_file=2)
        assert len(samples) >= 1

    def test_target_is_abc(self):
        samples = load_bharatbbq(max_rows_per_file=3)
        for s in samples:
            assert s.target in {"A", "B", "C"}, f"Unexpected target: {s.target}"

    def test_has_category_metadata(self):
        samples = load_bharatbbq(max_rows_per_file=2)
        for s in samples:
            assert "category" in s.metadata

    def test_no_dev_limit(self):
        """Without max_rows_per_file, should load thousands of samples."""
        samples = load_bharatbbq()
        assert len(samples) > 100, (
            f"Expected full dataset, got only {len(samples)} samples — "
            "check for leftover .head(5)"
        )


class TestBiasMetrics:
    """Verify bias metric computations on synthetic data."""

    def _make_df(self, rows):
        return pd.DataFrame(rows)

    def test_bias_score_amb_all_stereotyped(self):
        """All predictions match target → bias score = 1."""
        df = self._make_df([
            {"context_type": "amb", "prediction": 1, "target_numeric": 1, "label": 2},
            {"context_type": "amb", "prediction": 2, "target_numeric": 2, "label": 3},
        ])
        score = bias_score_amb(df)
        assert score == 1.0

    def test_bias_score_amb_empty(self):
        """No ambiguous rows → score = 0."""
        df = self._make_df([
            {"context_type": "disamb", "prediction": 1, "target_numeric": 1, "label": 2},
        ])
        assert bias_score_amb(df) == 0.0

    def test_bias_score_disamb_empty(self):
        """No disambiguated rows → score = 0."""
        df = self._make_df([
            {"context_type": "amb", "prediction": 1, "target_numeric": 1, "label": 2},
        ])
        assert bias_score_disamb(df) == 0.0

    def test_compute_all_returns_four_keys(self):
        df = self._make_df([
            {
                "context_type": "amb", "prediction": 1,
                "target_numeric": 1, "label": 2,
                "pairing": "biased", "question_polarity": "neg",
            },
        ])
        result = compute_all_bias_metrics(df)
        assert set(result.keys()) == {"BSA", "BSD", "SBSA", "SBSD"}
