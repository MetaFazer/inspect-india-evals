"""Tests for india_evals.scorers.fairness — composite fairness index."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from india_evals.scorers.fairness import fairness_index


class TestFairnessIndex:
    """Verify composite fairness index computations."""

    def test_perfect_scores(self):
        result = fairness_index(
            multilingual_accuracy=1.0,
            bias_score_amb=0.0,       # 0 bias → normalised to 1.0
            safety_refusal_rate=1.0,
            dpi_accuracy=1.0,
        )
        assert result["fairness_index"] == 1.0

    def test_worst_scores(self):
        result = fairness_index(
            multilingual_accuracy=0.0,
            bias_score_amb=1.0,       # max bias → normalised to 0.0
            safety_refusal_rate=0.0,
            dpi_accuracy=0.0,
        )
        assert result["fairness_index"] == 0.0

    def test_mixed_scores(self):
        result = fairness_index(
            multilingual_accuracy=0.8,
            bias_score_amb=0.2,       # normalised: 1-0.2 = 0.8
            safety_refusal_rate=0.6,
            dpi_accuracy=0.4,
        )
        # Expected: (0.8 + 0.8 + 0.6 + 0.4) / 4 = 0.65
        assert result["fairness_index"] == 0.65

    def test_custom_weights(self):
        result = fairness_index(
            multilingual_accuracy=1.0,
            bias_score_amb=0.0,
            safety_refusal_rate=0.0,
            dpi_accuracy=0.0,
            weights={
                "multilingual": 1.0,
                "bias": 0.0,
                "safety": 0.0,
                "dpi": 0.0,
            },
        )
        assert result["fairness_index"] == 1.0

    def test_output_keys(self):
        result = fairness_index()
        expected_keys = {
            "multilingual_normalised",
            "bias_normalised",
            "safety_normalised",
            "dpi_normalised",
            "fairness_index",
        }
        assert set(result.keys()) == expected_keys

    def test_clamping(self):
        """Values outside 0-1 should be clamped."""
        result = fairness_index(
            multilingual_accuracy=1.5,   # clamped to 1.0
            bias_score_amb=-0.5,         # normalised: 1-0.5=0.5
            safety_refusal_rate=-0.2,    # clamped to 0.0
            dpi_accuracy=2.0,            # clamped to 1.0
        )
        assert 0.0 <= result["fairness_index"] <= 1.0
        assert result["multilingual_normalised"] == 1.0
        assert result["safety_normalised"] == 0.0
