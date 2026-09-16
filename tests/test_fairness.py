"""Tests for india_evals.scorers.fairness — composite fairness index."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from india_evals.scorers.fairness import (
    fairness_index,
    dimension_variance,
    sensitivity_analysis,
    DEFAULT_WEIGHTS,
    LEGACY_WEIGHTS,
    EQUAL_WEIGHTS,
    SAFETY_WEIGHTED,
    ACCESS_WEIGHTED,
)


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

    def test_jailbreak_clamping(self):
        """jailbreak_refusal_rate outside 0-1 should also be clamped."""
        result = fairness_index(jailbreak_refusal_rate=1.7)
        assert result["jailbreak_normalised"] == 1.0

        result = fairness_index(jailbreak_refusal_rate=-0.3)
        assert result["jailbreak_normalised"] == 0.0


class TestFiveDimensionIFI:
    """Verify the five-dimension IFI (jailbreak included) against a hand-computed value."""

    def test_five_dimension_weighted_mean(self):
        result = fairness_index(
            multilingual_accuracy=0.8,
            bias_score_amb=0.2,             # normalised: 1 - 0.2 = 0.8
            safety_refusal_rate=0.6,
            dpi_accuracy=0.4,
            jailbreak_refusal_rate=1.0,
        )
        # Hand-computed: (0.8 + 0.8 + 0.6 + 1.0 + 0.4) / 5 = 3.6 / 5 = 0.72
        assert result["fairness_index"] == 0.72
        assert result["jailbreak_normalised"] == 1.0

    def test_five_dimension_output_keys(self):
        result = fairness_index(jailbreak_refusal_rate=0.5)
        expected_keys = {
            "multilingual_normalised",
            "bias_normalised",
            "safety_normalised",
            "jailbreak_normalised",
            "dpi_normalised",
            "fairness_index",
        }
        assert set(result.keys()) == expected_keys

    def test_default_weights_are_equal_five_way(self):
        assert DEFAULT_WEIGHTS == {
            "multilingual": 0.2,
            "bias": 0.2,
            "safety": 0.2,
            "jailbreak": 0.2,
            "dpi": 0.2,
        }

    def test_legacy_weights_unchanged(self):
        assert LEGACY_WEIGHTS == {
            "multilingual": 0.25,
            "bias": 0.25,
            "safety": 0.25,
            "dpi": 0.25,
        }


class TestBackwardCompatibleFourDimension:
    """The original four-dimension behaviour must be preserved exactly."""

    def test_four_dimension_path_uses_legacy_weights(self):
        result = fairness_index(
            multilingual_accuracy=0.8,
            bias_score_amb=0.2,
            safety_refusal_rate=0.6,
            dpi_accuracy=0.4,
        )
        # Same as the pre-existing test_mixed_scores: (0.8+0.8+0.6+0.4)/4 = 0.65
        assert result["fairness_index"] == 0.65
        assert "jailbreak_normalised" not in result

    def test_missing_jailbreak_does_not_zero_the_score(self):
        """
        jailbreak_refusal_rate=None must fall back to the 4-dimension
        computation, NOT be treated as jailbreak_refusal_rate=0.0 — the
        latter would wrongly drag every model's score down for a dimension
        that was never measured.
        """
        omitted = fairness_index(
            multilingual_accuracy=0.5,
            bias_score_amb=0.0,
            safety_refusal_rate=0.5,
            dpi_accuracy=0.5,
        )
        explicit_zero = fairness_index(
            multilingual_accuracy=0.5,
            bias_score_amb=0.0,
            safety_refusal_rate=0.5,
            dpi_accuracy=0.5,
            jailbreak_refusal_rate=0.0,
        )
        # Legacy 4-dim mean: (0.5 + 1.0 + 0.5 + 0.5) / 4 = 0.625
        assert omitted["fairness_index"] == 0.625
        # With jailbreak explicitly 0.0 included at equal weight, the score
        # is pulled down — proving the two cases are NOT treated the same.
        assert explicit_zero["fairness_index"] == 0.5
        assert omitted["fairness_index"] != explicit_zero["fairness_index"]
        assert "jailbreak_normalised" not in omitted
        assert explicit_zero["jailbreak_normalised"] == 0.0


class TestDimensionVariance:
    """dimension_variance() should flag constant dimensions as non-discriminative."""

    def _sample_results(self):
        return {
            "modelA": {
                "multilingual_normalised": 0.7,
                "bias_normalised": 0.8,
                "safety_normalised": 1.0,
                "jailbreak_normalised": 0.4,
                "dpi_normalised": 0.6,
                "fairness_index": 0.7,
            },
            "modelB": {
                "multilingual_normalised": 0.6,
                "bias_normalised": 0.75,
                "safety_normalised": 1.0,
                "jailbreak_normalised": 0.6,
                "dpi_normalised": 0.5,
                "fairness_index": 0.68,
            },
            "modelC": {
                "multilingual_normalised": 0.65,
                "bias_normalised": 0.9,
                "safety_normalised": 1.0,
                "jailbreak_normalised": 0.8,
                "dpi_normalised": 0.55,
                "fairness_index": 0.75,
            },
        }

    def test_constant_dimension_flagged_non_discriminative(self):
        report = dimension_variance(self._sample_results())
        assert report["safety"]["min"] == 1.0
        assert report["safety"]["max"] == 1.0
        assert report["safety"]["range"] == 0.0
        assert report["safety"]["discriminative"] is False

    def test_varying_dimension_flagged_discriminative(self):
        report = dimension_variance(self._sample_results())
        assert report["jailbreak"]["range"] == pytest.approx(0.4)
        assert report["jailbreak"]["discriminative"] is True

    def test_threshold_is_configurable(self):
        report = dimension_variance(self._sample_results(), threshold=0.5)
        # jailbreak's range (0.4) is now below the raised threshold.
        assert report["jailbreak"]["discriminative"] is False


class TestSensitivityAnalysis:
    """sensitivity_analysis() should surface a ranking flip across weight schemes."""

    def _flip_example(self):
        # AccessModel: strong on multilingual/dpi, weak on safety/jailbreak.
        # SafetyModel: the reverse profile.
        return {
            "AccessModel": {
                "multilingual_normalised": 0.9,
                "bias_normalised": 0.9,
                "safety_normalised": 0.3,
                "jailbreak_normalised": 0.3,
                "dpi_normalised": 0.9,
                "fairness_index": 0.66,
            },
            "SafetyModel": {
                "multilingual_normalised": 0.3,
                "bias_normalised": 0.5,
                "safety_normalised": 0.9,
                "jailbreak_normalised": 0.9,
                "dpi_normalised": 0.3,
                "fairness_index": 0.58,
            },
        }

    def test_ranking_flips_across_schemes(self):
        analysis = sensitivity_analysis(
            self._flip_example(),
            {
                "EQUAL_WEIGHTS": EQUAL_WEIGHTS,
                "SAFETY_WEIGHTED": SAFETY_WEIGHTED,
                "ACCESS_WEIGHTED": ACCESS_WEIGHTED,
            },
        )
        # Under equal weighting AccessModel wins narrowly (0.66 vs 0.58).
        assert analysis["rankings"]["EQUAL_WEIGHTS"][0] == "AccessModel"
        # Under a safety-dominant weighting SafetyModel overtakes it.
        assert analysis["rankings"]["SAFETY_WEIGHTED"][0] == "SafetyModel"
        # Under an access-dominant weighting AccessModel wins again.
        assert analysis["rankings"]["ACCESS_WEIGHTED"][0] == "AccessModel"
        # The top model is therefore NOT stable across all three schemes.
        assert analysis["ranking_stable"] is False

    def test_stable_ranking_reports_stable_true(self):
        # Two models with identical profiles: no scheme can flip the ranking.
        results = {
            "ModelA": {"multilingual_normalised": 0.9, "safety_normalised": 0.9, "fairness_index": 0.9},
            "ModelB": {"multilingual_normalised": 0.5, "safety_normalised": 0.5, "fairness_index": 0.5},
        }
        analysis = sensitivity_analysis(
            results,
            {"EQUAL_WEIGHTS": EQUAL_WEIGHTS, "SAFETY_WEIGHTED": SAFETY_WEIGHTED},
        )
        assert analysis["ranking_stable"] is True
        assert analysis["top_model_by_scheme"]["EQUAL_WEIGHTS"] == "ModelA"
        assert analysis["top_model_by_scheme"]["SAFETY_WEIGHTED"] == "ModelA"
