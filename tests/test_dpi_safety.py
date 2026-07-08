"""Tests for india_evals.dpi_safety — DPI-context safety evaluation."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from india_evals.dpi_safety.task import load_dpi_dataset


class TestDPIDataset:
    """Verify the DPI dataset loads correctly."""

    def test_loads_at_least_one(self):
        samples = load_dpi_dataset(max_rows=2)
        assert len(samples) >= 1

    def test_no_dev_limit(self):
        """Full dataset should have many rows (confirms .head(1) removed)."""
        samples = load_dpi_dataset()
        assert len(samples) > 5, (
            f"Expected full dataset, got only {len(samples)} — "
            "check for leftover .head(1)"
        )

    def test_target_values(self):
        """Each DPI sample target must be 'answer' or 'refuse'."""
        samples = load_dpi_dataset(max_rows=10)
        for s in samples:
            assert s.target in {"answer", "refuse"}, (
                f"Unexpected target: {s.target}"
            )

    def test_has_risk_level(self):
        samples = load_dpi_dataset(max_rows=5)
        for s in samples:
            assert "risk_level" in s.metadata
            assert s.metadata["risk_level"] in {"Low", "High"}

    def test_has_language_and_category(self):
        samples = load_dpi_dataset(max_rows=3)
        for s in samples:
            assert "language" in s.metadata
            assert "category" in s.metadata

    def test_low_risk_target_is_answer(self):
        """Low-risk samples should have target='answer'."""
        samples = load_dpi_dataset()
        low_risk = [s for s in samples if s.metadata["risk_level"] == "Low"]
        if low_risk:
            for s in low_risk:
                assert s.target == "answer"

    def test_high_risk_target_is_refuse(self):
        """High-risk samples should have target='refuse'."""
        samples = load_dpi_dataset()
        high_risk = [s for s in samples if s.metadata["risk_level"] == "High"]
        if high_risk:
            for s in high_risk:
                assert s.target == "refuse"
