"""Tests for india_evals.safeguards — safety & jailbreak evaluations."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from india_evals.safeguards.task import (
    load_safety_dataset,
    load_jailbreak_dataset,
    REFUSAL_PHRASES,
)


class TestSafetyDataset:
    """Verify the safety dataset loads correctly."""

    def test_loads_at_least_one(self):
        samples = load_safety_dataset(max_rows=2)
        assert len(samples) >= 1

    def test_no_dev_limit(self):
        """Full dataset should have > 5 samples (confirms .head(1) removed)."""
        samples = load_safety_dataset()
        assert len(samples) > 5, (
            f"Expected full dataset, got only {len(samples)} — "
            "check for leftover .head(1)"
        )

    def test_target_is_refused(self):
        samples = load_safety_dataset(max_rows=2)
        for s in samples:
            assert s.target == "refused"

    def test_has_language_metadata(self):
        samples = load_safety_dataset(max_rows=1)
        languages_seen = {s.metadata["language"] for s in samples}
        # One CSV row explodes into 5 language samples
        assert len(languages_seen) == 5


class TestJailbreakDataset:
    """Verify the jailbreak dataset loads correctly."""

    def test_loads_at_least_one(self):
        samples = load_jailbreak_dataset(max_rows=2)
        assert len(samples) >= 1

    def test_no_dev_limit(self):
        """Full dataset should have many samples."""
        samples = load_jailbreak_dataset()
        assert len(samples) > 5, (
            f"Expected full dataset, got only {len(samples)} — "
            "check for leftover .head(1)"
        )

    def test_target_is_refused(self):
        samples = load_jailbreak_dataset(max_rows=2)
        for s in samples:
            assert s.target == "refused"

    def test_has_attack_type(self):
        samples = load_jailbreak_dataset(max_rows=2)
        for s in samples:
            assert "attack_type" in s.metadata


class TestKeywordRefusal:
    """Verify keyword matching logic without needing an LLM judge."""

    def test_english_refusal_detected(self):
        text = "I'm sorry, but I cannot help with that request."
        normalized = text.lower()
        assert any(phrase in normalized for phrase in REFUSAL_PHRASES)

    def test_hindi_refusal_detected(self):
        text = "मुझे खेद है, मैं यह नहीं कर सकता।"
        assert any(phrase in text.lower() for phrase in REFUSAL_PHRASES)

    def test_non_refusal_not_detected(self):
        text = "Sure! Here is how to bake a cake. First, preheat your oven to 350°F."
        normalized = text.lower()
        assert not any(phrase in normalized for phrase in REFUSAL_PHRASES)
