"""
Consistency test for dataset composition figures.

Loads every dataset directly and asserts it against india_evals/_dataset_facts.py
— the single place these numbers are supposed to live. If a dataset file
changes, this test fails, forcing a deliberate update to _dataset_facts.py
(and the README) instead of letting the documented figures silently drift
away from the real data.

No model calls.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from india_evals import _dataset_facts as facts

INDIA_EVALS = REPO_ROOT / "india_evals"


class TestMultilingualDataset:
    @staticmethod
    @pytest.fixture(scope="class")
    def df():
        return pd.read_csv(INDIA_EVALS / "multilingual" / "datasets" / "mmlu_translated.csv")

    def test_total_rows(self, df):
        assert len(df) == facts.MULTILINGUAL["total_rows"]

    def test_language_codes(self, df):
        codes = sorted(df["language"].unique())
        assert codes == sorted(facts.MULTILINGUAL["language_codes"])
        assert len(codes) == facts.MULTILINGUAL["n_languages"]

    def test_questions_per_language(self, df):
        counts = df["language"].value_counts()
        assert set(counts.unique()) == {facts.MULTILINGUAL["questions_per_language"]}

    def test_n_subjects(self, df):
        assert df["subject"].nunique() == facts.MULTILINGUAL["n_subjects"]

    def test_n_indian_languages(self):
        n_indian = sum(1 for c in facts.MULTILINGUAL["language_codes"] if c != "en")
        assert n_indian == facts.MULTILINGUAL["n_indian_languages"]


class TestSafetyDataset:
    @staticmethod
    @pytest.fixture(scope="class")
    def df():
        return pd.read_csv(INDIA_EVALS / "safeguards" / "datasets" / "safety.csv")

    def test_total_rows(self, df):
        assert len(df) == facts.SAFETY["total_rows"]

    def test_total_samples(self, df):
        # Each row explodes into one sample per language column.
        assert len(df) * facts.SAFETY["n_language_columns"] == facts.SAFETY["total_samples"]

    def test_categories(self, df):
        assert df["category"].value_counts().to_dict() == facts.SAFETY["categories"]
        assert df["category"].nunique() == facts.SAFETY["n_categories"]

    def test_all_high_risk(self, df):
        assert df["risk_level"].value_counts().to_dict() == facts.SAFETY["risk_levels"]


class TestJailbreakDataset:
    @staticmethod
    @pytest.fixture(scope="class")
    def df():
        return pd.read_csv(INDIA_EVALS / "safeguards" / "datasets" / "jailbreak_multilingual.csv")

    def test_total_rows(self, df):
        assert len(df) == facts.JAILBREAK["total_rows"]

    def test_languages(self, df):
        codes = sorted(df["language"].unique())
        assert codes == sorted(facts.JAILBREAK["language_codes"])
        assert len(codes) == facts.JAILBREAK["n_languages"]

    def test_five_turns_all_populated(self, df):
        for i in range(1, facts.JAILBREAK["n_turns"] + 1):
            col = f"turn{i}"
            assert col in df.columns
            assert df[col].notna().all()
            assert (df[col].astype(str).str.strip() == "").sum() == 0
        # No turn6+ column should exist.
        assert f"turn{facts.JAILBREAK['n_turns'] + 1}" not in df.columns

    def test_attack_types(self, df):
        assert df["attack_type"].value_counts().to_dict() == facts.JAILBREAK["attack_types"]
        assert df["attack_type"].nunique() == facts.JAILBREAK["n_attack_types"]


class TestDPIDataset:
    @staticmethod
    @pytest.fixture(scope="class")
    def df():
        return pd.read_csv(INDIA_EVALS / "dpi_safety" / "datasets" / "dpi_dataset.csv")

    def test_total_rows(self, df):
        assert len(df) == facts.DPI["total_rows"]

    def test_english_only(self, df):
        assert df["Language"].value_counts().to_dict() == facts.DPI["languages"]
        assert df["Language"].nunique() == facts.DPI["n_languages"]

    def test_categories(self, df):
        assert df["Category"].value_counts().to_dict() == facts.DPI["categories"]
        assert df["Category"].nunique() == facts.DPI["n_categories"]

    def test_risk_levels_no_medium(self, df):
        assert df["Risk Level"].value_counts().to_dict() == facts.DPI["risk_levels"]
        assert "Medium" not in set(df["Risk Level"].unique())


class TestCulturalKnowledgeDataset:
    @staticmethod
    @pytest.fixture(scope="class")
    def data():
        path = INDIA_EVALS / "cultural_knowledge" / "datasets" / "Cultural_knowledge_rubric_dataset.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_total_rows(self, data):
        assert len(data) == facts.CULTURAL_KNOWLEDGE["total_rows"]

    def test_domains(self, data):
        counts = dict(Counter(item["Domain"] for item in data))
        assert counts == facts.CULTURAL_KNOWLEDGE["domains"]
        assert len(counts) == facts.CULTURAL_KNOWLEDGE["n_domains"]

    def test_rubric_length(self, data):
        lengths = {len(item["rubric"]) for item in data}
        assert lengths == {facts.CULTURAL_KNOWLEDGE["rubric_criteria_per_entry"]}


class TestBharatBBQDatasets:
    @staticmethod
    @pytest.fixture(scope="class")
    def csv_files():
        return sorted((INDIA_EVALS / "bias" / "datasets").glob("*.csv"))

    def test_file_count(self, csv_files):
        assert len(csv_files) == facts.BHARATBBQ["n_files"]

    def test_file_names_match(self, csv_files):
        names = {f.name for f in csv_files}
        assert names == set(facts.BHARATBBQ["files"].keys())

    def test_per_file_row_counts_and_total(self, csv_files):
        total = 0
        for f in csv_files:
            df = pd.read_csv(f)
            expected = facts.BHARATBBQ["files"][f.name]
            assert len(df) == expected, f"{f.name}: expected {expected} rows, got {len(df)}"
            total += len(df)
        assert total == facts.BHARATBBQ["total_rows"]

    def test_categories_match_recorded_facts(self, csv_files):
        categories = set()
        for f in csv_files:
            df = pd.read_csv(f)
            assert "Category" in df.columns
            categories.update(df["Category"].unique())
        assert categories == set(facts.BHARATBBQ["categories"])
