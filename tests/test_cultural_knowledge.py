"""
Tests for india_evals.cultural_knowledge

Covers:
    - Dataset loading (full and limited)
    - Sample structure (input, metadata fields)
    - Rubric judge prompt builder
    - Task instantiation
    - Scorer instantiation
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_rubric():
    return [
        "Mentions that article 21 guarantees the right to life and personal liberty",
        "Mentions that not just citizens",
        "Mentions that the Supreme Court has expanded its scope",
        "Mentions a dignified life",
    ]


@pytest.fixture
def sample_item():
    return {
        "Scenario Id": 1,
        "Domain": "Indian Constitution",
        "Questions": "What is the significance of Article 21?",
        "rubric": [
            "Mentions that article 21 guarantees the right to life and personal liberty",
            "Mentions that not just citizens",
            "Mentions that the Supreme Court has expanded its scope",
            "Mentions a dignified life",
        ],
    }


# ── Dataset loading ────────────────────────────────────────────────────────────

class TestLoadCulturalKnowledge:

    def test_loads_samples(self):
        from india_evals.cultural_knowledge.task import load_cultural_knowledge
        samples = load_cultural_knowledge()
        assert len(samples) > 0

    def test_max_rows_limits(self):
        from india_evals.cultural_knowledge.task import load_cultural_knowledge
        samples = load_cultural_knowledge(max_rows=5)
        assert len(samples) == 5

    def test_sample_has_input(self):
        from india_evals.cultural_knowledge.task import load_cultural_knowledge
        sample = load_cultural_knowledge(max_rows=1)[0]
        assert isinstance(sample.input, str)
        assert len(sample.input) > 10

    def test_sample_input_contains_question(self):
        from india_evals.cultural_knowledge.task import load_cultural_knowledge
        sample = load_cultural_knowledge(max_rows=1)[0]
        assert "Question:" in sample.input

    def test_sample_has_empty_target(self):
        from india_evals.cultural_knowledge.task import load_cultural_knowledge
        sample = load_cultural_knowledge(max_rows=1)[0]
        assert sample.target == ""

    def test_sample_metadata_id(self):
        from india_evals.cultural_knowledge.task import load_cultural_knowledge
        sample = load_cultural_knowledge(max_rows=1)[0]
        assert "id" in sample.metadata
        assert sample.metadata["id"] == 1

    def test_sample_metadata_domain(self):
        from india_evals.cultural_knowledge.task import load_cultural_knowledge
        sample = load_cultural_knowledge(max_rows=1)[0]
        assert "domain" in sample.metadata
        assert isinstance(sample.metadata["domain"], str)

    def test_sample_metadata_rubric(self):
        from india_evals.cultural_knowledge.task import load_cultural_knowledge
        sample = load_cultural_knowledge(max_rows=1)[0]
        assert "rubric" in sample.metadata
        rubric = sample.metadata["rubric"]
        assert isinstance(rubric, list)
        assert len(rubric) > 0

    def test_rubric_items_are_strings(self):
        from india_evals.cultural_knowledge.task import load_cultural_knowledge
        sample = load_cultural_knowledge(max_rows=3)[0]
        for criterion in sample.metadata["rubric"]:
            assert isinstance(criterion, str)

    def test_multiple_domains_present(self):
        from india_evals.cultural_knowledge.task import load_cultural_knowledge
        samples = load_cultural_knowledge()
        domains = {s.metadata["domain"] for s in samples}
        assert len(domains) > 1, "Expected multiple domains in dataset"

    def test_dataset_file_exists(self):
        dataset_path = (
            Path(__file__).parent.parent
            / "india_evals"
            / "cultural_knowledge"
            / "datasets"
            / "Cultural_knowledge_rubric_dataset.json"
        )
        assert dataset_path.exists(), f"Dataset not found at {dataset_path}"

    def test_dataset_is_valid_json(self):
        dataset_path = (
            Path(__file__).parent.parent
            / "india_evals"
            / "cultural_knowledge"
            / "datasets"
            / "Cultural_knowledge_rubric_dataset.json"
        )
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0


# ── Judge prompt builder ───────────────────────────────────────────────────────

class TestBuildJudgePrompt:

    def test_prompt_contains_question(self, sample_rubric):
        from india_evals.cultural_knowledge.task import _build_judge_prompt
        prompt = _build_judge_prompt("What is Article 21?", "Some answer", sample_rubric)
        assert "What is Article 21?" in prompt

    def test_prompt_contains_answer(self, sample_rubric):
        from india_evals.cultural_knowledge.task import _build_judge_prompt
        prompt = _build_judge_prompt("Q?", "My model answer here", sample_rubric)
        assert "My model answer here" in prompt

    def test_prompt_contains_all_rubric_criteria(self, sample_rubric):
        from india_evals.cultural_knowledge.task import _build_judge_prompt
        prompt = _build_judge_prompt("Q?", "A", sample_rubric)
        for criterion in sample_rubric:
            assert criterion in prompt

    def test_prompt_contains_json_format(self, sample_rubric):
        from india_evals.cultural_knowledge.task import _build_judge_prompt
        prompt = _build_judge_prompt("Q?", "A", sample_rubric)
        assert '"results"' in prompt
        assert '"passed"' in prompt

    def test_rubric_numbered(self, sample_rubric):
        from india_evals.cultural_knowledge.task import _build_judge_prompt
        prompt = _build_judge_prompt("Q?", "A", sample_rubric)
        assert "1." in prompt
        assert "4." in prompt


# ── Task instantiation ─────────────────────────────────────────────────────────

class TestCulturalKnowledgeTask:

    def test_task_returns_task_object(self):
        from inspect_ai import Task
        from india_evals.cultural_knowledge.task import cultural_knowledge
        t = cultural_knowledge()
        assert isinstance(t, Task)

    def test_task_has_dataset(self):
        from india_evals.cultural_knowledge.task import cultural_knowledge
        t = cultural_knowledge()
        assert t.dataset is not None
        assert len(t.dataset) > 0

    def test_task_has_scorer(self):
        from india_evals.cultural_knowledge.task import cultural_knowledge
        t = cultural_knowledge()
        assert t.scorer is not None


# ── Scorer instantiation ───────────────────────────────────────────────────────

class TestRubricScorer:

    def test_scorer_callable(self):
        from india_evals.cultural_knowledge.task import rubric_scorer
        s = rubric_scorer()
        assert callable(s)

    def test_scorer_accepts_custom_model(self):
        from india_evals.cultural_knowledge.task import rubric_scorer
        # Should not raise
        s = rubric_scorer(judge_model="ollama/llama3.2:3b")
        assert s is not None
