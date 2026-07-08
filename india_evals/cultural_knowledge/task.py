"""
Cultural Knowledge evaluation — rubric-based LLM-as-judge scoring.

Tests a model's knowledge of Indian domains:
    Indian Constitution, Healthcare, Economy, History, Science & Technology,
    Culture, Geography, and more.

Each question is scored against a rubric of 4 criteria by an LLM judge.
The judge returns a fraction (0.0–1.0) of criteria passed.

Dataset:
    india_evals/cultural_knowledge/datasets/Module4_rubric_dataset.json
    ~825 questions across multiple Indian knowledge domains.

Run:
    inspect eval india_evals/cultural_knowledge/task.py@cultural_knowledge \\
        --model ollama/llama3.2:3b

    # Quick smoke test (5 samples):
    inspect eval india_evals/cultural_knowledge/task.py@cultural_knowledge \\
        --model ollama/llama3.2:3b --limit 5
"""

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model, ChatMessageUser
from inspect_ai.scorer import Score, scorer, mean
from inspect_ai.solver import generate


# ── Dataset path ───────────────────────────────────────────────────────────────

_DATASET_DIR = Path(__file__).parent / "datasets"
_DATASET_FILE = _DATASET_DIR / "Cultural_knowledge_rubric_dataset.json"


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_cultural_knowledge(max_rows: int = 0) -> list[Sample]:
    """
    Load the cultural knowledge rubric dataset.

    Parameters
    ----------
    max_rows : int
        If > 0, only load this many rows (useful for quick tests).

    Returns
    -------
    list[Sample] — each sample has:
        input    : the question prompt
        target   : "" (open-ended; scored by rubric judge)
        metadata : id, domain, rubric
    """
    with open(_DATASET_FILE, encoding="utf-8") as f:
        data = json.load(f)

    if max_rows:
        data = data[:max_rows]

    samples = []
    for item in data:
        question = item["Questions"]
        prompt = (
            f"Question:\n\n{question}\n\n"
            "Provide a detailed and accurate answer."
        )
        samples.append(Sample(
            input=prompt,
            target="",          # open-ended — judged by rubric scorer
            metadata={
                "id":     item["Scenario Id"],
                "domain": item["Domain"],
                "rubric": item["rubric"],       # list[str] of criteria
            },
        ))

    return samples


# ── Rubric judge scorer ────────────────────────────────────────────────────────

def _build_judge_prompt(question: str, answer: str, rubric: list[str]) -> str:
    rubric_text = "\n".join(f"{i}. {c}" for i, c in enumerate(rubric, 1))
    return f"""You are an evaluation engine.

Question:

{question}

Model Answer:

{answer}

Rubric:

{rubric_text}

Evaluate the answer against EVERY rubric criterion.
For each criterion determine whether the answer satisfies it.

Return ONLY valid JSON in this exact format:
{{
  "results": [
    {{
      "criterion": "criterion text",
      "passed": true
    }}
  ]
}}

Rules:
- Return valid JSON only
- No markdown, no code fences, no explanations
- One result object per rubric criterion, in order"""


@scorer(metrics=[mean()])
def rubric_scorer(judge_model: str = "ollama/llama3.2:3b"):
    """
    LLM-as-judge scorer using a rubric of 4 criteria per question.

    Returns a score of 0.0–1.0 representing the fraction of rubric
    criteria the model's answer satisfied.
    """
    async def score(state, target) -> Score:
        answer  = state.output.completion
        rubric  = state.metadata.get("rubric", [])
        question = state.input_text.split("\n\n", 1)[0].replace("Question:", "").strip()

        if not rubric:
            return Score(value=0.0, explanation="No rubric found in metadata.")

        prompt = _build_judge_prompt(question, answer, rubric)

        try:
            judge  = get_model(judge_model)
            output = await judge.generate([ChatMessageUser(content=prompt)])
            text   = output.completion.strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                text = "\n".join(
                    line for line in text.splitlines()
                    if not line.strip().startswith("```")
                )

            result  = json.loads(text)
            results = result.get("results", [])

            if not results:
                return Score(value=0.0, explanation="Judge returned empty results.")

            n_passed = sum(1 for r in results if r.get("passed", False))
            score_val = n_passed / len(results)

            criteria_summary = "\n".join(
                f"  {'PASS' if r.get('passed') else 'FAIL'}: {r.get('criterion', '')}"
                for r in results
            )

            return Score(
                value=score_val,
                answer=f"{n_passed}/{len(results)} criteria passed",
                explanation=f"Rubric results:\n{criteria_summary}",
                metadata={
                    "n_passed": n_passed,
                    "n_total":  len(results),
                    "criteria": results,
                },
            )

        except json.JSONDecodeError as e:
            return Score(
                value=0.0,
                explanation=f"JSON parse error from judge: {e}",
            )
        except Exception as e:
            return Score(
                value=0.0,
                explanation=f"Scorer error: {e}",
            )

    return score


# ── Task ───────────────────────────────────────────────────────────────────────

@task
def cultural_knowledge():
    """
    Rubric-based Indian cultural knowledge evaluation.

    Covers: Constitution, Healthcare, Economy, History,
            Science & Technology, Culture, Geography, and more.
    """
    return Task(
        dataset=load_cultural_knowledge(),
        solver=generate(),
        scorer=rubric_scorer(),
    )
