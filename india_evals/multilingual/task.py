"""
Multilingual MMLU evaluation — tests factual accuracy across Indian languages.

Run:
    inspect eval india_evals/multilingual/task.py@multilingual \
        --model ollama/llama3.2:3b
"""

import re
from pathlib import Path

import pandas as pd
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import Score, scorer, accuracy
from inspect_ai.scorer import Target


# ── MMLU scorer (inlined for inspect eval compatibility) ───────────────────────
# Each task.py must be self-contained so Inspect AI can load it as a
# standalone file without requiring the package to be installed.

def _normalize_answer(text: str) -> str:
    """Extract first A/B/C/D token from model output."""
    if not text:
        return ""
    text = str(text).strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return ""


@scorer(metrics=[accuracy()])
def mmlu_accuracy():
    """Score a sample by comparing extracted letter to target."""

    async def score(state, target: Target):
        prediction = _normalize_answer(state.output.completion)
        expected   = _normalize_answer(target.text)
        correct    = prediction == expected
        return Score(
            value=1 if correct else 0,
            answer=prediction,
            explanation=f"predicted={prediction}, expected={expected}",
        )

    return score


def load_samples(max_rows: int = 0) -> list[Sample]:
    """
    Load the multilingual MMLU dataset.
    Each row becomes one Sample with language/subject metadata.
    """
    csv_path = Path(__file__).parent / "datasets" / "mmlu_translated.csv"
    df = pd.read_csv(csv_path)

    if max_rows:
        df = df.head(max_rows)

    samples = []
    for _, row in df.iterrows():
        prompt = f"""
Answer the following multiple-choice question.

Question:
{row['question']}

A. {row['A']}
B. {row['B']}
C. {row['C']}
D. {row['D']}

Reply with exactly one character:

A
B
C
D

No explanation.
"""
        samples.append(
            Sample(
                input=prompt,
                target=row["answer_letter"],
                metadata={
                    "language": row["language"],
                    "subject": row["subject"],
                    "question_id": int(row["id"]),
                },
            )
        )
    return samples


@task
def multilingual():
    """Multilingual MMLU accuracy across Indian languages."""
    return Task(
        dataset=load_samples(),
        solver=generate(),
        scorer=mmlu_accuracy(),
    )
