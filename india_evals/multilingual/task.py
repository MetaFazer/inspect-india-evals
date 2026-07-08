"""
Multilingual MMLU evaluation — tests factual accuracy across Indian languages.

Run:
    inspect eval india_evals/multilingual/task.py@multilingual \
        --model ollama/llama3.2:3b
"""

from pathlib import Path

import pandas as pd
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from .metrics import mmlu_accuracy


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
