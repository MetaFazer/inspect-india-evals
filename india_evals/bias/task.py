"""
BharatBBQ bias evaluation — measures stereotype and social bias across
Indian cultural categories.

Run:
    inspect eval india_evals/bias/task.py@bharatbbq \
        --model ollama/llama3.2:3b
"""

from pathlib import Path

import pandas as pd
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import match


TARGET_MAP = {
    1: "A",
    2: "B",
    3: "C",
}


def build_prompt(row) -> str:
    """Format a BharatBBQ row into a multiple-choice prompt."""
    return f"""
Context:
{row['Context']}

Question:
{row['Question']}

Options:

A. {row['Option1']}
B. {row['Option2']}
C. {row['Option3']}

Answer with only A, B, or C.
"""


def load_bharatbbq(max_rows_per_file: int = 0) -> list[Sample]:
    """
    Load all BharatBBQ category CSV files from the datasets/ directory.
    No dev limits — loads every row from every category file.
    """
    samples = []
    dataset_dir = Path(__file__).parent / "datasets"

    csv_files = sorted(dataset_dir.glob("*.csv"))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if max_rows_per_file:
            df = df.head(max_rows_per_file)

        for _, row in df.iterrows():
            samples.append(
                Sample(
                    input=build_prompt(row),
                    target=TARGET_MAP[int(row["Target"])],
                    metadata={
                        "category": row.get("Category"),
                        "context_type": row.get("Context_type"),
                        "source_file": csv_file.name,
                        "label": row.get("Label"),
                        "target_numeric": row.get("Target"),
                        "pairing": row.get("Pairing"),
                        "question_polarity": row.get("Question_polarity"),
                        "qid": row.get("Qid"),
                        "proper_noun": row.get("Proper_Noun"),
                    },
                )
            )

    return samples


@task
def bharatbbq():
    """BharatBBQ stereotype & social-bias benchmark."""
    return Task(
        dataset=load_bharatbbq(),
        solver=generate(),
        scorer=match(),
    )
