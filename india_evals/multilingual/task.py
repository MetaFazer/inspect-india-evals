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
    """Smartly extract target A/B/C/D option letter from model output across all 16 languages (15 Indian + English)."""
    if not text:
        return ""

    text = str(text).strip()

    # 1. Search for explicit answer indicators across 15 Indian languages + English
    indicators = [
        # English / General
        r'(?:THE CORRECT ANSWER IS|ANSWER IS|CORRECT OPTION|OPTION|CHOICE|SO THE ANSWER IS|HENCE THE ANSWER IS|THE ANSWER SHOULD BE)\s*[:\-]*\s*([A-D])\b',
        # Hindi, Marathi, Nepali, Konkani (Devanagari)
        r'(?:उत्तर|विकल्प|सही)\s*[:\-]*\s*([A-D])\b', r'(?:विकल्प|उत्तर)\s*([ABCD])\b',
        # Bengali & Assamese (Eastern Nagari)
        r'(?:উত্তর|বিকল্প|সঠিক)\s*[:\-]*\s*([A-D])\b',
        # Tamil
        r'(?:விடை|தேர்வு|விருப்பம்|சரியான)\s*[:\-]*\s*([A-D])\b',
        # Telugu
        r'(?:సమాధానం|ఎంపిక|సరైన)\s*[:\-]*\s*([A-D])\b',
        # Kannada
        r'(?:ಉತ್ತರ|ಆಯ್ಕೆ|ಸರಿಯಾದ)\s*[:\-]*\s*([A-D])\b',
        # Malayalam
        r'(?:ഉത്തരം|ഓപ്ഷൻ|ശരിയായ)\s*[:\-]*\s*([A-D])\b',
        # Gujarati
        r'(?:જવાબ|વિકલ્પ|સાચો)\s*[:\-]*\s*([A-D])\b',
        # Odia
        r'(?:ଉତ୍ତର|ବିକଳ୍ପ|ସଠିକ୍)\s*[:\-]*\s*([A-D])\b',
        # Punjabi (Gurmukhi)
        r'(?:ਜਵਾਬ|ਚੋਣ|ਸਹੀ)\s*[:\-]*\s*([A-D])\b',
        # Urdu (Perso-Arabic)
        r'(?:جواب|گزینہ|صحیح)\s*[:\-]*\s*([A-D])\b',
        # Manipuri
        r'(?:পাউখুম|ময়েক)\s*[:\-]*\s*([A-D])\b',
        # End-of-string single letter fallback
        r'\b([A-D])\b\s*$'
    ]
    
    for pat in indicators:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
            
    # 2. Look for (A), (B), (C), (D) or **A**, **B**, **C**, **D** near the end of response
    reversed_text = "\n".join(reversed([l.strip() for l in text.split("\n") if l.strip()]))
    m = re.search(r'[\(\*\s\:\-\[\'\"]([A-D])[\)\*\s\:\-\]\'\"]', reversed_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. Fallback: Take the last standalone A/B/C/D token in the completion
    all_letters = re.findall(r'\b([ABCD])\b', text)
    if all_letters:
        return all_letters[-1].upper()
        
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
