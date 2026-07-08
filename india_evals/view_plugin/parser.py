"""
parser.py — reads Inspect AI .eval log files and returns a normalised DataFrame.

Each row in the output DataFrame represents one evaluated sample, with columns:
    log_file, task, model, sample_id,
    language, category, subject, attack_type, risk_level,
    score_name, score_value, correct
"""

from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd


# ── Inspect AI log reader ──────────────────────────────────────────────────────

def _read_log(path: str | pathlib.Path):
    """Read an Inspect AI .eval log file and return the log object."""
    from inspect_ai.log import read_eval_log
    return read_eval_log(str(path), header_only=False)


def _score_value(score) -> Optional[float]:
    """Safely extract a numeric value from an Inspect AI Score object."""
    if score is None:
        return None
    v = getattr(score, "value", None)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    # Some scorers return "C" / "I" (correct/incorrect) strings
    if isinstance(v, str):
        v_up = v.strip().upper()
        if v_up in ("C", "CORRECT", "1", "REFUSED", "YES"):
            return 1.0
        if v_up in ("I", "INCORRECT", "-1", "0", "COMPLIED", "NO"):
            return 0.0
    return None


def parse_logs(paths: list[str | pathlib.Path]) -> pd.DataFrame:
    """
    Parse a list of .eval log files into a flat DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        log_file, task, model, sample_id,
        language, category, subject, attack_type, risk_level,
        score_name, score_value, correct
    """
    rows = []

    for path in paths:
        path = pathlib.Path(path)
        try:
            log = _read_log(path)
        except Exception as e:
            print(f"  ⚠  Could not read {path.name}: {e}")
            continue

        task  = getattr(log.eval, "task",  path.stem)
        model = getattr(log.eval, "model", "unknown")

        samples = log.samples or []
        for sample in samples:
            meta = sample.metadata or {}

            base = {
                "log_file":    path.name,
                "task":        task,
                "model":       str(model),
                "sample_id":   str(getattr(sample, "id", "")),
                "language":    str(meta.get("language",    "unknown")),
                "category":    str(meta.get("category",   meta.get("subject", "unknown"))),
                "subject":     str(meta.get("subject",    "")),
                "attack_type": str(meta.get("attack_type","unknown")),
                "risk_level":  str(meta.get("risk_level", "unknown")),
                "dataset":     str(meta.get("dataset",    "")),
            }

            scores = sample.scores or {}
            if scores:
                for score_name, score_obj in scores.items():
                    val = _score_value(score_obj)
                    row = dict(base)
                    row["score_name"]  = score_name
                    row["score_value"] = val
                    # normalise: 1.0 = correct/refused, 0.0 = wrong/complied, -1 = wrong (our tasks)
                    row["correct"] = (
                        1.0 if val is not None and val > 0 else
                        0.0 if val is not None else
                        None
                    )
                    rows.append(row)
            else:
                # No scores recorded (e.g., old log without scorer)
                row = dict(base)
                row["score_name"]  = None
                row["score_value"] = None
                row["correct"]     = None
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Clean up "unknown" for axis labels
    df = df.replace("unknown", pd.NA)
    return df


def find_logs(directory: str | pathlib.Path) -> list[pathlib.Path]:
    """Return all .eval files in a directory (non-recursive)."""
    d = pathlib.Path(directory)
    if not d.exists():
        return []
    return sorted(d.glob("*.eval"), key=lambda p: p.stat().st_mtime)
