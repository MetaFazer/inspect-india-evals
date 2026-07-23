# -*- coding: utf-8 -*-
"""
run_all.py — End-to-end evaluation runner with MLflow logging.

Runs all 6 india_evals tasks against 5 local Ollama models, parses
the Inspect AI log files, and logs all metrics to MLflow.

Usage:
    python run_all.py                    # run everything
    python run_all.py --limit 5          # quick test with 5 samples per task
    python run_all.py --models llama3.2:3b qwen3:4b   # subset of models
    python run_all.py --skip-eval        # only log existing results to MLflow

Prerequisites:
    pip install mlflow
    ollama pull qwen2.5:32b
    ollama pull llama3.3:70b
    ollama pull gemma2:27b
    ollama pull mistral-small:24b
    ollama pull llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("  ⚠  mlflow not installed. Operating in local mode (results saved to local summary JSON).")


# ── Configuration ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent

MODELS = [
    "ollama/qwen2.5:32b",
    "ollama/llama3.3:70b",
    "ollama/gemma2:27b",
    "ollama/mistral-small:24b",
    "ollama/llama3.1:8b",
]

TASKS = {
    "multilingual":       "india_evals/multilingual/task.py@multilingual",
    "bharatbbq":          "india_evals/bias/task.py@bharatbbq",
    "safety":             "india_evals/safeguards/task.py@multilingual_safety",
    "jailbreak":          "india_evals/safeguards/task.py@jailbreak_safety",
    "dpi":                "india_evals/dpi_safety/task.py@dpi_safety",
    "cultural_knowledge": "india_evals/cultural_knowledge/task.py@cultural_knowledge",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_env() -> dict:
    
    env = os.environ.copy()
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "ALL_PROXY"):
        env.pop(var, None)
    # Ensure localhost is always bypassed even if a proxy is set later
    env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    env["no_proxy"] = "localhost,127.0.0.1,::1"
    return env


def run_eval(task_spec: str, model: str, limit: int | None = None) -> str | None:
    """Run a single inspect eval and return the log file path."""
    cmd = ["inspect", "eval", task_spec, "--model", model]
    if limit:
        cmd += ["--limit", str(limit)]

    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=False, env=_clean_env())

    if result.returncode != 0:
        print(f"  ⚠  inspect eval failed (exit {result.returncode})")
        return None

    # Find the most recent log file
    log_dir = REPO_ROOT / "logs"
    if not log_dir.exists():
        return None

    logs = sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
    return str(logs[-1]) if logs else None



def parse_eval_log(log_path: str) -> dict:
    """
    Parse an Inspect AI .eval log file and extract metrics.
    .eval files are zip-compressed — use inspect_ai.log.read_eval_log(),
    not raw open(), to decode them correctly.
    Returns a dict of metric_name → value.
    """
    metrics = {}
    try:
        from inspect_ai.log import read_eval_log
        log = read_eval_log(log_path)

        if log.results and log.results.scores:
            for score_group in log.results.scores:
                scorer_name = score_group.name or "unknown"
                for metric_name, metric_data in (score_group.metrics or {}).items():
                    metrics[f"{scorer_name}/{metric_name}"] = metric_data.value

        if log.samples:
            metrics["total_samples"] = len(log.samples)
            lang_totals: dict = {}
            lang_refused: dict = {}
            for sample in log.samples:
                lang = (sample.metadata or {}).get("language", "unknown")
                if sample.scores:
                    for scorer_name, score_obj in sample.scores.items():
                        val = getattr(score_obj, "value", 0)
                        if isinstance(val, (int, float)):
                            lang_totals[lang] = lang_totals.get(lang, 0) + 1
                            lang_refused[lang] = lang_refused.get(lang, 0) + val
            for lang, total in lang_totals.items():
                if total > 0:
                    metrics[f"refusal_rate/{lang}"] = round(lang_refused[lang] / total, 4)

    except Exception as e:
        print(f"  ⚠  Failed to parse log: {e}")

    return metrics



def log_to_mlflow(model: str, task_name: str, metrics: dict):
    """Log metrics for a single model+task run to MLflow."""
    if not HAS_MLFLOW:
        return
    with mlflow.start_run(run_name=f"{model.split('/')[-1]}_{task_name}"):
        mlflow.set_tag("model", model)
        mlflow.set_tag("task", task_name)

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key.replace("/", "."), value)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run india_evals across models + log to MLflow")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (for quick testing)")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to evaluate")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()), help="Tasks to run")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval runs, only log existing results")
    parser.add_argument("--experiment", default="india_evals", help="MLflow experiment name")
    args = parser.parse_args()

    if HAS_MLFLOW:
        mlflow.set_experiment(args.experiment)

    print(f"\n{'#'*60}")
    print(f"  india_evals — End-to-End Evaluation")
    print(f"  Models:  {', '.join(args.models)}")
    print(f"  Tasks:   {', '.join(args.tasks)}")
    print(f"  Limit:   {args.limit or 'FULL DATASET'}")
    print(f"{'#'*60}\n")

    all_results = {}

    for model in args.models:
        model_results = {}

        for task_name in args.tasks:
            if task_name not in TASKS:
                print(f"  ⚠  Unknown task: {task_name}, skipping")
                continue

            task_spec = TASKS[task_name]

            if not args.skip_eval:
                log_path = run_eval(task_spec, model, args.limit)
            else:
                log_path = None

            if log_path:
                metrics = parse_eval_log(log_path)
                log_to_mlflow(model, task_name, metrics)
                model_results[task_name] = metrics
                print(f"  ✓  {task_name}: {len(metrics)} metrics logged")
            else:
                print(f"  ⚠  {task_name}: no results to log")

        all_results[model] = model_results

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}\n")

    for model, tasks in all_results.items():
        print(f"  {model}:")
        for task_name, metrics in tasks.items():
            accuracy = metrics.get("accuracy/value", metrics.get("refusal_scorer/accuracy", "N/A"))
            print(f"    {task_name}: accuracy={accuracy}")
        print()

    # ── Compute fairness index if we have all 4 dimensions ─────────
    sys.path.insert(0, str(REPO_ROOT))
    from india_evals.scorers.fairness import fairness_index

    for model, tasks in all_results.items():
        ml_acc = tasks.get("multilingual", {}).get("mmlu_accuracy/accuracy", 0.0)
        bias = tasks.get("bharatbbq", {}).get("match/accuracy", 0.0)
        safety = tasks.get("safety", {}).get("refusal_scorer/accuracy", 0.0)
        dpi = tasks.get("dpi", {}).get("dpi_scorer/accuracy", 0.0)

        fi = fairness_index(
            multilingual_accuracy=ml_acc,
            bias_score_amb=1.0 - bias,  # convert accuracy to bias score
            safety_refusal_rate=safety,
            dpi_accuracy=dpi,
        )

        print(f"  {model}: Fairness Index = {fi['fairness_index']}")

        if HAS_MLFLOW:
            with mlflow.start_run(run_name=f"{model.split('/')[-1]}_fairness_index"):
                mlflow.set_tag("model", model)
                mlflow.set_tag("task", "fairness_index")
                for k, v in fi.items():
                    mlflow.log_metric(k, v)

    print(f"\n  MLflow UI:  mlflow ui  →  http://localhost:5000")
    print(f"  Experiment: {args.experiment}\n")


if __name__ == "__main__":
    main()
