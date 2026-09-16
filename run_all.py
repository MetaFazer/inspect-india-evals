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
    "ollama/gemma2:27b",
    "ollama/qwen2.5:32b",
    "ollama/mashriram/sarvam-m",
    "ollama/deepseek-r1:14b",
]

# The judge must be independent of every model under test. Configurable via
# --judge-model or INDIA_EVALS_JUDGE; must never be one of MODELS (checked
# at startup in main()).
DEFAULT_JUDGE_MODEL = os.environ.get("INDIA_EVALS_JUDGE", "ollama/llama3.1:8b")

# Tasks whose scorer takes a judge_model parameter (the LLM-as-judge tasks).
JUDGED_TASKS = {"safety", "jailbreak", "dpi", "cultural_knowledge"}

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


def run_eval(
    task_spec: str,
    model: str,
    limit: int | None = None,
    judge_model: str | None = None,
    task_name: str | None = None,
) -> str | None:
    """Run a single inspect eval and return the log file path."""
    cmd = ["inspect", "eval", task_spec, "--model", model]
    if limit:
        cmd += ["--limit", str(limit)]
    if judge_model and task_name in JUDGED_TASKS:
        cmd += ["-T", f"judge_model={judge_model}"]

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


def find_latest_log(task_name: str, model_name: str) -> str | None:
    """Find the most recent .eval log file matching task_name and model_name."""
    log_dir = REPO_ROOT / "logs"
    if not log_dir.exists():
        return None

    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        return None

    clean_model = model_name.lower()

    for p in sorted(log_dir.glob("*.eval"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            log = read_eval_log(str(p))
            if log.eval:
                eval_task = (log.eval.task or "").lower()
                eval_model = (log.eval.model or "").lower()
                task_match = (eval_task == task_name.lower()) or (task_name == "safety" and eval_task == "multilingual_safety") or (task_name == "jailbreak" and eval_task == "jailbreak_safety")
                model_match = (eval_model == clean_model) or (clean_model.endswith(eval_model)) or (eval_model.endswith(clean_model))
                if task_match and model_match:
                    return str(p)
        except Exception:
            continue

    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run india_evals across models + log to MLflow")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (for quick testing)")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to evaluate")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()), help="Tasks to run")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval runs, only log existing results")
    parser.add_argument("--experiment", default="india_evals", help="MLflow experiment name")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                         help="LLM judge model for safety/dpi/cultural_knowledge scorers "
                              "(default: $INDIA_EVALS_JUDGE or ollama/llama3.1:8b). "
                              "Must not be one of --models.")
    args = parser.parse_args()

    if args.judge_model in args.models:
        print(
            f"  ⚠  ERROR: judge model '{args.judge_model}' is also one of the models "
            "under evaluation. The judge must be independent of every model being "
            "tested — pick a different --judge-model or INDIA_EVALS_JUDGE.",
            file=sys.stderr,
        )
        sys.exit(1)

    if HAS_MLFLOW:
        mlflow.set_experiment(args.experiment)

    print(f"\n{'#'*60}")
    print(f"  india_evals — End-to-End Evaluation")
    print(f"  Models:  {', '.join(args.models)}")
    print(f"  Tasks:   {', '.join(args.tasks)}")
    print(f"  Limit:   {args.limit or 'FULL DATASET'}")
    print(f"  Judge:   {args.judge_model}")
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
                log_path = run_eval(task_spec, model, args.limit, args.judge_model, task_name)
            else:
                log_path = find_latest_log(task_name, model)

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

    def get_accuracy(metrics: dict) -> float | None:
        """Find the main accuracy metric from a parsed eval log metrics dict."""
        for key in ("accuracy/accuracy", "accuracy/value", "refusal_scorer/accuracy", 
                    "dpi_scorer/accuracy", "match/accuracy", "accuracy/mean"):
            if key in metrics and isinstance(metrics[key], (int, float)):
                return float(metrics[key])
        for k, v in metrics.items():
            if ("accuracy" in k.lower() or "mean" in k.lower()) and isinstance(v, (int, float)):
                return float(v)
        return None

    for model, tasks in all_results.items():
        print(f"  {model}:")
        for task_name, metrics in tasks.items():
            acc = get_accuracy(metrics)
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            print(f"    {task_name}: accuracy={acc_str}")
        print()

    # ── Compute fairness index ─────────────────────────────────────────
    sys.path.insert(0, str(REPO_ROOT))
    from india_evals.scorers.fairness import (
        fairness_index,
        dimension_variance,
        sensitivity_analysis,
        EQUAL_WEIGHTS,
        SAFETY_WEIGHTED,
        ACCESS_WEIGHTED,
    )

    # Five-dimension (current) IFI per model, keyed for dimension_variance()
    # and sensitivity_analysis() below. ifi_v1_4dim is the legacy score
    # (pre-jailbreak) kept alongside ifi_v2_5dim so published figures stay
    # traceable to which version produced them.
    ifi_5dim_by_model: dict[str, dict] = {}

    for model, tasks in all_results.items():
        ml_acc = get_accuracy(tasks.get("multilingual", {})) or 0.0
        bias = get_accuracy(tasks.get("bharatbbq", {})) or 0.0
        safety = get_accuracy(tasks.get("safety", {})) or 0.0
        jailbreak = get_accuracy(tasks.get("jailbreak", {})) or 0.0
        dpi = get_accuracy(tasks.get("dpi", {})) or 0.0

        ifi_v1_4dim = fairness_index(
            multilingual_accuracy=ml_acc,
            bias_score_amb=1.0 - bias,
            safety_refusal_rate=safety,
            dpi_accuracy=dpi,
        )
        ifi_v2_5dim = fairness_index(
            multilingual_accuracy=ml_acc,
            bias_score_amb=1.0 - bias,
            safety_refusal_rate=safety,
            dpi_accuracy=dpi,
            jailbreak_refusal_rate=jailbreak,
        )
        ifi_5dim_by_model[model] = ifi_v2_5dim

        print(f"  {model}: IFI (v1, 4-dim, legacy) = {ifi_v1_4dim['fairness_index']}   "
              f"IFI (v2, 5-dim incl. jailbreak) = {ifi_v2_5dim['fairness_index']}")

        if HAS_MLFLOW:
            with mlflow.start_run(run_name=f"{model.split('/')[-1]}_fairness_index"):
                mlflow.set_tag("model", model)
                mlflow.set_tag("task", "fairness_index")
                mlflow.log_metric("ifi_v1_4dim", ifi_v1_4dim["fairness_index"])
                mlflow.log_metric("ifi_v2_5dim", ifi_v2_5dim["fairness_index"])
                for k, v in ifi_v1_4dim.items():
                    mlflow.log_metric(f"v1_4dim.{k}", v)
                for k, v in ifi_v2_5dim.items():
                    mlflow.log_metric(f"v2_5dim.{k}", v)

    # ── Which dimensions actually discriminate between models? ──────────
    if ifi_5dim_by_model:
        print(f"\n{'='*60}")
        print("  DIMENSION VARIANCE (5-dim IFI, does this dimension discriminate?)")
        print(f"{'='*60}\n")
        variance_report = dimension_variance(ifi_5dim_by_model)
        print(f"  {'dimension':<14} {'min':>6} {'max':>6} {'range':>7}   discriminative?")
        for dim, stats in variance_report.items():
            flag = "yes" if stats["discriminative"] else "NO — constant"
            print(f"  {dim:<14} {stats['min']:>6.3f} {stats['max']:>6.3f} {stats['range']:>7.3f}   {flag}")

        # ── Weight sensitivity analysis ──────────────────────────────────
        print(f"\n{'='*60}")
        print("  WEIGHT SENSITIVITY ANALYSIS")
        print(f"{'='*60}\n")
        analysis = sensitivity_analysis(
            ifi_5dim_by_model,
            {
                "EQUAL_WEIGHTS": EQUAL_WEIGHTS,
                "SAFETY_WEIGHTED": SAFETY_WEIGHTED,
                "ACCESS_WEIGHTED": ACCESS_WEIGHTED,
            },
        )
        for scheme, ranking in analysis["rankings"].items():
            scores = analysis["scores"][scheme]
            ranked = ", ".join(f"{m} ({scores[m]})" for m in ranking)
            print(f"  {scheme:<16} {ranked}")

        if analysis["ranking_stable"]:
            print(f"\n  Top-ranked model is STABLE across all three weight schemes: "
                  f"{next(iter(analysis['top_model_by_scheme'].values()))}")
        else:
            print(f"\n  Top-ranked model CHANGES depending on weighting — "
                  "the equal-weight ranking is not robust:")
            for scheme, top_model in analysis["top_model_by_scheme"].items():
                print(f"    {scheme:<16} → {top_model}")

    print(f"\n  MLflow UI:  mlflow ui  →  http://localhost:5000")
    print(f"  Experiment: {args.experiment}\n")


if __name__ == "__main__":
    main()
