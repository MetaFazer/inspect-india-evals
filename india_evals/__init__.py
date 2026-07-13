"""
india_evals — India-specific AI safety, bias, and fairness evaluations
for Inspect AI.
"""

from __future__ import annotations

from india_evals.multilingual.task import multilingual
from india_evals.bias.task import bharatbbq
from india_evals.safeguards.task import multilingual_safety, jailbreak_safety
from india_evals.dpi_safety.task import dpi_safety
from india_evals.cultural_knowledge.task import cultural_knowledge
from india_evals.scorers.fairness import fairness_index
from india_evals.view_plugin import render_report

__version__ = "0.1.0"

__all__ = [
    "multilingual",
    "bharatbbq",
    "multilingual_safety",
    "jailbreak_safety",
    "dpi_safety",
    "cultural_knowledge",
    "fairness_index",
    "render_report",
]

