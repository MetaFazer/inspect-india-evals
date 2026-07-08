"""
india_evals.view_plugin
========================
Standalone India heatmap report generator for Inspect AI eval logs.

Usage:
    python -m india_evals.view_plugin                  # scan ./logs/
    python -m india_evals.view_plugin path/to/logs/    # custom dir
    python -m india_evals.view_plugin run1.eval run2.eval   # specific files

Public API:
    from india_evals.view_plugin import render_report
    html_path = render_report("logs/", output="report.html")
"""

from india_evals.view_plugin.heatmap import render_report

__all__ = ["render_report"]
