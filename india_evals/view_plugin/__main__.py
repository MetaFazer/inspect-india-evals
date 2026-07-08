"""
CLI entry point for india_evals.view_plugin.

Usage:
    python -m india_evals.view_plugin                        # scan ./logs/
    python -m india_evals.view_plugin path/to/logs/          # custom dir
    python -m india_evals.view_plugin a.eval b.eval          # specific files
    python -m india_evals.view_plugin --no-browser logs/     # skip auto-open
    python -m india_evals.view_plugin --output my.html logs/ # custom output
"""

import argparse
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="python -m india_evals.view_plugin",
        description="Generate an interactive India heatmap report from Inspect AI eval logs.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["logs"],
        help="Directory containing .eval files, or specific .eval file paths (default: ./logs/)",
    )
    parser.add_argument(
        "--output", "-o",
        default="india_report.html",
        help="Output HTML file path (default: india_report.html)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the report in the browser automatically.",
    )
    args = parser.parse_args()

    # Check plotly
    try:
        import plotly  # noqa: F401
    except ImportError:
        print(
            "\n  ⚠  plotly is not installed.  Charts will be unavailable.\n"
            "     Install with:  pip install plotly\n"
        )

    # Resolve input paths
    input_paths = [pathlib.Path(p) for p in args.paths]

    # If all inputs are .eval files, pass directory as parent of first file
    if all(p.suffix == ".eval" for p in input_paths):
        # Pass them all directly via a temp wrapper
        from india_evals.view_plugin.parser import parse_logs
        from india_evals.view_plugin.heatmap import render_report as _render

        # Monkey-patch to pass files directly
        import india_evals.view_plugin.heatmap as _heatmap_mod
        _orig_find = None

        def _patched_render(logs_dir, output, open_browser):
            import pandas as pd
            df = parse_logs(input_paths)
            # Re-use the rest of render_report but with our df
            # Call the real render_report pointing at a non-existent dir that has 0 logs,
            # then we can't do that easily — so call from parser directly.
            return df

        # Simpler: just call render_report with the parent dir, let it find what it finds,
        # but override by writing logs directly.
        # The cleanest approach: render_report already calls find_logs(logs_dir)
        # We override find_logs to return our specific files.
        from india_evals.view_plugin import heatmap as hm
        import india_evals.view_plugin.parser as _parser_mod
        _orig_find_logs = _parser_mod.find_logs

        def _patched_find_logs(directory):
            return input_paths

        _parser_mod.find_logs = _patched_find_logs
        try:
            hm.render_report(
                logs_dir=input_paths[0].parent,
                output=args.output,
                open_browser=not args.no_browser,
            )
        finally:
            _parser_mod.find_logs = _orig_find_logs
    else:
        # Treat as directories
        logs_dir = input_paths[0]
        if not logs_dir.exists():
            print(f"  ⚠  Path does not exist: {logs_dir}")
            sys.exit(1)

        from india_evals.view_plugin.heatmap import render_report
        render_report(
            logs_dir=logs_dir,
            output=args.output,
            open_browser=not args.no_browser,
        )


if __name__ == "__main__":
    main()
