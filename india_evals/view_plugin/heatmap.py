"""
heatmap.py — Generates a self-contained interactive HTML report from Inspect AI
eval logs, with India-specific per-language × per-category heatmaps.

Requires: plotly  (pip install plotly)
"""

from __future__ import annotations

import pathlib
import webbrowser
from typing import Optional

import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ── Colour palette ─────────────────────────────────────────────────────────────

INDIA_BLUE   = "#1a3a5c"
SAFFRON      = "#FF9933"
INDIA_GREEN  = "#138808"
WHITE        = "#FFFFFF"
LIGHT_BG     = "#f4f6fa"
CARD_BG      = "#ffffff"

HEATMAP_COLORSCALE = [
    [0.0,  "#d73027"],   # red   → 0 (bad)
    [0.25, "#fc8d59"],
    [0.5,  "#fee090"],
    [0.75, "#91bfdb"],
    [1.0,  "#4575b4"],   # blue  → 1 (good)
]

REFUSAL_COLORSCALE = [
    [0.0,  "#d73027"],
    [0.5,  "#ffffbf"],
    [1.0,  "#1a9850"],   # green → 1 (refused = good for safety)
]


# ── Individual chart builders ──────────────────────────────────────────────────

def _heatmap_fig(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str = "correct",
    title: str = "",
    colorscale=None,
    zmin: float = 0.0,
    zmax: float = 1.0,
) -> Optional["go.Figure"]:
    """Create a single plotly heatmap figure from a DataFrame."""
    if not HAS_PLOTLY:
        return None

    sub = df[[x_col, y_col, value_col]].dropna()
    if sub.empty:
        return None

    pivot = sub.groupby([y_col, x_col])[value_col].mean().unstack(fill_value=None)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=colorscale or HEATMAP_COLORSCALE,
            zmin=zmin,
            zmax=zmax,
            text=[[f"{v:.2f}" if v is not None else "N/A"
                   for v in row]
                  for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 11},
            hoverongaps=False,
            colorbar=dict(title="Score", thickness=15, len=0.8),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INDIA_BLUE), x=0.02),
        plot_bgcolor=LIGHT_BG,
        paper_bgcolor=CARD_BG,
        height=max(300, len(pivot.index) * 45 + 120),
        margin=dict(l=160, r=60, t=60, b=80),
        xaxis=dict(side="bottom", tickangle=-30),
    )
    return fig


def _bar_fig(
    df: pd.DataFrame,
    x_col: str,
    value_col: str = "correct",
    title: str = "",
    color: str = INDIA_BLUE,
) -> Optional["go.Figure"]:
    """Bar chart of mean score per category."""
    if not HAS_PLOTLY:
        return None

    sub = df[[x_col, value_col]].dropna()
    if sub.empty:
        return None

    agg = sub.groupby(x_col)[value_col].mean().reset_index()
    agg.columns = [x_col, "mean_score"]
    agg = agg.sort_values("mean_score", ascending=True)

    fig = px.bar(
        agg, y=x_col, x="mean_score",
        orientation="h",
        title=title,
        labels={"mean_score": "Score (0–1)", x_col: ""},
        color="mean_score",
        color_continuous_scale=HEATMAP_COLORSCALE,
        range_color=[0, 1],
        text="mean_score",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        plot_bgcolor=LIGHT_BG, paper_bgcolor=CARD_BG,
        height=max(300, len(agg) * 38 + 100),
        margin=dict(l=160, r=80, t=60, b=40),
        title=dict(font=dict(size=16, color=INDIA_BLUE), x=0.02),
        coloraxis_showscale=False,
    )
    return fig


def _summary_table_fig(df: pd.DataFrame) -> Optional["go.Figure"]:
    """Model × Task summary table as a plotly table."""
    if not HAS_PLOTLY or df.empty:
        return None

    agg = (
        df.dropna(subset=["correct"])
          .groupby(["model", "task"])["correct"]
          .agg(["mean", "count"])
          .reset_index()
    )
    agg.columns = ["Model", "Task", "Accuracy", "Samples"]
    agg["Accuracy"] = agg["Accuracy"].map(lambda x: f"{x:.2%}")

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Model</b>", "<b>Task</b>", "<b>Accuracy</b>", "<b>Samples</b>"],
            fill_color=INDIA_BLUE,
            font=dict(color=WHITE, size=13),
            align="left",
            height=35,
        ),
        cells=dict(
            values=[agg[c] for c in ["Model", "Task", "Accuracy", "Samples"]],
            fill_color=[[LIGHT_BG if i % 2 == 0 else CARD_BG for i in range(len(agg))]],
            align="left",
            font=dict(size=12),
            height=30,
        ),
    ))
    fig.update_layout(
        title=dict(text="Summary: Model × Task", font=dict(size=16, color=INDIA_BLUE), x=0.02),
        paper_bgcolor=CARD_BG,
        height=max(250, len(agg) * 35 + 100),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ── HTML builder ───────────────────────────────────────────────────────────────

def _fig_to_div(fig) -> str:
    """Convert a plotly figure to an embeddable HTML div."""
    if fig is None:
        return "<p><em>No data available for this chart.</em></p>"
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def _tab_html(tabs: list[tuple[str, str]]) -> str:
    """
    Build tabbed layout.
    tabs = [(tab_label, content_html), ...]
    """
    tab_buttons = ""
    tab_panels  = ""

    for i, (label, content) in enumerate(tabs):
        active = "active" if i == 0 else ""
        tab_buttons += f'<button class="tab-btn {active}" onclick="showTab({i})">{label}</button>\n'
        display = "block" if i == 0 else "none"
        tab_panels += f'<div class="tab-panel" id="panel-{i}" style="display:{display}">{content}</div>\n'

    return f"""
<div class="tab-bar">{tab_buttons}</div>
<div class="tab-content">{tab_panels}</div>
<script>
function showTab(n) {{
    document.querySelectorAll('.tab-panel').forEach((p,i) => p.style.display = i===n ? 'block' : 'none');
    document.querySelectorAll('.tab-btn').forEach((b,i) => b.className = 'tab-btn' + (i===n ? ' active' : ''));
}}
</script>
"""


_CSS = f"""
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: {LIGHT_BG};
    color: #222;
  }}
  header {{
    background: {INDIA_BLUE};
    color: {WHITE};
    padding: 20px 40px;
    display: flex;
    align-items: center;
    gap: 16px;
  }}
  header .flag {{ font-size: 2rem; }}
  header h1 {{ font-size: 1.5rem; font-weight: 700; }}
  header p  {{ font-size: 0.9rem; opacity: 0.8; margin-top: 2px; }}
  .stripe {{
    height: 6px;
    background: linear-gradient(to right, {SAFFRON} 33%, {WHITE} 33% 66%, {INDIA_GREEN} 66%);
  }}
  .container {{ max-width: 1300px; margin: 30px auto; padding: 0 24px; }}
  .card {{
    background: {CARD_BG};
    border-radius: 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    padding: 24px;
    margin-bottom: 24px;
  }}
  .section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {INDIA_BLUE};
    border-left: 4px solid {SAFFRON};
    padding-left: 12px;
    margin-bottom: 16px;
  }}
  .tab-bar {{
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }}
  .tab-btn {{
    background: {LIGHT_BG};
    border: 1px solid #dde;
    border-radius: 6px;
    padding: 7px 18px;
    cursor: pointer;
    font-size: 0.88rem;
    color: {INDIA_BLUE};
    transition: all .15s;
  }}
  .tab-btn.active, .tab-btn:hover {{
    background: {INDIA_BLUE};
    color: {WHITE};
    border-color: {INDIA_BLUE};
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }}
  .stat-card {{
    background: {CARD_BG};
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,.07);
    border-top: 3px solid {SAFFRON};
  }}
  .stat-card .value {{ font-size: 1.8rem; font-weight: 700; color: {INDIA_BLUE}; }}
  .stat-card .label {{ font-size: 0.8rem; color: #666; margin-top: 4px; }}
  .no-data {{ color: #999; font-style: italic; padding: 16px 0; }}
  footer {{
    text-align: center;
    padding: 20px;
    color: #888;
    font-size: 0.8rem;
  }}
</style>
"""


# ── Main render function ───────────────────────────────────────────────────────

def render_report(
    logs_dir: str | pathlib.Path = "logs",
    output: str | pathlib.Path = "india_report.html",
    open_browser: bool = True,
) -> pathlib.Path:
    """
    Parse all .eval logs in logs_dir and render a self-contained HTML report.

    Parameters
    ----------
    logs_dir : str | Path
        Directory containing .eval files, OR a list-like of file paths.
    output : str | Path
        Output HTML file path.
    open_browser : bool
        If True, open the report in the default browser after rendering.

    Returns
    -------
    pathlib.Path — path to the written HTML file.
    """
    from india_evals.view_plugin.parser import find_logs, parse_logs

    # ── Find logs ──────────────────────────────────────────────────────────────
    logs_dir = pathlib.Path(logs_dir)
    log_files = find_logs(logs_dir)

    if not log_files:
        print(f"  ⚠  No .eval files found in: {logs_dir}")
        log_files = []

    print(f"\n  India Evals View Plugin")
    print(f"  Found {len(log_files)} log file(s) in {logs_dir}\n")

    df = parse_logs(log_files) if log_files else pd.DataFrame()

    # ── Stats ──────────────────────────────────────────────────────────────────
    n_logs    = len(log_files)
    n_samples = len(df) if not df.empty else 0
    n_models  = df["model"].nunique() if not df.empty else 0
    n_tasks   = df["task"].nunique()  if not df.empty else 0
    n_langs   = df["language"].nunique() if not df.empty else 0

    stats_html = f"""
<div class="stats-grid">
  <div class="stat-card"><div class="value">{n_logs}</div><div class="label">Eval Logs</div></div>
  <div class="stat-card"><div class="value">{n_samples}</div><div class="label">Samples</div></div>
  <div class="stat-card"><div class="value">{n_models}</div><div class="label">Models</div></div>
  <div class="stat-card"><div class="value">{n_tasks}</div><div class="label">Tasks</div></div>
  <div class="stat-card"><div class="value">{n_langs}</div><div class="label">Languages</div></div>
</div>
"""

    # ── Build charts ───────────────────────────────────────────────────────────
    # Fetch plotly CDN once
    plotly_cdn = (
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        if HAS_PLOTLY else ""
    )

    # Summary table
    summary_div = _fig_to_div(_summary_table_fig(df)) if not df.empty else "<p class='no-data'>No data.</p>"

    # ── Safety / Jailbreak heatmaps ────────────────────────────────────────────
    def _safety_panels(task_name: str, section_title: str) -> str:
        sub = df[df["task"].str.contains(task_name, case=False, na=False)] if not df.empty else pd.DataFrame()
        tabs = []
        if not sub.empty and sub["correct"].notna().any():
            # Language × Category heatmap
            fig1 = _heatmap_fig(
                sub, x_col="language", y_col="category", value_col="correct",
                title=f"{section_title}: Refusal Rate — Language × Category",
                colorscale=REFUSAL_COLORSCALE,
            )
            tabs.append(("Language × Category", _fig_to_div(fig1)))

            # By language bar
            fig2 = _bar_fig(sub, x_col="language", value_col="correct",
                            title=f"{section_title}: Refusal Rate by Language")
            tabs.append(("By Language", _fig_to_div(fig2)))

            # By category bar
            fig3 = _bar_fig(sub, x_col="category", value_col="correct",
                            title=f"{section_title}: Refusal Rate by Category")
            tabs.append(("By Category", _fig_to_div(fig3)))

            if sub["attack_type"].notna().any():
                fig4 = _heatmap_fig(
                    sub, x_col="language", y_col="attack_type", value_col="correct",
                    title=f"{section_title}: Refusal Rate — Language × Attack Type",
                    colorscale=REFUSAL_COLORSCALE,
                )
                tabs.append(("Language × Attack Type", _fig_to_div(fig4)))
        else:
            tabs.append(("No Data", "<p class='no-data'>Run the task first, then regenerate the report.</p>"))
        return _tab_html(tabs)

    safety_html   = _safety_panels("multilingual_safety", "Safety")
    jailbreak_html = _safety_panels("jailbreak_safety",   "Jailbreak")

    # ── Multilingual MMLU heatmaps ─────────────────────────────────────────────
    def _multilingual_panels() -> str:
        sub = df[df["task"].str.contains("multilingual", case=False, na=False)] if not df.empty else pd.DataFrame()
        tabs = []
        if not sub.empty and sub["correct"].notna().any():
            cat_col = "subject" if sub["subject"].notna().any() else "category"
            fig1 = _heatmap_fig(
                sub, x_col="language", y_col=cat_col, value_col="correct",
                title="Multilingual MMLU: Accuracy — Language × Subject",
            )
            tabs.append(("Language × Subject", _fig_to_div(fig1)))
            fig2 = _bar_fig(sub, x_col="language", value_col="correct",
                            title="Multilingual MMLU: Accuracy by Language")
            tabs.append(("By Language", _fig_to_div(fig2)))
        else:
            tabs.append(("No Data", "<p class='no-data'>Run the task first, then regenerate the report.</p>"))
        return _tab_html(tabs)

    multilingual_html = _multilingual_panels()

    # ── BharatBBQ bias heatmap ─────────────────────────────────────────────────
    def _bias_panels() -> str:
        sub = df[df["task"].str.contains("bharatbbq|bias", case=False, na=False)] if not df.empty else pd.DataFrame()
        tabs = []
        if not sub.empty and sub["correct"].notna().any():
            fig1 = _bar_fig(sub, x_col="category", value_col="correct",
                            title="BharatBBQ: Accuracy by Social Category")
            tabs.append(("By Category", _fig_to_div(fig1)))
        else:
            tabs.append(("No Data", "<p class='no-data'>Run the task first, then regenerate the report.</p>"))
        return _tab_html(tabs)

    bias_html = _bias_panels()

    # ── DPI Safety heatmap ─────────────────────────────────────────────────────
    def _dpi_panels() -> str:
        sub = df[df["task"].str.contains("dpi", case=False, na=False)] if not df.empty else pd.DataFrame()
        tabs = []
        if not sub.empty and sub["correct"].notna().any():
            fig1 = _heatmap_fig(
                sub, x_col="risk_level", y_col="category", value_col="correct",
                title="DPI Safety: Accuracy — Category × Risk Level",
                colorscale=REFUSAL_COLORSCALE,
            )
            tabs.append(("Category × Risk Level", _fig_to_div(fig1)))
            fig2 = _bar_fig(sub, x_col="language", value_col="correct",
                            title="DPI Safety: Accuracy by Language")
            tabs.append(("By Language", _fig_to_div(fig2)))
        else:
            tabs.append(("No Data", "<p class='no-data'>Run the task first, then regenerate the report.</p>"))
        return _tab_html(tabs)

    dpi_html = _dpi_panels()

    # ── Assemble full HTML ──────────────────────────────────────────────────────
    from datetime import datetime
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_list = "".join(f"<li>{p.name}</li>" for p in log_files) or "<li><em>none</em></li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>India Evals — Inspection Report</title>
  {plotly_cdn}
  {_CSS}
</head>
<body>

<header>
  <span class="flag">🇮🇳</span>
  <div>
    <h1>India Evals — Inspection Report</h1>
    <p>Per-language &times; per-category accuracy &amp; safety heatmaps &nbsp;|&nbsp; Generated {generated_at}</p>
  </div>
</header>
<div class="stripe"></div>

<div class="container">

  <!-- Stats strip -->
  {stats_html}

  <!-- Summary table -->
  <div class="card">
    <div class="section-title">📊 Summary Table</div>
    {summary_div}
  </div>

  <!-- Multilingual MMLU -->
  <div class="card">
    <div class="section-title">🌐 Multilingual MMLU Accuracy</div>
    {multilingual_html}
  </div>

  <!-- Safety -->
  <div class="card">
    <div class="section-title">🛡️ Multilingual Safety — Refusal Rates</div>
    {safety_html}
  </div>

  <!-- Jailbreak -->
  <div class="card">
    <div class="section-title">⚔️ Jailbreak Safety — Refusal Rates</div>
    {jailbreak_html}
  </div>

  <!-- Bias -->
  <div class="card">
    <div class="section-title">⚖️ BharatBBQ Bias — Social Category Accuracy</div>
    {bias_html}
  </div>

  <!-- DPI Safety -->
  <div class="card">
    <div class="section-title">🔒 DPI Safety — Aadhaar / UPI / Bhashini</div>
    {dpi_html}
  </div>

  <!-- Log file list -->
  <div class="card">
    <div class="section-title">📁 Eval Logs Included</div>
    <ul style="padding-left:20px;line-height:1.9;font-size:0.88rem;color:#555">{log_list}</ul>
  </div>

</div>

<footer>
  inspect-india-evals &nbsp;·&nbsp; UK AISI Submission &nbsp;·&nbsp; {generated_at}
</footer>

</body>
</html>
"""

    # ── Write output ────────────────────────────────────────────────────────────
    output = pathlib.Path(output)
    output.write_text(html, encoding="utf-8")
    print(f"  *  Report written to: {output.resolve()}")

    if open_browser:
        webbrowser.open(output.resolve().as_uri())

    return output
