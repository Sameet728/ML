"""
visualization/reports.py
=========================
Generates a professional HTML backtest report with:
  - All embedded Plotly charts
  - Performance metrics table
  - Strategy comparison table
  - Monthly returns table
  - Summary narrative
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

from config.settings import get_settings
from utils.logger import log


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Quant Research — Backtest Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg:      #0D1117;
    --panel:   #161B22;
    --border:  #30363D;
    --text:    #E6EDF3;
    --muted:   #8B949E;
    --cyan:    #00D4FF;
    --green:   #00CC6A;
    --red:     #FF4757;
    --amber:   #FFA502;
    --radius:  8px;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'Inter', system-ui, sans-serif; font-size: 14px;
    line-height: 1.6;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  header {{
    border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 32px;
  }}
  header h1 {{ font-size: 28px; font-weight: 700; color: var(--cyan); margin-bottom: 4px; }}
  header p  {{ color: var(--muted); font-size: 13px; }}
  .badge {{
    display: inline-block; padding: 2px 10px; border-radius: 100px;
    font-size: 11px; font-weight: 600;
    background: rgba(0, 212, 255, 0.12); color: var(--cyan); border: 1px solid var(--cyan);
    margin-left: 10px;
  }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px; margin: 32px 0;
  }}
  .kpi-card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 18px; text-align: center;
    transition: border-color 0.2s;
  }}
  .kpi-card:hover {{ border-color: var(--cyan); }}
  .kpi-value {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
  .kpi-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--red); }}
  .neutral  {{ color: var(--amber); }}
  .section {{ margin: 40px 0; }}
  .section-title {{
    font-size: 16px; font-weight: 600; color: var(--cyan);
    border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 20px;
  }}
  .chart-card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px; margin-bottom: 20px;
    overflow: hidden;
  }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
    background: var(--panel); border-radius: var(--radius); overflow: hidden;
  }}
  th {{
    background: rgba(0, 212, 255, 0.08); color: var(--cyan);
    padding: 10px 14px; text-align: left; font-weight: 600;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 9px 14px; border-bottom: 1px solid rgba(48, 54, 61, 0.5);
    color: var(--text);
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .tag-good   {{ color: var(--green); font-weight: 600; }}
  .tag-warn   {{ color: var(--amber); font-weight: 600; }}
  .tag-bad    {{ color: var(--red);   font-weight: 600; }}
  .narrative {{
    background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--cyan);
    border-radius: var(--radius); padding: 20px; margin: 20px 0;
    line-height: 1.8;
  }}
  .narrative p {{ margin-bottom: 12px; }}
  .narrative p:last-child {{ margin-bottom: 0; }}
  footer {{
    border-top: 1px solid var(--border); margin-top: 60px; padding-top: 20px;
    text-align: center; color: var(--muted); font-size: 12px;
  }}
  @media (max-width: 768px) {{
    .two-col {{ grid-template-columns: 1fr; }}
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>⚡ AI Quant Research Platform <span class="badge">BTCUSDT</span></h1>
    <p>Walk-Forward Backtest Report &nbsp;·&nbsp; Generated: {generated_at} &nbsp;·&nbsp; Period: {period}</p>
  </header>

  <!-- KPI Cards -->
  <div class="section">
    <div class="section-title">📊 Key Performance Indicators</div>
    <div class="kpi-grid">{kpi_cards}</div>
  </div>

  <!-- Equity Curve -->
  <div class="section">
    <div class="section-title">📈 Equity Curve</div>
    <div class="chart-card">{equity_chart}</div>
  </div>

  <!-- Drawdown + Rolling Sharpe -->
  <div class="section">
    <div class="section-title">📉 Risk Analysis</div>
    <div class="two-col">
      <div class="chart-card">{drawdown_chart}</div>
      <div class="chart-card">{rolling_sharpe_chart}</div>
    </div>
  </div>

  <!-- Monthly Heatmap -->
  <div class="section">
    <div class="section-title">📅 Monthly Returns</div>
    <div class="chart-card">{monthly_heatmap}</div>
    <div style="margin-top:16px">{monthly_table}</div>
  </div>

  <!-- Yearly Returns -->
  <div class="section">
    <div class="section-title">📊 Yearly Returns</div>
    <div class="chart-card">{yearly_chart}</div>
  </div>

  <!-- Model Analysis -->
  <div class="section">
    <div class="section-title">🤖 Model Analysis</div>
    <div class="two-col">
      <div class="chart-card">{feature_importance_chart}</div>
      <div class="chart-card">{probability_dist_chart}</div>
    </div>
    <div class="two-col" style="margin-top:20px">
      <div class="chart-card">{confusion_matrix_chart}</div>
      <div class="chart-card">{trade_dist_chart}</div>
    </div>
  </div>

  <!-- Regime Timeline -->
  <div class="section">
    <div class="section-title">🌊 Market Regime Analysis</div>
    <div class="chart-card">{regime_chart}</div>
  </div>

  <!-- Strategy Comparison -->
  <div class="section">
    <div class="section-title">⚔️ Benchmark Comparison</div>
    {comparison_table}
  </div>

  <!-- Narrative -->
  <div class="section">
    <div class="section-title">📝 Strategy Summary</div>
    <div class="narrative">{narrative}</div>
  </div>

  <footer>
    <p>AI Quant Research Platform · For Research Purposes Only · Not Financial Advice</p>
  </footer>
</div>
</body>
</html>"""


def _fig_to_html(fig: go.Figure, height: int = None) -> str:
    """Convert Plotly figure to embedded HTML div."""
    if fig is None:
        return "<p style='color:#8B949E'>Chart not available</p>"
    try:
        if height:
            fig.update_layout(height=height)
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": True, "displaylogo": False},
        )
    except Exception as e:
        log.warning(f"Chart render error: {e}")
        return f"<p style='color:#FF4757'>Chart error: {e}</p>"


def _kpi_card(value: str, label: str, color_class: str = "") -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-value {color_class}">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>"""


def _build_kpi_cards(metrics: Dict) -> str:
    cards = []
    def fmt_pct(v): return f"{v*100:+.1f}%" if not np.isnan(v) else "N/A"
    def fmt_f(v, d=3): return f"{v:.{d}f}" if not np.isnan(v) else "N/A"

    cagr_class = "positive" if metrics.get("cagr", 0) > 0.10 else "neutral" if metrics.get("cagr", 0) > 0 else "negative"
    cards.append(_kpi_card(fmt_pct(metrics.get("cagr", np.nan)), "CAGR", cagr_class))

    sh = metrics.get("sharpe", np.nan)
    sh_class = "positive" if sh > 1 else "neutral" if sh > 0 else "negative"
    cards.append(_kpi_card(fmt_f(sh), "Sharpe Ratio", sh_class))

    cards.append(_kpi_card(fmt_f(metrics.get("sortino", np.nan)), "Sortino Ratio"))

    dd = metrics.get("max_drawdown", np.nan)
    dd_class = "positive" if abs(dd) < 0.1 else "neutral" if abs(dd) < 0.2 else "negative"
    cards.append(_kpi_card(fmt_pct(dd), "Max Drawdown", dd_class))

    cards.append(_kpi_card(fmt_pct(metrics.get("avg_monthly_return", np.nan)), "Avg Monthly", "neutral"))
    cards.append(_kpi_card(fmt_pct(metrics.get("total_return", np.nan)), "Total Return", "positive"))
    cards.append(_kpi_card(fmt_f(metrics.get("calmar", np.nan)), "Calmar Ratio"))

    wr = metrics.get("win_rate", np.nan)
    wr_class = "positive" if wr > 0.5 else "negative"
    cards.append(_kpi_card(fmt_pct(wr) if not np.isnan(wr) else "N/A", "Win Rate", wr_class))

    cards.append(_kpi_card(fmt_f(metrics.get("profit_factor", np.nan)), "Profit Factor"))
    cards.append(_kpi_card(str(metrics.get("total_trades", 0)), "Total Trades"))
    cards.append(_kpi_card(fmt_pct(metrics.get("best_month", np.nan)), "Best Month", "positive"))
    cards.append(_kpi_card(fmt_pct(metrics.get("worst_month", np.nan)), "Worst Month", "negative"))

    return "\n".join(cards)


def _monthly_table_html(monthly_pivot: pd.DataFrame) -> str:
    """Generate monthly returns HTML table."""
    tbl = "<table><thead><tr><th>Year</th>"
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Annual"]
    for m in months:
        if m in monthly_pivot.columns:
            tbl += f"<th>{m}</th>"
    tbl += "</tr></thead><tbody>"

    for year, row in monthly_pivot.iterrows():
        tbl += f"<tr><td><strong>{year}</strong></td>"
        for m in months:
            if m not in monthly_pivot.columns:
                continue
            val = row.get(m, np.nan)
            if pd.isna(val):
                tbl += "<td>—</td>"
            else:
                pct = val * 100
                css = "tag-good" if pct > 0 else "tag-bad" if pct < 0 else ""
                tbl += f'<td class="{css}">{pct:+.1f}%</td>'
        tbl += "</tr>"
    tbl += "</tbody></table>"
    return tbl


def _comparison_table_html(comparison_df: pd.DataFrame) -> str:
    """Generate strategy comparison HTML table."""
    if comparison_df is None or comparison_df.empty:
        return "<p>No comparison data</p>"
    tbl = "<table><thead><tr><th>Strategy</th>"
    for col in comparison_df.columns:
        tbl += f"<th>{col.replace('_', ' ').title()}</th>"
    tbl += "</tr></thead><tbody>"
    for idx, row in comparison_df.iterrows():
        tbl += f"<tr><td><strong>{idx}</strong></td>"
        for val in row.values:
            tbl += f"<td>{val}</td>"
        tbl += "</tr>"
    tbl += "</tbody></table>"
    return tbl


def _build_narrative(metrics: Dict, cfg=None) -> str:
    cfg = cfg or get_settings()
    cagr   = metrics.get("cagr", 0)
    sharpe = metrics.get("sharpe", 0)
    maxdd  = metrics.get("max_drawdown", 0)
    avg_mo = metrics.get("avg_monthly_return", 0)

    meets_cagr   = 0.12 <= cagr <= 0.25
    meets_dd     = abs(maxdd) <= 0.15
    meets_sharpe = sharpe >= 1.0
    meets_monthly= 0.008 <= avg_mo <= 0.02

    summary_lines = []
    summary_lines.append(
        f"<p>This walk-forward backtest covers <strong>{metrics.get('years_tested', 0):.1f} years</strong> "
        f"of BTC/USDT hourly data, executing <strong>{metrics.get('total_trades', 0):,} trades</strong> "
        f"using an XGBoost-based ensemble with meta-labeling and regime-aware signal filtering.</p>"
    )

    # CAGR assessment
    if meets_cagr:
        summary_lines.append(
            f"<p>✅ <strong>CAGR ({cagr:.1%})</strong> is within the 12–25% target range.</p>"
        )
    elif cagr > 0.25:
        summary_lines.append(
            f"<p>⚠️ <strong>CAGR ({cagr:.1%})</strong> exceeds the 25% target — review for overfitting.</p>"
        )
    else:
        summary_lines.append(
            f"<p>🔴 <strong>CAGR ({cagr:.1%})</strong> is below the 12% target — consider parameter tuning.</p>"
        )

    # Drawdown assessment
    if meets_dd:
        summary_lines.append(
            f"<p>✅ <strong>Max Drawdown ({abs(maxdd):.1%})</strong> is within the 15% target.</p>"
        )
    else:
        summary_lines.append(
            f"<p>⚠️ <strong>Max Drawdown ({abs(maxdd):.1%})</strong> exceeds the 15% target — consider tightening risk controls.</p>"
        )

    # Sharpe assessment
    if meets_sharpe:
        summary_lines.append(f"<p>✅ <strong>Sharpe Ratio ({sharpe:.3f})</strong> is above 1.0 — good risk-adjusted return.</p>")
    else:
        summary_lines.append(f"<p>⚠️ <strong>Sharpe Ratio ({sharpe:.3f})</strong> is below 1.0 — returns may not justify risk.</p>")

    # Overall verdict
    n_met = sum([meets_cagr, meets_dd, meets_sharpe, meets_monthly])
    if n_met >= 3:
        summary_lines.append(
            "<p><strong>Overall verdict: 🟢 Strategy meets primary targets.</strong> "
            "Continue monitoring for regime changes and feature drift.</p>"
        )
    elif n_met >= 2:
        summary_lines.append(
            "<p><strong>Overall verdict: 🟡 Partial target achievement.</strong> "
            "Review hyperparameters and consider additional risk filters.</p>"
        )
    else:
        summary_lines.append(
            "<p><strong>Overall verdict: 🔴 Targets not met.</strong> "
            "Significant tuning or strategy redesign recommended.</p>"
        )

    summary_lines.append(
        "<p style='color:#8B949E;font-size:12px'>"
        "⚠️ Past backtest performance does not guarantee future results. "
        "This report is for research purposes only and not financial advice.</p>"
    )

    return "\n".join(summary_lines)


def generate_html_report(
    metrics: Dict,
    charts: Dict,
    monthly_pivot: pd.DataFrame,
    yearly_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    cfg=None,
) -> Path:
    """Generate and save the full HTML backtest report."""
    cfg = cfg or get_settings()

    log.info("Generating HTML report …")

    period = f"{monthly_pivot.index[0]} – {monthly_pivot.index[-1]}"

    html = HTML_TEMPLATE.format(
        generated_at    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        period          = period,
        kpi_cards       = _build_kpi_cards(metrics),
        equity_chart    = _fig_to_html(charts.get("equity_curve"),    height=480),
        drawdown_chart  = _fig_to_html(charts.get("drawdown"),        height=300),
        rolling_sharpe_chart = _fig_to_html(charts.get("rolling_sharpe"), height=300),
        monthly_heatmap = _fig_to_html(charts.get("monthly_heatmap"), height=400),
        monthly_table   = _monthly_table_html(monthly_pivot),
        yearly_chart    = _fig_to_html(charts.get("yearly_returns"),  height=380),
        feature_importance_chart = _fig_to_html(charts.get("feature_importance"), height=420),
        probability_dist_chart   = _fig_to_html(charts.get("probability_dist"),   height=380),
        confusion_matrix_chart   = _fig_to_html(charts.get("confusion_matrix"),   height=380),
        trade_dist_chart         = _fig_to_html(charts.get("trade_distribution"),  height=380),
        regime_chart    = _fig_to_html(charts.get("regime_timeline"), height=500),
        comparison_table= _comparison_table_html(comparison_df),
        narrative       = _build_narrative(metrics, cfg),
    )

    report_path = cfg.paths["reports"] / "backtest_report.html"
    report_path.write_text(html, encoding="utf-8")
    log.info(f"HTML report saved → {report_path}")

    return report_path
