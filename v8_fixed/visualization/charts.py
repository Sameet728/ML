"""
visualization/charts.py
========================
All 10 professional Plotly charts. No Seaborn — Plotly only.

Charts generated:
  1.  Equity Curve (vs benchmarks)
  2.  Drawdown Graph
  3.  Monthly Returns Heatmap
  4.  Yearly Returns Bar Chart
  5.  Rolling Sharpe Ratio
  6.  Trade P&L Distribution
  7.  Feature Importance (XGBoost + SHAP)
  8.  Regime Timeline
  9.  Confusion Matrix
  10. Probability Score Distribution
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.settings import get_settings
from utils.logger import log

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "strategy":   "#00D4FF",   # Bright cyan
    "benchmark1": "#FF6B35",   # Orange (BTC hold)
    "benchmark2": "#A8E6CF",   # Light green (EMA)
    "benchmark3": "#888",      # Gray (random)
    "profit":     "#00CC6A",   # Green
    "loss":       "#FF4757",   # Red
    "neutral":    "#FFA502",   # Amber
    "regime0":    "#FF4757",   # Bearish red
    "regime1":    "#FFA502",   # Ranging amber
    "regime2":    "#00CC6A",   # Bullish green
    "regime3":    "#FF6B35",   # High vol bull
    "regime4":    "#C0392B",   # High vol bear
    "bg":         "#0D1117",   # Dark background
    "panel":      "#161B22",   # Panel bg
    "border":     "#30363D",   # Border
    "text":       "#E6EDF3",   # Text
    "grid":       "#21262D",   # Gridline
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["panel"],
    font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif", size=12),
    legend=dict(bgcolor=COLORS["panel"], bordercolor=COLORS["border"], borderwidth=1),
    margin=dict(l=60, r=40, t=60, b=50),
    hoverlabel=dict(bgcolor=COLORS["panel"], font_color=COLORS["text"]),
)

AXIS_DEFAULTS = dict(
    gridcolor=COLORS["grid"],
    linecolor=COLORS["border"],
    tickcolor=COLORS["text"],
    zerolinecolor=COLORS["border"],
)


def _apply_layout(fig: go.Figure, title: str, height: int = 500) -> go.Figure:
    fig.update_layout(**LAYOUT_DEFAULTS, title=dict(text=title, font=dict(size=18)), height=height)
    fig.update_xaxes(**AXIS_DEFAULTS)
    fig.update_yaxes(**AXIS_DEFAULTS)
    return fig


# ── Chart 1: Equity Curve ─────────────────────────────────────────────────────

def chart_equity_curve(
    strategy_equity: pd.Series,
    benchmarks: Dict[str, pd.Series] = None,
    label: str = "AI Strategy",
) -> go.Figure:
    """Equity curve vs benchmarks (all normalized to 100)."""
    fig = go.Figure()

    # Normalize to 100
    def norm(s): return s / s.iloc[0] * 100

    # Strategy
    eq_norm = norm(strategy_equity)
    fig.add_trace(go.Scatter(
        x=eq_norm.index, y=eq_norm.values,
        name=label, line=dict(color=COLORS["strategy"], width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Value: %{y:.1f}<extra></extra>",
    ))

    # Benchmarks
    bench_colors = [COLORS["benchmark1"], COLORS["benchmark2"], COLORS["benchmark3"]]
    if benchmarks:
        for i, (bname, beq) in enumerate(benchmarks.items()):
            try:
                beq_aligned = beq.reindex(eq_norm.index, method="ffill").dropna()
                if len(beq_aligned) > 0:
                    fig.add_trace(go.Scatter(
                        x=beq_aligned.index, y=norm(beq_aligned).values,
                        name=bname, line=dict(
                            color=bench_colors[i % len(bench_colors)],
                            width=1.5, dash="dot"
                        ),
                    ))
            except Exception:
                pass

    # 100 baseline
    fig.add_hline(y=100, line_dash="dash", line_color=COLORS["border"], opacity=0.6)

    return _apply_layout(fig, "📈 Equity Curve (Normalized to 100)", height=500)


# ── Chart 2: Drawdown ─────────────────────────────────────────────────────────

def chart_drawdown(equity: pd.Series, label: str = "Strategy") -> go.Figure:
    """Drawdown chart with filled area."""
    from backtesting.metrics import compute_drawdown_series
    dd = compute_drawdown_series(equity) * 100  # Convert to %

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown (%)",
        fill="tozeroy",
        fillcolor="rgba(255, 71, 87, 0.3)",
        line=dict(color=COLORS["loss"], width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
    ))

    # Max DD line
    max_dd = dd.min()
    fig.add_hline(
        y=max_dd, line_dash="dash", line_color=COLORS["neutral"],
        annotation_text=f"Max DD: {max_dd:.2f}%",
        annotation_position="top right",
    )

    return _apply_layout(fig, "📉 Drawdown Chart", height=350)


# ── Chart 3: Monthly Returns Heatmap ─────────────────────────────────────────

def chart_monthly_heatmap(monthly_pivot: pd.DataFrame) -> go.Figure:
    """Monthly returns heatmap: green=positive, red=negative."""
    # Remove "Annual" column for heatmap
    plot_data = monthly_pivot.drop(columns=["Annual"], errors="ignore")
    z_vals    = plot_data.values * 100  # Convert to %

    # Color scale: red→white→green
    colorscale = [
        [0.0,  "#C0392B"],
        [0.35, "#E74C3C"],
        [0.5,  "#2D3748"],
        [0.65, "#27AE60"],
        [1.0,  "#00CC6A"],
    ]

    text_vals = np.where(
        np.isnan(z_vals), "",
        np.vectorize(lambda v: f"{v:+.1f}%")(z_vals)
    )

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=plot_data.columns.tolist(),
        y=[str(y) for y in plot_data.index],
        text=text_vals,
        texttemplate="%{text}",
        colorscale=colorscale,
        zmid=0,
        colorbar=dict(title="Return %", ticksuffix="%"),
        hovertemplate="Year: %{y} | Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    ))

    return _apply_layout(fig, "📅 Monthly Returns Heatmap", height=max(350, len(plot_data) * 35 + 100))


# ── Chart 4: Yearly Returns ───────────────────────────────────────────────────

def chart_yearly_returns(yearly_df: pd.DataFrame) -> go.Figure:
    """Yearly returns bar chart."""
    returns_pct = yearly_df["Return"].values * 100
    years       = [str(y) for y in yearly_df.index]
    colors      = [COLORS["profit"] if r >= 0 else COLORS["loss"] for r in returns_pct]

    fig = go.Figure(go.Bar(
        x=years, y=returns_pct,
        marker_color=colors,
        text=[f"{r:+.1f}%" for r in returns_pct],
        textposition="outside",
        hovertemplate="Year: %{x}<br>Return: %{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=COLORS["border"])
    fig.add_hline(y=12, line_dash="dash", line_color=COLORS["strategy"],
                  annotation_text="12% target", annotation_position="top right", opacity=0.6)

    return _apply_layout(fig, "📊 Yearly Returns", height=400)


# ── Chart 5: Rolling Sharpe ───────────────────────────────────────────────────

def chart_rolling_sharpe(returns: pd.Series, window_days: int = 90) -> go.Figure:
    """90-day rolling Sharpe ratio."""
    from backtesting.metrics import compute_rolling_sharpe
    rolling = compute_rolling_sharpe(returns, window_bars=window_days * 24).dropna()

    color_vals = [
        COLORS["profit"] if v >= 1 else
        COLORS["neutral"] if v >= 0 else
        COLORS["loss"]
        for v in rolling.values
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling.values,
        name="Rolling Sharpe (90d)",
        line=dict(color=COLORS["strategy"], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.3f}<extra></extra>",
    ))

    # Reference lines
    for level, color, label in [
        (1.0, COLORS["profit"],  "Good (1.0)"),
        (0.0, COLORS["neutral"], "Break-even"),
    ]:
        fig.add_hline(y=level, line_dash="dash", line_color=color,
                      annotation_text=label, opacity=0.7)

    return _apply_layout(fig, "📈 Rolling Sharpe Ratio (90-day)", height=380)


# ── Chart 6: Trade Distribution ───────────────────────────────────────────────

def chart_trade_distribution(trades_df: pd.DataFrame) -> go.Figure:
    """P&L distribution histogram."""
    if trades_df is None or len(trades_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No trade data", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_layout(fig, "Trade P&L Distribution")

    pnl = trades_df["PnL"].values if "PnL" in trades_df.columns else trades_df.iloc[:, 0].values

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl,
        nbinsx=50,
        name="Trade P&L",
        marker=dict(
            color=[COLORS["profit"] if v >= 0 else COLORS["loss"] for v in pnl],
            line=dict(width=0.5, color=COLORS["panel"]),
        ),
        hovertemplate="P&L: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    # Mean line
    fig.add_vline(x=np.mean(pnl), line_dash="dash", line_color=COLORS["strategy"],
                  annotation_text=f"Mean: {np.mean(pnl):.2f}")

    return _apply_layout(fig, "🎯 Trade P&L Distribution", height=380)


# ── Chart 7: Feature Importance ───────────────────────────────────────────────

def chart_feature_importance(
    importance: pd.Series,
    shap_values: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
) -> go.Figure:
    """Feature importance bar chart (XGBoost gain)."""
    top = importance.head(top_n)

    fig = go.Figure(go.Bar(
        x=top.values[::-1],
        y=top.index[::-1],
        orientation="h",
        marker=dict(
            color=top.values[::-1],
            colorscale=[[0, COLORS["neutral"]], [1, COLORS["strategy"]]],
            showscale=False,
        ),
        hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
    ))

    return _apply_layout(fig, f"🔍 Feature Importance (Top {top_n})", height=max(400, top_n * 22 + 80))


# ── Chart 8: Regime Timeline ──────────────────────────────────────────────────

def chart_regime_timeline(
    regime_df: pd.DataFrame,
    equity: pd.Series = None,
) -> go.Figure:
    """Regime classification timeline with equity overlay."""
    from models.regime import REGIME_LABELS

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # Equity curve (top)
    if equity is not None:
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            name="Equity", line=dict(color=COLORS["strategy"], width=2),
        ), row=1, col=1)

    # Regime coloring as scatter with color
    if "regime" in regime_df.columns:
        regime_colors = {
            0: COLORS["regime0"], 1: COLORS["regime1"],
            2: COLORS["regime2"], 3: COLORS["regime3"], 4: COLORS["regime4"],
        }
        for reg_id, reg_label in REGIME_LABELS.items():
            mask = regime_df["regime"] == reg_id
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=regime_df[mask].index,
                y=[reg_id] * mask.sum(),
                name=reg_label,
                mode="markers",
                marker=dict(color=regime_colors[reg_id], size=4),
                hovertemplate=f"{reg_label}<br>%{{x|%Y-%m-%d}}<extra></extra>",
            ), row=2, col=1)

    fig.update_layout(**LAYOUT_DEFAULTS, title=dict(text="🌊 Market Regime Timeline", font=dict(size=18)), height=550)
    fig.update_xaxes(**AXIS_DEFAULTS)
    fig.update_yaxes(**AXIS_DEFAULTS)
    return fig


# ── Chart 9: Confusion Matrix ─────────────────────────────────────────────────

def chart_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> go.Figure:
    """Model confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    labels = ["No Trade (0)", "Trade (1)"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels, y=labels,
        text=cm, texttemplate="%{text}",
        colorscale=[[0, COLORS["panel"]], [1, COLORS["strategy"]]],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))

    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual")
    return _apply_layout(fig, "📊 Confusion Matrix", height=380)


# ── Chart 10: Probability Distribution ───────────────────────────────────────

def chart_probability_distribution(
    probs: pd.Series,
    labels: pd.Series = None,
    threshold: float = 0.52,
) -> go.Figure:
    """Model probability score distribution (positive vs negative outcomes)."""
    fig = go.Figure()

    if labels is not None and len(labels) > 0:
        # Align to common index to avoid boolean Series indexer error
        common  = probs.index.intersection(labels.index)
        p_align = probs.loc[common]
        l_align = labels.loc[common].astype(int)

        pos_probs = p_align[l_align == 1].values
        neg_probs = p_align[l_align == 0].values

        if len(pos_probs) > 0:
            fig.add_trace(go.Histogram(
                x=pos_probs, name="Profitable (TP)", nbinsx=40,
                marker_color=COLORS["profit"], opacity=0.7,
            ))
        if len(neg_probs) > 0:
            fig.add_trace(go.Histogram(
                x=neg_probs, name="Unprofitable (SL/Timeout)", nbinsx=40,
                marker_color=COLORS["loss"], opacity=0.7,
            ))
    else:
        fig.add_trace(go.Histogram(x=probs.values, name="Probability", nbinsx=50,
                                    marker_color=COLORS["strategy"], opacity=0.8))

    fig.add_vline(x=threshold, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text=f"Threshold: {threshold}")

    fig.update_layout(barmode="overlay")
    return _apply_layout(fig, "🎲 Probability Score Distribution", height=380)


# ── Save all charts ───────────────────────────────────────────────────────────

def save_chart(fig: go.Figure, filename: str, cfg=None) -> Path:
    """Save chart as interactive HTML."""
    cfg  = cfg or get_settings()
    path = cfg.paths["reports"] / filename
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=False)
    log.debug(f"Chart saved → {path}")
    return path


def generate_all_charts(
    strategy_result: Dict,
    benchmarks: Dict,
    monthly_pivot: pd.DataFrame,
    yearly_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    feature_importance: pd.Series,
    oos_probs: pd.Series,
    oos_labels: pd.Series,
    trades_df: pd.DataFrame = None,
    cfg=None,
) -> Dict[str, go.Figure]:
    """Generate all 10 charts and return as dict. Each chart is isolated."""
    cfg     = cfg or get_settings()
    equity  = strategy_result["equity"]
    returns = strategy_result["returns"]
    figs    = {}

    log.info("Generating all charts …")
    bench_equities = {k: v["equity"] for k, v in benchmarks.items()}

    chart_tasks = [
        ("equity_curve",       lambda: chart_equity_curve(equity, bench_equities)),
        ("drawdown",           lambda: chart_drawdown(equity)),
        ("monthly_heatmap",    lambda: chart_monthly_heatmap(monthly_pivot)),
        ("yearly_returns",     lambda: chart_yearly_returns(yearly_df)),
        ("rolling_sharpe",     lambda: chart_rolling_sharpe(returns)),
        ("trade_distribution", lambda: chart_trade_distribution(trades_df)),
        ("feature_importance", lambda: chart_feature_importance(feature_importance)),
        ("regime_timeline",    lambda: chart_regime_timeline(regime_df, equity)),
        ("probability_dist",   lambda: chart_probability_distribution(oos_probs, oos_labels)),
    ]

    # Confusion matrix
    try:
        threshold = cfg.meta_min_confidence
        y_pred  = (oos_probs >= threshold).astype(int)
        common  = oos_labels.index.intersection(y_pred.index)
        if len(common) > 0:
            chart_tasks.append((
                "confusion_matrix",
                lambda: chart_confusion_matrix(oos_labels.loc[common], y_pred.loc[common])
            ))
    except Exception as e:
        log.warning(f"Confusion matrix prep failed: {e}")

    # Generate each chart independently — one failure won't kill the report
    for name, fn in chart_tasks:
        try:
            fig = fn()
            figs[name] = fig
            save_chart(fig, f"{name}.html", cfg)
        except Exception as e:
            log.warning(f"Chart '{name}' failed (skipping): {e}")

    log.info(f"All {len(figs)} charts generated and saved to reports/")
    return figs
