"""
dashboard/app.py
=================
Professional 6-page Streamlit dashboard for the AI Quant Research Platform.

Pages:
  1. 🏠 Overview      — equity curve, KPI cards, quick summary
  2. 📊 Trade Analysis — trade table, P&L distribution, streaks
  3. 🤖 Model Insights — feature importance, SHAP, confusion matrix, probability dist
  4. 🌊 Regime Analysis— regime timeline, per-regime performance
  5. 📅 Returns        — monthly heatmap, yearly bars, monthly table
  6. ⚠️  Risk Monitor  — drawdown, rolling Sharpe, Monte Carlo, drift report

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards   # type: ignore

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Quant Research",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #30363D;
}

/* Main background */
.stApp { background: #0D1117; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="metric-container"]:hover { border-color: #00D4FF; transition: border-color 0.2s; }
[data-testid="stMetricValue"] { font-size: 20px !important; font-weight: 700 !important; }

/* Plotly charts */
.js-plotly-plot { border-radius: 8px; }

/* Tables */
.stDataFrame { border: 1px solid #30363D; border-radius: 8px; }

/* Section headers */
h2 { color: #00D4FF !important; border-bottom: 1px solid #30363D; padding-bottom: 8px; }
h3 { color: #E6EDF3 !important; }

/* Info boxes */
.info-box {
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #E6EDF3;
}
.warn-box {
    background: rgba(255, 165, 2, 0.08);
    border: 1px solid rgba(255, 165, 2, 0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading pipeline results …")
def load_results():
    """Load cached pipeline results from disk."""
    import joblib
    results_path = ROOT / "reports" / "pipeline_results.pkl"
    if not results_path.exists():
        return None
    return joblib.load(results_path)


@st.cache_data(ttl=3600)
def load_equity() -> pd.Series:
    p = ROOT / "reports" / "equity_curve.parquet"
    if p.exists():
        return pd.read_parquet(p).squeeze()
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def load_returns() -> pd.Series:
    p = ROOT / "reports" / "returns.parquet"
    if p.exists():
        return pd.read_parquet(p).squeeze()
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def load_regime() -> pd.DataFrame:
    p = ROOT / "reports" / "regime_df.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_oos_signals() -> pd.DataFrame:
    p = ROOT / "reports" / "oos_signals.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_feature_importance() -> pd.Series:
    p = ROOT / "reports" / "feature_importance.parquet"
    if p.exists():
        return pd.read_parquet(p).squeeze()
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def load_metrics() -> dict:
    import joblib
    p = ROOT / "reports" / "metrics.pkl"
    if p.exists():
        return joblib.load(p)
    return {}


@st.cache_data(ttl=3600)
def load_benchmarks() -> dict:
    import joblib
    p = ROOT / "reports" / "benchmarks.pkl"
    if p.exists():
        return joblib.load(p)
    return {}


# ── Chart imports (lazy) ──────────────────────────────────────────────────────

def get_charts():
    from visualization.charts import (
        chart_equity_curve, chart_drawdown, chart_monthly_heatmap,
        chart_yearly_returns, chart_rolling_sharpe, chart_trade_distribution,
        chart_feature_importance, chart_regime_timeline, chart_confusion_matrix,
        chart_probability_distribution, COLORS,
    )
    return {
        "equity_curve": chart_equity_curve,
        "drawdown": chart_drawdown,
        "monthly_heatmap": chart_monthly_heatmap,
        "yearly_returns": chart_yearly_returns,
        "rolling_sharpe": chart_rolling_sharpe,
        "trade_distribution": chart_trade_distribution,
        "feature_importance": chart_feature_importance,
        "regime_timeline": chart_regime_timeline,
        "confusion_matrix": chart_confusion_matrix,
        "probability_distribution": chart_probability_distribution,
        "COLORS": COLORS,
    }


# ── Sidebar navigation ────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚡ AI Quant Research")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            options=[
                "🏠 Overview",
                "📊 Trade Analysis",
                "🤖 Model Insights",
                "🌊 Regime Analysis",
                "📅 Returns",
                "⚠️ Risk Monitor",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Market:** BTCUSDT")
        st.markdown("**Timeframe:** 1H")
        st.markdown("**Strategy:** Ensemble + Meta-Label")

        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.markdown(
            "<p style='color:#8B949E;font-size:11px;margin-top:20px'>"
            "⚠️ For research only. Not financial advice.</p>",
            unsafe_allow_html=True,
        )

    return page


# ── Helper: metric row ────────────────────────────────────────────────────────

def metric_row(metrics: dict, keys: list):
    cols = st.columns(len(keys))
    for i, (key, label, fmt) in enumerate(keys):
        val = metrics.get(key, None)
        if val is not None:
            if fmt == "pct":
                display = f"{val*100:+.2f}%"
                delta_color = "normal" if val > 0 else "inverse"
            elif fmt == "pct_abs":
                display = f"{abs(val)*100:.2f}%"
                delta_color = "inverse"
            else:
                display = f"{val:.3f}" if fmt == "float" else f"{val:,.0f}" if fmt == "int" else str(val)
                delta_color = "normal"
            cols[i].metric(label=label, value=display)
        else:
            cols[i].metric(label=label, value="N/A")


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_overview(equity, returns, metrics, benchmarks, regime_df):
    st.markdown("## 🏠 Overview")

    if equity.empty:
        st.warning("⚠️ No backtest results found. Run `python run_pipeline.py` first.")
        st.markdown("""
        <div class="info-box">
        <strong>Getting Started:</strong><br>
        1. Install dependencies: <code>pip install -r requirements.txt</code><br>
        2. Run full pipeline: <code>python run_pipeline.py --mode=full</code><br>
        3. Refresh this dashboard
        </div>
        """, unsafe_allow_html=True)
        return

    # KPI row 1
    st.markdown("### Key Metrics")
    metric_row(metrics, [
        ("cagr",             "CAGR",             "pct"),
        ("sharpe",           "Sharpe Ratio",      "float"),
        ("sortino",          "Sortino Ratio",      "float"),
        ("max_drawdown",     "Max Drawdown",       "pct_abs"),
        ("avg_monthly_return","Avg Monthly",       "pct"),
        ("total_return",     "Total Return",       "pct"),
    ])

    # KPI row 2
    metric_row(metrics, [
        ("win_rate",         "Win Rate",          "pct"),
        ("profit_factor",    "Profit Factor",      "float"),
        ("calmar",           "Calmar Ratio",       "float"),
        ("total_trades",     "Total Trades",       "int"),
        ("best_month",       "Best Month",         "pct"),
        ("worst_month",      "Worst Month",        "pct"),
    ])

    style_metric_cards(
        background_color="#161B22",
        border_color="#30363D",
        border_left_color="#00D4FF",
    )

    # Equity curve
    st.markdown("### Equity Curve")
    try:
        charts = get_charts()
        bench_equities = {k: v["equity"] for k, v in benchmarks.items() if "equity" in v}
        fig = charts["equity_curve"](equity, bench_equities)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

    # Target achievement
    st.markdown("### 🎯 Target Achievement")
    c1, c2, c3 = st.columns(3)
    cagr = metrics.get("cagr", 0)
    dd   = abs(metrics.get("max_drawdown", 1))
    sh   = metrics.get("sharpe", 0)

    c1.markdown(f"""
    <div class="{'info-box' if 0.12<=cagr<=0.25 else 'warn-box'}">
    <strong>CAGR Target (12–20%)</strong><br>
    Current: {cagr:.1%}<br>
    Status: {'✅ Met' if 0.12<=cagr<=0.25 else '⚠️ Not Met'}
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="{'info-box' if dd<=0.15 else 'warn-box'}">
    <strong>Max DD Target (≤15%)</strong><br>
    Current: {dd:.1%}<br>
    Status: {'✅ Met' if dd<=0.15 else '⚠️ Not Met'}
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="{'info-box' if sh>=1.0 else 'warn-box'}">
    <strong>Sharpe Target (≥1.0)</strong><br>
    Current: {sh:.3f}<br>
    Status: {'✅ Met' if sh>=1.0 else '⚠️ Not Met'}
    </div>""", unsafe_allow_html=True)


def page_trade_analysis(equity, returns, metrics, oos_signals):
    st.markdown("## 📊 Trade Analysis")

    charts = get_charts()

    c1, c2 = st.columns(2)
    with c1:
        if not oos_signals.empty and "PnL" in oos_signals.columns:
            fig = charts["trade_distribution"](oos_signals)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trade P&L data not available")

    with c2:
        # Trade stats
        st.markdown("#### Trade Statistics")
        trade_stats = {
            "Total Trades":      metrics.get("total_trades", "N/A"),
            "Win Rate":          f"{metrics.get('win_rate', 0):.2%}",
            "Profit Factor":     f"{metrics.get('profit_factor', 0):.3f}",
            "Avg Trade ($ PnL)": f"{metrics.get('avg_trade', 0):.2f}",
            "Avg Win":           f"{metrics.get('avg_win', 0):.2f}",
            "Avg Loss":          f"{metrics.get('avg_loss', 0):.2f}",
            "Max Win":           f"{metrics.get('max_win', 0):.2f}",
            "Max Loss":          f"{metrics.get('max_loss', 0):.2f}",
            "Expectancy":        f"{metrics.get('expectancy', 0):.4f}",
            "Longest Loss Streak":f"{metrics.get('longest_loss_streak', 0)} days",
        }
        st.dataframe(
            pd.DataFrame.from_dict(trade_stats, orient="index", columns=["Value"]),
            use_container_width=True,
        )

    # Trade log table
    st.markdown("#### Signal Quality Distribution")
    if not oos_signals.empty and "quality_score" in oos_signals.columns:
        trade_signals = oos_signals[oos_signals["signal"] == 1]
        if not trade_signals.empty:
            fig = go.Figure(go.Histogram(
                x=trade_signals["quality_score"],
                nbinsx=30,
                marker_color="#00D4FF",
                name="Quality Score",
            ))
            fig.update_layout(
                paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                font=dict(color="#E6EDF3"),
                title="Trade Quality Score Distribution",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)


def page_model_insights(feature_importance, oos_signals):
    st.markdown("## 🤖 Model Insights")
    charts = get_charts()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Feature Importance")
        if not feature_importance.empty:
            fig = charts["feature_importance"](feature_importance, top_n=20)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available")

    with c2:
        st.markdown("#### Probability Distribution")
        if not oos_signals.empty and "prob" in oos_signals.columns:
            probs  = oos_signals["prob"]
            labels = oos_signals.get("label", None)
            fig = charts["probability_distribution"](probs, labels)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Probability data not available")

    # Confusion matrix
    st.markdown("#### Confusion Matrix")
    if not oos_signals.empty and "label" in oos_signals.columns and "prob" in oos_signals.columns:
        from config.settings import get_settings
        cfg = get_settings()
        y_true = oos_signals["label"].dropna().astype(int)
        y_pred = (oos_signals["prob"].reindex(y_true.index) >= cfg.meta_min_confidence).astype(int)
        common = y_true.index.intersection(y_pred.index)
        if len(common) > 10:
            fig = charts["confusion_matrix"](y_true.loc[common], y_pred.loc[common])
            st.plotly_chart(fig, use_container_width=True)

    # Feature table
    if not feature_importance.empty:
        st.markdown("#### Top 20 Features")
        top20 = feature_importance.head(20).reset_index()
        top20.columns = ["Feature", "Importance"]
        top20["Importance"] = top20["Importance"].map(lambda x: f"{x:.6f}")
        st.dataframe(top20, use_container_width=True, hide_index=True)


def page_regime_analysis(equity, regime_df):
    st.markdown("## 🌊 Regime Analysis")
    charts = get_charts()

    if regime_df.empty:
        st.info("Regime data not available")
        return

    # Regime timeline
    fig = charts["regime_timeline"](regime_df, equity)
    st.plotly_chart(fig, use_container_width=True)

    # Regime distribution
    if "regime_label" in regime_df.columns:
        st.markdown("#### Regime Distribution")
        dist = regime_df["regime_label"].value_counts()
        c1, c2 = st.columns(2)

        with c1:
            fig2 = go.Figure(go.Pie(
                labels=dist.index.tolist(),
                values=dist.values.tolist(),
                hole=0.4,
                marker=dict(colors=["#FF4757","#FFA502","#00CC6A","#FF6B35","#C0392B"]),
            ))
            fig2.update_layout(
                paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                font=dict(color="#E6EDF3"), title="Regime Distribution",
                height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.markdown("**Regime Breakdown**")
            df = dist.to_frame("Count")
            df["Pct"] = (df["Count"] / df["Count"].sum() * 100).map(lambda x: f"{x:.1f}%")
            st.dataframe(df, use_container_width=True)

        # Regime confidence stats
        if "regime_confidence" in regime_df.columns:
            st.markdown("#### Regime Confidence Stats")
            conf_stats = regime_df.groupby("regime_label")["regime_confidence"].agg(["mean", "std", "min", "max"])
            conf_stats.columns = ["Mean Conf", "Std", "Min", "Max"]
            conf_stats = conf_stats.map(lambda x: f"{x:.3f}")
            st.dataframe(conf_stats, use_container_width=True)


def page_returns(equity, returns, metrics):
    st.markdown("## 📅 Returns Analysis")
    charts = get_charts()

    if equity.empty:
        st.info("No data available")
        return

    from backtesting.metrics import compute_monthly_returns, compute_yearly_returns

    monthly_pivot = compute_monthly_returns(equity)
    yearly_df     = compute_yearly_returns(equity)

    # Monthly heatmap
    st.markdown("#### Monthly Returns Heatmap")
    fig = charts["monthly_heatmap"](monthly_pivot)
    st.plotly_chart(fig, use_container_width=True)

    # Yearly returns
    st.markdown("#### Yearly Returns")
    fig2 = charts["yearly_returns"](yearly_df)
    st.plotly_chart(fig2, use_container_width=True)

    # Monthly table (formatted)
    st.markdown("#### Monthly Returns Table")
    display_pivot = monthly_pivot.copy()
    for col in display_pivot.columns:
        display_pivot[col] = display_pivot[col].map(
            lambda x: f"{x*100:+.2f}%" if not pd.isna(x) else "—"
        )
    st.dataframe(display_pivot, use_container_width=True)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Monthly",  f"{metrics.get('avg_monthly_return', 0)*100:+.2f}%")
    c2.metric("Best Month",   f"{metrics.get('best_month', 0)*100:+.2f}%")
    c3.metric("Worst Month",  f"{metrics.get('worst_month', 0)*100:+.2f}%")
    c4.metric("Positive Months", f"{metrics.get('positive_months', 0)}")


def page_risk_monitor(equity, returns, regime_df):
    st.markdown("## ⚠️ Risk Monitor")
    charts = get_charts()

    if equity.empty:
        st.info("No data available")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Drawdown Chart")
        fig = charts["drawdown"](equity)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Rolling Sharpe Ratio (90-day)")
        fig2 = charts["rolling_sharpe"](returns)
        st.plotly_chart(fig2, use_container_width=True)

    # Monte Carlo simulation
    st.markdown("#### Monte Carlo Simulation (1000 paths)")
    with st.spinner("Running Monte Carlo …"):
        daily_rets = equity.resample("D").last().pct_change().dropna()
        n_days  = len(daily_rets)
        n_sims  = 1000
        rng     = np.random.default_rng(42)
        sim_rets = rng.choice(daily_rets.values, size=(n_sims, n_days), replace=True)
        paths   = np.cumprod(1 + sim_rets, axis=1) * equity.iloc[0]

        # Plot percentile bands
        p5   = np.percentile(paths, 5,  axis=0)
        p25  = np.percentile(paths, 25, axis=0)
        p50  = np.percentile(paths, 50, axis=0)
        p75  = np.percentile(paths, 75, axis=0)
        p95  = np.percentile(paths, 95, axis=0)

        fig_mc = go.Figure()
        x_range = list(range(n_days))

        fig_mc.add_trace(go.Scatter(x=x_range+x_range[::-1],
            y=p95.tolist()+p5.tolist()[::-1], fill="toself",
            fillcolor="rgba(0,212,255,0.08)", line_color="rgba(0,0,0,0)", name="5–95%"))
        fig_mc.add_trace(go.Scatter(x=x_range+x_range[::-1],
            y=p75.tolist()+p25.tolist()[::-1], fill="toself",
            fillcolor="rgba(0,212,255,0.15)", line_color="rgba(0,0,0,0)", name="25–75%"))
        fig_mc.add_trace(go.Scatter(x=x_range, y=p50, name="Median",
            line=dict(color="#00D4FF", width=2)))
        fig_mc.add_trace(go.Scatter(x=x_range, y=equity.resample("D").last().values,
            name="Actual", line=dict(color="#FF6B35", width=2)))

        fig_mc.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3"), title="Monte Carlo Simulation",
            height=400, showlegend=True,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    # Drift monitoring status
    st.markdown("#### Feature Drift Status")
    drift_report_path = ROOT / "reports" / "drift_report.pkl"
    if drift_report_path.exists():
        import joblib
        drift = joblib.load(drift_report_path)
        status = drift.get("status", "Unknown")
        avg_psi = drift.get("avg_psi", 0)
        n_high  = drift.get("n_high_drift", 0)
        n_total = drift.get("n_features_checked", 1)

        color = "#FF4757" if drift.get("needs_retraining") else "#FFA502" if avg_psi > 0.1 else "#00CC6A"
        st.markdown(f"""
        <div style="background:#161B22;border:1px solid {color};border-radius:8px;padding:16px;margin:8px 0">
        <strong style="color:{color}">{status}</strong><br>
        Avg PSI: {avg_psi:.4f} &nbsp;|&nbsp;
        High-drift features: {n_high}/{n_total}<br>
        Top drifting: {", ".join(drift.get("high_drift_features", [])[:5])}
        </div>""", unsafe_allow_html=True)

        if drift.get("needs_retraining"):
            st.warning("⚠️ Model retraining is recommended due to significant feature drift!")
    else:
        st.info("No drift report found. Run the full pipeline to generate one.")


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    # Load data
    equity    = load_equity()
    returns   = load_returns()
    regime_df = load_regime()
    oos_sigs  = load_oos_signals()
    feat_imp  = load_feature_importance()
    metrics   = load_metrics()
    benchmarks= load_benchmarks()

    # Route to page
    if page == "🏠 Overview":
        page_overview(equity, returns, metrics, benchmarks, regime_df)
    elif page == "📊 Trade Analysis":
        page_trade_analysis(equity, returns, metrics, oos_sigs)
    elif page == "🤖 Model Insights":
        page_model_insights(feat_imp, oos_sigs)
    elif page == "🌊 Regime Analysis":
        page_regime_analysis(equity, regime_df)
    elif page == "📅 Returns":
        page_returns(equity, returns, metrics)
    elif page == "⚠️ Risk Monitor":
        page_risk_monitor(equity, returns, regime_df)


if __name__ == "__main__":
    main()
