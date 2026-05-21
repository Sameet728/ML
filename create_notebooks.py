"""
create_notebooks.py
====================
Helper script to create all Jupyter notebooks.
Run once: python create_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent / "notebooks"
ROOT.mkdir(parents=True, exist_ok=True)


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": cells,
    }


def code(src, idx=0):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src, "id": f"c{idx}"}


def md(src, idx=0):
    return {"cell_type": "markdown", "metadata": {}, "source": src, "id": f"m{idx}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 1: Data Exploration
# ═══════════════════════════════════════════════════════════════════════════════
nb1 = nb([
    md("# 01 — Data Exploration\n\nExplore raw BTC/USDT OHLCV data quality and characteristics over 5+ years.", 0),
    code("""\
import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.settings import get_settings

cfg = get_settings()
print(f"Period: {cfg.start_date} to {cfg.end_date}")
print(f"Symbol: {cfg.primary_symbol}")""", 1),
    code("""\
# Load or download data
try:
    from data.preprocessor import load_processed
    btc = load_processed('btc_1h_clean')
    print("Loaded from cache")
except FileNotFoundError:
    from data.downloader import download_btc
    from data.preprocessor import clean_ohlcv
    btc = download_btc()
    btc = clean_ohlcv(btc)
    print("Downloaded fresh")

print(f"Shape: {btc.shape}")
print(f"Date range: {btc.index[0]} to {btc.index[-1]}")
print(f"Missing: {btc.isna().sum().sum()}")
btc.tail()""", 2),
    code("""\
# Basic statistics
btc.describe().round(4)""", 3),
    code("""\
# Interactive candlestick chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.75, 0.25], vertical_spacing=0.04)

fig.add_trace(go.Candlestick(
    x=btc.index, open=btc['open'], high=btc['high'],
    low=btc['low'], close=btc['close'], name='BTC/USDT',
    increasing_line_color='#00CC6A', decreasing_line_color='#FF4757',
), row=1, col=1)

fig.add_trace(go.Bar(
    x=btc.index, y=btc['volume'],
    name='Volume', marker_color='rgba(0,212,255,0.4)',
), row=2, col=1)

fig.update_layout(
    template='plotly_dark', height=600,
    title='BTC/USDT — 5-Year OHLCV',
    xaxis_rangeslider_visible=False,
)
fig.show()""", 4),
    code("""\
# Annual returns summary
yearly = btc['close'].resample('YE').last().pct_change().dropna()
print("Annual Returns:")
for year, ret in zip(yearly.index.year, yearly.values):
    icon = "🟢" if ret > 0 else "🔴"
    print(f"  {icon} {year}: {ret:+.1%}")""", 5),
    code("""\
# Volatility regime chart
log_ret = np.log(btc['close'] / btc['close'].shift(1)).dropna()
vol_30d = log_ret.rolling(30 * 24).std() * np.sqrt(8760)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=vol_30d.index, y=vol_30d.values,
    name="30d Ann. Volatility",
    line=dict(color="#FFA502", width=1.5),
    fill='tozeroy', fillcolor='rgba(255,165,2,0.1)',
))
fig.add_hline(y=vol_30d.mean(), line_dash="dash", line_color="white",
    annotation_text=f"Mean: {vol_30d.mean():.1%}")
fig.update_layout(
    template="plotly_dark", height=350,
    title="30-Day Annualized Volatility",
    yaxis_tickformat=".0%",
)
fig.show()""", 6),
    code("""\
# Weekend vs weekday volume analysis
btc_copy = btc.copy()
btc_copy['is_weekend'] = btc_copy.index.dayofweek >= 5
weekend_vol   = btc_copy[btc_copy['is_weekend']]['volume'].mean()
weekday_vol   = btc_copy[~btc_copy['is_weekend']]['volume'].mean()
print(f"Avg weekday volume:  {weekday_vol:,.0f}")
print(f"Avg weekend volume:  {weekend_vol:,.0f}")
print(f"Weekend/weekday ratio: {weekend_vol/weekday_vol:.2f}")""", 7),
])

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 2: Feature Analysis
# ═══════════════════════════════════════════════════════════════════════════════
nb2 = nb([
    md("# 02 — Feature Analysis\n\nExplore engineered features: distributions, correlations, and importance.", 0),
    code("""\
import sys; sys.path.insert(0, '..')
import pandas as pd, numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from features.pipeline import load_feature_matrix
    X, ohlcv = load_feature_matrix()
    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
except FileNotFoundError:
    print("Run run_pipeline.py --mode=features first")""", 1),
    code("""\
# Feature overview
stats = X.describe().T
stats['null_pct'] = (X.isna().mean() * 100).round(2)
stats[['mean', 'std', 'min', 'max', 'null_pct']].round(4).head(30)""", 2),
    code("""\
# Distribution of key features
sample_features = ['rsi', 'macd_histogram', 'atr_pct', 'bb_width',
                   'rel_vol_20', 'trend_strength', 'adx', 'roc']
sample_features = [f for f in sample_features if f in X.columns]

n_cols = 4
n_rows = (len(sample_features) + n_cols - 1) // n_cols
fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=sample_features)

for i, feat in enumerate(sample_features):
    r, c = divmod(i, n_cols)
    data = X[feat].dropna()
    fig.add_trace(go.Histogram(
        x=data, name=feat, nbinsx=50,
        marker_color='#00D4FF', opacity=0.7,
    ), row=r+1, col=c+1)

fig.update_layout(
    template='plotly_dark', height=400,
    title='Feature Distributions', showlegend=False,
)
fig.show()""", 3),
    code("""\
# Correlation heatmap (top 25 features by variance)
top_cols = X.var().sort_values(ascending=False).head(25).index.tolist()
corr = X[top_cols].corr().round(2)

fig = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.index,
    colorscale='RdBu', zmid=0,
    colorbar=dict(title='Corr'),
    hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>',
))
fig.update_layout(
    template='plotly_dark', height=700,
    title='Feature Correlation Matrix (Top 25 by Variance)',
)
fig.show()""", 4),
    code("""\
# Regime feature analysis
if 'vol_regime' in X.columns and 'trend_regime' in X.columns:
    vol_dist   = X['vol_regime'].value_counts()
    trend_dist = X['trend_regime'].value_counts()
    print("Volatility Regime distribution:")
    for v, c in vol_dist.items():
        print(f"  Regime {int(v)}: {c:,} bars ({c/len(X):.1%})")
    print("\\nTrend Regime distribution:")
    for v, c in trend_dist.items():
        labels = {0: 'Bearish', 1: 'Ranging', 2: 'Bullish'}
        print(f"  {labels.get(int(v), v)}: {c:,} bars ({c/len(X):.1%})")""", 5),
    code("""\
# Time feature analysis: avg return by hour
ohlcv_feat = ohlcv.join(X[['hour', 'session_overlap', 'is_weekend']], how='left')
ohlcv_feat['ret'] = ohlcv_feat['close'].pct_change()
hourly_ret = ohlcv_feat.groupby('hour')['ret'].mean() * 100

fig = go.Figure(go.Bar(
    x=hourly_ret.index, y=hourly_ret.values,
    marker_color=['#00CC6A' if v >= 0 else '#FF4757' for v in hourly_ret.values],
    name='Avg Hourly Return (%)',
))
fig.update_layout(
    template='plotly_dark', height=380,
    title='Average Return by Hour of Day (UTC)',
    xaxis_title='Hour (UTC)', yaxis_title='Avg Return (%)',
)
fig.show()""", 6),
])

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 3: Model Training
# ═══════════════════════════════════════════════════════════════════════════════
nb3 = nb([
    md("# 03 — Model Training\n\nInteractive model training, evaluation, and comparison.", 0),
    code("""\
import sys; sys.path.insert(0, '..')
import pandas as pd, numpy as np
import plotly.graph_objects as go
from sklearn.metrics import classification_report
from features.pipeline import load_feature_matrix
from models.labeling import compute_triple_barrier_labels, get_label_stats
from models.xgboost_model import XGBoostModel
from models.rf_model import RandomForestModel
from models.ensemble import EnsembleModel
from training.trainer import evaluate_predictions
from config.settings import get_settings

cfg = get_settings()
X, ohlcv = load_feature_matrix()
print(f"Loaded feature matrix: {X.shape}")""", 1),
    code("""\
# Generate triple barrier labels
atr = X['atr'] if 'atr' in X.columns else ohlcv['close'].pct_change().abs().rolling(14).mean() * ohlcv['close']
label_df = compute_triple_barrier_labels(ohlcv, atr,
    tp_mult=cfg.barrier_atr_mult_tp,
    sl_mult=cfg.barrier_atr_mult_sl,
    horizon=cfg.barrier_horizon_bars,
)
y = label_df['label'].dropna()
common = X.index.intersection(y.index)
X_clean, y_clean = X.loc[common], y.loc[common]
get_label_stats(y_clean)""", 2),
    code("""\
# Chronological train/val/test split (no walk-forward for exploration)
n = len(X_clean)
n_tr = int(n * 0.60)
n_va = int(n * 0.80)

X_tr, y_tr = X_clean.iloc[:n_tr],     y_clean.iloc[:n_tr]
X_va, y_va = X_clean.iloc[n_tr:n_va], y_clean.iloc[n_tr:n_va]
X_te, y_te = X_clean.iloc[n_va:],     y_clean.iloc[n_va:]

print(f"Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")
print(f"Date ranges:")
print(f"  Train: {X_tr.index[0].date()} to {X_tr.index[-1].date()}")
print(f"  Val:   {X_va.index[0].date()} to {X_va.index[-1].date()}")
print(f"  Test:  {X_te.index[0].date()} to {X_te.index[-1].date()}")""", 3),
    code("""\
# Feature selection
from training.feature_selector import select_features
X_tr_sel, X_te_sel, selected = select_features(X_tr, y_tr, X_te, top_n=60)
X_va_sel = X_va.reindex(columns=selected, fill_value=0)
print(f"Selected features: {len(selected)}")""", 4),
    code("""\
# Train XGBoost
print("Training XGBoost...")
xgb = XGBoostModel()
xgb.fit(X_tr_sel, y_tr, X_va_sel, y_va)

# Evaluate
probs_xgb = pd.Series(xgb.predict_proba(X_te_sel)[:, 1], index=X_te_sel.index)
metrics_xgb = evaluate_predictions(y_te, probs_xgb, threshold=0.55, name="XGBoost")""", 5),
    code("""\
# Train Random Forest
print("Training Random Forest...")
rf = RandomForestModel()
rf.fit(X_tr_sel, y_tr, X_va_sel, y_va)

probs_rf = pd.Series(rf.predict_proba(X_te_sel)[:, 1], index=X_te_sel.index)
metrics_rf = evaluate_predictions(y_te, probs_rf, threshold=0.55, name="RandomForest")""", 6),
    code("""\
# Ensemble
ensemble = EnsembleModel(models=[xgb, rf], weights=[0.6, 0.4])
probs_ens = pd.Series(ensemble.predict_proba(X_te_sel)[:, 1], index=X_te_sel.index)
metrics_ens = evaluate_predictions(y_te, probs_ens, threshold=0.55, name="Ensemble")

# Comparison table
comparison = pd.DataFrame([metrics_xgb, metrics_rf, metrics_ens]).set_index('threshold')
comparison.index = ['XGBoost', 'RandomForest', 'Ensemble']
comparison[['accuracy', 'f1', 'precision', 'recall', 'auc', 'n_signals']]""", 7),
    code("""\
# Feature Importance
imp = xgb.feature_importance().head(25)
fig = go.Figure(go.Bar(
    x=imp.values[::-1], y=imp.index[::-1],
    orientation='h',
    marker=dict(
        color=imp.values[::-1],
        colorscale=[[0, '#FFA502'], [1, '#00D4FF']],
        showscale=False,
    ),
))
fig.update_layout(
    template='plotly_dark', height=600,
    title='XGBoost Feature Importance (Top 25 — Gain)',
)
fig.show()""", 8),
    code("""\
# SHAP values
shap_vals = xgb.shap_values(X_te_sel.head(500))
if shap_vals is not None:
    mean_shap = np.abs(shap_vals).mean(axis=0)
    shap_ser = pd.Series(mean_shap, index=selected).sort_values(ascending=False).head(20)
    fig = go.Figure(go.Bar(
        x=shap_ser.values[::-1], y=shap_ser.index[::-1],
        orientation='h', marker_color='#FFA502',
    ))
    fig.update_layout(template='plotly_dark', height=550,
        title='SHAP Feature Importance (Top 20)')
    fig.show()
else:
    print("SHAP not available — install: pip install shap")""", 9),
])

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 4: Backtest Analysis
# ═══════════════════════════════════════════════════════════════════════════════
nb4 = nb([
    md("# 04 — Backtest Analysis\n\nAnalyze full walk-forward backtest results.", 0),
    code("""\
import sys; sys.path.insert(0, '..')
import pandas as pd, numpy as np, joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from config.settings import get_settings
from backtesting.metrics import (
    compute_monthly_returns, compute_yearly_returns,
    compute_rolling_sharpe, compute_drawdown_series,
)

cfg = get_settings()
rpt = cfg.paths['reports']

# Load results
try:
    equity  = pd.read_parquet(rpt / 'equity_curve.parquet').squeeze()
    returns = pd.read_parquet(rpt / 'returns.parquet').squeeze()
    metrics = joblib.load(rpt / 'metrics.pkl')
    print(f"Loaded results: {equity.index[0].date()} to {equity.index[-1].date()}")
    print(f"Total return: {(equity.iloc[-1]/equity.iloc[0]-1):.2%}")
except FileNotFoundError:
    print("Run: python run_pipeline.py --mode=full")""", 1),
    code("""\
# Print all metrics
print("=" * 50)
for k, v in metrics.items():
    if k in ('label', 'feature_details'):
        continue
    if isinstance(v, float):
        pct_keys = ('return', 'drawdown', 'rate', 'month', 'cagr')
        if any(p in k for p in pct_keys):
            print(f"  {k:<30}: {v:+.2%}")
        else:
            print(f"  {k:<30}: {v:.4f}")
    else:
        print(f"  {k:<30}: {v}")""", 2),
    code("""\
# Equity + Drawdown chart
dd = compute_drawdown_series(equity) * 100
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.04)

fig.add_trace(go.Scatter(
    x=equity.index, y=equity.values,
    name='AI Strategy', line=dict(color='#00D4FF', width=2.5),
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=dd.index, y=dd.values,
    name='Drawdown %', fill='tozeroy',
    fillcolor='rgba(255,71,87,0.2)', line=dict(color='#FF4757', width=1.5),
), row=2, col=1)

fig.update_layout(template='plotly_dark', height=600,
    title='Walk-Forward Equity Curve + Drawdown')
fig.show()""", 3),
    code("""\
# Monthly returns heatmap
monthly_pivot = compute_monthly_returns(equity)
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Annual']
plot_cols = [c for c in month_names if c in monthly_pivot.columns and c != 'Annual']
plot_data = monthly_pivot[plot_cols] * 100

fig = go.Figure(go.Heatmap(
    z=plot_data.values,
    x=plot_data.columns.tolist(),
    y=[str(y) for y in plot_data.index],
    colorscale=[[0,'#C0392B'],[0.35,'#E74C3C'],[0.5,'#2D3748'],[0.65,'#27AE60'],[1,'#00CC6A']],
    zmid=0,
    text=np.where(np.isnan(plot_data.values), '', np.vectorize(lambda v: f'{v:+.1f}%')(plot_data.values)),
    texttemplate='%{text}',
    colorbar=dict(title='Return %', ticksuffix='%'),
))
fig.update_layout(template='plotly_dark', height=400, title='Monthly Returns Heatmap')
fig.show()""", 4),
    code("""\
# Yearly returns
yearly_df = compute_yearly_returns(equity)
colors = ['#00CC6A' if r >= 0 else '#FF4757' for r in yearly_df['Return'].values]
fig = go.Figure(go.Bar(
    x=[str(y) for y in yearly_df.index],
    y=yearly_df['Return'].values * 100,
    marker_color=colors,
    text=[f"{r*100:+.1f}%" for r in yearly_df['Return'].values],
    textposition='outside',
))
fig.add_hline(y=12, line_dash='dash', line_color='#00D4FF',
    annotation_text='12% target')
fig.update_layout(template='plotly_dark', height=400,
    title='Yearly Returns', yaxis_title='Return (%)')
fig.show()""", 5),
    code("""\
# Rolling Sharpe
rolling_sharpe = compute_rolling_sharpe(returns, window_bars=24*90).dropna()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=rolling_sharpe.index, y=rolling_sharpe.values,
    name='90-day Rolling Sharpe', line=dict(color='#00D4FF', width=2),
))
for level, color, label in [(1.0,'#00CC6A','Good (1.0)'), (0,'#FFA502','Break-even')]:
    fig.add_hline(y=level, line_dash='dash', line_color=color, annotation_text=label)
fig.update_layout(template='plotly_dark', height=380, title='Rolling Sharpe (90-day)')
fig.show()""", 6),
    code("""\
# Walk-forward fold summary
fold_file = rpt / 'fold_summary.csv'
if fold_file.exists():
    fold_df = pd.read_csv(fold_file)
    print("Walk-Forward Fold Results:")
    print(fold_df.to_string(index=False))
    print(f"\\nMean Val Sharpe: {fold_df['val_sharpe'].mean():.4f}")
    print(f"Std Val Sharpe:  {fold_df['val_sharpe'].std():.4f}")
    print(f"Total Signals:   {fold_df['n_signals'].sum():,}")""", 7),
])

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 5: Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════════════
nb5 = nb([
    md("# 05 — Monte Carlo Simulation\n\nBootstrap trade returns to estimate the distribution of outcomes.", 0),
    code("""\
import sys; sys.path.insert(0, '..')
import pandas as pd, numpy as np, joblib
import plotly.graph_objects as go
from pathlib import Path
from config.settings import get_settings
from backtesting.metrics import compute_drawdown_series

cfg = get_settings()
rpt = cfg.paths['reports']

try:
    equity  = pd.read_parquet(rpt / 'equity_curve.parquet').squeeze()
    returns = pd.read_parquet(rpt / 'returns.parquet').squeeze()
    print(f"Equity loaded: {equity.index[0].date()} to {equity.index[-1].date()}")
    print(f"Initial: ${equity.iloc[0]:,.0f}  |  Final: ${equity.iloc[-1]:,.0f}")
except FileNotFoundError:
    print("Run the full pipeline first: python run_pipeline.py")""", 1),
    code("""\
# Monte Carlo configuration
N_SIMS  = 2000   # Number of simulation paths
N_DAYS  = len(equity.resample('D').last())  # Same number of trading days
rng     = np.random.default_rng(42)

daily_rets = equity.resample('D').last().pct_change().dropna()
print(f"Daily returns: {len(daily_rets)} days")
print(f"Daily mean: {daily_rets.mean():.4%}")
print(f"Daily std:  {daily_rets.std():.4%}")""", 2),
    code("""\
# Run bootstrap simulation
print(f"Running {N_SIMS:,} Monte Carlo paths...")
sim_rets = rng.choice(daily_rets.values, size=(N_SIMS, N_DAYS), replace=True)
paths    = np.cumprod(1 + sim_rets, axis=1) * equity.iloc[0]

# Compute statistics on final values
final_vals = paths[:, -1]
print(f"\\nFinal Portfolio Distribution ({N_SIMS:,} simulations):")
for pct in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  P{pct:2d}: ${np.percentile(final_vals, pct):,.0f}")
print(f"  Actual: ${equity.iloc[-1]:,.0f}")
print(f"\\nProbability of profit:         {(final_vals > equity.iloc[0]).mean():.1%}")
print(f"Probability of > 10% return:   {(final_vals > equity.iloc[0]*1.10).mean():.1%}")
print(f"Probability of > 20% return:   {(final_vals > equity.iloc[0]*1.20).mean():.1%}")""", 3),
    code("""\
# Plot Monte Carlo paths with percentile bands
p5  = np.percentile(paths,  5, axis=0)
p25 = np.percentile(paths, 25, axis=0)
p50 = np.percentile(paths, 50, axis=0)
p75 = np.percentile(paths, 75, axis=0)
p95 = np.percentile(paths, 95, axis=0)
x   = list(range(N_DAYS))

fig = go.Figure()

# Bands
fig.add_trace(go.Scatter(
    x=x + x[::-1], y=p95.tolist() + p5.tolist()[::-1],
    fill='toself', fillcolor='rgba(0,212,255,0.06)',
    line_color='rgba(0,0,0,0)', name='5–95% Band',
))
fig.add_trace(go.Scatter(
    x=x + x[::-1], y=p75.tolist() + p25.tolist()[::-1],
    fill='toself', fillcolor='rgba(0,212,255,0.15)',
    line_color='rgba(0,0,0,0)', name='25–75% Band',
))

# Median + actual
fig.add_trace(go.Scatter(x=x, y=p50, name='Median Simulation',
    line=dict(color='#00D4FF', width=2)))
fig.add_trace(go.Scatter(
    x=list(range(N_DAYS)), y=equity.resample('D').last().values[:N_DAYS],
    name='Actual Equity', line=dict(color='#FF6B35', width=2.5),
))

fig.update_layout(
    template='plotly_dark', height=520,
    title=f'Monte Carlo Simulation — {N_SIMS:,} Bootstrap Paths',
    yaxis_title='Portfolio Value ($)',
    xaxis_title='Trading Days',
)
fig.show()""", 4),
    code("""\
# Final value distribution histogram
fig = go.Figure(go.Histogram(
    x=final_vals, nbinsx=80,
    marker_color='#00D4FF', opacity=0.8,
))
fig.add_vline(x=equity.iloc[-1], line_dash='dash', line_color='#FF6B35',
    annotation_text=f'Actual: ${equity.iloc[-1]:,.0f}')
fig.add_vline(x=equity.iloc[0], line_dash='dash', line_color='gray',
    annotation_text='Initial')
fig.add_vline(x=np.median(final_vals), line_dash='dash', line_color='#00CC6A',
    annotation_text=f'Median: ${np.median(final_vals):,.0f}')
fig.update_layout(
    template='plotly_dark', height=400,
    title='Distribution of Final Portfolio Values (Monte Carlo)',
    xaxis_title='Final Value ($)', yaxis_title='Count',
)
fig.show()""", 5),
    code("""\
# Max drawdown distribution across simulations
print("Computing max drawdowns for all simulations...")
max_dds = []
for path in paths:
    eq_s = pd.Series(path)
    dd   = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min()
    max_dds.append(dd * 100)
max_dds = np.array(max_dds)

actual_dd = compute_drawdown_series(equity).min() * 100

fig = go.Figure(go.Histogram(x=max_dds, nbinsx=60,
    marker_color='#FF4757', opacity=0.8))
fig.add_vline(x=np.percentile(max_dds, 95), line_dash='dash',
    line_color='#FFA502', annotation_text=f'P95 DD: {np.percentile(max_dds, 95):.1f}%')
fig.add_vline(x=actual_dd, line_dash='solid', line_color='#FF6B35',
    annotation_text=f'Actual DD: {actual_dd:.1f}%')
fig.update_layout(template='plotly_dark', height=400,
    title='Monte Carlo Max Drawdown Distribution',
    xaxis_title='Max Drawdown (%)', yaxis_title='Count')
fig.show()

print(f"\\nMax Drawdown Stats:")
print(f"  P25 DD: {np.percentile(max_dds, 25):.2f}%")
print(f"  P50 DD: {np.percentile(max_dds, 50):.2f}%")
print(f"  P75 DD: {np.percentile(max_dds, 75):.2f}%")
print(f"  P95 DD: {np.percentile(max_dds, 95):.2f}%")
print(f"  Actual DD: {actual_dd:.2f}%")""", 6),
])

# Save all notebooks
notebooks = {
    "01_data_exploration.ipynb":  nb1,
    "02_feature_analysis.ipynb":  nb2,
    "03_model_training.ipynb":    nb3,
    "04_backtest_analysis.ipynb": nb4,
    "05_monte_carlo.ipynb":       nb5,
}

for fname, content in notebooks.items():
    path = ROOT / fname
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(f"Created: {path.name}")

print("\nAll 5 notebooks created successfully!")
