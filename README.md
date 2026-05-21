# AI Quant Research Platform

> **Professional-grade AI backtesting & research system for BTCUSDT**  
> Walk-forward retraining · Meta-labeling · Regime detection · VectorBT · Streamlit

---

## 🎯 What This Is

A fully modular quantitative research platform that:
- Downloads 5+ years of BTC/USDT hourly data
- Engineers 80+ technical, time, and advanced features
- Detects market regimes (Trending Bullish/Bearish, Ranging, High-Vol)
- Trains an XGBoost + Random Forest + LR ensemble with meta-labeling
- Performs walk-forward rolling retraining (no future data leakage)
- Runs realistic backtesting (fees, slippage, ATR stops)
- Produces 10 Plotly charts + HTML report + 6-page Streamlit dashboard
- Monitors feature drift and model decay

**Target:** ~1% monthly return · 12–20% CAGR · ≤15% max drawdown

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Windows note:** If `numba` fails, install it separately:
> ```bash
> pip install numba
> ```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```
This runs all 7 phases. First run takes **30–90 minutes** (data download + Optuna tuning).

### 3. Launch the dashboard
```bash
python run_pipeline.py --mode=dashboard
# OR directly:
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
BTCML/
│
├── config/
│   └── settings.py          ← All parameters (edit this to tune strategy)
│
├── data/
│   ├── downloader.py        ← CCXT (Binance) + yfinance fallback
│   ├── preprocessor.py      ← Cleaning, alignment, parquet persistence
│   ├── raw/                 ← Downloaded OHLCV (auto-created)
│   └── processed/           ← Clean feature matrix (auto-created)
│
├── features/
│   ├── technical.py         ← 40+ indicators (pandas-ta)
│   ├── time_features.py     ← Session, hour, DOW, cyclic encoding
│   ├── advanced.py          ← Rolling returns, z-score, regimes, 4H merge
│   └── pipeline.py          ← Orchestrator → returns (X, ohlcv)
│
├── models/
│   ├── labeling.py          ← Triple Barrier Method (ATR-based)
│   ├── regime.py            ← ATR+EMA rules + optional HMM (with confidence)
│   ├── xgboost_model.py     ← XGBoost + SHAP + calibration
│   ├── rf_model.py          ← Random Forest
│   ├── lr_model.py          ← Logistic Regression baseline
│   ├── ensemble.py          ← Sharpe-weighted soft voting
│   ├── meta_label.py        ← Meta-labeling + Trade Quality Score (0–100)
│   └── ann_model.py         ← PyTorch MLP (scaffolded, disabled by default)
│
├── training/
│   ├── trainer.py           ← Full training pipeline + time-decay weights
│   ├── optimizer.py         ← Optuna (maximizes OOS Sharpe, not accuracy)
│   └── feature_selector.py  ← Variance + correlation + XGB importance filter
│
├── retraining/
│   └── walk_forward.py      ← Purged K-Fold + rolling walk-forward engine
│
├── backtesting/
│   ├── engine.py            ← VectorBT + 3 benchmarks
│   ├── risk.py              ← Position sizing + circuit breaker + SL/TP
│   └── metrics.py           ← All 15 metrics + monthly/yearly tables
│
├── monitoring/
│   └── drift.py             ← PSI + KL divergence drift detection
│
├── visualization/
│   ├── charts.py            ← 10 Plotly charts (dark theme, Plotly only)
│   └── reports.py           ← Professional HTML report generator
│
├── dashboard/
│   └── app.py               ← 6-page Streamlit dashboard
│
├── utils/
│   ├── logger.py            ← Loguru structured logging
│   ├── cache.py             ← Disk caching (@cached decorator)
│   └── validators.py        ← Data integrity checks
│
├── notebooks/               ← Jupyter analysis notebooks
├── reports/                 ← Output: HTML report, charts, parquet files
├── run_pipeline.py          ← ← ← MAIN ENTRY POINT
└── requirements.txt
```

---

## ⚙️ Pipeline Modes

| Command | Description |
|---------|-------------|
| `python run_pipeline.py` | Full pipeline (default) |
| `python run_pipeline.py --mode=data` | Download + preprocess data only |
| `python run_pipeline.py --mode=features` | Feature engineering only |
| `python run_pipeline.py --mode=train` | Training only |
| `python run_pipeline.py --mode=backtest` | Backtest on saved results |
| `python run_pipeline.py --mode=report` | Generate report only |
| `python run_pipeline.py --mode=dashboard` | Launch Streamlit |
| `python run_pipeline.py --no-optimize` | Skip Optuna (faster, ~10–20 min) |
| `python run_pipeline.py --force-refresh` | Re-download all data |

---

## 🔧 Key Configuration

Edit [`config/settings.py`](config/settings.py):

```python
# Risk per trade
base_risk_pct = 0.5          # 0.5% of equity per trade

# Triple barrier
barrier_atr_mult_tp = 2.0    # Take-profit: 2× ATR
barrier_atr_mult_sl = 1.0    # Stop-loss: 1× ATR
barrier_horizon_bars = 24    # Max 24 candles (24 hours)

# Walk-forward
train_window_months = 24     # 2-year training window
test_window_months  = 3      # 3-month OOS test
step_months         = 3      # Roll forward 3 months each fold

# Optuna
optuna_n_trials = 50         # Reduce to 20 for faster runs

# Signal quality
quality_score_threshold = 55  # Min score (0–100) to enter trade
meta_min_confidence     = 0.55 # Min ML probability
```

---

## 📊 Dashboard Pages

| Page | Content |
|------|---------|
| 🏠 Overview | Equity curve, 12 KPI cards, target achievement |
| 📊 Trade Analysis | P&L distribution, quality scores, trade stats |
| 🤖 Model Insights | Feature importance, confusion matrix, probability dist |
| 🌊 Regime Analysis | Regime timeline, confidence, per-regime performance |
| 📅 Returns | Monthly heatmap, yearly bars, returns table |
| ⚠️ Risk Monitor | Drawdown, rolling Sharpe, Monte Carlo (1000 paths), drift status |

---

## 📈 Strategy Architecture

```
CCXT/yfinance (5yr BTC 1H + 4H)
         │
         ▼
  Feature Engineering (80+ features)
  [Technical + Time + Advanced + HTF]
         │
         ├── Triple Barrier Labeling (ATR × 2/1, 24-bar horizon)
         │
         ├── Regime Detection (ATR percentile + EMA + ADX → 5 states + confidence)
         │
         ▼
  Purged K-Fold Walk-Forward (24m train, 3m test, 3m step)
  └── Optuna (50 trials, maximize OOS Sharpe)
  └── XGBoost + RF + LR → Sharpe-weighted Ensemble
  └── Meta-Label model (gates primary signals)
         │
         ▼
  Trade Quality Score (0–100)
  [Primary prob × Meta prob × Regime conf × Vol × Trend]
         │
         ▼
  Risk Engine
  [Confidence-scaled position sizing: 0.25%–1%]
  [ATR SL/TP · Circuit breaker · Consecutive loss limit]
         │
         ▼
  VectorBT Backtest (fees: 0.04%, slippage: 0.02%)
         │
         ▼
  Analytics + Streamlit Dashboard
```

---

## 📋 Performance Metrics Generated

1. CAGR
2. Sharpe Ratio
3. Sortino Ratio
4. Calmar Ratio
5. Max Drawdown
6. Win Rate
7. Profit Factor
8. Expectancy
9. Avg Trade
10. Recovery Factor
11. Best / Worst Month
12. Longest Losing Streak
13. Monthly Returns Table
14. Yearly Returns Table
15. Trade Count

---

## ⚠️ Important Disclaimers

- **Not financial advice.** This is a research tool only.
- Backtest results do not guarantee future performance.
- Realistic assumptions (fees, slippage) are used, but execution risk exists.
- Walk-forward validation reduces but does not eliminate overfitting risk.
- Monitor for feature drift after model deployment.

---

## 🛠️ Troubleshooting

**`numba` import error (labeling.py)**  
→ `pip install numba` or the labeling falls back to pure Python automatically.

**VectorBT compatibility issue**  
→ `pip install vectorbt==0.26.2` (pin to tested version)

**Streamlit `streamlit-extras` missing**  
→ `pip install streamlit-extras`

**CCXT rate limit errors**  
→ Pipeline automatically retries with exponential backoff. yfinance fallback activates after 5 failures.

**Out of memory (OOM) on large feature matrix**  
→ Set `optuna_n_trials = 20` and reduce `train_window_months = 18` in settings.

---

*Built with ❤️ for quantitative research. Use responsibly.*
