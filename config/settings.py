"""
config/settings.py
==================
Central configuration for the AI Quant Research Platform.
All tunable parameters live here — change one file, affects everything.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


# ── Root paths ──────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent
DATA_RAW_DIR    = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR   = ROOT_DIR / "data" / "processed"
MODEL_DIR       = ROOT_DIR / "models" / "saved"
REPORT_DIR      = ROOT_DIR / "reports"
CACHE_DIR       = ROOT_DIR / ".cache"

for _d in [DATA_RAW_DIR, DATA_PROC_DIR, MODEL_DIR, REPORT_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Main settings dataclass ──────────────────────────────────────────────────
@dataclass
class Settings:
    # ── Market & timeframe ─────────────────────────────────────────────────
    primary_symbol: str          = "BTC/USDT"           # CCXT format
    primary_symbol_vbt: str      = "BTCUSDT"            # VectorBT / Binance
    optional_symbol_yf: str      = "GC=F"               # Gold — yfinance ticker
    enable_gold: bool            = True                  # Enable XAUUSD portfolio mode
    primary_timeframe: str       = "1h"                  # Primary OHLCV timeframe
    htf_timeframe: str           = "4h"                  # Higher timeframe filter

    # ── Date range ─────────────────────────────────────────────────────────
    start_date: str              = "2020-01-01"          # 5+ years of history
    end_date: str                = ""                    # "" = today

    # ── Feature engineering ────────────────────────────────────────────────
    ema_periods: List[int]       = field(default_factory=lambda: [20, 50, 100, 200])
    rsi_period: int              = 14
    atr_period: int              = 14
    bb_period: int               = 20
    bb_std: float                = 2.0
    macd_fast: int               = 12
    macd_slow: int               = 26
    macd_signal: int             = 9
    roc_period: int              = 10
    vol_lookback: int            = 20                    # Historical volatility window
    z_score_window: int          = 50                    # Z-score normalization window

    # ── Triple Barrier Labeling ────────────────────────────────────────────
    barrier_atr_mult_tp: float   = 2.0                   # Take-profit barrier (ATR ×)
    barrier_atr_mult_sl: float   = 1.0                   # Stop-loss barrier (ATR ×)
    barrier_horizon_bars: int    = 24                    # Max candles to wait (24h)

    # ── Regime detection ───────────────────────────────────────────────────
    regime_atr_low_pct: float    = 33.0                  # Percentile threshold → low vol
    regime_atr_high_pct: float   = 67.0                  # Percentile threshold → high vol
    regime_hmm_states: int       = 3                     # HMM hidden states
    regime_adx_period: int       = 14
    regime_adx_threshold: float  = 25.0                  # ADX > 25 → trending

    # ── Meta-labeling ─────────────────────────────────────────────────────
    enable_meta_labeling: bool   = True
    meta_min_confidence: float   = 0.52                  # Primary model prob threshold (lowered from 0.55)
    meta_prob_threshold: float   = 0.50                  # Meta model filter threshold

    # ── Trade quality scoring ──────────────────────────────────────────────
    # Observed mean quality ~32-35 across folds; threshold 55 was too strict (54 trades).
    # Lowered to 30 to target ~300-500 trades for realistic monthly return.
    quality_score_threshold: int = 30                    # Min score (0–100) to trade

    # ── Position sizing ────────────────────────────────────────────────────
    # Sharpe=3.87, MaxDD=-1.0% → significant risk budget remaining.
    # Doubling base_risk_pct from 0.5% → 1.0% targets ~1% avg monthly return.
    base_risk_pct: float         = 1.0                   # Base risk per trade (%)
    risk_scale_low: float        = 0.5                   # Risk at confidence 52–65%
    risk_scale_mid: float        = 1.0                   # Risk at confidence 65–75%
    risk_scale_high: float       = 2.0                   # Risk at confidence 75%+
    confidence_low_thresh: float = 0.65
    confidence_high_thresh: float= 0.75
    max_position_risk_pct: float = 2.0                   # Hard cap per trade (raised from 1.0)

    # ── Risk management ────────────────────────────────────────────────────
    atr_sl_mult: float           = 1.5                   # ATR SL multiplier
    atr_tp_mult: float           = 3.0                   # ATR TP multiplier
    max_drawdown_circuit_pct: float = 15.0               # Pause trading if DD hits this
    consecutive_loss_limit: int  = 5                     # Max consecutive losses
    skip_uncertain_regimes: bool = True                  # Skip low-confidence regimes
    regime_confidence_min: float = 0.45                  # Min regime confidence (relaxed from 0.60; 'Ranging' dominates)

    # ── Walk-forward / Retraining ──────────────────────────────────────────
    train_window_months: int     = 24                    # Training window
    test_window_months: int      = 3                     # Out-of-sample window
    step_months: int             = 3                     # Rolling step

    # ── Purged K-Fold (within training window) ─────────────────────────────
    purged_embargo_bars: int     = 24                    # Embargo gap (bars) after each fold
    cv_n_splits: int             = 5                     # Folds in purged cross-validation

    # ── Optuna optimization ────────────────────────────────────────────────
    optuna_n_trials: int         = 50
    optuna_direction: str        = "maximize"            # Maximize Sharpe
    optuna_timeout_sec: int      = 1800                  # 30min max per fold

    # ── Backtesting ────────────────────────────────────────────────────────
    backtest_fees_pct: float     = 0.04                  # 0.04% per side (Binance taker)
    backtest_slippage_pct: float = 0.02                  # 0.02% slippage
    initial_capital: float       = 10_000.0

    # ── Benchmarks ────────────────────────────────────────────────────────
    benchmarks: List[str]        = field(default_factory=lambda: [
        "btc_buy_hold", "ema_crossover", "random_strategy"
    ])

    # ── Feature drift monitoring ───────────────────────────────────────────
    drift_psi_threshold: float   = 0.2                   # PSI > 0.2 → significant drift
    drift_kl_threshold: float    = 0.5                   # KL divergence threshold
    drift_window_days: int       = 30                    # Window for drift check

    # ── Model toggles ──────────────────────────────────────────────────────
    enable_random_forest: bool   = True
    enable_logistic_reg: bool    = True
    enable_ensemble: bool        = True
    enable_ann: bool             = False                 # Scaffolded — enable later

    # ── Misc ───────────────────────────────────────────────────────────────
    random_seed: int             = 42
    n_jobs: int                  = -1                    # Use all CPU cores
    log_level: str               = "INFO"
    cache_enabled: bool          = True
    report_format: str           = "html"                # "html" or "pdf"

    def __post_init__(self):
        import datetime
        if not self.end_date:
            self.end_date = datetime.date.today().isoformat()

    @property
    def paths(self):
        return {
            "root":       ROOT_DIR,
            "data_raw":   DATA_RAW_DIR,
            "data_proc":  DATA_PROC_DIR,
            "models":     MODEL_DIR,
            "reports":    REPORT_DIR,
            "cache":      CACHE_DIR,
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values and return updated instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    for k, v in kwargs.items():
        if hasattr(_settings, k):
            setattr(_settings, k, v)
        else:
            raise ValueError(f"Unknown setting: {k}")
    return _settings
