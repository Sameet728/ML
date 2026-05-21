"""
backtesting/engine.py
======================
VectorBT-based backtesting engine with:
  - Realistic fees + slippage
  - ATR-based exits (SL/TP simulation via signal truncation)
  - Fixed fractional position sizing
  - Benchmark comparisons (BTC buy & hold, EMA crossover, random)
  - Returns full portfolio stats

Note on SL/TP in VectorBT:
  VectorBT's basic Portfolio supports stop-loss and take-profit natively
  via sl_stop and tp_stop parameters. We pass these as arrays.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from config.settings import get_settings
from utils.logger import log


def run_backtest(
    ohlcv: pd.DataFrame,
    signal_df: pd.DataFrame,
    cfg=None,
    label: str = "Strategy",
) -> Dict:
    """
    Run VectorBT backtest.

    Args:
        ohlcv     : OHLCV DataFrame with DatetimeIndex
        signal_df : Signal DataFrame from risk.build_final_signal()
        cfg       : Settings
        label     : Name for this backtest

    Returns:
        Dict with: portfolio, equity_curve, returns, stats
    """
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("Install vectorbt: pip install vectorbt")

    cfg = cfg or get_settings()

    # ── Align signals to OHLCV ──
    close  = ohlcv["close"]
    signal = signal_df["signal"].reindex(close.index, fill_value=0)
    sl_pct = (signal_df["sl_price"] / signal_df["entry_price"] - 1).abs()
    tp_pct = (signal_df["tp_price"] / signal_df["entry_price"] - 1).abs()

    sl_pct = sl_pct.reindex(close.index).fillna(cfg.atr_sl_mult * 0.02)
    tp_pct = tp_pct.reindex(close.index).fillna(cfg.atr_tp_mult * 0.02)

    # ── Position size as fraction of portfolio ──
    pos_size = signal_df["position_size"].reindex(close.index, fill_value=0)
    # Convert risk% to portfolio fraction (assuming 1× leverage, no margin)
    # pos_size here is % of equity risked; convert to size via SL distance
    sl_dist  = sl_pct.replace(0, 0.01)
    size_pct = (pos_size / 100) / sl_dist  # Kelly-inspired fraction
    size_pct = size_pct.clip(0, 0.25)  # Cap at 25% of portfolio per trade

    entries = signal == 1

    log.info(
        f"[{label}] Running backtest: {entries.sum():,} entries, "
        f"initial_capital={cfg.initial_capital:,.0f}, "
        f"fees={cfg.backtest_fees_pct}%, slippage={cfg.backtest_slippage_pct}%"
    )

    try:
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=pd.Series(False, index=close.index),  # Exits via SL/TP
            sl_stop=sl_pct,
            tp_stop=tp_pct,
            size=size_pct,
            size_type="percent",
            init_cash=cfg.initial_capital,
            fees=cfg.backtest_fees_pct / 100,
            slippage=cfg.backtest_slippage_pct / 100,
            freq="1h",
        )
    except Exception as e:
        log.warning(f"VectorBT advanced mode failed ({e}), falling back to simple mode …")
        portfolio = _simple_backtest(close, entries, size_pct, cfg)

    equity  = portfolio.value()
    returns = equity.pct_change().fillna(0)

    stats = _extract_stats(portfolio, equity, returns, label)

    log.info(
        f"[{label}] CAGR={stats['cagr']:.2%} | "
        f"Sharpe={stats['sharpe']:.3f} | "
        f"MaxDD={stats['max_drawdown']:.2%} | "
        f"Trades={stats['total_trades']:,}"
    )

    return {
        "label":     label,
        "portfolio": portfolio,
        "equity":    equity,
        "returns":   returns,
        "stats":     stats,
    }


def _simple_backtest(close, entries, size_pct, cfg):
    """Fallback simple backtest using basic VectorBT from_signals."""
    import vectorbt as vbt
    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=~entries,
        size=size_pct.clip(0, 0.1),
        size_type="percent",
        init_cash=cfg.initial_capital,
        fees=cfg.backtest_fees_pct / 100,
        slippage=cfg.backtest_slippage_pct / 100,
        freq="1h",
    )


def run_benchmark_btc_hold(ohlcv: pd.DataFrame, cfg=None) -> Dict:
    """BTC buy-and-hold benchmark."""
    import vectorbt as vbt
    cfg = cfg or get_settings()
    close   = ohlcv["close"]
    entries = pd.Series(False, index=close.index)
    entries.iloc[0] = True
    exits   = pd.Series(False, index=close.index)
    exits.iloc[-1] = True

    portfolio = vbt.Portfolio.from_signals(
        close, entries=entries, exits=exits,
        size=1.0, size_type="percent",
        init_cash=cfg.initial_capital,
        fees=cfg.backtest_fees_pct / 100,
        freq="1h",
    )
    equity  = portfolio.value()
    returns = equity.pct_change().fillna(0)
    stats   = _extract_stats(portfolio, equity, returns, "BTC Buy & Hold")
    return {"label": "BTC Buy & Hold", "equity": equity, "returns": returns, "stats": stats}


def run_benchmark_ema_crossover(ohlcv: pd.DataFrame, cfg=None) -> Dict:
    """Simple EMA 20/50 crossover benchmark."""
    import vectorbt as vbt
    cfg   = cfg or get_settings()
    close = ohlcv["close"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    entries = (ema20 > ema50) & (ema20.shift(1) <= ema50.shift(1))
    exits   = (ema20 < ema50) & (ema20.shift(1) >= ema50.shift(1))

    portfolio = vbt.Portfolio.from_signals(
        close, entries=entries, exits=exits,
        size=0.95, size_type="percent",
        init_cash=cfg.initial_capital,
        fees=cfg.backtest_fees_pct / 100,
        slippage=cfg.backtest_slippage_pct / 100,
        freq="1h",
    )
    equity  = portfolio.value()
    returns = equity.pct_change().fillna(0)
    stats   = _extract_stats(portfolio, equity, returns, "EMA Crossover")
    return {"label": "EMA Crossover", "equity": equity, "returns": returns, "stats": stats}


def run_benchmark_random(ohlcv: pd.DataFrame, cfg=None, seed: int = 42) -> Dict:
    """Random entry benchmark for sanity check."""
    import vectorbt as vbt
    cfg   = cfg or get_settings()
    close = ohlcv["close"]
    rng   = np.random.default_rng(seed)

    # Random entries with same frequency as ~1 trade/day
    n_entries = len(close) // 24
    entry_idx = rng.choice(len(close), size=n_entries, replace=False)
    entries   = pd.Series(False, index=close.index)
    entries.iloc[sorted(entry_idx)] = True
    exits = entries.shift(12).fillna(False)

    portfolio = vbt.Portfolio.from_signals(
        close, entries=entries, exits=exits,
        size=0.05, size_type="percent",
        init_cash=cfg.initial_capital,
        fees=cfg.backtest_fees_pct / 100,
        freq="1h",
    )
    equity  = portfolio.value()
    returns = equity.pct_change().fillna(0)
    stats   = _extract_stats(portfolio, equity, returns, "Random Strategy")
    return {"label": "Random Strategy", "equity": equity, "returns": returns, "stats": stats}


def run_all_benchmarks(ohlcv: pd.DataFrame, cfg=None) -> Dict[str, Dict]:
    """Run all benchmarks and return dict."""
    cfg = cfg or get_settings()
    benchmarks = {}

    log.info("Running benchmarks …")
    try:
        benchmarks["btc_buy_hold"]    = run_benchmark_btc_hold(ohlcv, cfg)
        benchmarks["ema_crossover"]   = run_benchmark_ema_crossover(ohlcv, cfg)
        benchmarks["random_strategy"] = run_benchmark_random(ohlcv, cfg)
    except Exception as e:
        log.warning(f"Some benchmarks failed: {e}")

    return benchmarks


def _extract_stats(portfolio, equity: pd.Series, returns: pd.Series, label: str) -> Dict:
    """Extract key statistics from VectorBT portfolio."""
    try:
        vbt_stats = portfolio.stats()
    except Exception:
        vbt_stats = {}

    # Helper lambdas
    def safe(key, default=np.nan):
        try:
            val = vbt_stats.get(key, default) if isinstance(vbt_stats, dict) else getattr(vbt_stats, key, default)
            return float(val) if val is not None else default
        except Exception:
            return default

    # CAGR
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 0.1)) - 1

    # Max drawdown
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # Sharpe (annualized, hourly)
    ann_factor = np.sqrt(24 * 365)
    sharpe     = returns.mean() / (returns.std() + 1e-10) * ann_factor

    # Sortino
    neg_rets   = returns[returns < 0]
    sortino    = returns.mean() / (neg_rets.std() + 1e-10) * ann_factor

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Win rate / profit factor from VectorBT trade records
    try:
        trades    = portfolio.trades.records_readable
        wins      = trades[trades["PnL"] > 0]
        losses    = trades[trades["PnL"] < 0]
        win_rate  = len(wins) / max(len(trades), 1)
        gross_profit = wins["PnL"].sum()
        gross_loss   = abs(losses["PnL"].sum())
        profit_factor = gross_profit / max(gross_loss, 1e-8)
        total_trades  = len(trades)
        avg_trade     = trades["PnL"].mean() if len(trades) > 0 else 0
        expectancy    = win_rate * wins["PnL"].mean() + (1 - win_rate) * losses["PnL"].mean() if len(losses) > 0 else 0
    except Exception:
        win_rate = profit_factor = total_trades = avg_trade = expectancy = np.nan

    # Recovery factor
    total_pnl     = equity.iloc[-1] - equity.iloc[0]
    recovery      = total_pnl / abs(equity.iloc[0] * max_dd) if max_dd != 0 else np.nan

    return {
        "label":          label,
        "cagr":           float(cagr),
        "sharpe":         float(sharpe),
        "sortino":        float(sortino),
        "calmar":         float(calmar),
        "max_drawdown":   float(max_dd),
        "total_return":   float(equity.iloc[-1] / equity.iloc[0] - 1),
        "win_rate":       float(win_rate) if not np.isnan(win_rate) else np.nan,
        "profit_factor":  float(profit_factor) if not np.isnan(profit_factor) else np.nan,
        "total_trades":   int(total_trades) if not np.isnan(total_trades) else 0,
        "avg_trade":      float(avg_trade) if not np.isnan(avg_trade) else np.nan,
        "expectancy":     float(expectancy) if not np.isnan(expectancy) else np.nan,
        "recovery_factor":float(recovery) if not np.isnan(recovery) else np.nan,
        "final_equity":   float(equity.iloc[-1]),
        "initial_equity": float(equity.iloc[0]),
    }
