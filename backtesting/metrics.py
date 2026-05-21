"""
backtesting/metrics.py
=======================
Comprehensive performance analytics:
  - All 15 required metrics
  - Monthly / yearly returns tables
  - Rolling Sharpe
  - Longest losing streak
  - Best / worst month
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from utils.logger import log


def compute_monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """
    Compute month-by-month returns.
    Returns a pivot table: rows=Year, cols=Month.
    """
    monthly = equity.resample("ME").last().pct_change().dropna()
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")

    table = pd.DataFrame({
        "Year":   monthly.index.year,
        "Month":  monthly.index.month,
        "Return": monthly.values,
    })

    pivot = table.pivot(index="Year", columns="Month", values="Return")
    pivot.columns = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec",
    ][:len(pivot.columns)]

    # Annual return column
    pivot["Annual"] = (1 + pivot.fillna(0)).prod(axis=1) - 1

    return pivot


def compute_yearly_returns(equity: pd.Series) -> pd.DataFrame:
    """Annual returns table."""
    yearly = equity.resample("YE").last().pct_change().dropna()
    df = pd.DataFrame({
        "Year":   yearly.index.year,
        "Return": yearly.values,
    }).set_index("Year")
    df["Return_pct"] = (df["Return"] * 100).round(2)
    return df


def compute_rolling_sharpe(
    returns: pd.Series,
    window_bars: int = 24 * 90,  # 90 trading days @ 24 bars/day
) -> pd.Series:
    """Rolling Sharpe ratio on hourly returns."""
    ann = np.sqrt(24 * 365)
    roll = returns.rolling(window_bars, min_periods=window_bars // 4)
    return (roll.mean() / (roll.std() + 1e-10)) * ann


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute drawdown series (negative %)."""
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def compute_longest_losing_streak(equity: pd.Series, freq: str = "D") -> int:
    """Compute longest streak of negative daily returns."""
    daily = equity.resample(freq).last().pct_change().dropna()
    streak, max_streak = 0, 0
    for r in daily.values:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def compute_all_metrics(
    equity: pd.Series,
    returns: pd.Series,
    trades_df: pd.DataFrame = None,
    label: str = "Strategy",
) -> Dict:
    """
    Compute all 15 performance metrics + extended stats.

    Args:
        equity  : Portfolio equity curve (absolute values)
        returns : Hourly returns series
        trades_df: Optional trade records (from VBT portfolio.trades.records_readable)
        label   : Strategy name

    Returns: Dict of all metrics
    """
    log.info(f"Computing metrics for [{label}] …")

    # ── Time span ──
    n_years  = (equity.index[-1] - equity.index[0]).days / 365.25
    n_months = n_years * 12

    # ── Core return metrics ──
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr         = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

    # ── Risk-adjusted ──
    ann_factor  = np.sqrt(24 * 365)
    sharpe      = returns.mean() / (returns.std() + 1e-10) * ann_factor

    neg_rets    = returns[returns < 0]
    sortino     = returns.mean() / (neg_rets.std() + 1e-10) * ann_factor

    dd_series   = compute_drawdown_series(equity)
    max_dd      = dd_series.min()
    calmar      = cagr / abs(max_dd) if max_dd != 0 else np.nan
    recovery    = (equity.iloc[-1] - equity.iloc[0]) / abs(equity.iloc[0] * max_dd) if max_dd != 0 else np.nan

    # ── Monthly stats ──
    monthly_ret  = equity.resample("ME").last().pct_change().dropna()
    avg_monthly  = monthly_ret.mean()
    best_month   = monthly_ret.max()
    worst_month  = monthly_ret.min()
    pos_months   = (monthly_ret > 0).sum()
    neg_months   = (monthly_ret < 0).sum()

    # ── Trade stats (if trade records available) ──
    win_rate = profit_factor = total_trades = avg_trade = expectancy = np.nan
    avg_win = avg_loss = max_win = max_loss = np.nan

    if trades_df is not None and len(trades_df) > 0:
        try:
            wins   = trades_df[trades_df["PnL"] > 0]["PnL"]
            losses = trades_df[trades_df["PnL"] < 0]["PnL"]

            total_trades  = len(trades_df)
            win_rate      = len(wins) / total_trades
            profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
            avg_trade     = trades_df["PnL"].mean()
            avg_win       = wins.mean() if len(wins) > 0 else 0
            avg_loss      = losses.mean() if len(losses) > 0 else 0
            max_win       = wins.max() if len(wins) > 0 else 0
            max_loss      = losses.min() if len(losses) > 0 else 0
            expectancy    = win_rate * avg_win + (1 - win_rate) * avg_loss
        except Exception as e:
            log.warning(f"Trade stats computation error: {e}")

    # ── Streak ──
    longest_loss_streak = compute_longest_losing_streak(equity)

    metrics = {
        # 1. Return
        "total_return":         float(total_return),
        "cagr":                 float(cagr),
        "avg_monthly_return":   float(avg_monthly),

        # 2. Risk
        "max_drawdown":         float(max_dd),
        "avg_drawdown":         float(dd_series[dd_series < 0].mean() if (dd_series < 0).any() else 0),

        # 3. Risk-adjusted
        "sharpe":               float(sharpe),
        "sortino":              float(sortino),
        "calmar":               float(calmar) if not np.isnan(calmar) else 0,
        "recovery_factor":      float(recovery) if not np.isnan(recovery) else 0,

        # 4. Trade stats
        "total_trades":         int(total_trades) if not np.isnan(total_trades) else 0,
        "win_rate":             float(win_rate) if not np.isnan(win_rate) else 0,
        "profit_factor":        float(profit_factor) if not np.isnan(profit_factor) else 0,
        "avg_trade":            float(avg_trade) if not np.isnan(avg_trade) else 0,
        "expectancy":           float(expectancy) if not np.isnan(expectancy) else 0,
        "avg_win":              float(avg_win) if not np.isnan(avg_win) else 0,
        "avg_loss":             float(avg_loss) if not np.isnan(avg_loss) else 0,
        "max_win":              float(max_win) if not np.isnan(max_win) else 0,
        "max_loss":             float(max_loss) if not np.isnan(max_loss) else 0,

        # 5. Monthly
        "best_month":           float(best_month),
        "worst_month":          float(worst_month),
        "positive_months":      int(pos_months),
        "negative_months":      int(neg_months),
        "longest_loss_streak":  int(longest_loss_streak),

        # 6. Meta
        "label":                label,
        "years_tested":         round(n_years, 2),
        "final_equity":         float(equity.iloc[-1]),
        "initial_equity":       float(equity.iloc[0]),
    }

    # Print summary
    log.info(
        f"\n{'='*50}\n"
        f"  {label}\n"
        f"{'='*50}\n"
        f"  CAGR:           {cagr:.2%}\n"
        f"  Sharpe:         {sharpe:.3f}\n"
        f"  Sortino:        {sortino:.3f}\n"
        f"  Max Drawdown:   {max_dd:.2%}\n"
        f"  Calmar:         {calmar:.3f}\n"
        f"  Win Rate:       {win_rate:.2%}\n"
        f"  Profit Factor:  {profit_factor:.3f}\n"
        f"  Avg Monthly:    {avg_monthly:.2%}\n"
        f"  Total Trades:   {total_trades:,}\n"
        f"  Best Month:     {best_month:.2%}\n"
        f"  Worst Month:    {worst_month:.2%}\n"
        f"  Loss Streak:    {longest_loss_streak} days\n"
        f"{'='*50}"
    )

    return metrics


def compare_strategies(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a comparison table across multiple strategies/benchmarks.

    Args:
        results: Dict of {name: backtest_result_dict}

    Returns:
        DataFrame with strategies as rows and metrics as columns
    """
    rows = []
    key_metrics = [
        "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
        "win_rate", "profit_factor", "avg_monthly_return",
        "total_trades", "best_month", "worst_month",
    ]

    for name, result in results.items():
        stats = result.get("stats", {})
        row   = {"Strategy": name}
        for m in key_metrics:
            val = stats.get(m, np.nan)
            row[m] = val
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Strategy")

    # Format percentages
    pct_cols = ["cagr", "max_drawdown", "avg_monthly_return",
                "win_rate", "best_month", "worst_month"]
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")

    return df
