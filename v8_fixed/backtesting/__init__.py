"""
backtesting/__init__.py
"""
from backtesting.engine import run_backtest, run_all_benchmarks
from backtesting.risk import build_final_signal, compute_position_size
from backtesting.metrics import (
    compute_all_metrics, compute_monthly_returns,
    compute_yearly_returns, compare_strategies, compute_rolling_sharpe,
)

__all__ = [
    "run_backtest", "run_all_benchmarks",
    "build_final_signal", "compute_position_size",
    "compute_all_metrics", "compute_monthly_returns",
    "compute_yearly_returns", "compare_strategies", "compute_rolling_sharpe",
]
