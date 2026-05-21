"""
models/labeling.py
==================
Triple Barrier Method for trade labeling.

For each bar, we check forward N bars to see which barrier is hit first:
  - Upper barrier: entry + TP_mult × ATR  → label = 1 (BUY)
  - Lower barrier: entry - SL_mult × ATR  → label = 0 (NO BUY)
  - Vertical barrier: N bars elapsed       → label = 0 (timeout)

This avoids the naive "predict next candle direction" problem.
No future data leakage: labels are aligned to entry bar.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple
from config.settings import get_settings
from utils.logger import log


# ── Numba-accelerated barrier engine ────────────────────────────────────────

@njit(cache=True)
def _compute_barriers_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core triple-barrier labeling loop.
    Returns:
        labels    : int8 array (1=TP hit, 0=SL/timeout)
        exit_bars : int32 array (number of bars to exit)
    """
    n = len(close)
    labels    = np.zeros(n, dtype=np.int8)
    exit_bars = np.zeros(n, dtype=np.int32)

    for i in range(n - horizon):
        entry = close[i]
        atr_i = atr[i]
        if atr_i <= 0 or np.isnan(atr_i):
            exit_bars[i] = horizon
            continue

        upper = entry + tp_mult * atr_i
        lower = entry - sl_mult * atr_i

        for j in range(1, horizon + 1):
            idx = i + j
            if idx >= n:
                break
            if high[idx] >= upper:
                labels[i]    = 1
                exit_bars[i] = j
                break
            if low[idx] <= lower:
                labels[i]    = 0
                exit_bars[i] = j
                break
        else:
            exit_bars[i] = horizon

    return labels, exit_bars


def compute_triple_barrier_labels(
    ohlcv: pd.DataFrame,
    atr: pd.Series,
    tp_mult: float = None,
    sl_mult: float = None,
    horizon: int   = None,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for the full dataset.

    Args:
        ohlcv    : OHLCV DataFrame with DatetimeIndex
        atr      : ATR Series aligned to ohlcv
        tp_mult  : Take-profit ATR multiplier
        sl_mult  : Stop-loss ATR multiplier
        horizon  : Max bars to look forward

    Returns:
        DataFrame with columns:
          label      : 1 = TP hit, 0 = SL or timeout
          exit_bar   : Number of bars to exit
          tp_price   : Upper barrier price
          sl_price   : Lower barrier price
          entry_price: Entry price (close at bar i)
    """
    cfg = get_settings()
    tp_mult  = tp_mult  or cfg.barrier_atr_mult_tp
    sl_mult  = sl_mult  or cfg.barrier_atr_mult_sl
    horizon  = horizon  or cfg.barrier_horizon_bars

    log.info(
        f"Computing triple barriers: TP={tp_mult}×ATR, SL={sl_mult}×ATR, "
        f"horizon={horizon} bars …"
    )

    # Align ATR to ohlcv index
    atr = atr.reindex(ohlcv.index).ffill().bfill()

    high  = ohlcv["high"].values.astype(np.float64)
    low   = ohlcv["low"].values.astype(np.float64)
    close = ohlcv["close"].values.astype(np.float64)
    atr_v = atr.values.astype(np.float64)

    try:
        labels_arr, exit_arr = _compute_barriers_numba(
            high, low, close, atr_v, tp_mult, sl_mult, horizon
        )
    except Exception:
        # Pure Python fallback (slower)
        log.warning("Numba JIT failed — using pure Python fallback (slower)")
        labels_arr, exit_arr = _compute_barriers_python(
            high, low, close, atr_v, tp_mult, sl_mult, horizon
        )

    result = pd.DataFrame({
        "label":       labels_arr,
        "exit_bar":    exit_arr,
        "entry_price": close,
        "tp_price":    close + tp_mult * atr_v,
        "sl_price":    close - sl_mult * atr_v,
    }, index=ohlcv.index)

    # Last `horizon` rows have unreliable labels
    result.iloc[-horizon:, result.columns.get_loc("label")] = np.nan

    pos_rate = result["label"].dropna().mean()
    log.info(
        f"Labels computed: {len(result):,} bars, "
        f"positive rate = {pos_rate:.2%}, "
        f"avg exit = {result['exit_bar'].mean():.1f} bars"
    )
    return result


def _compute_barriers_python(
    high, low, close, atr, tp_mult, sl_mult, horizon
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure Python fallback for triple barrier computation."""
    n = len(close)
    labels    = np.zeros(n, dtype=np.int8)
    exit_bars = np.zeros(n, dtype=np.int32)

    for i in range(n - horizon):
        entry = close[i]
        atr_i = atr[i]
        if atr_i <= 0 or np.isnan(atr_i):
            exit_bars[i] = horizon
            continue

        upper = entry + tp_mult * atr_i
        lower = entry - sl_mult * atr_i
        hit   = False

        for j in range(1, horizon + 1):
            idx = i + j
            if idx >= n:
                break
            if high[idx] >= upper:
                labels[i], exit_bars[i], hit = 1, j, True
                break
            if low[idx] <= lower:
                labels[i], exit_bars[i], hit = 0, j, True
                break
        if not hit:
            exit_bars[i] = horizon

    return labels, exit_bars


def get_label_stats(labels: pd.Series) -> dict:
    """Print and return label statistics."""
    clean = labels.dropna()
    stats = {
        "total":    len(clean),
        "positive": int(clean.sum()),
        "negative": int((clean == 0).sum()),
        "pos_rate": float(clean.mean()),
    }
    log.info(
        f"Label stats: {stats['total']:,} labels | "
        f"Positive: {stats['positive']:,} ({stats['pos_rate']:.2%}) | "
        f"Negative: {stats['negative']:,}"
    )
    return stats
