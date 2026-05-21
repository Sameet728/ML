"""
backtesting/risk.py
====================
Risk management layer applied BEFORE backtesting:

1. Position sizing: confidence-scaled fractional risk
2. ATR-based stop-loss price computation
3. Circuit breaker: halt trading if drawdown > threshold
4. Consecutive loss limiter
5. Regime filter: reduce size in uncertain/high-vol regimes
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from config.settings import get_settings
from utils.logger import log


def compute_position_size(
    quality_scores: pd.Series,
    probs: pd.Series,
    regime_df: pd.DataFrame,
    cfg=None,
) -> pd.Series:
    """
    Confidence-scaled position sizing.

    Risk tiers based on probability:
      prob < conf_low  → base_risk × risk_scale_low
      prob < conf_high → base_risk × risk_scale_mid
      prob >= conf_high→ base_risk × risk_scale_high

    Additional modifiers:
      - High-vol regime (3 or 4): × 0.5
      - Low-quality score: linear scale down below threshold
      - Weekend: × 0.7

    Returns:
        position_size_pct: Series of risk % per trade (0 to max_risk)
    """
    cfg = cfg or get_settings()

    size = pd.Series(0.0, index=probs.index)

    # Base tier from probability
    low_conf  = probs < cfg.confidence_low_thresh
    mid_conf  = (probs >= cfg.confidence_low_thresh) & (probs < cfg.confidence_high_thresh)
    high_conf = probs >= cfg.confidence_high_thresh

    size[low_conf]  = cfg.risk_scale_low
    size[mid_conf]  = cfg.risk_scale_mid
    size[high_conf] = cfg.risk_scale_high

    # Quality score modifier
    if quality_scores is not None:
        q_thresh = cfg.quality_score_threshold
        q_norm = np.clip((quality_scores - q_thresh) / (100 - q_thresh), 0, 1)
        q_factor = 0.5 + 0.5 * q_norm   # Scale from 0.5× to 1.0×
        size = size * q_factor

    # Regime modifier
    if regime_df is not None and "regime" in regime_df.columns:
        high_vol_regime = regime_df["regime"].reindex(probs.index).isin([3, 4])
        size[high_vol_regime] *= 0.5

        # Low regime confidence → reduce size
        if "regime_confidence" in regime_df.columns:
            conf = regime_df["regime_confidence"].reindex(probs.index).fillna(0.5)
            low_conf_mask = conf < cfg.regime_confidence_min
            size[low_conf_mask] *= 0.5

    # Cap at maximum
    size = size.clip(0, cfg.max_position_risk_pct)

    # Zero out non-trade bars
    no_signal = probs < cfg.meta_min_confidence
    size[no_signal] = 0.0

    return size


def apply_circuit_breaker(
    signals: pd.Series,
    equity_curve: pd.Series = None,
    cfg=None,
) -> pd.Series:
    """
    Circuit breaker: pause trading if rolling drawdown exceeds threshold.
    Uses equity curve if provided; otherwise uses a consecutive-loss proxy.
    Returns filtered signals.
    """
    cfg = cfg or get_settings()
    filtered = signals.copy()

    if equity_curve is not None and len(equity_curve) == len(signals):
        rolling_max  = equity_curve.cummax()
        drawdown_pct = (equity_curve - rolling_max) / rolling_max * 100
        in_circuit   = drawdown_pct < -cfg.max_drawdown_circuit_pct
        filtered[in_circuit] = 0
        n_halted = in_circuit.sum()
        if n_halted > 0:
            log.info(f"Circuit breaker: halted {n_halted:,} signals (DD > {cfg.max_drawdown_circuit_pct}%)")

    return filtered


def apply_consecutive_loss_limit(
    signals: pd.Series,
    labels: pd.Series,
    cfg=None,
) -> pd.Series:
    """
    After N consecutive losses, suppress signals for 24 bars (cool-down).
    Reduces drawdown during losing streaks.
    """
    cfg = cfg or get_settings()
    limit = cfg.consecutive_loss_limit
    filtered = signals.copy()

    # Identify actual trade outcomes where we had a signal
    trade_mask    = signals > 0
    trade_indices = np.where(trade_mask)[0]

    consec_losses = 0
    in_cooldown   = 0
    cooldown_bars = 24

    for i in range(len(filtered)):
        if in_cooldown > 0:
            filtered.iloc[i] = 0
            if signals.iloc[i] > 0:
                in_cooldown -= 1
            continue

        if signals.iloc[i] > 0 and i < len(labels):
            outcome = labels.iloc[i]
            if outcome == 0:
                consec_losses += 1
            else:
                consec_losses = 0

            if consec_losses >= limit:
                in_cooldown   = cooldown_bars
                consec_losses = 0
                log.debug(f"Consecutive loss limit hit at bar {i}, entering cooldown")

    return filtered


def compute_atr_stops(
    ohlcv: pd.DataFrame,
    atr: pd.Series,
    entry_price: pd.Series,
    cfg=None,
) -> pd.DataFrame:
    """
    Compute ATR-based stop-loss and take-profit prices.

    Returns DataFrame with:
        sl_price : Stop-loss price
        tp_price : Take-profit price
    """
    cfg = cfg or get_settings()
    atr_aligned = atr.reindex(ohlcv.index).ffill()

    stops = pd.DataFrame(index=ohlcv.index)
    stops["sl_price"] = entry_price - cfg.atr_sl_mult * atr_aligned
    stops["tp_price"] = entry_price + cfg.atr_tp_mult * atr_aligned
    stops["sl_price"] = stops["sl_price"].clip(lower=0)

    return stops


def build_final_signal(
    oos_probs: pd.Series,
    oos_quality: pd.Series,
    regime_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg=None,
) -> pd.DataFrame:
    """
    Produce the final signal DataFrame for the backtesting engine.

    Columns:
        signal       : 1 = enter long, 0 = no trade
        position_size: risk % of equity
        entry_price  : close at bar i
        sl_price     : ATR stop-loss
        tp_price     : ATR take-profit

    Signal pipeline:
        OOS Prob → Quality Filter → Regime Filter → Final Signal
    """
    cfg = cfg or get_settings()
    log.info("Building final trading signals …")

    # Align all series to ohlcv index
    probs   = oos_probs.reindex(ohlcv.index).fillna(0)
    quality = oos_quality.reindex(ohlcv.index).fillna(0)
    r_df    = regime_df.reindex(ohlcv.index).ffill()

    # ── Step 1: Probability threshold ──
    raw_signal = (probs >= cfg.meta_min_confidence).astype(float)

    # ── Step 2: Quality score filter ──
    quality_ok = quality >= cfg.quality_score_threshold
    filtered_signal = raw_signal * quality_ok.astype(float)

    # ── Step 3: Regime confidence filter ──
    if cfg.skip_uncertain_regimes and "regime_confidence" in r_df.columns:
        reg_conf = r_df["regime_confidence"].fillna(0)
        uncertain = reg_conf < cfg.regime_confidence_min
        filtered_signal[uncertain] = 0

    # ── Step 4: Weekend filter (reduce not eliminate) ──
    is_weekend = ohlcv.index.dayofweek >= 5
    filtered_signal[is_weekend] *= 0.5  # Allow but reduce size

    # ── Step 5: Position sizing ──
    pos_size = compute_position_size(quality, probs, r_df, cfg)
    pos_size = pos_size * filtered_signal  # Zero out non-signal bars

    # ── Step 6: ATR stops ──
    atr = ohlcv["close"].pct_change().abs().rolling(14).mean() * ohlcv["close"]
    if "atr" in ohlcv.columns:
        atr = ohlcv["atr"]

    close = ohlcv["close"]
    stops = compute_atr_stops(ohlcv, atr, close, cfg)

    result = pd.DataFrame({
        "signal":        filtered_signal.astype(int),
        "prob":          probs,
        "quality_score": quality,
        "position_size": pos_size,
        "entry_price":   close,
        "sl_price":      stops["sl_price"],
        "tp_price":      stops["tp_price"],
    }, index=ohlcv.index)

    n_signals = result["signal"].sum()
    log.info(
        f"Signal pipeline complete: {n_signals:,} signals | "
        f"avg_quality={quality[result['signal'] == 1].mean():.1f} | "
        f"avg_prob={probs[result['signal'] == 1].mean():.3f}"
    )
    return result
