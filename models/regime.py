"""
models/regime.py
================
Market regime detection with confidence scoring.

Regime classification:
  0 = Trending Bearish
  1 = Ranging / Choppy
  2 = Trending Bullish
  3 = High Volatility Bullish
  4 = High Volatility Bearish

Methods:
  - ATR percentile (primary, no extra deps)
  - EMA alignment + ADX (trend state)
  - Hidden Markov Model (optional, 3 states)

Outputs:
  regime        : int (0–4)
  regime_confidence : float (0–1)
  regime_label  : str
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from config.settings import get_settings
from utils.logger import log


REGIME_LABELS = {
    0: "Trending Bearish",
    1: "Ranging",
    2: "Trending Bullish",
    3: "High Vol Bullish",
    4: "High Vol Bearish",
}


# ── ATR + EMA Regime Engine ─────────────────────────────────────────────────

def detect_regime_rules(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    Rule-based regime detection using ATR percentile + EMA alignment.
    Fast, interpretable, no extra dependencies.

    Returns DataFrame with:
      regime             : int 0–4
      regime_confidence  : float 0–1
      regime_label       : str
    """
    cfg = cfg or get_settings()
    result = pd.DataFrame(index=df.index)

    # ── ATR percentile ──
    atr = df["atr"] if "atr" in df.columns else df["close"].pct_change().abs().rolling(14).mean()
    atr_pct = atr.rolling(500, min_periods=50).rank(pct=True).fillna(0.5)

    high_vol = atr_pct > (cfg.regime_atr_high_pct / 100)
    low_vol  = atr_pct < (cfg.regime_atr_low_pct  / 100)

    # ── EMA alignment ──
    if all(c in df.columns for c in ["ema_20", "ema_50", "ema_100", "ema_200"]):
        bull = (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"])
        bear = (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"])
    else:
        close = df["close"]
        ema20  = close.ewm(span=20, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        bull   = ema20 > ema200
        bear   = ema20 < ema200

    # ── ADX strength ──
    adx = df.get("adx", pd.Series(25.0, index=df.index))
    strong_trend = adx > cfg.regime_adx_threshold

    # ── Regime assignment ──
    regime = pd.Series(1, index=df.index, dtype=int)  # Default: Ranging

    # High volatility states (override trend)
    regime[high_vol & bull] = 3  # High vol bullish
    regime[high_vol & bear] = 4  # High vol bearish

    # Normal volatility trend states
    normal_vol = ~high_vol & ~low_vol
    regime[normal_vol & bull & strong_trend] = 2   # Trending bullish
    regime[normal_vol & bear & strong_trend] = 0   # Trending bearish

    # ── Confidence score ──
    # Confidence = f(ATR percentile extremity, ADX strength, EMA alignment)
    alignment_score = np.zeros(len(df))
    if "ema_alignment" in df.columns:
        alignment_score = df["ema_alignment"].values / 3.0  # 0–1

    adx_score = np.clip(adx.values / 50.0, 0, 1)

    # Volatility certainty: extreme percentiles → more confident
    vol_score = (atr_pct - 0.5).abs().values * 2  # 0 at 50th pct, 1 at extremes

    # Combined confidence (weighted average)
    confidence = 0.4 * adx_score + 0.35 * alignment_score + 0.25 * vol_score
    confidence = np.clip(confidence, 0, 1)

    result["regime"]            = regime.values
    result["regime_confidence"] = confidence
    result["regime_label"]      = regime.map(REGIME_LABELS)

    # Rolling regime stability (how stable is regime over last 24 bars)
    result["regime_stability"] = (
        result["regime"].rolling(24).apply(
            lambda x: (x == x.iloc[-1]).mean(), raw=False
        )
    ).fillna(0.5)

    log.info(
        f"Regime detection complete. Distribution:\n"
        f"{result['regime_label'].value_counts().to_string()}"
    )
    return result


# ── Hidden Markov Model Regime ───────────────────────────────────────────────

def detect_regime_hmm(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    3-state Hidden Markov Model regime detection.
    Uses log returns + volatility as observation features.
    States are mapped to regime labels post-hoc by avg return.

    Falls back to rule-based if hmmlearn not available.
    """
    cfg = cfg or get_settings()

    try:
        from hmmlearn import hmm
    except ImportError:
        log.warning("hmmlearn not installed — falling back to rule-based regime")
        return detect_regime_rules(df, cfg)

    close   = df["close"]
    log_ret = np.log(close / close.shift(1)).fillna(0)
    vol     = log_ret.rolling(5).std().fillna(0)

    X_hmm = np.column_stack([log_ret.values, vol.values])

    # Remove any NaN/inf rows for HMM fitting
    valid = np.isfinite(X_hmm).all(axis=1)
    X_fit = X_hmm[valid]

    n_states = cfg.regime_hmm_states
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=cfg.random_seed,
    )

    try:
        model.fit(X_fit)
        hidden_states = model.predict(X_hmm)
    except Exception as e:
        log.warning(f"HMM fitting failed ({e}) — using rule-based fallback")
        return detect_regime_rules(df, cfg)

    # ── Map HMM states to regime labels ──
    # Sort by mean return: state with highest return → Bullish
    state_returns = {}
    for s in range(n_states):
        mask = hidden_states == s
        state_returns[s] = log_ret.values[mask].mean() if mask.sum() > 0 else 0

    sorted_states = sorted(state_returns, key=state_returns.get)
    state_to_regime = {
        sorted_states[0]: 0,  # Lowest return → Bearish
        sorted_states[1]: 1,  # Middle → Ranging
        sorted_states[2]: 2,  # Highest return → Bullish
    }

    hmm_regime = np.array([state_to_regime[s] for s in hidden_states])

    # Compute HMM posterior probability as confidence
    try:
        posteriors = model.predict_proba(X_hmm)
        mapped_idx = np.array([state_to_regime[s] for s in range(n_states)])
        confidence = posteriors.max(axis=1)
    except Exception:
        confidence = np.full(len(df), 0.6)

    result = pd.DataFrame({
        "regime":            hmm_regime,
        "regime_confidence": confidence,
    }, index=df.index)
    result["regime_label"]    = result["regime"].map(REGIME_LABELS)
    result["regime_stability"] = (
        result["regime"].rolling(24).apply(
            lambda x: (x == x.iloc[-1]).mean(), raw=False
        )
    ).fillna(0.5)

    log.info(
        f"HMM regime detection complete ({n_states} states).\n"
        f"{result['regime_label'].value_counts().to_string()}"
    )
    return result


# ── Public API ───────────────────────────────────────────────────────────────

def detect_regime(
    df: pd.DataFrame,
    method: str = "rules",
    cfg=None,
) -> pd.DataFrame:
    """
    Detect market regime.

    Args:
        df     : Feature DataFrame (with EMA, ATR, ADX columns)
        method : "rules" (default) or "hmm"
        cfg    : Settings

    Returns:
        DataFrame with regime, regime_confidence, regime_label, regime_stability
    """
    cfg = cfg or get_settings()
    if method == "hmm":
        return detect_regime_hmm(df, cfg)
    return detect_regime_rules(df, cfg)


def regime_filter(
    signals: pd.Series,
    regime_df: pd.DataFrame,
    cfg=None,
) -> pd.Series:
    """
    Filter trading signals based on regime confidence.
    Suppresses signals when:
      - Regime confidence < min threshold
      - High volatility regime (unless configured to allow)

    Returns filtered signal Series.
    """
    cfg = cfg or get_settings()

    filtered = signals.copy().astype(float)

    if not cfg.skip_uncertain_regimes:
        return filtered

    # Suppress low-confidence regime bars
    low_conf = regime_df["regime_confidence"] < cfg.regime_confidence_min
    filtered[low_conf] = 0.0

    # Optional: suppress high-vol regimes (3, 4) to reduce DD
    high_vol_regime = regime_df["regime"].isin([3, 4])
    # In high-vol, reduce signal by 50% (don't fully remove — captures vol moves)
    filtered[high_vol_regime] *= 0.5

    n_filtered = (signals > 0).sum() - (filtered > 0).sum()
    log.info(f"Regime filter: suppressed {n_filtered:,} signals")
    return filtered
