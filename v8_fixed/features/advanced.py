"""
features/advanced.py
=====================
Advanced derived features:
  - Rolling returns (multi-horizon)
  - Z-score normalization
  - Volatility regime encoding
  - Trend regime encoding
  - 4H trend filter merge
  - Market state encoding
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from config.settings import get_settings
from utils.logger import log


def add_rolling_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling log returns at multiple horizons."""
    close = df["close"]
    log_ret = np.log(close / close.shift(1))

    for h in [1, 4, 8, 12, 24, 48, 72]:
        df[f"log_ret_{h}h"] = np.log(close / close.shift(h))

    # Rolling realized volatility (std of log returns)
    df["ret_vol_12h"] = log_ret.rolling(12).std()
    df["ret_vol_24h"] = log_ret.rolling(24).std()

    # Return skewness and kurtosis over 24-bar window
    df["ret_skew_24h"] = log_ret.rolling(24).skew()
    df["ret_kurt_24h"] = log_ret.rolling(24).kurt()

    return df


def add_zscore_features(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    Z-score normalization of key features using rolling window.
    This is crucial for regime-invariant feature scaling.
    """
    cfg = cfg or get_settings()
    w = cfg.z_score_window

    # Features to normalize
    cols_to_zscore = [
        "close", "atr", "rsi", "macd_histogram", "roc",
        "hist_vol_20", "bb_width", "rel_vol_20", "adx",
    ]

    for col in cols_to_zscore:
        if col in df.columns:
            roll = df[col].rolling(w, min_periods=10)
            df[f"zscore_{col}"] = (df[col] - roll.mean()) / (roll.std() + 1e-8)

    return df


def add_volatility_regime(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    ATR percentile-based volatility regime encoding.
    0 = Low vol, 1 = Normal vol, 2 = High vol
    """
    cfg = cfg or get_settings()

    # Rolling ATR percentile (lookback = 500 bars)
    atr = df["atr"] if "atr" in df.columns else df["close"].pct_change().abs().rolling(14).mean()
    roll_atr = atr.rolling(500, min_periods=50)

    low_thresh  = roll_atr.quantile(cfg.regime_atr_low_pct  / 100)
    high_thresh = roll_atr.quantile(cfg.regime_atr_high_pct / 100)

    df["vol_regime"] = 1.0  # Normal
    df.loc[atr < low_thresh, "vol_regime"]  = 0.0  # Low vol
    df.loc[atr > high_thresh, "vol_regime"] = 2.0  # High vol

    # Continuous percentile version (for model)
    df["atr_percentile"] = roll_atr.rank(pct=True)

    return df


def add_trend_regime(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    Trend regime based on EMA alignment + ADX.
    Returns:
      trend_regime: 0=Bearish, 1=Ranging, 2=Bullish
      trend_strength: 0–1 continuous
    """
    cfg = cfg or get_settings()

    # Requires EMA features to be computed
    if "ema_20" not in df.columns or "ema_200" not in df.columns:
        log.warning("EMA columns missing — skipping trend regime")
        return df

    adx = df.get("adx", pd.Series(25, index=df.index))

    # Bullish: EMA 20 > EMA 50 > EMA 100 > EMA 200 AND ADX > threshold
    bull = (
        (df["ema_20"] > df["ema_50"]) &
        (df["ema_50"] > df["ema_100"]) &
        (df["ema_100"] > df["ema_200"]) &
        (adx > cfg.regime_adx_threshold)
    )
    # Bearish: reverse
    bear = (
        (df["ema_20"] < df["ema_50"]) &
        (df["ema_50"] < df["ema_100"]) &
        (df["ema_100"] < df["ema_200"]) &
        (adx > cfg.regime_adx_threshold)
    )

    df["trend_regime"] = 1.0  # Ranging default
    df.loc[bull, "trend_regime"] = 2.0   # Bullish
    df.loc[bear, "trend_regime"] = 0.0   # Bearish

    # Continuous trend strength: normalized EMA stack alignment
    ema_stack = (
        (df["ema_20"] > df["ema_50"]).astype(float) * 0.33 +
        (df["ema_50"] > df["ema_100"]).astype(float) * 0.33 +
        (df["ema_100"] > df["ema_200"]).astype(float) * 0.34
    )
    adx_norm = np.clip(adx / 60.0, 0, 1)
    df["trend_strength"] = ema_stack * adx_norm

    return df


def add_market_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combined market state encoding from vol + trend regime.
    market_state: 0–5 (6 states)
      0: Low vol + Ranging
      1: Normal vol + Bullish
      2: Normal vol + Bearish
      3: Normal vol + Ranging
      4: High vol + any trend
      5: High vol + Ranging
    """
    if "vol_regime" not in df.columns or "trend_regime" not in df.columns:
        return df

    cond = (df["vol_regime"].astype(int).astype(str) +
            "_" + df["trend_regime"].astype(int).astype(str))

    state_map = {
        "0_0": 0, "0_1": 0, "0_2": 0,   # Low vol → state 0
        "1_2": 1,                          # Normal vol bullish
        "1_0": 2,                          # Normal vol bearish
        "1_1": 3,                          # Normal vol ranging
        "2_2": 4,                          # High vol bullish
        "2_0": 4,                          # High vol bearish
        "2_1": 5,                          # High vol ranging
    }
    df["market_state"] = cond.map(state_map).fillna(3).astype(float)

    return df


def add_institutional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds advanced institutional-grade features.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

    # 1. Volatility Breakout & Contraction
    # Ratio of current candle range to 50-period ATR
    candle_range = high - low
    atr_50 = df["atr"].rolling(50, min_periods=10).mean() if "atr" in df.columns else candle_range.rolling(50, min_periods=10).mean()
    df["vol_breakout_ratio"] = candle_range / (atr_50 + 1e-8)

    # 2. Pseudo Orderbook Imbalance (Close location within candle + Volume Momentum)
    # 0 = close at low (selling pressure), 1 = close at high (buying pressure)
    close_pos = (close - low) / (candle_range + 1e-8)
    # Multiply by volume anomaly (volume / 20-period moving average volume)
    vol_ma_20 = volume.rolling(20, min_periods=5).mean()
    vol_anomaly = volume / (vol_ma_20 + 1e-8)
    df["orderbook_imbalance_proxy"] = (close_pos - 0.5) * vol_anomaly

    # 3. Return Auto-correlation (Short term memory: mean reverting vs momentum)
    log_ret = np.log(close / close.shift(1))
    # 10-period rolling autocorrelation of 1-period returns
    df["ret_autocorr_10"] = log_ret.rolling(10).apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x)>1 else 0)

    # 4. Price/Volume Divergence
    # 10-period correlation between price and volume
    df["price_vol_corr_10"] = close.rolling(10).corr(volume).fillna(0)

    return df


def merge_4h_features(
    df_1h: pd.DataFrame,
    df_4h_on_1h: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge pre-aligned 4H features into the 1H DataFrame.
    Only include key 4H signals to avoid noise.
    """
    if df_4h_on_1h is None or df_4h_on_1h.empty:
        log.warning("4H data not available — skipping HTF merge")
        return df_1h

    df = df_1h.copy()

    # Select useful 4H columns
    useful_4h_cols = [c for c in df_4h_on_1h.columns if any(
        k in c for k in ["close", "ema", "rsi", "atr", "trend", "vol_regime"]
    )]

    # Add _4h suffix to avoid collision
    rename = {c: f"htf_{c}" for c in useful_4h_cols}
    df_4h_sel = df_4h_on_1h[useful_4h_cols].rename(columns=rename)

    # Align on common index
    common_idx = df.index.intersection(df_4h_sel.index)
    df.loc[common_idx, list(rename.values())] = df_4h_sel.loc[common_idx]

    log.debug(f"Merged {len(rename)} 4H features into 1H frame")
    return df


def compute_all_advanced(
    df: pd.DataFrame,
    df_4h_on_1h: pd.DataFrame = None,
    cfg=None,
) -> pd.DataFrame:
    """Compute all advanced features."""
    cfg = cfg or get_settings()
    log.info("Computing advanced features …")

    df = add_rolling_returns(df)
    df = add_zscore_features(df, cfg)
    df = add_volatility_regime(df, cfg)
    df = add_trend_regime(df, cfg)
    df = add_market_state(df)
    df = add_institutional_features(df)

    if df_4h_on_1h is not None:
        df = merge_4h_features(df, df_4h_on_1h)

    log.info("Advanced features computed.")
    return df
