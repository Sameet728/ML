"""
features/technical.py
=====================
All technical indicator features computed with pandas-ta.
No TA-Lib dependency — pure Python/pandas.

Features:
  Trend    : EMA 20/50/100/200, spreads, distance
  Momentum : RSI, MACD histogram, ROC, momentum slope
  Volatility: ATR, Bollinger Band width, hist vol, candle range %
  Volume   : relative volume, vol MAs, vol spike flag
  Candle   : wick ratio, body %, bullish/bearish pressure
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import pandas_ta as ta
from config.settings import get_settings
from utils.logger import log


def add_trend_features(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """EMA-based trend features."""
    cfg = cfg or get_settings()
    close = df["close"]

    for p in cfg.ema_periods:
        df[f"ema_{p}"] = ta.ema(close, length=p)

    # EMA spreads
    df["ema_20_50_spread"]   = df["ema_20"]  - df["ema_50"]
    df["ema_50_100_spread"]  = df["ema_50"]  - df["ema_100"]
    df["ema_100_200_spread"] = df["ema_100"] - df["ema_200"]

    # Price distance from EMAs (normalized by ATR)
    atr = ta.atr(df["high"], df["low"], df["close"], length=cfg.atr_period)
    atr = atr.replace(0, np.nan).ffill()
    for p in cfg.ema_periods:
        df[f"dist_ema_{p}"] = (close - df[f"ema_{p}"]) / atr

    # EMA alignment score: count of ema_20 > ema_50 > ema_100 > ema_200
    df["ema_alignment"] = (
        (df["ema_20"] > df["ema_50"]).astype(int) +
        (df["ema_50"] > df["ema_100"]).astype(int) +
        (df["ema_100"] > df["ema_200"]).astype(int)
    ).astype(float)

    # Slope: (current - N bars ago) / ATR
    df["ema_20_slope"] = df["ema_20"].diff(5) / atr
    df["ema_50_slope"] = df["ema_50"].diff(10) / atr

    return df


def add_momentum_features(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """RSI, MACD, ROC, momentum slope."""
    cfg = cfg or get_settings()
    close = df["close"]

    # RSI
    df["rsi"] = ta.rsi(close, length=cfg.rsi_period)
    df["rsi_centered"] = df["rsi"] - 50  # Centered at zero

    # RSI slope
    df["rsi_slope"] = df["rsi"].diff(3)

    # MACD
    macd = ta.macd(close,
                   fast=cfg.macd_fast,
                   slow=cfg.macd_slow,
                   signal=cfg.macd_signal)
    if macd is not None and not macd.empty:
        cols = macd.columns.tolist()
        df["macd_line"]      = macd.iloc[:, 0]
        df["macd_signal"]    = macd.iloc[:, 2]
        df["macd_histogram"] = macd.iloc[:, 1]
        df["macd_hist_slope"] = df["macd_histogram"].diff(3)

    # Rate of Change
    df["roc"] = ta.roc(close, length=cfg.roc_period)

    # Momentum (close - close[N])
    df["momentum_10"] = close.diff(10)
    df["momentum_24"] = close.diff(24)

    # Momentum slope
    df["momentum_slope"] = df["momentum_10"].diff(5)

    # Stochastic RSI
    stochrsi = ta.stochrsi(close, length=cfg.rsi_period)
    if stochrsi is not None and not stochrsi.empty:
        df["stochrsi_k"] = stochrsi.iloc[:, 0]
        df["stochrsi_d"] = stochrsi.iloc[:, 1]

    return df


def add_volatility_features(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """ATR, Bollinger Bands, historical volatility, candle range."""
    cfg = cfg or get_settings()
    close = df["close"]

    # ATR
    df["atr"] = ta.atr(df["high"], df["low"], close, length=cfg.atr_period)
    df["atr_pct"] = df["atr"] / close * 100  # ATR as % of price

    # Normalized ATR (z-score over rolling 100-bar window)
    df["atr_zscore"] = (
        (df["atr"] - df["atr"].rolling(100).mean()) /
        (df["atr"].rolling(100).std() + 1e-8)
    )

    # Bollinger Bands
    bb = ta.bbands(close, length=cfg.bb_period, std=cfg.bb_std)
    if bb is not None and not bb.empty:
        df["bb_upper"]  = bb.iloc[:, 0]
        df["bb_mid"]    = bb.iloc[:, 1]
        df["bb_lower"]  = bb.iloc[:, 2]
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct_b"]  = (close - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-8
        )

    # Historical volatility (annualized log-return std)
    log_ret = np.log(close / close.shift(1))
    df["hist_vol_20"]  = log_ret.rolling(20).std()  * np.sqrt(8760)  # Hourly → annual
    df["hist_vol_50"]  = log_ret.rolling(50).std()  * np.sqrt(8760)

    # Candle range % of price
    df["candle_range_pct"] = (df["high"] - df["low"]) / close * 100

    # True range normalized
    df["true_range"] = ta.true_range(df["high"], df["low"], close)
    df["true_range_pct"] = df["true_range"] / close * 100

    return df


def add_volume_features(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Volume-based features."""
    vol = df["volume"]

    # Volume moving averages
    df["vol_ma_10"]  = vol.rolling(10).mean()
    df["vol_ma_20"]  = vol.rolling(20).mean()
    df["vol_ma_50"]  = vol.rolling(50).mean()

    # Relative volume
    df["rel_vol_10"] = vol / (df["vol_ma_10"] + 1e-8)
    df["rel_vol_20"] = vol / (df["vol_ma_20"] + 1e-8)

    # Volume spike flag (> 2× 20-bar avg)
    df["vol_spike"] = (vol > 2 * df["vol_ma_20"]).astype(float)

    # Volume trend (slope)
    df["vol_slope"] = vol.rolling(10).mean().diff(5) / (df["vol_ma_20"] + 1e-8)

    # Price-volume pressure: close > open with high volume
    df["bullish_vol_pressure"] = ((df["close"] > df["open"]) & (vol > df["vol_ma_20"])).astype(float)
    df["bearish_vol_pressure"] = ((df["close"] < df["open"]) & (vol > df["vol_ma_20"])).astype(float)

    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candle structure features."""
    body  = (df["close"] - df["open"]).abs()
    total = (df["high"] - df["low"]).replace(0, np.nan)
    close = df["close"]

    # Body as % of total range
    df["body_pct"] = body / total

    # Upper wick ratio
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    df["upper_wick_pct"] = upper_wick / (total + 1e-8)

    # Lower wick ratio
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["lower_wick_pct"] = lower_wick / (total + 1e-8)

    # Candle direction: +1 bullish, -1 bearish
    df["candle_dir"] = np.sign(df["close"] - df["open"])

    # Consecutive same-direction candles
    df["consec_bull"] = (
        df["candle_dir"].rolling(5).apply(lambda x: (x == 1).sum(), raw=True)
    )
    df["consec_bear"] = (
        df["candle_dir"].rolling(5).apply(lambda x: (x == -1).sum(), raw=True)
    )

    # Doji flag (body < 10% of range)
    df["doji"] = (df["body_pct"] < 0.10).astype(float)

    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ADX for trend strength."""
    adx = ta.adx(df["high"], df["low"], df["close"], length=period)
    if adx is not None and not adx.empty:
        df["adx"]    = adx.iloc[:, 0]
        df["dmp"]    = adx.iloc[:, 1]
        df["dmn"]    = adx.iloc[:, 2]
    return df


def compute_all_technical(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    Compute all technical features on a copy of df.
    Input: clean OHLCV DataFrame
    Output: DataFrame with all technical columns added
    """
    cfg = cfg or get_settings()
    df = df.copy()
    log.info("Computing technical features …")

    df = add_trend_features(df, cfg)
    df = add_momentum_features(df, cfg)
    df = add_volatility_features(df, cfg)
    df = add_volume_features(df, cfg)
    df = add_candle_features(df)
    df = add_adx(df, period=cfg.regime_adx_period)

    n_added = len(df.columns) - 5  # Original OHLCV = 5
    log.info(f"Technical features computed: {n_added} indicators added")
    return df
