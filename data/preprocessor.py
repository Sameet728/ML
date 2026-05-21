"""
data/preprocessor.py
====================
Cleans, validates, and aligns raw OHLCV data.
Produces clean DataFrames ready for feature engineering.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from config.settings import get_settings
from utils.logger import log
from utils.validators import validate_ohlcv, check_timestamp_continuity


def clean_ohlcv(df: pd.DataFrame, name: str = "OHLCV") -> pd.DataFrame:
    """
    Clean raw OHLCV DataFrame:
    - Normalize column names to lowercase
    - Ensure UTC DatetimeIndex
    - Remove duplicates
    - Forward-fill small gaps (≤ 3 bars)
    - Drop rows with zero volume or zero prices
    - Cap extreme outliers (> 10σ)
    """
    df = df.copy()

    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    col_aliases = {
        "timestamp": "timestamp", "time": "timestamp",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        "adj close": "close", "adj_close": "close",
    }
    df = df.rename(columns=col_aliases)

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"[{name}] Missing column: {col}")

    df = df[required].astype(float)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    # Deduplicate & sort
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Drop zero-volume or zero-price rows
    df = df[(df["volume"] > 0) & (df["close"] > 0)]

    # Forward-fill small gaps
    df = df.ffill(limit=3)

    # Drop remaining NaNs
    df = df.dropna()

    # Cap extreme close outliers (> 10σ rolling)
    roll = df["close"].rolling(200, min_periods=50)
    mean, std = roll.mean(), roll.std()
    df = df[(df["close"] - mean).abs() <= 10 * std]

    log.info(f"[{name}] Cleaned: {len(df):,} bars ({df.index[0]} → {df.index[-1]})")
    return df


def align_timeframes(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align 1H and 4H DataFrames so that each 1H bar has the
    corresponding 4H context. 4H values are forward-filled into 1H index.

    Returns: (df_1h_aligned, df_4h_on_1h_index)
    """
    # Rename 4H columns to avoid collision
    df_4h_renamed = df_4h.rename(columns={
        c: f"{c}_4h" for c in df_4h.columns
    })

    # Reindex 4H to 1H index using forward fill (no future leakage)
    df_4h_on_1h = df_4h_renamed.reindex(df_1h.index, method="ffill")

    # Ensure date ranges overlap
    common_start = max(df_1h.index[0], df_4h_on_1h.index[0])
    df_1h_aligned = df_1h.loc[common_start:]
    df_4h_on_1h   = df_4h_on_1h.loc[common_start:]

    log.info(
        f"Aligned 1H ({len(df_1h_aligned):,} bars) with 4H → "
        f"{df_4h_on_1h.notna().all(axis=1).sum():,} fully merged rows"
    )
    return df_1h_aligned, df_4h_on_1h


def split_dataset(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / validation / test segments.
    Args:
        df: Full DataFrame
        train_end: 'YYYY-MM-DD' end of training period
        val_end:   'YYYY-MM-DD' end of validation period
    Returns: (train_df, val_df, test_df)
    """
    train = df.loc[: pd.Timestamp(train_end, tz="UTC")]
    val   = df.loc[pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(hours=1)
                   : pd.Timestamp(val_end, tz="UTC")]
    test  = df.loc[pd.Timestamp(val_end, tz="UTC") + pd.Timedelta(hours=1):]

    log.info(
        f"Split → Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}"
    )
    return train, val, test


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV to daily for reporting purposes."""
    daily = df.resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return daily


def preprocess_all(
    raw_data: dict,
    align_htf: bool = True,
) -> dict:
    """
    Full preprocessing pipeline.
    Input: raw_data dict from downloader.download_all()
    Output: dict with cleaned, aligned DataFrames
    """
    result = {}

    # Clean BTC 1H
    btc_1h = clean_ohlcv(raw_data["btc_1h"], name="BTC_1H")
    validate_ohlcv(btc_1h, name="BTC_1H")

    # Clean BTC 4H
    btc_4h = clean_ohlcv(raw_data["btc_4h"], name="BTC_4H")
    validate_ohlcv(btc_4h, name="BTC_4H")

    # Align
    if align_htf:
        btc_1h, btc_4h_aligned = align_timeframes(btc_1h, btc_4h)
        result["btc_4h_on_1h"] = btc_4h_aligned
    else:
        result["btc_4h_on_1h"] = None

    result["btc_1h"] = btc_1h
    result["btc_4h"] = btc_4h

    # Gold (optional)
    if "gold_1d" in raw_data:
        gold = clean_ohlcv(raw_data["gold_1d"], name="GOLD_1D")
        validate_ohlcv(gold, name="GOLD_1D")
        result["gold_1d"] = gold
        log.info(f"Gold data ready: {len(gold):,} daily bars")

    # Save processed
    cfg = get_settings()
    proc_dir = cfg.paths["data_proc"]
    result["btc_1h"].to_parquet(proc_dir / "btc_1h_clean.parquet")
    result["btc_4h"].to_parquet(proc_dir / "btc_4h_clean.parquet")
    if result.get("btc_4h_on_1h") is not None:
        result["btc_4h_on_1h"].to_parquet(proc_dir / "btc_4h_on_1h.parquet")
    if "gold_1d" in result:
        result["gold_1d"].to_parquet(proc_dir / "gold_1d_clean.parquet")

    log.info("Preprocessing complete. All files saved to data/processed/")
    return result


def load_processed(name: str) -> pd.DataFrame:
    """Load a previously processed parquet file."""
    cfg = get_settings()
    path = cfg.paths["data_proc"] / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_parquet(path)
