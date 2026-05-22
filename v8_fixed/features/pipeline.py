"""
features/pipeline.py
====================
Orchestrates the full feature engineering pipeline.
Returns a clean feature matrix X and raw OHLCV aligned.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from config.settings import get_settings
from features.technical   import compute_all_technical
from features.time_features import add_time_features
from features.advanced    import compute_all_advanced
from utils.logger import log
from utils.validators import check_no_nan, check_feature_variance


# Columns to ALWAYS exclude from feature matrix (OHLCV + meta)
ALWAYS_EXCLUDE = [
    "open", "high", "low", "close", "volume",
    # Raw 4H OHLCV (keep processed 4H features only)
    "open_4h", "high_4h", "low_4h", "close_4h", "volume_4h",
]


def build_feature_matrix(
    df: pd.DataFrame,
    df_4h_on_1h: Optional[pd.DataFrame] = None,
    cfg=None,
    drop_na: bool = True,
    return_feature_names: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feature engineering pipeline.

    Args:
        df           : Clean 1H OHLCV DataFrame
        df_4h_on_1h  : 4H OHLCV reindexed to 1H (from preprocessor)
        cfg          : Settings (uses global if None)
        drop_na      : Drop rows with any NaN feature
        return_feature_names: If True, also return feature names list

    Returns:
        (X, ohlcv)   : Feature matrix and aligned OHLCV
    """
    cfg = cfg or get_settings()

    log.info("Starting feature pipeline …")

    # Step 1: Technical indicators
    df_feat = compute_all_technical(df, cfg)

    # Step 2: If 4H data available, compute 4H technicals too
    if df_4h_on_1h is not None and not df_4h_on_1h.empty:
        # Build a synthetic 4H OHLCV from the aligned 4H columns
        df_4h_cols = df_4h_on_1h.copy()
        # Rename back to OHLCV for technical computation
        rename_back = {f"{c}_4h": c for c in ["open", "high", "low", "close", "volume"]
                       if f"{c}_4h" in df_4h_cols.columns}
        if len(rename_back) == 5:
            df_4h_ohlcv = df_4h_cols.rename(columns=rename_back)[list(rename_back.values())]
            df_4h_tech  = compute_all_technical(df_4h_ohlcv, cfg)
            # Rename and merge key 4H technicals
            key_4h = ["ema_20", "ema_50", "ema_200", "rsi", "atr", "atr_pct",
                      "bb_width", "adx", "trend_strength"]
            for col in key_4h:
                if col in df_4h_tech.columns:
                    df_feat[f"htf_{col}"] = df_4h_tech[col].reindex(df_feat.index, method="ffill")

    # Step 3: Time features
    df_feat = add_time_features(df_feat)

    # Step 4: Advanced features
    df_feat = compute_all_advanced(df_feat, df_4h_on_1h, cfg)

    # Step 5: Separate OHLCV from features
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    ohlcv = df_feat[ohlcv_cols].copy()

    # Build feature matrix (exclude OHLCV and meta cols)
    exclude = set(ALWAYS_EXCLUDE)
    feature_cols = [c for c in df_feat.columns if c not in exclude]
    X = df_feat[feature_cols].copy()

    # Step 6: Replace inf values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Step 7: Drop low-variance features
    low_var = check_feature_variance(X, threshold=1e-10, name="FeatureMatrix")
    X = X.drop(columns=low_var)

    # Step 8: Drop NaN rows
    if drop_na:
        valid_mask = X.notna().all(axis=1)
        n_dropped  = (~valid_mask).sum()
        if n_dropped > 0:
            log.info(f"Dropped {n_dropped:,} rows with NaN features (warmup period)")
        X    = X[valid_mask]
        ohlcv = ohlcv[valid_mask]

    log.info(
        f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features "
        f"({X.index[0]} → {X.index[-1]})"
    )

    return X, ohlcv


def get_feature_names(X: pd.DataFrame) -> List[str]:
    """Return sorted list of feature column names."""
    return sorted(X.columns.tolist())


def save_feature_matrix(X: pd.DataFrame, ohlcv: pd.DataFrame) -> None:
    """Persist feature matrix to parquet."""
    cfg = get_settings()
    proc = cfg.paths["data_proc"]
    X.to_parquet(proc / "features.parquet")
    ohlcv.to_parquet(proc / "ohlcv_aligned.parquet")
    log.info(f"Feature matrix saved → {proc / 'features.parquet'}")


def load_feature_matrix() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load persisted feature matrix."""
    cfg = get_settings()
    proc = cfg.paths["data_proc"]
    X     = pd.read_parquet(proc / "features.parquet")
    ohlcv = pd.read_parquet(proc / "ohlcv_aligned.parquet")
    log.info(f"Feature matrix loaded: {X.shape}")
    return X, ohlcv
