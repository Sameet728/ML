"""
utils/validators.py
===================
Data integrity checks throughout the pipeline.
Runs before any model training to catch leakage/NaN issues early.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from utils.logger import log


class ValidationError(Exception):
    pass


def check_no_nan(df: pd.DataFrame, name: str = "DataFrame", raise_on_fail: bool = True) -> bool:
    """Ensure no NaN values in any column."""
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        msg = f"[{name}] NaN found in columns: {nan_cols}"
        if raise_on_fail:
            raise ValidationError(msg)
        log.warning(msg)
        return False
    log.debug(f"[{name}] NaN check passed.")
    return True


def check_timestamp_continuity(
    df: pd.DataFrame,
    freq: str = "1h",
    name: str = "DataFrame",
    max_gap_hours: int = 48,
) -> Tuple[bool, int]:
    """
    Verify timestamp continuity.
    Returns (passed, gap_count).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        log.warning(f"[{name}] Index is not DatetimeIndex — skipping continuity check.")
        return True, 0

    expected_delta = pd.tseries.frequencies.to_offset(freq)
    diffs = df.index.to_series().diff().dropna()
    large_gaps = diffs[diffs > pd.Timedelta(hours=max_gap_hours)]
    gap_count = len(large_gaps)

    if gap_count > 0:
        log.warning(
            f"[{name}] {gap_count} gaps > {max_gap_hours}h detected:\n"
            f"{large_gaps.head(5).to_string()}"
        )
        return False, gap_count

    log.debug(f"[{name}] Timestamp continuity check passed.")
    return True, 0


def check_label_balance(
    labels: pd.Series,
    name: str = "Labels",
    min_positive_pct: float = 0.1,
    max_positive_pct: float = 0.9,
) -> bool:
    """Warn if class imbalance is extreme (< 10% or > 90% positive)."""
    pos_pct = labels.mean()
    msg = f"[{name}] Label balance: {pos_pct:.1%} positive ({labels.sum()}/{len(labels)})"
    log.info(msg)

    if pos_pct < min_positive_pct or pos_pct > max_positive_pct:
        log.warning(
            f"[{name}] Extreme imbalance detected. "
            f"Consider adjusting barrier parameters or class weights."
        )
        return False
    return True


def check_no_future_leakage(
    features: pd.DataFrame,
    labels: pd.Series,
    name: str = "Dataset",
) -> bool:
    """
    Sanity check: features must be entirely BEFORE the label date.
    Checks that the feature DataFrame and label Series share the same index.
    """
    if not features.index.equals(labels.index):
        log.warning(f"[{name}] Feature/label index mismatch!")
        return False
    log.debug(f"[{name}] Leakage check passed.")
    return True


def check_feature_variance(
    df: pd.DataFrame,
    threshold: float = 1e-8,
    name: str = "Features",
) -> List[str]:
    """Return list of near-zero variance columns."""
    low_var = [c for c in df.columns if df[c].var() < threshold]
    if low_var:
        log.warning(f"[{name}] Near-zero variance columns: {low_var}")
    else:
        log.debug(f"[{name}] Variance check passed.")
    return low_var


def validate_ohlcv(df: pd.DataFrame, name: str = "OHLCV") -> bool:
    """Basic OHLCV sanity: H >= L, H >= O/C, L <= O/C, V >= 0."""
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(f"[{name}] Missing OHLCV columns: {missing}")

    bad_hl  = (df["high"] < df["low"]).sum()
    bad_hc  = (df["high"] < df["close"]).sum()
    bad_lc  = (df["low"]  > df["close"]).sum()
    bad_vol = (df["volume"] < 0).sum()

    issues = bad_hl + bad_hc + bad_lc + bad_vol
    if issues > 0:
        log.warning(
            f"[{name}] OHLCV integrity issues: H<L={bad_hl}, H<C={bad_hc}, "
            f"L>C={bad_lc}, V<0={bad_vol}"
        )
        return False

    log.debug(f"[{name}] OHLCV integrity check passed.")
    return True


def run_all_checks(
    df: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    freq: str = "1h",
    name: str = "Pipeline",
) -> bool:
    """Run all validation checks. Returns True if all pass."""
    passed = True

    passed &= validate_ohlcv(df, name=name)
    ts_ok, _ = check_timestamp_continuity(df, freq=freq, name=name)
    passed &= ts_ok

    if labels is not None:
        passed &= check_no_future_leakage(df, labels, name=name)
        passed &= check_label_balance(labels, name=name)

    log.info(f"[{name}] Validation {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed
