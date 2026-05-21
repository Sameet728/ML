"""
features/time_features.py
==========================
Calendar and session-based features.

Features:
  - Hour of day (0–23)
  - Day of week (0=Monday, 6=Sunday)
  - Is weekend flag
  - Trading session: Asia / London / NY / Overlap
  - London-NY overlap flag
  - Month, quarter
  - Days since epoch (trend proxy)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from utils.logger import log


# ── Session definitions (UTC hours) ─────────────────────────────────────────
# Tokyo:   00:00 – 09:00 UTC
# London:  08:00 – 17:00 UTC
# New York: 13:00 – 22:00 UTC
# Overlap: 13:00 – 17:00 UTC

SESSION_MAP = {
    "asia":    (0,  8),
    "london":  (8,  17),
    "ny":      (13, 22),
    "overlap": (13, 17),   # London + NY overlap — most liquid
}


def _in_session(hour: int, session: str) -> int:
    start, end = SESSION_MAP[session]
    return int(start <= hour < end)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all time-based features to DataFrame."""
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        log.warning("Index is not DatetimeIndex — time features may be incorrect")

    idx = df.index

    # Basic time components
    df["hour"]    = idx.hour.astype(float)
    df["dow"]     = idx.dayofweek.astype(float)   # 0=Mon, 6=Sun
    df["month"]   = idx.month.astype(float)
    df["quarter"] = idx.quarter.astype(float)
    df["week"]    = idx.isocalendar().week.astype(float)

    # Is weekend
    df["is_weekend"] = (idx.dayofweek >= 5).astype(float)

    # Crypto never closes but weekends behave differently
    df["is_weekday"] = (idx.dayofweek < 5).astype(float)

    # Trading session flags
    hours = idx.hour
    df["session_asia"]    = hours.map(lambda h: _in_session(h, "asia")).astype(float)
    df["session_london"]  = hours.map(lambda h: _in_session(h, "london")).astype(float)
    df["session_ny"]      = hours.map(lambda h: _in_session(h, "ny")).astype(float)
    df["session_overlap"] = hours.map(lambda h: _in_session(h, "overlap")).astype(float)

    # Cyclic encoding of hour (sin/cos) — avoids discontinuity at 0/23
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Cyclic encoding of day of week
    dow = idx.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Cyclic encoding of month
    df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)

    # Days since start of dataset (gentle time trend proxy)
    df["days_elapsed"] = (idx - idx[0]).total_seconds() / 86400

    log.debug(f"Time features added: {len(df.columns)} total columns")
    return df
