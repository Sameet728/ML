"""
data/downloader.py
==================
Downloads historical OHLCV data from:
  1. Binance (via CCXT) — primary, no API key needed
  2. yfinance — fallback for BTC and Gold

Handles pagination, rate limits, and de-duplication.
Saves raw CSVs to data/raw/.
"""

from __future__ import annotations
import time
import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from config.settings import get_settings
from utils.logger import log
from utils.cache import cached


# ── CCXT Binance downloader ──────────────────────────────────────────────────

def _ccxt_download(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """
    Download OHLCV from Binance using CCXT public API.
    No API key required for historical data.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("Install ccxt: pip install ccxt")

    exchange = getattr(ccxt, exchange_id)({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    since_ms  = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms    = int(pd.Timestamp(end).timestamp()   * 1000)
    limit     = 1000  # Binance max per request

    tf_ms_map = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
                 "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    tf_ms = tf_ms_map.get(timeframe, 3_600_000)

    all_bars: list = []
    log.info(f"Downloading {symbol} {timeframe} from Binance ({start} → {end}) …")

    current_since = since_ms
    retries = 0
    max_retries = 5

    while current_since < end_ms:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            if not bars:
                break
            all_bars.extend(bars)
            last_ts = bars[-1][0]
            if last_ts >= end_ms or len(bars) < limit:
                break
            current_since = last_ts + tf_ms
            time.sleep(exchange.rateLimit / 1000)
            retries = 0
        except Exception as e:
            retries += 1
            if retries > max_retries:
                log.error(f"Max retries exceeded: {e}")
                break
            wait = 2 ** retries
            log.warning(f"CCXT error (retry {retries}/{max_retries}, wait {wait}s): {e}")
            time.sleep(wait)

    if not all_bars:
        raise ValueError(f"No data returned from CCXT for {symbol}")

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[df.index < pd.Timestamp(end, tz="UTC")]
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.astype(float)

    log.info(f"Downloaded {len(df):,} bars for {symbol} {timeframe}")
    return df


# ── yfinance downloader ──────────────────────────────────────────────────────

def _yfinance_download(
    ticker: str,
    interval: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download OHLCV from yfinance (fallback).
    Handles: BTC-USD, GC=F (Gold), etc.
    Note: yfinance only provides 1h data for 2 years max — use 1d for longer.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    log.info(f"Downloading {ticker} {interval} from yfinance ({start} → {end}) …")

    # yfinance 1h is limited to ~730 days — chunk if needed
    if interval in ("1h", "60m") and (
        pd.Timestamp(end) - pd.Timestamp(start)
    ).days > 700:
        log.warning("yfinance 1h limited to ~2 years. Chunking into 2-year windows …")
        chunks = []
        chunk_start = pd.Timestamp(start)
        chunk_end = pd.Timestamp(end)
        step = pd.DateOffset(days=700)
        while chunk_start < chunk_end:
            cs = chunk_start.strftime("%Y-%m-%d")
            ce = min(chunk_start + step, chunk_end).strftime("%Y-%m-%d")
            try:
                df_chunk = yf.download(ticker, start=cs, end=ce, interval=interval, progress=False, auto_adjust=True)
                if not df_chunk.empty:
                    chunks.append(df_chunk)
            except Exception as e:
                log.warning(f"yfinance chunk error ({cs}→{ce}): {e}")
            chunk_start += step
        if not chunks:
            raise ValueError(f"No data from yfinance for {ticker}")
        df = pd.concat(chunks)
    else:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data from yfinance for {ticker}")

    # Normalize columns
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    col_map = {"adj close": "close", "open": "open", "high": "high", "low": "low",
               "close": "close", "volume": "volume"}
    df = df.rename(columns=col_map)
    df = df[["open", "high", "low", "close", "volume"]].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index().astype(float)
    df = df.dropna()

    log.info(f"Downloaded {len(df):,} bars for {ticker} {interval}")
    return df


# ── Public API ───────────────────────────────────────────────────────────────

def download_btc(
    timeframe: str = None,
    start: str = None,
    end: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download BTC/USDT data. Tries CCXT first, yfinance fallback."""
    cfg = get_settings()
    timeframe = timeframe or cfg.primary_timeframe
    start     = start     or cfg.start_date
    end       = end       or cfg.end_date

    raw_path = cfg.paths["data_raw"] / f"BTC_USDT_{timeframe}_{start}_{end}.parquet"

    if raw_path.exists() and not force_refresh:
        log.info(f"Loading BTC from cache: {raw_path.name}")
        return pd.read_parquet(raw_path)

    # Try CCXT
    try:
        df = _ccxt_download(cfg.primary_symbol, timeframe, start, end)
    except Exception as e:
        log.warning(f"CCXT failed ({e}). Falling back to yfinance …")
        yf_interval = "1h" if timeframe == "1h" else "1d"
        df = _yfinance_download("BTC-USD", yf_interval, start, end)

    df.to_parquet(raw_path)
    log.info(f"Saved BTC raw data → {raw_path}")
    return df


def download_btc_4h(
    start: str = None,
    end: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download BTC 4H data for higher-timeframe trend filter."""
    cfg = get_settings()
    start = start or cfg.start_date
    end   = end   or cfg.end_date

    raw_path = cfg.paths["data_raw"] / f"BTC_USDT_4h_{start}_{end}.parquet"

    if raw_path.exists() and not force_refresh:
        log.info(f"Loading BTC 4H from cache: {raw_path.name}")
        return pd.read_parquet(raw_path)

    try:
        df = _ccxt_download(cfg.primary_symbol, "4h", start, end)
    except Exception as e:
        log.warning(f"CCXT 4H failed ({e}). Falling back to yfinance 1h aggregated …")
        df_1h = _yfinance_download("BTC-USD", "1h", start, end)
        df = df_1h.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()

    df.to_parquet(raw_path)
    log.info(f"Saved BTC 4H raw data → {raw_path}")
    return df


def download_gold(
    start: str = None,
    end: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download Gold (XAUUSD) daily data via yfinance."""
    cfg = get_settings()
    start = start or cfg.start_date
    end   = end   or cfg.end_date

    raw_path = cfg.paths["data_raw"] / f"XAUUSD_1d_{start}_{end}.parquet"

    if raw_path.exists() and not force_refresh:
        log.info(f"Loading Gold from cache: {raw_path.name}")
        return pd.read_parquet(raw_path)

    df = _yfinance_download(cfg.optional_symbol_yf, "1d", start, end)
    df.to_parquet(raw_path)
    log.info(f"Saved Gold raw data → {raw_path}")
    return df


def download_all(force_refresh: bool = False) -> dict:
    """Download all required datasets and return as dict."""
    cfg = get_settings()
    result = {
        "btc_1h":  download_btc(force_refresh=force_refresh),
        "btc_4h":  download_btc_4h(force_refresh=force_refresh),
    }
    if cfg.enable_gold:
        result["gold_1d"] = download_gold(force_refresh=force_refresh)
    return result
