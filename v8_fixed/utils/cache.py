"""
utils/cache.py
==============
Disk-based caching using diskcache + pickle fallback.
Expensive ops (data download, feature engineering) are cached
to avoid re-running during iteration.
"""

from __future__ import annotations
import hashlib
import pickle
import functools
from pathlib import Path
from typing import Any, Callable, Optional

from config.settings import get_settings
from utils.logger import log


def _get_cache_dir() -> Path:
    return get_settings().paths["cache"]


def cache_key(*args, **kwargs) -> str:
    """Generate a deterministic cache key from arguments."""
    payload = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(payload.encode()).hexdigest()


def load_cache(key: str) -> Optional[Any]:
    """Load a cached value from disk."""
    cfg = get_settings()
    if not cfg.cache_enabled:
        return None
    path = _get_cache_dir() / f"{key}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                log.debug(f"Cache HIT: {key}")
                return pickle.load(f)
        except Exception as e:
            log.warning(f"Cache read error ({key}): {e}")
    return None


def save_cache(key: str, value: Any) -> None:
    """Save a value to disk cache."""
    cfg = get_settings()
    if not cfg.cache_enabled:
        return
    path = _get_cache_dir() / f"{key}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=5)
        log.debug(f"Cache SAVE: {key}")
    except Exception as e:
        log.warning(f"Cache write error ({key}): {e}")


def cached(func: Callable) -> Callable:
    """
    Decorator: cache function results to disk using argument-based key.
    Usage:
        @cached
        def expensive_computation(x, y):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__module__}.{func.__qualname__}_{cache_key(*args, **kwargs)}"
        result = load_cache(key)
        if result is not None:
            return result
        result = func(*args, **kwargs)
        save_cache(key, result)
        return result
    return wrapper


def clear_cache(prefix: str = "") -> int:
    """Clear all cached files (optionally filtered by prefix). Returns count."""
    cache_dir = _get_cache_dir()
    pattern = f"{prefix}*.pkl" if prefix else "*.pkl"
    files = list(cache_dir.glob(pattern))
    for f in files:
        f.unlink(missing_ok=True)
    log.info(f"Cleared {len(files)} cache files (prefix='{prefix}')")
    return len(files)
