"""
training/feature_selector.py
=============================
Feature selection pipeline:
  1. Remove near-zero variance features
  2. Remove highly correlated pairs (keep higher-importance one)
  3. XGBoost-based importance filter
  4. Returns selected feature names
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple

from xgboost import XGBClassifier
from config.settings import get_settings
from utils.logger import log


def remove_low_variance(
    X: pd.DataFrame,
    threshold: float = 1e-8,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove features with near-zero variance."""
    low_var = [c for c in X.columns if X[c].var() < threshold]
    if low_var:
        log.debug(f"Removing {len(low_var)} low-variance features: {low_var[:5]} …")
        X = X.drop(columns=low_var)
    return X, low_var


def remove_correlated(
    X: pd.DataFrame,
    threshold: float = 0.95,
    importance: pd.Series = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features.
    When two features have corr > threshold, keep the one with higher importance.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > threshold].tolist()
        for partner in partners:
            if importance is not None:
                # Drop the one with lower importance
                if importance.get(col, 0) >= importance.get(partner, 0):
                    to_drop.add(partner)
                else:
                    to_drop.add(col)
            else:
                to_drop.add(col)  # Drop first encountered

    to_drop = list(to_drop)
    if to_drop:
        log.debug(f"Removing {len(to_drop)} correlated features (threshold={threshold})")
        X = X.drop(columns=to_drop)
    return X, to_drop


def xgb_importance_filter(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = None,
    min_importance: float = 0.001,
    cfg=None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Quick XGBoost fit to rank features.
    Removes features with importance below threshold.
    """
    cfg = cfg or get_settings()

    quick_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        verbosity=0,
        random_state=cfg.random_seed,
        n_jobs=cfg.n_jobs,
    )

    try:
        quick_model.fit(X.values, y.values, verbose=False)
        scores = quick_model.get_booster().get_score(importance_type="gain")
        imp = pd.Series(scores, name="importance").reindex(X.columns).fillna(0)
    except Exception as e:
        log.warning(f"XGB importance filter failed: {e}. Skipping.")
        return X, pd.Series(1.0, index=X.columns)

    # Normalize
    total = imp.sum()
    if total > 0:
        imp = imp / total

    # Filter low importance
    keep_mask = imp >= min_importance
    if top_n:
        keep_mask = imp.rank(ascending=False) <= top_n

    dropped = imp[~keep_mask].index.tolist()
    if dropped:
        log.debug(f"Dropping {len(dropped)} low-importance features")

    return X[imp[keep_mask].index], imp


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame = None,
    top_n: int = 80,
    cfg=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Full feature selection pipeline.

    Args:
        X_train : Training features
        y_train : Training labels
        X_test  : Optional test set (same features selected)
        top_n   : Maximum features to keep

    Returns:
        (X_train_sel, X_test_sel, selected_feature_names)
    """
    cfg = cfg or get_settings()
    log.info(f"Feature selection: starting with {X_train.shape[1]} features …")

    # Step 1: Low variance
    X_train, _ = remove_low_variance(X_train)

    # Step 2: Quick importance for correlation tie-breaking
    _, imp = xgb_importance_filter(X_train, y_train, top_n=None, min_importance=0)

    # Step 3: Remove high correlation
    X_train, _ = remove_correlated(X_train, threshold=0.95, importance=imp)

    # Step 4: Importance filter
    X_train, _ = xgb_importance_filter(X_train, y_train, top_n=top_n, min_importance=0.001)

    selected = list(X_train.columns)
    log.info(f"Feature selection complete: {len(selected)} features retained")

    # Apply same selection to test set
    if X_test is not None:
        avail = [c for c in selected if c in X_test.columns]
        missing = set(selected) - set(avail)
        if missing:
            log.warning(f"Test set missing {len(missing)} features — filling with 0")
        X_test_sel = X_test.reindex(columns=selected, fill_value=0)
        return X_train, X_test_sel, selected

    return X_train, None, selected
