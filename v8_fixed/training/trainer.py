"""
training/trainer.py
====================
Training pipeline: builds models, runs Purged K-Fold CV,
evaluates on validation, returns predictions + metrics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from config.settings import get_settings
from models.xgboost_model import XGBoostModel
from models.rf_model import RandomForestModel
from models.lr_model import LogisticRegressionModel
from models.ann_model import ANNModel
from models.ensemble import EnsembleModel
from models.meta_label import MetaLabelModel
from training.feature_selector import select_features
from utils.logger import log

# NOTE: training.optimizer is NOT imported at module level to avoid
# circular imports (optimizer → walk_forward → trainer). Import lazily.


def compute_sample_weights(
    y: pd.Series,
    decay_days: int = 90,
    timestamps: pd.DatetimeIndex = None,
) -> np.ndarray:
    """
    Time-decay sample weights: more recent samples get higher weight.
    Reduces influence of stale market regimes.

    NOTE: Explicitly converts to numpy to handle pandas 2.x where
    TimedeltaIndex.total_seconds() returns a pandas Index, not ndarray.
    """
    n = len(y)
    if timestamps is not None and len(timestamps) > 0:
        # pandas 2.x: .total_seconds() on TimedeltaIndex returns an Index,
        # not a numpy array — must call .to_numpy() explicitly.
        td = timestamps[-1] - timestamps
        days_ago = np.asarray(td.total_seconds(), dtype=np.float64) / 86400.0
        weights  = np.exp(-days_ago / max(decay_days, 1))
    else:
        weights = np.exp(-np.linspace(1, 0, n))

    # Normalize — weights is guaranteed numpy here
    total = weights.sum()
    if total > 0:
        weights = weights / total * n
    return weights.astype(np.float32)


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg=None,
    xgb_params: dict = None,
) -> Tuple[EnsembleModel, MetaLabelModel, Dict[str, float]]:
    """
    Train XGBoost + RF + LR, build ensemble, train meta-label model.

    Returns:
        ensemble   : Trained EnsembleModel
        meta_model : Trained MetaLabelModel
        sharpe_dict: Per-model validation Sharpe
    """
    cfg = cfg or get_settings()

    # ── 1. Feature selection ──
    X_train_sel, X_val_sel, selected_features = select_features(X_train, y_train, X_val)
    log.info(f"Features after selection: {len(selected_features)} / {len(X_train.columns)}")

    # ── 2. Sample weights ──
    weights = compute_sample_weights(y_train, timestamps=X_train.index)

    models   = []
    sharpes  = {}

    # ── 3. XGBoost ──
    log.info("Training XGBoost …")
    xgb = XGBoostModel(params=xgb_params)
    xgb.fit(X_train_sel, y_train, X_val_sel, y_val, sample_weight=weights)
    models.append(xgb)
    sharpes["xgboost"] = _quick_val_sharpe(xgb, X_val_sel, y_val)

    # ── 4. Random Forest ──
    if cfg.enable_random_forest:
        log.info("Training RandomForest …")
        rf = RandomForestModel()
        rf.fit(X_train_sel, y_train, X_val_sel, y_val)
        models.append(rf)
        sharpes["random_forest"] = _quick_val_sharpe(rf, X_val_sel, y_val)

    # ── 5. Logistic Regression ──
    if cfg.enable_logistic_reg:
        log.info("Training LogisticRegression …")
        lr = LogisticRegressionModel()
        lr.fit(X_train_sel, y_train)
        models.append(lr)
        sharpes["logistic_reg"] = _quick_val_sharpe(lr, X_val_sel, y_val)

    # ── 5.5 Artificial Neural Network (ANN) ──
    if cfg.enable_ann:
        log.info("Training ANN …")
        ann = ANNModel()
        ann.fit(X_train_sel, y_train)
        models.append(ann)
        sharpes["ann"] = _quick_val_sharpe(ann, X_val_sel, y_val)

    # ── 6. Build ensemble ──
    ensemble = EnsembleModel(models=models)
    if cfg.enable_ensemble:
        ensemble.set_weights_from_sharpe(sharpes)
    else:
        ensemble.set_weights_from_sharpe({k: 1.0 for k in sharpes})

    log.info(f"Ensemble Sharpe (val): {_quick_val_sharpe(ensemble, X_val_sel, y_val):.4f}")

    # ── 7. Meta-label model ──
    meta_model = None
    if cfg.enable_meta_labeling:
        log.info("Training Meta-label model …")
        primary_probs_train = pd.Series(
            ensemble.predict_proba(X_train_sel)[:, 1],
            index=X_train_sel.index,
        )
        meta_model = MetaLabelModel()
        meta_model.fit(X_train_sel, y_train, primary_probs_train)

    # ── 8. Log summary ──
    for name, s in sharpes.items():
        log.info(f"  {name: <20} val_sharpe = {s:.4f}")

    return ensemble, meta_model, sharpes, selected_features


def _quick_val_sharpe(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Quick proxy Sharpe: simulates trade returns on val set.
    Uses predicted prob > 0.5 as signal, actual label as realized P&L proxy.
    """
    try:
        probs = model.predict_proba(X_val)[:, 1]
        # Simulate: if signal & correct → +1, if signal & wrong → -1.5 (SL hit)
        # Ratio = TP_mult / SL_mult = 2.0/1.0 from default settings
        cfg = get_settings()
        rr = cfg.barrier_atr_mult_tp / cfg.barrier_atr_mult_sl

        preds = (probs >= 0.5).astype(int)
        rets  = np.where(
            preds == 1,
            np.where(y_val.values == 1, rr, -1.0),
            0.0,
        )
        n_trades = (preds == 1).sum()
        if n_trades < 5:
            return 0.0
        trade_rets = rets[preds == 1]
        sharpe = trade_rets.mean() / (trade_rets.std() + 1e-8) * np.sqrt(252)
        return float(np.clip(sharpe, -5, 10))
    except Exception as e:
        log.warning(f"Sharpe computation failed: {e}")
        return 0.0


def evaluate_predictions(
    y_true: pd.Series,
    probs: pd.Series,
    threshold: float = 0.5,
    name: str = "Model",
) -> Dict:
    """Compute classification metrics for evaluation."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    preds = (probs >= threshold).astype(int)
    metrics = {
        "accuracy":  float(accuracy_score(y_true, preds)),
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall":    float(recall_score(y_true, preds, zero_division=0)),
        "auc":       float(roc_auc_score(y_true, probs)) if len(y_true.unique()) > 1 else 0.5,
        "n_signals": int(preds.sum()),
        "threshold": threshold,
    }

    log.info(
        f"[{name}] acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
        f"auc={metrics['auc']:.3f} signals={metrics['n_signals']:,}"
    )
    return metrics
