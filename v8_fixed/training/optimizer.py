"""
training/optimizer.py
=====================
Optuna hyperparameter optimization.
Optimizes for OUT-OF-SAMPLE Sharpe ratio (NOT accuracy, NOT in-sample profit).
Uses Purged K-Fold cross-validation within the training window.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage

from config.settings import get_settings, ROOT_DIR
from models.xgboost_model import XGBoostModel
from training.feature_selector import select_features
from utils.logger import log

# Suppress Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── SQLite DB for persistent study storage ────────────────────────────────────
_OPTUNA_DB = ROOT_DIR / ".cache" / "optuna_studies.db"
_OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
_STORAGE_URL = f"sqlite:///{_OPTUNA_DB.as_posix()}"


def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    cfg,
) -> float:
    """
    Optuna objective function.
    Returns: mean Sharpe across Purged K-Fold splits.
    Pruned early if intermediate fold is very poor.
    """
    # ── Hyperparameter search space ──
    params = {
        "n_estimators":     trial.suggest_int("n_estimators",     100,  500, step=50),
        "max_depth":        trial.suggest_int("max_depth",         3,    8),
        "learning_rate":    trial.suggest_float("learning_rate",   0.01, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample",       0.6,  1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,  1.0),
        "min_child_weight": trial.suggest_int("min_child_weight",  3,    30),
        "gamma":            trial.suggest_float("gamma",           0.0,  0.5),
        "reg_alpha":        trial.suggest_float("reg_alpha",       0.0,  1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda",      0.1,  5.0),
    }

    # ── Purged K-Fold cross-validation (lazy import to avoid circular dependency) ──
    from retraining.walk_forward import PurgedKFold
    pkf = PurgedKFold(n_splits=cfg.cv_n_splits, embargo_bars=cfg.purged_embargo_bars)
    fold_sharpes: list = []

    for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

        if len(y_tr) < 200 or len(y_te) < 30:
            continue

        # Feature selection on this fold
        X_tr_sel, X_te_sel, _ = select_features(X_tr, y_tr, X_te)

        model = XGBoostModel(params=params, calibrate=False)
        try:
            model.fit(X_tr_sel, y_tr)
            sharpe = _fold_sharpe(model, X_te_sel, y_te, cfg)
        except Exception as e:
            log.debug(f"Trial {trial.number} fold {fold_idx} failed: {e}")
            sharpe = -5.0

        fold_sharpes.append(sharpe)

        # Pruning: if first fold is terrible, skip rest
        trial.report(np.mean(fold_sharpes), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if not fold_sharpes:
        return -5.0

    mean_sharpe = float(np.mean(fold_sharpes))
    std_sharpe  = float(np.std(fold_sharpes))

    # Penalize inconsistent results (high std across folds)
    adjusted = mean_sharpe - 0.3 * std_sharpe

    log.debug(
        f"Trial {trial.number}: "
        f"mean_sharpe={mean_sharpe:.3f} ± {std_sharpe:.3f} → adj={adjusted:.3f}"
    )
    return adjusted


def _fold_sharpe(model, X_test: pd.DataFrame, y_test: pd.Series, cfg) -> float:
    """Compute proxy Sharpe on a test fold."""
    try:
        probs  = model.predict_proba(X_test)[:, 1]
        rr     = cfg.barrier_atr_mult_tp / cfg.barrier_atr_mult_sl
        preds  = (probs >= 0.52).astype(int)
        n_sig  = preds.sum()
        if n_sig < 5:
            return 0.0
        rets   = np.where(y_test.values == 1, rr, -1.0)
        trade_rets = rets[preds == 1]
        sharpe = trade_rets.mean() / (trade_rets.std() + 1e-8) * np.sqrt(252)
        return float(np.clip(sharpe, -5, 10))
    except Exception:
        return 0.0


def run_optuna_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg=None,
    study_name: str = "xgb_optuna",
) -> Dict:
    """
    Run Optuna to find best XGBoost hyperparameters.
    Studies are persisted to SQLite so they survive shutdown and resume
    automatically — completed trials are skipped, only remaining ones run.

    Returns:
        best_params: Dict of best hyperparameters
    """
    cfg = cfg or get_settings()

    # Each fold gets a unique study name so they don't interfere
    try:
        storage = RDBStorage(
            url=_STORAGE_URL,
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        existing = optuna.get_all_study_names(storage=storage)
        already_done = study_name in existing
    except Exception:
        storage       = None
        already_done  = False

    # Count already-completed trials to skip if resuming
    completed_trials = 0
    if already_done and storage is not None:
        try:
            existing_study = optuna.load_study(study_name=study_name, storage=storage)
            completed_trials = len([
                t for t in existing_study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
        except Exception:
            completed_trials = 0

    remaining = max(0, cfg.optuna_n_trials - completed_trials)

    if remaining == 0:
        log.info(f"Optuna [{study_name}]: all {cfg.optuna_n_trials} trials already done — loading best params.")
        study = optuna.load_study(study_name=study_name, storage=storage)
        best  = study.best_params
        log.info(f"Best params (cached): {best}")
        return {
            "best_params": best,
            "best_value":  study.best_value,
            "n_trials":    len(study.trials),
            "study":       study,
        }

    if completed_trials > 0:
        log.info(
            f"Optuna [{study_name}]: resuming — {completed_trials} trials done, "
            f"{remaining} remaining …"
        )
    else:
        log.info(
            f"Optuna [{study_name}]: starting fresh — {cfg.optuna_n_trials} trials, "
            f"timeout={cfg.optuna_timeout_sec}s …"
        )

    # Create or load the study
    study = optuna.create_study(
        direction=cfg.optuna_direction,
        sampler=TPESampler(seed=cfg.random_seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,   # ← KEY: resume if study already exists
    )

    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, cfg),
        n_trials=remaining,
        timeout=cfg.optuna_timeout_sec,
        n_jobs=1,  # Parallel trials cause issues with XGB n_jobs=-1
        show_progress_bar=True,
    )

    best = study.best_params
    log.info(
        f"Optuna [{study_name}] complete. Best Sharpe = {study.best_value:.4f}\n"
        f"Best params: {best}"
    )

    return {
        "best_params":  best,
        "best_value":   study.best_value,
        "n_trials":     len(study.trials),
        "study":        study,
    }
