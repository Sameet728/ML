"""
models/ensemble.py
==================
Soft-voting ensemble of XGBoost + RF + LR.
Weights are determined by validation Sharpe ratio.
Also implements probability threshold tuning.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import joblib
from config.settings import get_settings
from utils.logger import log


class EnsembleModel:
    """
    Weighted soft-voting ensemble.
    Members: XGBoost (primary), Random Forest, Logistic Regression.
    Weights: proportional to validation Sharpe ratio of each model.
    """

    def __init__(self, models: List = None, weights: List[float] = None):
        self._models  = models or []
        self._weights = weights
        self._feature_names: list = []

    def add_model(self, model, weight: float = 1.0) -> None:
        self._models.append(model)
        if self._weights is None:
            self._weights = [weight]
        else:
            self._weights.append(weight)

    def set_weights_from_sharpe(self, sharpe_scores: Dict[str, float]) -> None:
        """
        Set ensemble weights proportional to Sharpe ratios.
        Negative Sharpe → weight = 0 (exclude bad models).
        """
        model_names = list(sharpe_scores.keys())
        raw_weights = np.array([max(s, 0) for s in sharpe_scores.values()])
        total = raw_weights.sum()
        if total == 0:
            self._weights = [1.0 / len(self._models)] * len(self._models)
        else:
            self._weights = (raw_weights / total).tolist()
        log.info(f"Ensemble weights: {dict(zip(model_names, self._weights))}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of model probabilities."""
        if not self._models:
            raise ValueError("No models in ensemble")

        weights = self._weights or [1.0] * len(self._models)
        total_w = sum(weights)

        proba = np.zeros((len(X), 2))
        for model, w in zip(self._models, weights):
            try:
                proba += (w / total_w) * model.predict_proba(X)
            except Exception as e:
                log.warning(f"Model predict_proba failed: {e}")

        return proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        returns: pd.Series = None,
        thresholds: np.ndarray = None,
    ) -> float:
        """
        Find optimal probability threshold on validation set.
        Optimizes for Sharpe (if returns provided) or F1.
        """
        if thresholds is None:
            thresholds = np.arange(0.45, 0.80, 0.01)

        probs = self.predict_proba(X_val)[:, 1]
        best_thresh, best_score = 0.5, -np.inf

        for t in thresholds:
            preds = (probs >= t).astype(int)
            n_trades = preds.sum()
            if n_trades < 10:
                continue

            if returns is not None:
                trade_rets = returns.values[preds == 1]
                if len(trade_rets) < 5:
                    continue
                sharpe = trade_rets.mean() / (trade_rets.std() + 1e-8) * np.sqrt(252)
                score = sharpe
            else:
                from sklearn.metrics import f1_score
                score = f1_score(y_val, preds, zero_division=0)

            if score > best_score:
                best_score, best_thresh = score, t

        log.info(f"Optimal threshold: {best_thresh:.2f} (score={best_score:.4f})")
        return best_thresh

    def feature_importance(self, method: str = "mean") -> pd.Series:
        """Aggregate feature importance across models."""
        imps = []
        for model in self._models:
            try:
                imp = model.feature_importance()
                if isinstance(imp, pd.Series):
                    imps.append(imp)
            except Exception:
                pass
        if not imps:
            return pd.Series(dtype=float)

        df = pd.concat(imps, axis=1).fillna(0)
        return df.mean(axis=1).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._weights, path / "ensemble_weights.pkl")
        for i, m in enumerate(self._models):
            model_name = type(m).__name__.lower()
            try:
                m.save(path / f"{model_name}_{i}")
            except Exception as e:
                log.warning(f"Could not save model {i}: {e}")
        log.info(f"Ensemble saved → {path}")

    def load_weights(self, path: Path) -> None:
        self._weights = joblib.load(Path(path) / "ensemble_weights.pkl")
