"""
models/lr_model.py
==================
Logistic Regression baseline with same interface.
Good for interpretability and ensemble baseline.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config.settings import get_settings
from utils.logger import log


class LogisticRegressionModel:

    def __init__(self, params: Dict[str, Any] = None):
        cfg = get_settings()
        default = {
            "C":            0.1,
            # penalty='l2' is the default — omitting it avoids sklearn 1.8 FutureWarning
            # about specifying penalty without l1_ratio.
            "solver":       "lbfgs",
            "max_iter":     1000,
            "class_weight": "balanced",
        }
        p = {**default, **(params or {})}
        p["random_state"] = cfg.random_seed

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(**p)),
        ])
        self._feature_names: list = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val=None, y_val=None,
        sample_weight=None,
    ) -> "LogisticRegressionModel":
        self._feature_names = list(X_train.columns)
        fit_kwargs = {}
        if sample_weight is not None:
            # Always coerce to numpy to avoid pandas Index issues
            fit_kwargs["clf__sample_weight"] = np.asarray(sample_weight, dtype=np.float32)
        self._pipeline.fit(X_train.values, y_train.values, **fit_kwargs)
        log.info(f"LogisticRegression trained: {len(X_train):,} samples")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict_proba(X[self._feature_names].values)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def feature_importance(self) -> pd.Series:
        coef = self._pipeline.named_steps["clf"].coef_[0]
        return pd.Series(
            np.abs(coef), index=self._feature_names, name="importance"
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline,      path / "lr_pipeline.pkl")
        joblib.dump(self._feature_names, path / "feature_names.pkl")

    def load(self, path: Path) -> "LogisticRegressionModel":
        path = Path(path)
        self._pipeline      = joblib.load(path / "lr_pipeline.pkl")
        self._feature_names = joblib.load(path / "feature_names.pkl")
        return self
