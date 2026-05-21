"""
models/rf_model.py
==================
Random Forest classifier wrapper with same interface as XGBoostModel.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import sklearn as _sklearn
_SK_MAJOR, _SK_MINOR = [int(x) for x in _sklearn.__version__.split(".")[:2]]

from config.settings import get_settings
from utils.logger import log


DEFAULT_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth":    10,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs":       -1,
}


class RandomForestModel:

    def __init__(self, params: Dict[str, Any] = None, calibrate: bool = True):
        cfg = get_settings()
        self.params    = {**DEFAULT_RF_PARAMS, **(params or {})}
        self.params["random_state"] = cfg.random_seed
        self.calibrate = calibrate
        self._model    = None
        self._calibrated = None
        self._feature_names: list = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series    = None,
        sample_weight: np.ndarray = None,
    ) -> "RandomForestModel":
        self._feature_names = list(X_train.columns)
        self._model = RandomForestClassifier(**self.params)
        sw = np.asarray(sample_weight, dtype=np.float32) if sample_weight is not None else None
        self._model.fit(X_train.values, y_train.values, sample_weight=sw)

        if self.calibrate and X_val is not None and len(y_val) > 50:
            try:
                if _SK_MAJOR > 1 or (_SK_MAJOR == 1 and _SK_MINOR >= 2):
                    cal = CalibratedClassifierCV(self._model, method="isotonic")
                else:
                    cal = CalibratedClassifierCV(self._model, method="isotonic", cv="prefit")
                cal.fit(X_val.values, y_val.values)
                self._calibrated = cal
            except Exception as e:
                log.debug(f"RF calibration skipped: {e}")
                self._calibrated = None

        log.info(f"RandomForest trained: {len(X_train):,} samples")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_vals = X[self._feature_names].values
        if self._calibrated is not None:
            return self._calibrated.predict_proba(X_vals)
        return self._model.predict_proba(X_vals)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
            name="importance",
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model,      path / "rf_model.pkl")
        joblib.dump(self._calibrated, path / "rf_calibrated.pkl")
        joblib.dump(self._feature_names, path / "feature_names.pkl")
        log.info(f"RF model saved → {path}")

    def load(self, path: Path) -> "RandomForestModel":
        path = Path(path)
        self._model         = joblib.load(path / "rf_model.pkl")
        self._calibrated    = joblib.load(path / "rf_calibrated.pkl")
        self._feature_names = joblib.load(path / "feature_names.pkl")
        return self
