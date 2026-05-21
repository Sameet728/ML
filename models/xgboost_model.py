"""
models/xgboost_model.py
========================
XGBoost binary classifier wrapper with:
  - Probability calibration (Platt scaling) — sklearn-version-aware
  - SHAP feature importance
  - XGBoost 1.x / 2.x / 3.x API compatibility
  - Save/load utilities
  - Sklearn-compatible interface
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import xgboost as xgb_lib
from xgboost import XGBClassifier

from config.settings import get_settings
from utils.logger import log

# Detect XGBoost major version once at module load
_XGB_MAJOR = int(xgb_lib.__version__.split(".")[0])


def _make_calibrated_cv(base_estimator):
    """
    Build a CalibratedClassifierCV that works across sklearn versions.
    sklearn < 1.2:  CalibratedClassifierCV(estimator, cv='prefit')
    sklearn >= 1.2: CalibratedClassifierCV(estimator, cv=None)   [prefit default changed]
    """
    from sklearn.calibration import CalibratedClassifierCV
    import sklearn
    sk_major, sk_minor = [int(x) for x in sklearn.__version__.split(".")[:2]]

    try:
        if sk_major > 1 or (sk_major == 1 and sk_minor >= 2):
            # Newer sklearn: cv='prefit' was moved to a positional concept
            # Just pass the already-fitted estimator without cv='prefit'
            cal = CalibratedClassifierCV(base_estimator, method="sigmoid")
        else:
            cal = CalibratedClassifierCV(base_estimator, method="sigmoid", cv="prefit")
    except Exception:
        # Last resort: no calibration
        return None
    return cal


DEFAULT_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 1.0,   # Adjusted dynamically for class imbalance
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    # NOTE: "use_label_encoder" removed in XGBoost 2.x
    "verbosity":        0,
    "n_jobs":           -1,
}


class XGBoostModel:
    """
    XGBoost classifier with probability calibration and SHAP support.

    API compatibility:
      XGBoost ≥ 2.x : early_stopping_rounds → constructor param
      XGBoost 1.x   : early_stopping_rounds → fit() param
    """

    def __init__(self, params: Dict[str, Any] = None, calibrate: bool = True):
        cfg = get_settings()
        self.params    = {**DEFAULT_PARAMS, **(params or {})}
        self.params["random_state"] = cfg.random_seed
        self.calibrate = calibrate
        self._model: Optional[XGBClassifier] = None
        self._calibrated = None
        self._feature_names: list = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series    = None,
        sample_weight: np.ndarray = None,
    ) -> "XGBoostModel":
        """
        Train the XGBoost model.
        Automatically adjusts scale_pos_weight for class imbalance.
        """
        self._feature_names = list(X_train.columns)

        # Auto-adjust class weight for imbalanced labels
        pos_rate = float(y_train.mean())
        if 0.01 < pos_rate < 0.99:
            self.params["scale_pos_weight"] = (1 - pos_rate) / max(pos_rate, 1e-6)

        has_val = (X_val is not None and y_val is not None
                   and len(y_val) >= 20 and len(X_val) >= 20)

        eval_set = [(X_train.values, y_train.values)]
        if has_val:
            eval_set.append((X_val.values, y_val.values))

        # ── XGBoost version-aware early stopping ─────────────────────────────
        # v1.x: fit(early_stopping_rounds=N)
        # v2.x+: XGBClassifier(early_stopping_rounds=N)
        constructor_params = {**self.params}
        fit_kwargs: dict   = dict(eval_set=eval_set, verbose=False)

        if has_val:
            if _XGB_MAJOR >= 2:
                constructor_params["early_stopping_rounds"] = 30
            else:
                fit_kwargs["early_stopping_rounds"] = 30

        self._model = XGBClassifier(**constructor_params)

        # Ensure sample_weight is always a plain numpy array
        # (pandas 2.x TimedeltaIndex arithmetic returns pandas Index objects)
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)

        self._model.fit(X_train.values, y_train.values, **fit_kwargs)

        # Platt calibration — sklearn-version-aware helper
        if self.calibrate and has_val:
            try:
                log.debug("Calibrating probabilities with Platt scaling …")
                cal = _make_calibrated_cv(self._model)
                if cal is not None:
                    cal.fit(X_val.values, y_val.values)
                    self._calibrated = cal
            except Exception as e:
                log.debug(f"Calibration skipped: {e}")
                self._calibrated = None

        best_iter = getattr(self._model, "best_iteration",
                            self.params["n_estimators"])
        log.info(
            f"XGBoost trained: {len(X_train):,} samples, "
            f"best_iteration={best_iter}"
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability array of shape (n, 2)."""
        X_vals = X[self._feature_names].values
        if self._calibrated is not None:
            try:
                return self._calibrated.predict_proba(X_vals)
            except Exception:
                pass
        return self._model.predict_proba(X_vals)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Return feature importance as named Series."""
        try:
            imp = self._model.get_booster().get_score(importance_type=importance_type)
            return pd.Series(imp, name="importance").sort_values(ascending=False)
        except Exception:
            # Fallback: use feature_importances_ attribute
            fi = getattr(self._model, "feature_importances_", None)
            if fi is not None:
                return pd.Series(fi, index=self._feature_names,
                                 name="importance").sort_values(ascending=False)
            return pd.Series(dtype=float)

    def shap_values(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Compute SHAP values using TreeExplainer."""
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            return explainer.shap_values(X[self._feature_names].values)
        except Exception as e:
            log.warning(f"SHAP computation failed: {e}")
            return None

    def save(self, path: Path) -> None:
        """Save model artifacts to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model,         path / "xgb_model.pkl")
        joblib.dump(self._calibrated,    path / "xgb_calibrated.pkl")
        joblib.dump(self._feature_names, path / "feature_names.pkl")
        log.info(f"XGBoost model saved → {path}")

    def load(self, path: Path) -> "XGBoostModel":
        """Load model artifacts from disk."""
        path = Path(path)
        self._model         = joblib.load(path / "xgb_model.pkl")
        self._calibrated    = joblib.load(path / "xgb_calibrated.pkl")
        self._feature_names = joblib.load(path / "feature_names.pkl")
        log.info(f"XGBoost model loaded ← {path}")
        return self
