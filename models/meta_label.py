"""
models/meta_label.py
====================
Meta-labeling: secondary model that filters primary model signals.

Architecture:
  1. Primary model (XGBoost) predicts BUY probability
  2. Only bars where primary prob > threshold are candidates
  3. Meta model predicts: given this setup, will it ACTUALLY profit?

The meta model only trains on POSITIVE primary predictions,
making it a specialized "gate" rather than a replacement.

This technique is from "Advances in Financial Machine Learning" (Lopez de Prado).

Also computes Trade Quality Score (0–100) from multiple factors.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List

import joblib
from xgboost import XGBClassifier
from config.settings import get_settings
from utils.logger import log


class MetaLabelModel:
    """
    Meta-labeling filter.

    Trains on: [primary_prob, regime features, volatility features, quality features]
    Predicts:  P(trade will be profitable | primary model said BUY)

    Combined with primary probability → Trade Quality Score.
    """

    META_FEATURES = [
        # Primary model output
        "primary_prob",
        # Regime context
        "regime", "regime_confidence", "regime_stability",
        "vol_regime", "trend_regime", "trend_strength",
        # Volatility context
        "atr_pct", "atr_percentile", "hist_vol_20", "bb_width",
        # Momentum quality
        "rsi", "rsi_slope", "macd_histogram", "adx",
        # Market timing
        "session_overlap", "is_weekend", "hour",
        # Rolling return quality
        "log_ret_24h", "ret_vol_24h",
    ]

    def __init__(self, params: dict = None):
        cfg = get_settings()
        self.params = {
            "n_estimators":     200,
            "max_depth":        4,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "scale_pos_weight": 1.0,
            "objective":        "binary:logistic",
            "verbosity":        0,
            "random_state":     cfg.random_seed,
            "n_jobs":           -1,
            **(params or {}),
        }
        self._model: Optional[XGBClassifier] = None
        self._available_features: List[str] = []

    def _build_meta_features(
        self,
        X: pd.DataFrame,
        primary_probs: pd.Series,
    ) -> pd.DataFrame:
        """Assemble meta-model feature matrix."""
        meta = pd.DataFrame(index=X.index)
        meta["primary_prob"] = primary_probs.values

        for feat in self.META_FEATURES:
            if feat == "primary_prob":
                continue
            if feat in X.columns:
                meta[feat] = X[feat].values

        return meta.fillna(meta.median())

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        primary_probs_train: pd.Series,
        primary_threshold: float = None,
    ) -> "MetaLabelModel":
        """
        Train meta-model on primary POSITIVE signals only.

        Args:
            X_train            : Full feature matrix
            y_train            : True labels (from triple barrier)
            primary_probs_train: Primary model probabilities
            primary_threshold  : Only train on bars where primary prob > threshold
        """
        cfg = get_settings()
        threshold = primary_threshold or cfg.meta_min_confidence

        # Filter to primary positive signals only
        pos_mask = primary_probs_train >= threshold
        n_pos    = pos_mask.sum()

        if n_pos < 50:
            log.warning(
                f"Meta-labeling: only {n_pos} primary signals above threshold "
                f"{threshold}. Skipping meta-label training."
            )
            return self

        X_meta_train = self._build_meta_features(
            X_train[pos_mask], primary_probs_train[pos_mask]
        )
        y_meta_train = y_train[pos_mask]
        self._available_features = list(X_meta_train.columns)

        # Adjust class weight
        pos_rate = y_meta_train.mean()
        if 0.01 < pos_rate < 0.99:
            self.params["scale_pos_weight"] = (1 - pos_rate) / max(pos_rate, 1e-6)

        self._model = XGBClassifier(**self.params)
        self._model.fit(
            X_meta_train.values,
            y_meta_train.values,
            verbose=False,
        )

        log.info(
            f"Meta-label model trained on {n_pos:,} primary signals | "
            f"pos_rate={pos_rate:.2%}"
        )
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        primary_probs: pd.Series,
    ) -> pd.Series:
        """
        Return meta-model probability for each bar.
        Bars where primary prob < threshold get 0.0.
        """
        if self._model is None:
            log.warning("Meta-model not trained — returning primary probs")
            return primary_probs.clip(0, 1)

        cfg = get_settings()
        threshold = cfg.meta_min_confidence

        result = pd.Series(0.0, index=X.index)
        pos_mask = primary_probs >= threshold

        if pos_mask.sum() > 0:
            X_meta = self._build_meta_features(
                X[pos_mask], primary_probs[pos_mask]
            )
            # Align to available features
            for col in self._available_features:
                if col not in X_meta.columns:
                    X_meta[col] = 0.0
            X_meta = X_meta[self._available_features]
            meta_probs = self._model.predict_proba(X_meta.values)[:, 1]
            result[pos_mask] = meta_probs

        return result

    def compute_trade_quality_score(
        self,
        X: pd.DataFrame,
        primary_probs: pd.Series,
        regime_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute Trade Quality Score (0–100) for each bar.

        Components:
          30% — Primary model probability
          25% — Meta-model probability
          20% — Regime confidence
          15% — Volatility favorability (low ATR percentile preferred)
          10% — Trend strength

        Only bars with primary_prob > threshold get non-zero scores.
        """
        cfg = get_settings()
        meta_probs = self.predict_proba(X, primary_probs)

        # Component 1: Primary model (0–1 → 0–30)
        c1 = np.clip(primary_probs.values, 0, 1) * 30

        # Component 2: Meta model (0–1 → 0–25)
        c2 = np.clip(meta_probs.values, 0, 1) * 25

        # Component 3: Regime confidence (0–1 → 0–20)
        reg_conf = regime_df.get("regime_confidence", pd.Series(0.5, index=X.index))
        c3 = np.clip(reg_conf.reindex(X.index).fillna(0.5).values, 0, 1) * 20

        # Component 4: Volatility favorability
        # Best: mid-range ATR (not too high, not too low)
        atr_pct = X.get("atr_percentile", pd.Series(0.5, index=X.index))
        atr_pct_v = np.clip(atr_pct.values, 0, 1)
        vol_fav = 1 - (atr_pct_v - 0.4).clip(0) * 2  # Penalize very high vol
        c4 = np.clip(vol_fav, 0, 1) * 15

        # Component 5: Trend strength (0–1 → 0–10)
        ts = X.get("trend_strength", pd.Series(0.5, index=X.index))
        c5 = np.clip(ts.values, 0, 1) * 10

        score = c1 + c2 + c3 + c4 + c5
        quality_series = pd.Series(score, index=X.index, name="quality_score")

        log.info(
            f"Trade quality scores: mean={quality_series.mean():.1f}, "
            f"above_{cfg.quality_score_threshold}="
            f"{(quality_series >= cfg.quality_score_threshold).sum():,}"
        )
        return quality_series

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._model:
            joblib.dump(self._model,              path / "meta_model.pkl")
            joblib.dump(self._available_features, path / "meta_features.pkl")

    def load(self, path: Path) -> "MetaLabelModel":
        path = Path(path)
        if (path / "meta_model.pkl").exists():
            self._model              = joblib.load(path / "meta_model.pkl")
            self._available_features = joblib.load(path / "meta_features.pkl")
        return self
