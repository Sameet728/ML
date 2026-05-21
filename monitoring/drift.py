"""
monitoring/drift.py
====================
Feature drift monitoring for self-learning systems.

Detects when the feature distribution has shifted significantly
from the training distribution — a signal to retrain the model.

Methods:
  1. PSI (Population Stability Index) — industry standard for model monitoring
  2. KL Divergence — information-theoretic measure of distribution shift
  3. Rolling mean/std monitoring

PSI interpretation:
  PSI < 0.1   → No significant change
  0.1 ≤ PSI < 0.2 → Moderate change (monitor closely)
  PSI ≥ 0.2   → Significant change → RETRAIN

Reference: "Credit Risk Scorecards" (Siddiqi, 2006)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from config.settings import get_settings
from utils.logger import log


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI).

    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

    Args:
        expected : Training (reference) distribution values
        actual   : Current (monitoring) distribution values
        n_bins   : Number of bins for binning

    Returns:
        PSI value (float)
    """
    # Remove NaN/inf
    expected = expected[np.isfinite(expected)]
    actual   = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Create bins from expected distribution
    min_val  = min(expected.min(), actual.min())
    max_val  = max(expected.max(), actual.max())
    bins     = np.linspace(min_val, max_val, n_bins + 1)
    bins[0]  -= 1e-8  # Ensure min val is included
    bins[-1] += 1e-8

    # Compute bucket proportions
    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual,   bins=bins)

    exp_pct = exp_counts / max(len(expected), 1)
    act_pct = act_counts / max(len(actual),   1)

    # Replace zeros to avoid log(0)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def compute_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    n_bins: int = 50,
) -> float:
    """
    KL Divergence: D_KL(P || Q) = Σ P(x) × log(P(x) / Q(x))

    P = current distribution (actual)
    Q = reference distribution (expected)
    """
    p = p[np.isfinite(p)]
    q = q[np.isfinite(q)]

    if len(p) == 0 or len(q) == 0:
        return 0.0

    bins = np.linspace(
        min(p.min(), q.min()) - 1e-8,
        max(p.max(), q.max()) + 1e-8,
        n_bins + 1,
    )

    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)

    # Smooth to avoid division by zero
    eps    = 1e-8
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Normalize
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    kl = np.sum(p_hist * np.log(p_hist / q_hist))
    return float(kl)


class FeatureDriftMonitor:
    """
    Monitors feature distribution drift between training and production.

    Usage:
        monitor = FeatureDriftMonitor()
        monitor.set_reference(X_train)  # After training

        # Later, in production / monitoring:
        report = monitor.check_drift(X_current)
        if report["needs_retraining"]:
            trigger_retrain()
    """

    def __init__(self, cfg=None):
        self.cfg            = cfg or get_settings()
        self._reference_df: Optional[pd.DataFrame] = None
        self._reference_stats: Dict = {}
        self.drift_history: List[Dict] = []

    def set_reference(self, X_train: pd.DataFrame) -> None:
        """Store training distribution as reference."""
        self._reference_df    = X_train.copy()
        self._reference_stats = {}
        for col in X_train.columns:
            vals = X_train[col].values[np.isfinite(X_train[col].values)]
            self._reference_stats[col] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "p25":  float(np.percentile(vals, 25)),
                "p75":  float(np.percentile(vals, 75)),
                "values": vals,  # Store for PSI/KL computation
            }
        log.info(f"Drift monitor: reference set from {len(X_train):,} training samples")

    def check_drift(
        self,
        X_current: pd.DataFrame,
        timestamp: pd.Timestamp = None,
    ) -> Dict:
        """
        Check for drift between reference and current distribution.

        Returns:
            report: Dict with per-feature PSI/KL and overall drift flag
        """
        if self._reference_df is None:
            log.warning("Drift monitor: no reference set — call set_reference() first")
            return {"needs_retraining": False, "reason": "No reference"}

        cfg = self.cfg
        feature_reports = {}
        high_drift_features = []

        common_cols = [c for c in X_current.columns if c in self._reference_stats]

        for col in common_cols:
            ref_vals = self._reference_stats[col]["values"]
            cur_vals = X_current[col].values[np.isfinite(X_current[col].values)]

            if len(cur_vals) < 30:
                continue

            psi = compute_psi(ref_vals, cur_vals)
            kl  = compute_kl_divergence(ref_vals, cur_vals)

            cur_mean = float(np.mean(cur_vals))
            ref_mean = self._reference_stats[col]["mean"]
            ref_std  = self._reference_stats[col]["std"]
            mean_shift = abs(cur_mean - ref_mean) / max(ref_std, 1e-8)

            feature_reports[col] = {
                "psi":        round(psi, 4),
                "kl":         round(kl, 4),
                "mean_shift": round(mean_shift, 4),
                "drift_level": (
                    "high"     if psi >= cfg.drift_psi_threshold else
                    "moderate" if psi >= 0.1 else
                    "low"
                ),
            }

            if psi >= cfg.drift_psi_threshold:
                high_drift_features.append(col)

        # Overall drift assessment
        n_high_drift = len(high_drift_features)
        n_features   = len(common_cols)
        drift_pct    = n_high_drift / max(n_features, 1)

        needs_retraining = drift_pct > 0.20  # > 20% features with high drift

        # Summary stats
        all_psi = [v["psi"] for v in feature_reports.values()]
        avg_psi = float(np.mean(all_psi)) if all_psi else 0.0

        report = {
            "timestamp":          timestamp or pd.Timestamp.utcnow(),
            "n_features_checked": n_features,
            "n_high_drift":       n_high_drift,
            "drift_pct":          round(drift_pct, 4),
            "avg_psi":            round(avg_psi, 4),
            "needs_retraining":   needs_retraining,
            "high_drift_features": high_drift_features[:10],
            "feature_details":    feature_reports,
            "status": (
                "🔴 RETRAIN NEEDED"  if needs_retraining else
                "🟡 MONITORING"      if drift_pct > 0.05 else
                "🟢 STABLE"
            ),
        }

        self.drift_history.append({
            "timestamp":        report["timestamp"],
            "avg_psi":          avg_psi,
            "n_high_drift":     n_high_drift,
            "needs_retraining": needs_retraining,
        })

        level = "warning" if needs_retraining else "info"
        getattr(log, level)(
            f"Drift check: {report['status']} | "
            f"avg_PSI={avg_psi:.4f} | "
            f"high_drift_features={n_high_drift}/{n_features}"
        )

        return report

    def get_drift_history_df(self) -> pd.DataFrame:
        """Return drift monitoring history as DataFrame."""
        if not self.drift_history:
            return pd.DataFrame()
        return pd.DataFrame(self.drift_history).set_index("timestamp")

    def get_top_drifting_features(
        self,
        report: Dict,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Return DataFrame of top-drifting features sorted by PSI."""
        details = report.get("feature_details", {})
        if not details:
            return pd.DataFrame()

        rows = [
            {"feature": k, **v}
            for k, v in details.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("psi", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


def check_model_decay(
    recent_returns: pd.Series,
    historical_returns: pd.Series,
    window: int = 30,
) -> Dict:
    """
    Simple model performance decay detection.
    Compares recent Sharpe vs historical Sharpe.

    Returns: decay report dict
    """
    ann = np.sqrt(24 * 365)
    hist_sharpe   = historical_returns.mean() / (historical_returns.std() + 1e-10) * ann
    recent_sharpe = recent_returns.mean()     / (recent_returns.std()     + 1e-10) * ann

    decay_ratio = recent_sharpe / max(abs(hist_sharpe), 0.01)

    result = {
        "hist_sharpe":   round(float(hist_sharpe),   4),
        "recent_sharpe": round(float(recent_sharpe),  4),
        "decay_ratio":   round(float(decay_ratio),    4),
        "decaying":      decay_ratio < 0.5,  # Recent Sharpe < 50% of historical
        "status": (
            "🔴 DECAYING" if decay_ratio < 0.5 else
            "🟡 WEAKENING" if decay_ratio < 0.8 else
            "🟢 STABLE"
        ),
    }

    log.info(
        f"Model decay check: {result['status']} | "
        f"hist_sharpe={hist_sharpe:.3f} | recent_sharpe={recent_sharpe:.3f}"
    )
    return result
