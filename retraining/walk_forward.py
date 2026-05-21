"""
retraining/walk_forward.py
===========================
Implements two complementary time-series validation methods:

1. PurgedKFold — Purged K-Fold Cross-Validation
   - Removes training samples that overlap with test period (prevents leakage)
   - Adds embargo gap (N bars) after each test fold
   - Used WITHIN a training window for Optuna optimization

2. WalkForwardEngine — Rolling Walk-Forward Retraining
   - Train: [t0, t0 + train_window]
   - Test:  [t0 + train_window, t0 + train_window + test_window]
   - Roll forward by step_months
   - Aggregates all OOS predictions into a unified signal series
   - NO future data leakage by construction

References:
  - Lopez de Prado (2018) "Advances in Financial Machine Learning", Ch.7
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import joblib

from config.settings import get_settings
from utils.logger import log

# NOTE: training.trainer and training.optimizer are imported lazily inside
# WalkForwardEngine.run() to avoid circular imports:
#   walk_forward → trainer → (implicitly) walk_forward


# ── Purged K-Fold ────────────────────────────────────────────────────────────

class PurgedKFold:
    """
    Purged K-Fold cross-validator for financial time series.

    Prevents look-ahead bias by:
    1. Keeping folds in chronological order (no shuffling)
    2. Removing training samples whose label period overlaps with test period
    3. Adding embargo gap after each test fold

    Usage:
        pkf = PurgedKFold(n_splits=5, embargo_bars=24)
        for train_idx, test_idx in pkf.split(X, y):
            ...
    """

    def __init__(self, n_splits: int = 5, embargo_bars: int = 24):
        self.n_splits     = n_splits
        self.embargo_bars = embargo_bars

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) for each fold."""
        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            # Test fold: k-th chunk
            test_start  = k * fold_size
            test_end    = test_start + fold_size if k < self.n_splits - 1 else n
            test_idx    = indices[test_start:test_end]

            # Training: all bars NOT in test OR embargo zone
            embargo_end = min(test_end + self.embargo_bars, n)
            purge_zone  = set(range(test_start, embargo_end))

            train_idx = np.array([
                i for i in indices if i not in purge_zone
            ])

            if len(train_idx) < 50 or len(test_idx) < 10:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ── Walk-Forward Result Container ────────────────────────────────────────────

@dataclass
class FoldResult:
    fold_id:       int
    train_start:   pd.Timestamp
    train_end:     pd.Timestamp
    test_start:    pd.Timestamp
    test_end:      pd.Timestamp
    predictions:   pd.Series         # OOS probability predictions
    quality_scores: pd.Series        # OOS trade quality scores (0–100)
    labels:        pd.Series         # Actual labels
    regime_df:     pd.DataFrame      # Regime data for this fold
    val_sharpe:    float
    selected_features: List[str]
    best_params:   dict = field(default_factory=dict)


# ── Walk-Forward Engine ──────────────────────────────────────────────────────

class WalkForwardEngine:
    """
    Rolling walk-forward retraining engine.

    Each fold:
      - Train on [train_start, train_end]
      - Optionally optimize hyperparams with Optuna (within train window)
      - Test on [test_start, test_end]  ← OOS predictions only
      - Roll forward by step_months

    Output: unified OOS signal series covering the full backtest period.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or get_settings()
        self.fold_results: List[FoldResult] = []

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_df: pd.DataFrame,
        optimize: bool = True,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """
        Execute walk-forward retraining.

        Args:
            X          : Full feature matrix (DatetimeIndex)
            y          : Full label series (aligned to X)
            regime_df  : Regime DataFrame (aligned to X)
            optimize   : Run Optuna on each fold's training window

        Returns:
            oos_probs    : Out-of-sample probability series (full period)
            oos_quality  : Out-of-sample trade quality scores
            oos_labels   : Actual labels for OOS period
            oos_regime   : Regime data for OOS period
        """
        cfg = self.cfg
        folds = list(self._generate_folds(X))

        # Lazy imports to break circular dependency
        from training.trainer import train_all_models
        from training.optimizer import run_optuna_optimization

        # ── Load cached Optuna params (saved from previous runs) ──────────────────
        import json
        from pathlib import Path
        _param_cache_path = Path(cfg.paths["reports"]) / "optuna_best_params.json"
        _param_cache: dict = {}
        if _param_cache_path.exists():
            try:
                with open(_param_cache_path) as _f:
                    _param_cache = json.load(_f)
                log.info(
                    f"Loaded Optuna param cache: {len(_param_cache)} fold(s) pre-saved "
                    f"({sorted(int(k) for k in _param_cache)}) — will skip Optuna for these."
                )
            except Exception as _e:
                log.warning(f"Could not load param cache: {_e}")
                _param_cache = {}

        log.info(
            f"Walk-forward: {len(folds)} folds, "
            f"train={cfg.train_window_months}m, "
            f"test={cfg.test_window_months}m, "
            f"step={cfg.step_months}m"
        )

        all_probs   = []
        all_quality = []
        all_labels  = []
        all_regime  = []

        for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
            log.info(
                f"=== Fold {fold_idx+1}/{len(folds)} | "
                f"Train: {tr_start.date()} → {tr_end.date()} | "
                f"Test:  {te_start.date()} → {te_end.date()} ==="
            )

            # Slice data
            X_train = X.loc[tr_start:tr_end]
            y_train = y.loc[tr_start:tr_end].dropna()
            X_train = X_train.loc[y_train.index]

            X_test  = X.loc[te_start:te_end]
            y_test  = y.loc[te_start:te_end].dropna()
            X_test  = X_test.loc[y_test.index]
            r_test  = regime_df.loc[te_start:te_end]

            if len(X_train) < 500 or len(X_test) < 50:
                log.warning(f"Fold {fold_idx+1}: insufficient data, skipping")
                continue

            # ── Optional Optuna optimization (with JSON cache fallback) ─────────
            best_params = {}
            if optimize:
                fold_key = str(fold_idx + 1)
                if fold_key in _param_cache:
                    best_params = _param_cache[fold_key]
                    log.info(
                        f"Fold {fold_idx+1}: using cached Optuna params (skipping optimization)"
                    )
                else:
                    log.info(f"Running Optuna on fold {fold_idx+1} ...")
                    try:
                        opt_result = run_optuna_optimization(
                            X_train, y_train, cfg,
                            study_name=f"fold_{fold_idx+1}"
                        )
                        best_params = opt_result["best_params"]
                        # Save to cache immediately so progress is preserved
                        _param_cache[fold_key] = best_params
                        with open(_param_cache_path, "w") as _f:
                            json.dump(_param_cache, _f, indent=2)
                        log.info(f"Fold {fold_idx+1}: params saved to cache")
                    except Exception as e:
                        log.warning(f"Optuna failed on fold {fold_idx+1}: {e}")

            # ── Split train into train/val (80/20) for early stopping ──
            val_split = int(len(X_train) * 0.8)
            X_tr2, y_tr2 = X_train.iloc[:val_split], y_train.iloc[:val_split]
            X_val2, y_val2 = X_train.iloc[val_split:], y_train.iloc[val_split:]

            # ── Train all models ──
            try:
                ensemble, meta_model, sharpes, selected_feats = train_all_models(
                    X_tr2, y_tr2, X_val2, y_val2,
                    cfg=cfg,
                    xgb_params=best_params if best_params else None,
                )
            except Exception as e:
                log.error(f"Training failed on fold {fold_idx+1}: {e}")
                continue

            val_sharpe = sharpes.get("xgboost", 0.0)

            # ── OOS Predictions ──
            # Align test features to selected
            X_test_sel = X_test.reindex(columns=selected_feats, fill_value=0)

            try:
                probs_arr = ensemble.predict_proba(X_test_sel)[:, 1]
                oos_probs_fold = pd.Series(probs_arr, index=X_test_sel.index, name="prob")
            except Exception as e:
                log.error(f"Prediction failed on fold {fold_idx+1}: {e}")
                continue

            # ── Trade quality scores ──
            quality_fold = pd.Series(50.0, index=X_test_sel.index, name="quality")
            if meta_model is not None:
                try:
                    r_test_aligned = r_test.reindex(X_test_sel.index)
                    quality_fold = meta_model.compute_trade_quality_score(
                        X_test_sel, oos_probs_fold, r_test_aligned
                    )
                except Exception as e:
                    log.warning(f"Quality score failed: {e}")

            # Store fold result
            fold = FoldResult(
                fold_id=fold_idx,
                train_start=tr_start, train_end=tr_end,
                test_start=te_start,  test_end=te_end,
                predictions=oos_probs_fold,
                quality_scores=quality_fold,
                labels=y_test,
                regime_df=r_test,
                val_sharpe=val_sharpe,
                selected_features=selected_feats,
                best_params=best_params,
            )
            self.fold_results.append(fold)

            all_probs.append(oos_probs_fold)
            all_quality.append(quality_fold)
            all_labels.append(y_test)
            all_regime.append(r_test)

            log.info(
                f"Fold {fold_idx+1} done. Val Sharpe={val_sharpe:.3f}, "
                f"OOS predictions={len(oos_probs_fold):,}"
            )

        # ── Combine all OOS results ──
        if not all_probs:
            raise RuntimeError("Walk-forward produced no valid folds!")

        oos_probs   = pd.concat(all_probs).sort_index()
        oos_quality = pd.concat(all_quality).sort_index()
        oos_labels  = pd.concat(all_labels).sort_index()
        oos_regime  = pd.concat(all_regime).sort_index()

        # Remove duplicates (overlapping test windows)
        oos_probs   = oos_probs[~oos_probs.index.duplicated(keep="last")]
        oos_quality = oos_quality[~oos_quality.index.duplicated(keep="last")]
        oos_labels  = oos_labels[~oos_labels.index.duplicated(keep="last")]
        oos_regime  = oos_regime[~oos_regime.index.duplicated(keep="last")]

        log.info(
            f"Walk-forward complete: {len(self.fold_results)} folds, "
            f"{len(oos_probs):,} OOS predictions, "
            f"mean_val_sharpe={np.mean([f.val_sharpe for f in self.fold_results]):.3f}"
        )

        return oos_probs, oos_quality, oos_labels, oos_regime

    def _generate_folds(
        self, X: pd.DataFrame
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate (train_start, train_end, test_start, test_end) tuples."""
        cfg = self.cfg
        folds = []

        idx = X.index
        start = idx[0]
        end   = idx[-1]

        train_delta = pd.DateOffset(months=cfg.train_window_months)
        test_delta  = pd.DateOffset(months=cfg.test_window_months)
        step_delta  = pd.DateOffset(months=cfg.step_months)

        cur = start
        while True:
            tr_start = cur
            tr_end   = cur + train_delta
            te_start = tr_end + pd.Timedelta(hours=1)
            te_end   = tr_end + test_delta

            if te_end > end:
                break

            # Check data availability
            tr_data = X.loc[tr_start:tr_end]
            te_data = X.loc[te_start:te_end]

            if len(tr_data) >= 500 and len(te_data) >= 50:
                folds.append((tr_start, tr_end, te_start, te_end))

            cur += step_delta

        return folds

    def get_fold_summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all fold results."""
        if not self.fold_results:
            return pd.DataFrame()

        rows = []
        for f in self.fold_results:
            rows.append({
                "fold":       f.fold_id + 1,
                "train_from": f.train_start.date(),
                "train_to":   f.train_end.date(),
                "test_from":  f.test_start.date(),
                "test_to":    f.test_end.date(),
                "val_sharpe": round(f.val_sharpe, 4),
                "n_signals":  int((f.predictions > 0.5).sum()),
                "n_features": len(f.selected_features),
            })
        return pd.DataFrame(rows)

    def save_results(self, path: Path) -> None:
        """Save all fold results."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.fold_results, path / "fold_results.pkl")
        self.get_fold_summary().to_csv(path / "fold_summary.csv", index=False)
        log.info(f"Walk-forward results saved → {path}")

    def load_results(self, path: Path) -> None:
        """Load fold results from disk."""
        self.fold_results = joblib.load(Path(path) / "fold_results.pkl")
