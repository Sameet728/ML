"""
run_pipeline.py
================
Main entry point for the AI Quant Research Platform.

Modes:
  --mode=full      Complete pipeline (data → features → train → backtest → report)
  --mode=data      Data download + preprocessing only
  --mode=features  Feature engineering only (requires data)
  --mode=train     Training only (requires features)
  --mode=backtest  Backtesting only (requires trained models)
  --mode=report    Report generation only (requires backtest results)
  --mode=dashboard Launch Streamlit dashboard

Examples:
  python run_pipeline.py
  python run_pipeline.py --mode=full
  python run_pipeline.py --mode=data --force-refresh
  python run_pipeline.py --mode=backtest --no-optimize
  python run_pipeline.py --mode=dashboard
"""

from __future__ import annotations
import argparse
import sys
import time
import traceback
from pathlib import Path

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Quant Research Platform — Main Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", default="full",
        choices=["full", "data", "features", "train", "backtest", "report", "dashboard"],
        help="Pipeline mode to run (default: full)",
    )
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-download data even if cached")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Skip Optuna hyperparameter optimization (faster)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable disk caching")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip walk-forward training; load saved OOS signals from disk "
                             "(only valid with --mode=backtest after a prior full run)")
    parser.add_argument("--regime-method", default="rules",
                        choices=["rules", "hmm"],
                        help="Regime detection method")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    return parser.parse_args()


def setup_environment(args):
    from config.settings import update_settings
    from utils.logger import setup_logger

    # Apply CLI overrides
    if args.no_cache:
        update_settings(cache_enabled=False)
    update_settings(log_level=args.log_level)
    setup_logger(args.log_level)


# ── Phase handlers ────────────────────────────────────────────────────────────

def run_data(args) -> dict:
    """Phase 1: Download + preprocess data."""
    from utils.logger import log
    from data.downloader import download_all
    from data.preprocessor import preprocess_all

    log.info("=" * 60)
    log.info("PHASE 1: DATA DOWNLOAD & PREPROCESSING")
    log.info("=" * 60)

    raw  = download_all(force_refresh=args.force_refresh)
    data = preprocess_all(raw, align_htf=True)
    log.info("Phase 1 complete ✓")
    return data


def run_features(data: dict) -> tuple:
    """Phase 2: Feature engineering."""
    from utils.logger import log
    from features.pipeline import build_feature_matrix, save_feature_matrix

    log.info("=" * 60)
    log.info("PHASE 2: FEATURE ENGINEERING")
    log.info("=" * 60)

    X, ohlcv = build_feature_matrix(
        df=data["btc_1h"],
        df_4h_on_1h=data.get("btc_4h_on_1h"),
    )
    save_feature_matrix(X, ohlcv)
    log.info(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features ✓")
    return X, ohlcv


def run_labeling_and_regime(X: "pd.DataFrame", ohlcv: "pd.DataFrame", regime_method: str) -> tuple:
    """Phase 3: Labeling + regime detection."""
    import pandas as pd
    from utils.logger import log
    from models.labeling import compute_triple_barrier_labels
    from models.regime import detect_regime
    from config.settings import get_settings

    log.info("=" * 60)
    log.info("PHASE 3: LABELING + REGIME DETECTION")
    log.info("=" * 60)

    cfg = get_settings()

    # ATR for barrier computation
    atr = X["atr"] if "atr" in X.columns else ohlcv["close"].pct_change().abs().rolling(14).mean() * ohlcv["close"]

    label_df = compute_triple_barrier_labels(ohlcv, atr)
    y = label_df["label"].dropna()

    # Regime detection (on full X)
    regime_df = detect_regime(X, method=regime_method, cfg=cfg)

    # Align everything
    common_idx = X.index.intersection(y.index)
    X_aligned       = X.loc[common_idx]
    y_aligned       = y.loc[common_idx]
    ohlcv_aligned   = ohlcv.loc[common_idx]
    regime_aligned  = regime_df.reindex(common_idx)

    log.info(f"Labeled: {len(y_aligned):,} samples, pos_rate={y_aligned.mean():.2%} ✓")
    return X_aligned, y_aligned, ohlcv_aligned, regime_aligned, label_df


def run_walk_forward(X, y, regime_df, optimize: bool = True) -> tuple:
    """Phase 4: Walk-forward retraining."""
    from utils.logger import log
    from retraining.walk_forward import WalkForwardEngine
    from config.settings import get_settings

    log.info("=" * 60)
    log.info("PHASE 4: WALK-FORWARD RETRAINING")
    log.info("=" * 60)

    cfg    = get_settings()
    engine = WalkForwardEngine(cfg)
    oos_probs, oos_quality, oos_labels, oos_regime = engine.run(
        X, y, regime_df, optimize=optimize
    )

    # Save fold summary
    fold_summary = engine.get_fold_summary()
    fold_summary.to_csv(cfg.paths["reports"] / "fold_summary.csv", index=False)

    log.info(f"Walk-forward complete: {len(engine.fold_results)} folds ✓")
    return oos_probs, oos_quality, oos_labels, oos_regime, engine


def run_backtesting(ohlcv, X, oos_probs, oos_quality, oos_labels, regime_df) -> tuple:
    """Phase 5: Backtesting + benchmarks."""
    from utils.logger import log
    from backtesting.risk import build_final_signal
    from backtesting.engine import run_backtest, run_all_benchmarks
    from backtesting.metrics import (
        compute_all_metrics, compute_monthly_returns,
        compute_yearly_returns, compare_strategies,
    )
    from config.settings import get_settings
    import joblib

    log.info("=" * 60)
    log.info("PHASE 5: BACKTESTING")
    log.info("=" * 60)

    cfg = get_settings()

    # Build signals
    signal_df = build_final_signal(oos_probs, oos_quality, regime_df, ohlcv, cfg)

    # Strategy backtest
    result = run_backtest(ohlcv, signal_df, cfg, label="AI Strategy")

    # Benchmarks
    benchmarks = run_all_benchmarks(ohlcv, cfg)

    # Compute metrics
    try:
        trades_df = result["portfolio"].trades.records_readable
    except Exception:
        trades_df = None

    metrics = compute_all_metrics(
        result["equity"],
        result["returns"],
        trades_df=trades_df,
        label="AI Strategy",
    )

    monthly_pivot  = compute_monthly_returns(result["equity"])
    yearly_df      = compute_yearly_returns(result["equity"])
    comparison_df  = compare_strategies({
        "AI Strategy":  {"stats": metrics},
        **{k: {"stats": v.get("stats", {})} for k, v in benchmarks.items()},
    })

    # Save artifacts for dashboard
    rpt = cfg.paths["reports"]
    result["equity"].to_frame("equity").to_parquet(rpt / "equity_curve.parquet")
    result["returns"].to_frame("returns").to_parquet(rpt / "returns.parquet")
    regime_df.to_parquet(rpt / "regime_df.parquet")
    joblib.dump(metrics,     rpt / "metrics.pkl")
    joblib.dump(benchmarks,  rpt / "benchmarks.pkl")

    # Save OOS signals with labels
    import pandas as pd
    oos_signals = signal_df.copy()
    oos_signals["label"] = oos_labels.reindex(signal_df.index)
    oos_signals.to_parquet(rpt / "oos_signals.parquet")

    log.info("Phase 5 complete ✓")
    return result, benchmarks, metrics, monthly_pivot, yearly_df, comparison_df, signal_df


def run_feature_importance_export(engine) -> "pd.Series":
    """Extract and save feature importance from walk-forward."""
    import pandas as pd
    from utils.logger import log
    from config.settings import get_settings

    cfg = get_settings()
    all_imp = []
    for fold in engine.fold_results:
        # Feature importance not stored directly — placeholder
        pass

    # Try to load from last fold's model (if saved)
    imp_path = cfg.paths["reports"] / "feature_importance.parquet"
    if imp_path.exists():
        return pd.read_parquet(imp_path).squeeze()

    # Return uniform placeholder
    return pd.Series(dtype=float)


def run_visualization(result, benchmarks, monthly_pivot, yearly_df, regime_df, oos_probs, oos_labels, feature_imp):
    """Phase 6: Generate all charts + HTML report."""
    from utils.logger import log
    from visualization.charts import generate_all_charts
    from visualization.reports import generate_html_report
    from backtesting.metrics import compare_strategies

    log.info("=" * 60)
    log.info("PHASE 6: VISUALIZATION & REPORTING")
    log.info("=" * 60)

    bench_for_charts = {k: {"equity": v["equity"]} for k, v in benchmarks.items() if "equity" in v}
    comparison_df    = compare_strategies({
        "AI Strategy": {"stats": result.get("stats", {})},
        **{k: {"stats": v.get("stats", {})} for k, v in benchmarks.items()},
    })

    try:
        trades_df = result["portfolio"].trades.records_readable
    except Exception:
        trades_df = None

    charts = generate_all_charts(
        strategy_result=result,
        benchmarks={k: {"equity": v["equity"]} for k, v in benchmarks.items() if "equity" in v},
        monthly_pivot=monthly_pivot,
        yearly_df=yearly_df,
        regime_df=regime_df,
        feature_importance=feature_imp,
        oos_probs=oos_probs,
        oos_labels=oos_labels,
        trades_df=trades_df,
    )

    from config.settings import get_settings
    import joblib
    cfg = get_settings()

    # Compute metrics again for report
    metrics = joblib.load(cfg.paths["reports"] / "metrics.pkl")

    report_path = generate_html_report(
        metrics=metrics,
        charts=charts,
        monthly_pivot=monthly_pivot,
        yearly_df=yearly_df,
        comparison_df=comparison_df,
    )

    log.info(f"Report saved → {report_path} ✓")
    return charts, report_path


def run_drift_check(X_train, X_recent) -> dict:
    """Phase 7: Feature drift monitoring."""
    from utils.logger import log
    from monitoring.drift import FeatureDriftMonitor
    from config.settings import get_settings
    import joblib

    log.info("=" * 60)
    log.info("PHASE 7: DRIFT MONITORING")
    log.info("=" * 60)

    cfg = get_settings()
    monitor = FeatureDriftMonitor(cfg)
    monitor.set_reference(X_train)
    report = monitor.check_drift(X_recent)
    joblib.dump(report, cfg.paths["reports"] / "drift_report.pkl")

    log.info(f"Drift check: {report['status']} ✓")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_environment(args)

    from utils.logger import log
    from config.settings import get_settings
    cfg = get_settings()

    log.info("=" * 60)
    log.info("  AI QUANT RESEARCH PLATFORM — STARTING")
    log.info(f"  Mode: {args.mode.upper()}")
    log.info(f"  Symbol: {cfg.primary_symbol}")
    log.info(f"  Period: {cfg.start_date} → {cfg.end_date}")
    log.info("=" * 60)

    t_start = time.time()

    try:
        # ── Dashboard mode ──
        if args.mode == "dashboard":
            import subprocess
            log.info("Launching Streamlit dashboard …")
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                str(ROOT / "dashboard" / "app.py"),
                "--server.port", "8501",
                "--server.headless", "false",
            ])
            return

        # ── Data ──
        if args.mode in ("full", "data"):
            data = run_data(args)
        else:
            # Load cached processed data
            from data.preprocessor import load_processed
            data = {
                "btc_1h":       load_processed("btc_1h_clean"),
                "btc_4h":       load_processed("btc_4h_clean"),
                "btc_4h_on_1h": load_processed("btc_4h_on_1h"),
            }
            if cfg.enable_gold:
                try:
                    data["gold_1d"] = load_processed("gold_1d_clean")
                except FileNotFoundError:
                    pass

        if args.mode == "data":
            log.info("Data mode complete.")
            return

        # ── Features ──
        if args.mode in ("full", "features"):
            X, ohlcv = run_features(data)
        else:
            from features.pipeline import load_feature_matrix
            X, ohlcv = load_feature_matrix()

        if args.mode == "features":
            log.info("Features mode complete.")
            return

        # ── Labeling + Regime ──
        X, y, ohlcv, regime_df, label_df = run_labeling_and_regime(
            X, ohlcv, args.regime_method
        )

        # ── Walk-Forward Training ──
        use_cached = (args.mode == "backtest" and getattr(args, "skip_train", False))
        oos_cache  = cfg.paths["reports"] / "oos_signals.parquet"

        if use_cached and oos_cache.exists():
            import pandas as pd, joblib
            log.info("Loading cached OOS signals (--skip-train) …")
            oos_df      = pd.read_parquet(oos_cache)
            oos_probs   = oos_df["prob"]   if "prob"          in oos_df.columns else oos_df.iloc[:, 0]
            oos_quality = oos_df["quality_score"] if "quality_score" in oos_df.columns else pd.Series(dtype=float)
            oos_labels  = oos_df["label"].dropna() if "label" in oos_df.columns else pd.Series(dtype=float)
            oos_regime  = regime_df
            feat_imp_p  = cfg.paths["reports"] / "feature_importance.parquet"
            feat_imp    = pd.read_parquet(feat_imp_p).squeeze() if feat_imp_p.exists() else pd.Series(dtype=float)
        elif args.mode in ("full", "train", "backtest"):
            optimize = not args.no_optimize
            oos_probs, oos_quality, oos_labels, oos_regime, engine = \
                run_walk_forward(X, y, regime_df, optimize=optimize)

            # Feature importance
            feat_imp = run_feature_importance_export(engine)
            import joblib
            feat_imp.to_frame("importance").to_parquet(cfg.paths["reports"] / "feature_importance.parquet")
        else:
            import pandas as pd, joblib
            oos_probs   = pd.read_parquet(cfg.paths["reports"] / "oos_signals.parquet")["prob"]
            oos_quality = pd.read_parquet(cfg.paths["reports"] / "oos_signals.parquet")["quality_score"]
            oos_labels  = pd.read_parquet(cfg.paths["reports"] / "oos_signals.parquet")["label"].dropna()
            oos_regime  = regime_df
            feat_imp    = pd.read_parquet(cfg.paths["reports"] / "feature_importance.parquet").squeeze()

        if args.mode == "train":
            log.info("Train mode complete.")
            return

        # ── Backtesting ──
        result, benchmarks, metrics, monthly_pivot, yearly_df, comparison_df, signal_df = \
            run_backtesting(ohlcv, X, oos_probs, oos_quality, oos_labels, regime_df)

        if args.mode == "backtest":
            log.info("Backtest mode complete.")

        # ── Visualization ──
        charts, report_path = run_visualization(
            result, benchmarks, monthly_pivot, yearly_df,
            regime_df, oos_probs, oos_labels, feat_imp,
        )

        # ── Drift monitoring ──
        try:
            n_train = int(len(X) * 0.7)
            run_drift_check(X.iloc[:n_train], X.iloc[n_train:])
        except Exception as e:
            log.warning(f"Drift check failed: {e}")

        # ── Final summary ──
        elapsed = time.time() - t_start
        log.info("=" * 60)
        log.info(f"  PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
        log.info(f"  CAGR:          {metrics.get('cagr', 0):.2%}")
        log.info(f"  Sharpe:        {metrics.get('sharpe', 0):.3f}")
        log.info(f"  Max Drawdown:  {metrics.get('max_drawdown', 0):.2%}")
        log.info(f"  Avg Monthly:   {metrics.get('avg_monthly_return', 0):.2%}")
        log.info(f"  Win Rate:      {metrics.get('win_rate', 0):.2%}")
        log.info(f"  Total Trades:  {metrics.get('total_trades', 0):,}")
        log.info(f"  Report:        {report_path}")
        log.info("=" * 60)
        log.info("  Launch dashboard: python run_pipeline.py --mode=dashboard")
        log.info("=" * 60)

    except KeyboardInterrupt:
        from utils.logger import log
        log.info("Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        from utils.logger import log
        log.error(f"Pipeline failed: {e}")
        log.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
