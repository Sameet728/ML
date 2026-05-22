"""
live_trading/train_master.py
==============================
Downloads the latest data, computes features, and trains
ONE Master Model on the entire dataset using the optimal hyperparameters.
Saves the Master Model for use by the 24/7 live bot.
"""

import sys
import json
import joblib
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from data.downloader import download_all
from data.preprocessor import preprocess_all
from features.pipeline import build_feature_matrix
from models.labeling import compute_triple_barrier_labels
from training.trainer import train_all_models

def main():
    print("🚀 Starting Master Model Training...")
    cfg = get_settings()
    
    # 1. Load best params
    params_file = ROOT / "vps_final_results" / "reports" / "optuna_best_params.json"
    if not params_file.exists():
        print(f"❌ Error: Could not find optimal parameters at {params_file}")
        sys.exit(1)
        
    with open(params_file, "r") as f:
        best_params = json.load(f)
        
    print(f"✅ Loaded optimal hyperparameters: {list(best_params.keys())}")
    
    # 2. Get latest data
    print("📥 Downloading latest live data from Binance...")
    raw_data = download_all(force_refresh=True)
    processed_data = preprocess_all(raw_data, align_htf=True)
    
    # 3. Build features
    print("⚙️ Computing live features...")
    X, ohlcv = build_feature_matrix(
        df=processed_data["btc_1h"],
        df_4h_on_1h=processed_data.get("btc_4h_on_1h")
    )
    
    # 4. Generate Labels
    print("🏷️ Generating targets for master model...")
    atr = X["atr"] if "atr" in X.columns else ohlcv["close"].pct_change().abs().rolling(14).mean() * ohlcv["close"]
    label_df = compute_triple_barrier_labels(ohlcv, atr)
    y = label_df["label"].dropna()
    
    # Align
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # 5. Train Master Model
    print(f"🧠 Training Master Models on {len(X):,} historical hours...")
    # We use all data for both train and val because this is the final production model
    ensemble, meta_model, sharpes, selected_features = train_all_models(
        X_train=X, y_train=y,
        X_val=X, y_val=y,
        cfg=cfg,
        xgb_params=best_params
    )
    
    # 6. Save everything needed for live trading
    save_dir = ROOT / "live_trading" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("💾 Saving Master Models to disk...")
    joblib.dump(ensemble, save_dir / "master_ensemble.pkl")
    joblib.dump(meta_model, save_dir / "master_meta_model.pkl")
    joblib.dump(selected_features, save_dir / "selected_features.pkl")
    
    print(f"\n🎉 SUCCESS! Master Model trained and saved to: {save_dir}")
    print("You can now run the 24/7 Live Bot!")

if __name__ == "__main__":
    main()
