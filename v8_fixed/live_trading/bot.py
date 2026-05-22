"""
live_trading/bot.py
==============================
The 24/7 Live Trading Bot.
Runs continuously, downloading new data every hour, computing features,
and generating a live BUY/SELL/HOLD signal using the trained Master Model.
"""

import sys
import time
import joblib
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from data.downloader import download_all
from data.preprocessor import preprocess_all
from features.pipeline import build_feature_matrix

def generate_live_signal():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Waking up to check market...")
    cfg = get_settings()
    
    models_dir = ROOT / "live_trading" / "models"
    ensemble_path = models_dir / "master_ensemble.pkl"
    meta_path = models_dir / "master_meta_model.pkl"
    feat_path = models_dir / "selected_features.pkl"
    
    if not ensemble_path.exists():
        print("❌ Error: Master Model not found. Please run `python live_trading/train_master.py` first!")
        return
        
    print("📥 Fetching latest data from Binance...")
    raw_data = download_all(force_refresh=True)
    processed_data = preprocess_all(raw_data, align_htf=True)
    
    print("⚙️ Computing real-time features...")
    X, ohlcv = build_feature_matrix(
        df=processed_data["btc_1h"],
        df_4h_on_1h=processed_data.get("btc_4h_on_1h")
    )
    
    print("🧠 Loading Master Models...")
    ensemble = joblib.load(ensemble_path)
    meta_model = joblib.load(meta_path)
    selected_features = joblib.load(feat_path)
    
    # Get the absolute latest hour
    current_hour_features = X[selected_features].iloc[[-1]]
    current_time = current_hour_features.index[0]
    current_price = ohlcv["close"].iloc[-1]
    
    print(f"🎯 Evaluating signal for: {current_time} | Current Price: ${current_price:,.2f}")
    
    # Predict Primary
    primary_prob = ensemble.predict_proba(current_hour_features)[0, 1]
    is_primary_signal = primary_prob > 0.5
    
    # Predict Meta (if primary triggered)
    meta_prob = 0.0
    if is_primary_signal and meta_model:
        meta_prob = meta_model.predict_proba(current_hour_features, pd.Series([primary_prob], index=[current_time]))[0, 1]
    
    # Decision
    is_trade = is_primary_signal and (meta_prob >= cfg.meta_label_threshold)
    
    signal_str = "HOLD ⏸️"
    if is_trade:
        signal_str = "BUY 🟢"
        
    print("\n" + "="*50)
    print("         LIVE SIGNAL RESULTS")
    print("="*50)
    print(f" 🕒 Time:          {current_time}")
    print(f" 💰 BTC Price:     ${current_price:,.2f}")
    print(f" 🤖 Primary Prob:  {primary_prob:.1%}")
    print(f" 🛡️ Meta Prob:     {meta_prob:.1%}")
    print(f" 🚀 DECISION:      {signal_str}")
    print("="*50 + "\n")
    
    # Save to CSV
    log_file = ROOT / "live_trading" / "live_signals.csv"
    record = pd.DataFrame([{
        "Timestamp": current_time,
        "Price": current_price,
        "Primary_Prob": round(primary_prob, 3),
        "Meta_Prob": round(meta_prob, 3),
        "Signal": signal_str
    }])
    
    if not log_file.exists():
        record.to_csv(log_file, index=False)
    else:
        record.to_csv(log_file, mode='a', header=False, index=False)

def main():
    print("🤖 24/7 Live Trading Bot Started")
    print("The bot will automatically check for new signals at the start of every hour.")
    
    # Run once immediately
    try:
        generate_live_signal()
    except Exception as e:
        print(f"⚠️ Initial run failed: {e}")
        
    try:
        import schedule
    except ImportError:
        print("❌ Error: Missing 'schedule' library. Run `pip install schedule`")
        sys.exit(1)
        
    # Schedule to run at the 1st minute of every hour
    schedule.every().hour.at(":01").do(generate_live_signal)
    
    print("⏳ Entering infinite standby loop... Press Ctrl+C to stop.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped manually. Goodbye!")
            break

if __name__ == "__main__":
    main()
