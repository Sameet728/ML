"""
live_trading/export_trades.py
==============================
Reads the out-of-sample signals from the backtest and generates
a comprehensive CSV of all trades taken up to the exact current time.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from data.preprocessor import load_processed
from backtesting.engine import run_backtest

def main():
    cfg = get_settings()
    reports_dir = ROOT / "vps_final_results" / "reports"
    signals_file = reports_dir / "oos_signals.parquet"
    
    if not signals_file.exists():
        print(f"❌ Error: {signals_file} not found.")
        print("Please ensure you downloaded the VPS results to vps_final_results/reports/")
        sys.exit(1)
        
    print("📥 Loading market data...")
    try:
        # Load local processed data (or we could fetch fresh, but this is for historical trades)
        ohlcv = load_processed("btc_1h_clean")
    except Exception as e:
        print(f"❌ Failed to load OHLCV data: {e}")
        print("Please ensure you have run data preprocessing locally.")
        sys.exit(1)
        
    print("📊 Loading out-of-sample signals...")
    signal_df = pd.read_parquet(signals_file)
    
    # Align
    common_idx = ohlcv.index.intersection(signal_df.index)
    ohlcv = ohlcv.loc[common_idx]
    signal_df = signal_df.loc[common_idx]
    
    print("⚙️ Reconstructing VectorBT Portfolio...")
    # Override initial capital to $1000
    cfg.initial_capital = 1000
    # This runs the backtest engine using the exact signals we saved
    result = run_backtest(ohlcv, signal_df, cfg, label="Trade Exporter")
    portfolio = result["portfolio"]
    
    print("📝 Extracting trade records...")
    try:
        trades_df = portfolio.trades.records_readable.copy()
    except Exception as e:
        print(f"❌ Failed to extract trades from portfolio: {e}")
        sys.exit(1)
        
    if trades_df.empty:
        print("⚠️ No trades were found in the portfolio.")
        sys.exit(0)
        
    # Calculate Custom Columns
    rr_ratio = cfg.barrier_atr_mult_tp / cfg.barrier_atr_mult_sl
    risk_pct = cfg.base_risk_pct
    
    # Calculate running capital (Start with 1000, add PnL sequentially)
    # Note: If trades overlap, this is an approximation of realized equity at trade exit.
    running_pnl = trades_df["PnL"].cumsum()
    capital_before = 1000 + running_pnl.shift(1).fillna(0)
    capital_after = 1000 + running_pnl
    
    risk_usd = capital_before * (risk_pct / 100)
    
    # Map SL and TP prices from the signal_df using the Entry Timestamp
    sl_prices = trades_df["Entry Timestamp"].map(signal_df["sl_price"]).round(2)
    tp_prices = trades_df["Entry Timestamp"].map(signal_df["tp_price"]).round(2)
    
    lot_size = trades_df.get("Size", pd.Series(0, index=trades_df.index))
    actual_risk = (trades_df["Avg Entry Price"] - sl_prices).abs() * lot_size
    realized_rr = trades_df["PnL"] / actual_risk.replace(0, np.nan)
    
    # Format the CSV beautifully with requested columns
    export_df = pd.DataFrame({
        "Trade ID": trades_df.get("Exit Trade Id", trades_df.index),
        "Entry Date": trades_df["Entry Timestamp"],
        "Exit Date": trades_df["Exit Timestamp"],
        "Direction": trades_df["Direction"].str.upper(),
        "Lot Size (BTC)": lot_size.round(4),
        "Entry Price": trades_df["Avg Entry Price"].round(2),
        "SL Price": sl_prices,
        "TP Price": tp_prices,
        "Exit Price": trades_df["Avg Exit Price"].round(2),
        "Entry Fees": trades_df.get("Entry Fees", pd.Series(0, index=trades_df.index)).round(2),
        "Exit Fees": trades_df.get("Exit Fees", pd.Series(0, index=trades_df.index)).round(2),
        "Target RR": round(rr_ratio, 2),
        "Realized RR": realized_rr.round(2),
        "Risk (%)": risk_pct,
        "Planned Risk ($)": risk_usd.round(2),
        "Actual Risk ($)": actual_risk.round(2),
        "Capital Before": capital_before.round(2),
        "PnL ($)": trades_df["PnL"].round(2),
        "Capital After": capital_after.round(2),
        "Return (%)": (trades_df["Return"] * 100).round(2)
    })
    
    output_path = ROOT / "live_trading" / "historical_trades_list_v7_weekend_fix.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)
    
    # Append summary stats
    equity = result["equity"]
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    
    monthly_equity = equity.resample("ME").last()
    monthly_returns = monthly_equity.pct_change().dropna() * 100
    
    yearly_equity = equity.resample("YE").last()
    yearly_returns = yearly_equity.pct_change().dropna() * 100
    
    with open(output_path, "a", newline="") as f:
        f.write("\n\n--- SUMMARY STATISTICS ---\n")
        f.write(f"Total Return (%), {round(total_return * 100, 2)}%\n")
        f.write("Initial Capital, $1000\n")
        f.write(f"Final Capital, ${round(equity.iloc[-1], 2)}\n")
        
        f.write("\n--- YEARLY RETURNS ---\n")
        f.write("Year, Return (%)\n")
        for date, ret in yearly_returns.items():
            f.write(f"{date.year}, {round(ret, 2)}%\n")
            
        f.write("\n--- MONTHLY RETURNS ---\n")
        f.write("Month, Return (%)\n")
        for date, ret in monthly_returns.items():
            f.write(f"{date.strftime('%Y-%m')}, {round(ret, 2)}%\n")
    
    print(f"\n✅ SUCCESS! Exported {len(export_df)} trades to:")
    print(f"   {output_path}")
    print("\nSneak peek of the last 5 trades:")
    print(export_df.tail(5).to_string(index=False))

if __name__ == "__main__":
    main()
