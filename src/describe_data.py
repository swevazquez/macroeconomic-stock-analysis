import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Tickers (must match merged files)
tickers = ["S&P500", "NASDAQ", "Tech", "Financials", "Energy", "AAPL", "MSFT"]

processed_dir = os.path.join(project_root, "data", "processed")
outputs_dir = os.path.join(project_root, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

for ticker in tickers:
    try:
        path = os.path.join(processed_dir, f"{ticker}_merged.csv")
        
        if not os.path.exists(path):
            print(f"⚠️  File not found for {ticker}_merged.csv — did you run integrate_data.py?")
            continue
            
        df = pd.read_csv(path, parse_dates=["Date"])

        # Basic descriptive stats (numeric only)
        stats = df.describe()
        print(f"\n📊 Descriptive Statistics for {ticker}\n")
        print(stats)

        # Save to CSV
        stats.to_csv(os.path.join(outputs_dir, f"{ticker}_descriptive_stats.csv"))
        print(f"✅ Saved descriptive stats for {ticker}")

    except Exception as e:
        print(f"❌ Failed to describe {ticker}: {e}")