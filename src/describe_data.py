import pandas as pd
import os

# Tickers (must match merged files)
tickers = ["S&P500", "NASDAQ", "Tech", "Financials", "Energy", "AAPL", "MSFT"]

for ticker in tickers:
    try:
        path = f"../data/processed/{ticker}_merged.csv"
        df = pd.read_csv(path, parse_dates=["Date"])

        # Basic descriptive stats (numeric only)
        stats = df.describe()
        print(f"\n📊 Descriptive Statistics for {ticker}\n")
        print(stats)

        # Save to CSV
        stats.to_csv(f"../outputs/{ticker}_descriptive_stats.csv")

    except Exception as e:
        print(f"❌ Failed to describe {ticker}: {e}")