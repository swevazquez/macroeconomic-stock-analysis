import pandas as pd
import os

# Load cleaned macroeconomic data
macro_path = "../data/processed/fred_cleaned.csv"
macro_df = pd.read_csv(macro_path, parse_dates=["DATE"])
macro_df.rename(columns={"DATE": "Date"}, inplace=True)

# Stock tickers (must match the cleaned filenames)
tickers = ["S&P500", "NASDAQ", "Tech", "Financials", "Energy", "AAPL", "MSFT"]

# Create processed directory if not exists
os.makedirs("../data/processed", exist_ok=True)

for ticker in tickers:
    try:
        stock_path = f"../data/raw/{ticker}.csv"
        df = pd.read_csv(stock_path, header=2, parse_dates=["Date"])

        # Rename Unnamed columns to standard names
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]

        # Merge with macroeconomic data on Date
        merged_df = pd.merge(df, macro_df, on="Date", how="left")

        # Save the integrated file
        output_path = f"../data/processed/{ticker}_merged.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"✅ Merged data saved for {ticker}")

    except Exception as e:
        print(f"❌ Failed to integrate {ticker}: {e}")