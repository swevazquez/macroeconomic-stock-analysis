import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load cleaned macroeconomic data
macro_path = os.path.join(project_root, "data", "processed", "fred_cleaned.csv")

if not os.path.exists(macro_path):
    print(f"❌ Error: FRED cleaned data file not found at {macro_path}")
    print("   Please run 'python src/clean_data.py' first to clean the FRED data.")
    exit(1)

macro_df = pd.read_csv(macro_path, parse_dates=["DATE"])
macro_df.rename(columns={"DATE": "Date"}, inplace=True)

# Stock tickers (must match the cleaned filenames)
tickers = ["S&P500", "NASDAQ", "Tech", "Financials", "Energy", "AAPL", "MSFT"]

# Create processed directory if not exists
processed_dir = os.path.join(project_root, "data", "processed")
raw_dir = os.path.join(project_root, "data", "raw")
os.makedirs(processed_dir, exist_ok=True)

for ticker in tickers:
    try:
        stock_path = os.path.join(raw_dir, f"{ticker}.csv")
        
        if not os.path.exists(stock_path):
            print(f"⚠️  File not found for {ticker}.csv — did you fetch it?")
            continue
            
        df = pd.read_csv(stock_path, header=2, parse_dates=["Date"])

        # Rename Unnamed columns to standard names
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]

        # Merge with macroeconomic data on Date
        merged_df = pd.merge(df, macro_df, on="Date", how="left")

        # Save the integrated file
        output_path = os.path.join(processed_dir, f"{ticker}_merged.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"✅ Merged data saved for {ticker}")

    except Exception as e:
        print(f"❌ Failed to integrate {ticker}: {e}")