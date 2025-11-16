import yfinance as yf
import pandas as pd
import os
from config import YAHOO_TICKERS

def fetch_yahoo_data(start='2010-01-01', end='2024-12-31'):
    data = {}
    for name, ticker in YAHOO_TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        df = yf.download(ticker, start=start, end=end)
        df['Symbol'] = name
        data[name] = df
    return data

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create raw data directory if it doesn't exist
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    data = fetch_yahoo_data()
    for name, df in data.items():
        output_path = os.path.join(raw_dir, f"{name}.csv")
        df.to_csv(output_path)
        print(f"✅ Saved {name} to {output_path}")
    print("✅ Yahoo Finance data saved.")
