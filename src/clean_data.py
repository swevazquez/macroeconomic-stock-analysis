import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load macroeconomic data
fred_path = os.path.join(project_root, "data", "raw", "fred_data.csv")

if not os.path.exists(fred_path):
    print(f"❌ Error: FRED data file not found at {fred_path}")
    print("   Please run 'python src/fetch_fred.py' first to download the data.")
    exit(1)

fred_df = pd.read_csv(fred_path, parse_dates=["DATE"])
fred_df.set_index("DATE", inplace=True)

# Forward-fill missing macroeconomic values
fred_df_ffill = fred_df.ffill()

# Show missing value summary
missing_summary = fred_df_ffill.isnull().sum().sort_values(ascending=False)
print("Missing values after forward-fill:\n")
print(missing_summary)

# Save cleaned macro data
processed_dir = os.path.join(project_root, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)
fred_df_ffill.to_csv(os.path.join(processed_dir, "fred_cleaned.csv"))
print("Cleaned FRED data saved.")

# If you want to load and check stock data:
tickers = ["S&P500", "NASDAQ", "Energy", "Financials", "Tech", "AAPL", "MSFT"]
raw_dir = os.path.join(project_root, "data", "raw")

for ticker in tickers:
    path = os.path.join(raw_dir, f"{ticker}.csv")
    try:
        if not os.path.exists(path):
            print(f"⚠️  File not found for {ticker}.csv — did you fetch it?")
            continue
        df = pd.read_csv(path, header=2, parse_dates=["Date"])
        missing = df.isnull().sum()
        print(f"\nMissing values for {ticker}:\n{missing}")
    except Exception as e:
        print(f"❌ Failed to load {ticker}.csv — {str(e)}")