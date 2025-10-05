import pandas as pd
import os

# Load macroeconomic data
fred_path = "../data/raw/fred_data.csv"
fred_df = pd.read_csv(fred_path, parse_dates=["DATE"])
fred_df.set_index("DATE", inplace=True)

# Forward-fill missing macroeconomic values
fred_df_ffill = fred_df.ffill()

# Show missing value summary
missing_summary = fred_df_ffill.isnull().sum().sort_values(ascending=False)
print("Missing values after forward-fill:\n")
print(missing_summary)

# Save cleaned macro data
os.makedirs("../data/processed", exist_ok=True)
fred_df_ffill.to_csv("../data/processed/fred_cleaned.csv")
print("Cleaned FRED data saved.")

# If you want to load and check stock data:
tickers = ["S&P500", "NASDAQ", "Energy", "Financials", "Tech", "AAPL", "MSFT"]

for ticker in tickers:
    path = f"../data/raw/{ticker}.csv"
    try:
        df = pd.read_csv(path, header=2, parse_dates=["Date"])
        missing = df.isnull().sum()
        print(f"\nMissing values for {ticker}:\n{missing}")
    except FileNotFoundError:
        print(f"❌ File not found for {ticker}.csv — did you fetch it?")
    except Exception as e:
        print(f"❌ Failed to load {ticker}.csv — {str(e)}")