import yfinance as yf
import pandas as pd
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
    data = fetch_yahoo_data()
    for name, df in data.items():
        df.to_csv(f"../data/raw/{name}.csv")
    print("Yahoo Finance data saved.")
