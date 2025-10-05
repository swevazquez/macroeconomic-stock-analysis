import pandas_datareader.data as web
import pandas as pd
from datetime import datetime
from config import FRED_INDICATORS

def fetch_fred_data(start='2000-01-01', end='2024-12-31'):
    data = pd.DataFrame()
    for name, code in FRED_INDICATORS.items():
        print(f"Fetching {name}...")
        df = web.DataReader(code, 'fred', start, end)
        df.rename(columns={code: name}, inplace=True)
        data = pd.concat([data, df], axis=1)
    return data

if __name__ == "__main__":
    fred_data = fetch_fred_data()
    fred_data.to_csv("../data/raw/fred_data.csv")
    print("FRED data saved.")
