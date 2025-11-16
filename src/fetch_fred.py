import pandas_datareader.data as web
import pandas as pd
import os
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
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create raw data directory if it doesn't exist
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    fred_data = fetch_fred_data()
    output_path = os.path.join(raw_dir, "fred_data.csv")
    fred_data.to_csv(output_path)
    print(f"✅ FRED data saved to {output_path}")
