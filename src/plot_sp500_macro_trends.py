# src/plot_sp500_macro_trends.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned and merged S&P500 + macro data
data_path = "../data/processed/S&P500_merged.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])

# Sort and resample monthly
df = df.sort_values("Date")
df.set_index("Date", inplace=True)
monthly_df = df[["Close", "M2_Money_Supply", "GDP"]].resample("M").mean()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(monthly_df.index, monthly_df["Close"], label="S&P 500 Close", linewidth=2)
plt.plot(monthly_df.index, monthly_df["M2_Money_Supply"], label="M2 Money Supply", linestyle="--")
plt.plot(monthly_df.index, monthly_df["GDP"], label="GDP", linestyle=":")

plt.title("S&P 500 Trends vs M2 and GDP")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Output
output_path = "../figures/S&P500_macro_trends.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()