# src/plot_sp500_macro_trends.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load cleaned and merged S&P500 + macro data
data_path = os.path.join(project_root, "data", "processed", "S&P500_merged.csv")

if not os.path.exists(data_path):
    print(f"❌ Error: Merged data file not found at {data_path}")
    print("   Please run 'python src/integrate_data.py' first to create merged data.")
    exit(1)

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
outputs_plots_dir = os.path.join(project_root, "outputs", "plots")
os.makedirs(outputs_plots_dir, exist_ok=True)
output_path = os.path.join(outputs_plots_dir, "S&P500_macro_trends.png")
plt.savefig(output_path, dpi=300)
print(f"✅ Plot saved to {output_path}")
plt.show()