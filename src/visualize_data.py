import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Set output folder
outputs_plots_dir = os.path.join(project_root, "outputs", "plots")
os.makedirs(outputs_plots_dir, exist_ok=True)

# Ticker to visualize
ticker = "S&P500"
data_path = os.path.join(project_root, "data", "processed", f"{ticker}_merged.csv")

if not os.path.exists(data_path):
    print(f"❌ Error: Merged data file not found at {data_path}")
    print("   Please run 'python src/integrate_data.py' first to create merged data.")
    exit(1)

df = pd.read_csv(data_path, parse_dates=["Date"])

# 1. Histogram of Close prices
plt.figure(figsize=(8, 5))
sns.histplot(df["Close"].dropna(), bins=50, kde=True)
plt.title(f"{ticker} - Close Price Distribution")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(outputs_plots_dir, f"{ticker}_close_hist.png"))
plt.close()

# 2. Line plot of M2 and GDP over time
plt.figure(figsize=(10, 6))
plt.plot(df["Date"], df["M2_Money_Supply"], label="M2 Money Supply (Billions)")
plt.plot(df["Date"], df["GDP"], label="GDP (Billions)")
plt.title(f"{ticker} - Macroeconomic Indicators Over Time")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outputs_plots_dir, f"{ticker}_macro_trends.png"))
plt.close()

# 3. Correlation heatmap
corr = df.drop(columns=["Symbol", "Date"]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title(f"{ticker} - Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(outputs_plots_dir, f"{ticker}_correlation_heatmap.png"))
plt.close()

print(f"✅ Visualizations saved in {outputs_plots_dir}")