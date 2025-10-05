import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set output folder
os.makedirs("../outputs/plots", exist_ok=True)

# Ticker to visualize
ticker = "S&P500"
df = pd.read_csv(f"../data/processed/{ticker}_merged.csv", parse_dates=["Date"])

# 1. Histogram of Close prices
plt.figure(figsize=(8, 5))
sns.histplot(df["Close"].dropna(), bins=50, kde=True)
plt.title(f"{ticker} - Close Price Distribution")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"../outputs/plots/{ticker}_close_hist.png")
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
plt.savefig(f"../outputs/plots/{ticker}_macro_trends.png")
plt.close()

# 3. Correlation heatmap
corr = df.drop(columns=["Symbol", "Date"]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title(f"{ticker} - Correlation Matrix")
plt.tight_layout()
plt.savefig(f"../outputs/plots/{ticker}_correlation_heatmap.png")
plt.close()

print("✅ Visualizations saved in outputs/plots/")