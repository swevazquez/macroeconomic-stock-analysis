# 📈 Macroeconomic Stock Analysis

This repository contains the full workflow for our data mining project, **"Uncovering Hidden Signals: How Macroeconomic Indicators Influence Stock and Sector Performance"**. The objective is to investigate how macroeconomic indicators affect stock prices across major indices, sectors, and individual companies using Python and various data mining techniques.

---

## 📁 Project Structure

```
.
├── data/
│   ├── raw/               # Raw data from FRED and Yahoo Finance
│   └── processed/         # Cleaned and merged datasets
├── notebooks/             # (Optional) Jupyter notebooks
├── outputs/               # Generated outputs (descriptive stats, plots, etc.)
├── src/                   # Python source files
│   ├── config.py
│   ├── fetch_fred.py
│   ├── fetch_yfinance.py
│   ├── clean_data.py
│   ├── integrate_data.py
│   ├── describe_data.py
│   ├── visualize_data.py
│   ├── transform_data.py
│   └── frequent_patterns.py
├── frequent_patterns/     # Output CSVs for pattern mining
├── visualizations/        # Charts and plots
├── requirements.txt       # Required Python packages
└── README.md              # This file
```

---

## 🧰 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/macroeconomic-stock-analysis.git
cd macroeconomic-stock-analysis
```

### 2. Set up your Python environment

It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Fetch data

- **FRED Data** (macroeconomic indicators):

```bash
cd src
python fetch_fred.py
```

- **Yahoo Finance Data** (indices, sectors, stocks):

```bash
python fetch_yfinance.py
```

---

## 🔄 Run Full Pipeline

```bash
python clean_data.py
python integrate_data.py
python describe_data.py
python visualize_data.py
python transform_data.py
python frequent_patterns.py
```

---

## 📊 Data Sources

- [FRED](https://fred.stlouisfed.org) — for CPI, Unemployment, M2 Money Supply, GDP
- [Yahoo Finance](https://finance.yahoo.com) via `yfinance` — for stock and ETF data

---

## 🧪 Analysis Techniques

- Data cleaning and preprocessing
- Descriptive statistics
- Data visualization
- Frequent pattern mining (Apriori, FP-Growth using `mlxtend`)
- Return calculations and log transforms
- Sector-based comparisons

---

## 📈 Frequent Pattern Summary

| Binary Feature         | True Ratio |
|------------------------|------------|
| Close_High             | 49.9%      |
| Volume_High            | 49.9%      |
| M2_High                | 1.6%       |
| GDP_High               | 1.5%       |
| CPI_High               | 1.6%       |
| Unemployment_Low       | 1.5%       |

| Algorithm   | Frequent Itemsets | Rules |
|-------------|-------------------|-------|
| Apriori     | 3                 | 2     |
| FP-Growth   | 3                 | 2     |

---

## 👥 Team Members

- [Add additional names here]

---

## 📝 Notes

- Some Yahoo Finance exports may require `header=2` in `pd.read_csv()`
- Cleaned files are stored in `data/processed/`
- Pattern mining output saved in `frequent_patterns/`

---

## 📌 To-Do

- Finish regression or clustering (optional)
- Finalize full report with results discussion
- Continue tuning pattern mining thresholds

---

## 📄 License

MIT License (or update if needed)