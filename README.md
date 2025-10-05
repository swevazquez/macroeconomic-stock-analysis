# Macroeconomic Stock Analysis

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
│   └── describe_data.py
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

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Fetch data

- **FRED Data** (macroeconomic indicators):

```bash
cd src
python fetch_fred.py
```

- **Yahoo Finance Data** (indices, sectors, stocks):

```bash
python fetch_yfinance.py
```

### 5. Clean and integrate data

```bash
python clean_data.py
python integrate_data.py
```

### 6. Generate descriptive statistics

```bash
python describe_data.py
```

---

## 📊 Data Sources

- [FRED](https://fred.stlouisfed.org) — for CPI, Unemployment, M2 Money Supply, GDP
- [BLS](https://www.bls.gov) — for labor market data
- [Yahoo Finance](https://finance.yahoo.com) via `yfinance` — for stock and ETF data

---

## 🧪 Analysis Techniques (Planned)

- Data cleaning and preprocessing
- Descriptive statistics and visualization
- Frequent pattern mining (Apriori, FP-Growth)
- Regression or clustering (as needed)
- Sector-based comparison
- Search for counterintuitive macro-stock relationships

---

## 👥 Team Members

- **Eliezer Vazquez** — evazquez@example.com
- [Add additional names here]

---

## 📝 Notes

- You may need to manually download or clean files in `data/raw/` if issues arise (e.g., corrupted headers)
- Always check file paths before running scripts
- We use `header=2` in `pandas.read_csv()` to skip extra metadata rows in Yahoo Finance exports

---

## 📌 To-Do (Next Steps)

- Normalize and binarize variables for pattern mining
- Visualize correlations and macro impacts
- Apply Apriori and FP-Growth using `mlxtend`
- Create midterm and final report summaries

---

## 📄 License

MIT License (or update if needed)
