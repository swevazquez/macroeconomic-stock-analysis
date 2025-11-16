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
│   ├── frequent_patterns.py
│   ├── clustering_analysis.py
│   ├── dimensionality_reduction.py
│   ├── outlier_detection.py
│   └── integrated_analysis.py
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

### Basic Pipeline
```bash
cd src
python clean_data.py
python integrate_data.py
python describe_data.py
python visualize_data.py
python transform_data.py
python frequent_patterns.py
```

### Advanced Analysis Pipeline
```bash
cd src
# Clustering Analysis (K-means, Hierarchical, DBSCAN)
python clustering_analysis.py

# Dimensionality Reduction (PCA, t-SNE, UMAP)
python dimensionality_reduction.py

# Outlier Detection (Isolation Forest, One-Class SVM, Statistical methods)
python outlier_detection.py

# Integrated Analysis (combines all methods)
python integrated_analysis.py
```

---

## 📊 Data Sources

- [FRED](https://fred.stlouisfed.org) — for CPI, Unemployment, M2 Money Supply, GDP
- [Yahoo Finance](https://finance.yahoo.com) via `yfinance` — for stock and ETF data

---

## 🧪 Analysis Techniques

### Basic Analysis
- Data cleaning and preprocessing
- Descriptive statistics
- Data visualization
- Frequent pattern mining (Apriori, FP-Growth using `mlxtend`)
- Return calculations and log transforms
- Sector-based comparisons

### Advanced Analysis

#### 1. Clustering Analysis
- **K-Means Clustering**: Partition-based clustering with optimal k selection
- **Hierarchical Clustering**: Agglomerative clustering with dendrogram visualization
- **DBSCAN**: Density-based clustering for identifying noise points and irregular clusters
- **Evaluation Metrics**: Silhouette score, Calinski-Harabasz index
- **Visualization**: 2D scatter plots with cluster coloring, dendrograms
- **Interpretation**: Cluster characteristics analysis and comparison between methods

#### 2. Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Linear dimensionality reduction with variance explained analysis
- **t-SNE**: Non-linear dimensionality reduction preserving local structure
- **UMAP**: Uniform Manifold Approximation preserving both local and global structure
- **Visualization**: Before-and-after 2D/3D scatter plots
- **Analysis**: Component loadings, variance explained, structure preservation

#### 3. Outlier Detection
- **Isolation Forest**: Tree-based anomaly detection
- **One-Class SVM**: Support vector machine for novelty detection
- **Statistical Methods**: IQR (Interquartile Range) and Z-score based detection
- **Local Outlier Factor (LOF)**: Density-based local outlier detection
- **Visualization**: Outlier highlighting in reduced dimensions
- **Analysis**: Outlier characteristics, temporal patterns, method comparison

#### 4. Integrated Analysis
- Combines clustering, dimensionality reduction, and outlier detection
- Analyzes impact of outlier removal on clustering quality
- Generates comprehensive insights and strategic recommendations
- Identifies limitations and proposes next steps

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

## 📊 Output Files

### Clustering Analysis
- `outputs/plots/clustering_results.png` - Visualization of all clustering methods
- `outputs/plots/dendrogram.png` - Hierarchical clustering dendrogram
- `outputs/cluster_stats_*.csv` - Cluster characteristics for each method
- `outputs/clustering_comparison.csv` - Comparison of clustering methods

### Dimensionality Reduction
- `outputs/plots/dimensionality_reduction_comparison.png` - Before/after visualizations
- `outputs/plots/pca_variance_analysis.png` - PCA scree plot and cumulative variance
- `outputs/plots/pca_3d_visualization.png` - 3D PCA visualization
- `outputs/pca_components_analysis.csv` - Component loadings analysis
- `outputs/dimensionality_reduction_comparison.csv` - Method comparison

### Outlier Detection
- `outputs/plots/outlier_detection_comparison.png` - Outlier visualization for all methods
- `outputs/plots/outlier_feature_distributions.png` - Feature distributions for inliers/outliers
- `outputs/outlier_stats_*.csv` - Outlier characteristics for each method
- `outputs/outlier_detection_comparison.csv` - Method comparison
- `outputs/outlier_method_agreement.csv` - Agreement matrix between methods

### Integrated Analysis
- `outputs/plots/integrated_analysis.png` - Combined visualization
- `outputs/comprehensive_insights.csv` - Summary of all insights

## 📌 To-Do

- ✅ Clustering analysis (K-means, Hierarchical, DBSCAN)
- ✅ Dimensionality reduction (PCA, t-SNE, UMAP)
- ✅ Outlier detection (Isolation Forest, One-Class SVM, Statistical methods)
- ✅ Integrated analysis and insights generation
- Finalize full report with results discussion
- Continue tuning pattern mining thresholds

---

## 📄 License

MIT License (or update if needed)