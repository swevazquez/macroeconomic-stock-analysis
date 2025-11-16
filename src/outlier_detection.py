"""
Outlier Detection Module
Applies Isolation Forest, One-Class SVM, and statistical methods to identify anomalies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def prepare_data(file_path, drop_na=True):
    """
    Load and prepare data for outlier detection.
    
    Args:
        file_path: Path to merged CSV file
        drop_na: Whether to drop rows with missing values
    
    Returns:
        DataFrame with features and scaler
    """
    df = pd.read_csv(file_path, parse_dates=["Date"])
    
    # Select numeric features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'CPI', 'Unemployment', 'M2_Money_Supply', 'GDP']
    
    # Filter to available columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Extract features
    X = df[feature_cols].copy()
    
    if drop_na:
        # Drop rows with any missing values
        mask = X.notna().all(axis=1)
        X = X[mask]
        df_clean = df[mask].copy()
    else:
        # Forward fill missing values
        X = X.ffill().bfill()
        df_clean = df.copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X, df_clean, feature_cols, scaler


def statistical_outlier_detection(X, feature_cols, method='iqr', threshold=3):
    """
    Detect outliers using statistical methods (IQR or Z-score).
    
    Args:
        X: Feature matrix (not scaled) - can be DataFrame or numpy array
        feature_cols: Feature column names
        method: 'iqr' or 'zscore'
        threshold: Threshold for z-score (default 3)
    
    Returns:
        Outlier labels (-1 for outliers, 1 for inliers)
    """
    print(f"\n{'='*60}")
    print(f"STATISTICAL OUTLIER DETECTION ({method.upper()})")
    print(f"{'='*60}")
    
    # Convert to numpy array if DataFrame
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    outlier_mask = np.zeros(len(X_array), dtype=bool)
    
    if method == 'iqr':
        print("Using Interquartile Range (IQR) method")
        for i, col in enumerate(feature_cols):
            Q1 = np.percentile(X_array[:, i], 25)
            Q3 = np.percentile(X_array[:, i], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            col_outliers = (X_array[:, i] < lower_bound) | (X_array[:, i] > upper_bound)
            outlier_mask |= col_outliers
            
            n_outliers = col_outliers.sum()
            if n_outliers > 0:
                print(f"  {col}: {n_outliers} outliers ({n_outliers/len(X_array)*100:.2f}%)")
    
    elif method == 'zscore':
        print(f"Using Z-score method (threshold: {threshold})")
        from scipy import stats
        
        for i, col in enumerate(feature_cols):
            z_scores = np.abs(stats.zscore(X_array[:, i]))
            col_outliers = z_scores > threshold
            outlier_mask |= col_outliers
            
            n_outliers = col_outliers.sum()
            if n_outliers > 0:
                print(f"  {col}: {n_outliers} outliers ({n_outliers/len(X_array)*100:.2f}%)")
    
    labels = np.where(outlier_mask, -1, 1)
    n_outliers = outlier_mask.sum()
    
    print(f"\nTotal outliers detected: {n_outliers} ({n_outliers/len(X_array)*100:.2f}%)")
    
    return labels, n_outliers


def isolation_forest_detection(X_scaled, contamination=0.1, random_state=42):
    """
    Detect outliers using Isolation Forest.
    
    Args:
        X_scaled: Standardized feature matrix
        contamination: Expected proportion of outliers
        random_state: Random seed
    
    Returns:
        IsolationForest model and outlier labels
    """
    print(f"\n{'='*60}")
    print("ISOLATION FOREST OUTLIER DETECTION")
    print(f"{'='*60}")
    print(f"Contamination rate: {contamination} ({contamination*100:.1f}%)")
    
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    labels = iso_forest.fit_predict(X_scaled)
    
    n_outliers = (labels == -1).sum()
    print(f"Outliers detected: {n_outliers} ({n_outliers/len(X_scaled)*100:.2f}%)")
    
    # Get anomaly scores
    scores = iso_forest.score_samples(X_scaled)
    
    return iso_forest, labels, scores, n_outliers


def oneclass_svm_detection(X_scaled, nu=0.1, kernel='rbf', gamma='scale'):
    """
    Detect outliers using One-Class SVM.
    
    Args:
        X_scaled: Standardized feature matrix
        nu: Upper bound on fraction of outliers
        kernel: Kernel type ('rbf', 'linear', 'poly')
        gamma: Kernel coefficient
    
    Returns:
        OneClassSVM model and outlier labels
    """
    print(f"\n{'='*60}")
    print("ONE-CLASS SVM OUTLIER DETECTION")
    print(f"{'='*60}")
    print(f"nu (outlier fraction): {nu}")
    print(f"Kernel: {kernel}")
    
    oc_svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    labels = oc_svm.fit_predict(X_scaled)
    
    n_outliers = (labels == -1).sum()
    print(f"Outliers detected: {n_outliers} ({n_outliers/len(X_scaled)*100:.2f}%)")
    
    # Get decision function scores
    scores = oc_svm.decision_function(X_scaled)
    
    return oc_svm, labels, scores, n_outliers


def local_outlier_factor_detection(X_scaled, n_neighbors=20, contamination=0.1):
    """
    Detect outliers using Local Outlier Factor (LOF).
    
    Args:
        X_scaled: Standardized feature matrix
        n_neighbors: Number of neighbors to consider
        contamination: Expected proportion of outliers
    
    Returns:
        LOF model and outlier labels
    """
    print(f"\n{'='*60}")
    print("LOCAL OUTLIER FACTOR (LOF) DETECTION")
    print(f"{'='*60}")
    print(f"n_neighbors: {n_neighbors}")
    print(f"Contamination rate: {contamination} ({contamination*100:.1f}%)")
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(X_scaled)
    
    n_outliers = (labels == -1).sum()
    print(f"Outliers detected: {n_outliers} ({n_outliers/len(X_scaled)*100:.2f}%)")
    
    # Get outlier scores (negative scores indicate outliers)
    scores = -lof.negative_outlier_factor_
    
    return lof, labels, scores, n_outliers


def visualize_outliers(X_scaled, labels_dict, method_names, feature_cols, 
                       df_clean, output_dir="../outputs/plots"):
    """
    Visualize detected outliers.
    
    Args:
        X_scaled: Standardized feature matrix
        labels_dict: Dictionary of method names to labels
        method_names: List of method names
        feature_cols: Feature column names
        df_clean: Cleaned dataframe
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use PCA for 2D visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    n_methods = len(method_names)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, method_name in enumerate(method_names[:4]):  # Show up to 4 methods
        labels = labels_dict[method_name]
        
        # Separate inliers and outliers
        inlier_mask = labels == 1
        outlier_mask = labels == -1
        
        # Plot inliers
        axes[idx].scatter(X_pca[inlier_mask, 0], X_pca[inlier_mask, 1], 
                         c='blue', alpha=0.5, s=10, label='Inliers')
        
        # Plot outliers
        if outlier_mask.sum() > 0:
            axes[idx].scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1], 
                             c='red', alpha=0.8, s=30, marker='x', 
                             label=f'Outliers ({outlier_mask.sum()})', linewidths=2)
        
        axes[idx].set_title(f'{method_name} Outlier Detection', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/outlier_detection_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Outlier detection comparison saved to {output_dir}/outlier_detection_comparison.png")
    plt.close()
    
    # Create box plots for key features
    key_features = ['Close', 'Volume', 'CPI', 'Unemployment']
    key_features = [f for f in key_features if f in feature_cols]
    
    if key_features:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(key_features[:4]):
            feature_idx = feature_cols.index(feature)
            feature_values = X_scaled[:, feature_idx]
            
            # Get outliers from Isolation Forest (as reference)
            if 'Isolation Forest' in labels_dict:
                outlier_mask = labels_dict['Isolation Forest'] == -1
            else:
                outlier_mask = np.zeros(len(feature_values), dtype=bool)
            
            # Create box plot
            bp = axes[idx].boxplot([feature_values[~outlier_mask], feature_values[outlier_mask]], 
                                  labels=['Inliers', 'Outliers'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Standardized Value')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/outlier_feature_distributions.png", dpi=300, bbox_inches='tight')
        print(f"✅ Feature distribution plots saved to {output_dir}/outlier_feature_distributions.png")
        plt.close()


def analyze_outlier_characteristics(df_clean, labels_dict, feature_cols, output_dir="../outputs"):
    """
    Analyze characteristics of detected outliers.
    
    Args:
        df_clean: Cleaned dataframe
        labels_dict: Dictionary of method names to labels
        feature_cols: Feature column names
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("OUTLIER CHARACTERISTICS ANALYSIS")
    print(f"{'='*60}")
    
    for method_name, labels in labels_dict.items():
        outlier_mask = labels == -1
        n_outliers = outlier_mask.sum()
        
        if n_outliers == 0:
            print(f"\n{method_name}: No outliers detected")
            continue
        
        print(f"\n{method_name}: {n_outliers} outliers")
        outlier_df = df_clean[outlier_mask].copy()
        inlier_df = df_clean[~outlier_mask].copy()
        
        print("\nOutlier Statistics (mean values):")
        outlier_stats = {}
        for col in feature_cols:
            if col in outlier_df.columns:
                outlier_mean = outlier_df[col].mean()
                inlier_mean = inlier_df[col].mean()
                diff_pct = ((outlier_mean - inlier_mean) / inlier_mean) * 100 if inlier_mean != 0 else 0
                outlier_stats[col] = {
                    'Outlier_Mean': outlier_mean,
                    'Inlier_Mean': inlier_mean,
                    'Difference_Pct': diff_pct
                }
                print(f"  {col}: {outlier_mean:.2f} (vs inliers: {diff_pct:+.1f}%)")
        
        # Save statistics
        stats_df = pd.DataFrame(outlier_stats).T
        stats_df.to_csv(f"{output_dir}/outlier_stats_{method_name.lower().replace(' ', '_')}.csv")
        
        # Check dates of outliers
        if 'Date' in outlier_df.columns:
            print(f"\nOutlier date range: {outlier_df['Date'].min()} to {outlier_df['Date'].max()}")
            print(f"Inlier date range: {inlier_df['Date'].min()} to {inlier_df['Date'].max()}")


def compare_outlier_methods(labels_dict, output_dir="../outputs"):
    """
    Compare results from different outlier detection methods.
    
    Args:
        labels_dict: Dictionary of method names to labels
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("OUTLIER DETECTION METHODS COMPARISON")
    print(f"{'='*60}")
    
    methods = list(labels_dict.keys())
    n_methods = len(methods)
    
    # Count outliers per method
    outlier_counts = {method: (labels == -1).sum() for method, labels in labels_dict.items()}
    
    # Create agreement matrix
    agreement_matrix = np.zeros((n_methods, n_methods))
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            labels1 = labels_dict[method1]
            labels2 = labels_dict[method2]
            # Agreement: both say outlier or both say inlier
            agreement = ((labels1 == labels2).sum() / len(labels1)) * 100
            agreement_matrix[i, j] = agreement
    
    # Print comparison
    comparison_data = []
    for method in methods:
        comparison_data.append({
            'Method': method,
            'N_Outliers': outlier_counts[method],
            'Outlier_Pct': (outlier_counts[method] / len(labels_dict[method])) * 100
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nOutlier Counts:")
    print(comparison_df.to_string(index=False))
    
    # Print agreement matrix
    print("\nMethod Agreement Matrix (%):")
    agreement_df = pd.DataFrame(agreement_matrix, index=methods, columns=methods)
    print(agreement_df.round(2).to_string())
    
    # Save comparison
    comparison_df.to_csv(f"{output_dir}/outlier_detection_comparison.csv", index=False)
    agreement_df.to_csv(f"{output_dir}/outlier_method_agreement.csv")
    
    print(f"\n✅ Comparison saved to {output_dir}/outlier_detection_comparison.csv")
    print(f"✅ Agreement matrix saved to {output_dir}/outlier_method_agreement.csv")
    
    return comparison_df, agreement_df


def main():
    """Main function to run outlier detection analysis."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuration
    ticker = "S&P500"  # Can be modified to analyze different datasets
    file_path = os.path.join(project_root, "data", "processed", f"{ticker}_merged.csv")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Merged data file not found at {file_path}")
        print("   Please run 'python src/integrate_data.py' first to create merged data.")
        return None
    
    print(f"\n{'='*60}")
    print(f"OUTLIER DETECTION ANALYSIS FOR {ticker}")
    print(f"{'='*60}")
    
    # Prepare data
    print("\n📊 Preparing data...")
    try:
        X_scaled, X, df_clean, feature_cols, scaler = prepare_data(file_path, drop_na=True)
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return None
    print(f"Data shape: {X_scaled.shape}")
    print(f"Features: {', '.join(feature_cols)}")
    
    # Define what constitutes an outlier in this context
    print(f"\n{'='*60}")
    print("OUTLIER DEFINITION")
    print(f"{'='*60}")
    print("In the context of macroeconomic stock analysis, outliers represent:")
    print("1. Extreme market events (crashes, bubbles, flash crashes)")
    print("2. Unusual macroeconomic conditions (recessions, hyperinflation)")
    print("3. Data quality issues (missing values, recording errors)")
    print("4. Rare but valid market conditions (market corrections, policy shocks)")
    print(f"{'='*60}")
    
    # Apply different outlier detection methods
    labels_dict = {}
    
    # Statistical methods
    print("\n🔍 Applying statistical outlier detection...")
    labels_iqr, n_iqr = statistical_outlier_detection(X, feature_cols, method='iqr')
    labels_dict['IQR Method'] = labels_iqr
    
    labels_zscore, n_zscore = statistical_outlier_detection(X, feature_cols, method='zscore', threshold=3)
    labels_dict['Z-Score Method'] = labels_zscore
    
    # Isolation Forest
    print("\n🔍 Applying Isolation Forest...")
    iso_forest, labels_iso, scores_iso, n_iso = isolation_forest_detection(
        X_scaled, contamination=0.1, random_state=42
    )
    labels_dict['Isolation Forest'] = labels_iso
    
    # One-Class SVM
    print("\n🔍 Applying One-Class SVM...")
    oc_svm, labels_svm, scores_svm, n_svm = oneclass_svm_detection(
        X_scaled, nu=0.1, kernel='rbf', gamma='scale'
    )
    labels_dict['One-Class SVM'] = labels_svm
    
    # Local Outlier Factor
    print("\n🔍 Applying Local Outlier Factor...")
    lof, labels_lof, scores_lof, n_lof = local_outlier_factor_detection(
        X_scaled, n_neighbors=20, contamination=0.1
    )
    labels_dict['Local Outlier Factor'] = labels_lof
    
    # Set up output directories
    output_plots_dir = os.path.join(project_root, "outputs", "plots")
    output_dir = os.path.join(project_root, "outputs")
    
    # Visualize results
    print("\n📈 Creating visualizations...")
    visualize_outliers(X_scaled, labels_dict, list(labels_dict.keys()), 
                      feature_cols, df_clean, output_dir=output_plots_dir)
    
    # Analyze outlier characteristics
    print("\n🔍 Analyzing outlier characteristics...")
    analyze_outlier_characteristics(df_clean, labels_dict, feature_cols, output_dir=output_dir)
    
    # Compare methods
    compare_outlier_methods(labels_dict, output_dir=output_dir)
    
    print(f"\n{'='*60}")
    print("✅ OUTLIER DETECTION ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    
    return {
        'labels_dict': labels_dict,
        'X_scaled': X_scaled,
        'df_clean': df_clean,
        'feature_cols': feature_cols
    }


if __name__ == "__main__":
    main()

