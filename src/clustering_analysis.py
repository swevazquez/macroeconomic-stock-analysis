"""
Clustering Analysis Module
Applies K-means, Hierarchical Clustering, and DBSCAN to identify patterns in stock and macroeconomic data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def prepare_data(file_path, drop_na=True):
    """
    Load and prepare data for clustering.
    
    Args:
        file_path: Path to merged CSV file
        drop_na: Whether to drop rows with missing values
    
    Returns:
        DataFrame with features and scaler
    """
    df = pd.read_csv(file_path, parse_dates=["Date"])
    
    # Select numeric features for clustering
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


def kmeans_clustering(X_scaled, n_clusters=5, random_state=42):
    """
    Apply K-means clustering.
    
    Args:
        X_scaled: Standardized feature matrix
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        KMeans model and cluster labels
    """
    print(f"\n{'='*60}")
    print("K-MEANS CLUSTERING")
    print(f"{'='*60}")
    print(f"Number of clusters: {n_clusters}")
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Evaluate clustering
    silhouette = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {calinski:.4f}")
    print(f"Cluster distribution:\n{pd.Series(labels).value_counts().sort_index()}")
    
    return kmeans, labels, silhouette, calinski


def hierarchical_clustering(X_scaled, n_clusters=5, linkage_method='ward'):
    """
    Apply Hierarchical Clustering.
    
    Args:
        X_scaled: Standardized feature matrix
        n_clusters: Number of clusters
        linkage_method: Linkage criterion ('ward', 'complete', 'average')
    
    Returns:
        AgglomerativeClustering model and cluster labels
    """
    print(f"\n{'='*60}")
    print("HIERARCHICAL CLUSTERING")
    print(f"{'='*60}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Linkage method: {linkage_method}")
    
    # Apply hierarchical clustering
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage=linkage_method
    )
    labels = hierarchical.fit_predict(X_scaled)
    
    # Evaluate clustering
    silhouette = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {calinski:.4f}")
    print(f"Cluster distribution:\n{pd.Series(labels).value_counts().sort_index()}")
    
    return hierarchical, labels, silhouette, calinski


def dbscan_clustering(X_scaled, eps=0.5, min_samples=5):
    """
    Apply DBSCAN clustering.
    
    Args:
        X_scaled: Standardized feature matrix
        eps: Maximum distance between samples in the same neighborhood
        min_samples: Minimum number of samples in a neighborhood
    
    Returns:
        DBSCAN model and cluster labels
    """
    print(f"\n{'='*60}")
    print("DBSCAN CLUSTERING")
    print(f"{'='*60}")
    print(f"eps: {eps}")
    print(f"min_samples: {min_samples}")
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    # Count clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
    
    if n_clusters > 1:
        # Only compute scores if we have at least 2 clusters
        mask = labels != -1
        if mask.sum() > 1:
            silhouette = silhouette_score(X_scaled[mask], labels[mask])
            calinski = calinski_harabasz_score(X_scaled[mask], labels[mask])
            print(f"Silhouette Score (excluding noise): {silhouette:.4f}")
            print(f"Calinski-Harabasz Score (excluding noise): {calinski:.4f}")
        else:
            silhouette = -1
            calinski = 0
    else:
        silhouette = -1
        calinski = 0
    
    print(f"Cluster distribution:\n{pd.Series(labels).value_counts().sort_index()}")
    
    return dbscan, labels, silhouette, calinski, n_clusters, n_noise


def find_optimal_k(X_scaled, max_k=10):
    """
    Find optimal number of clusters using elbow method and silhouette analysis.
    
    Args:
        X_scaled: Standardized feature matrix
        max_k: Maximum number of clusters to test
    
    Returns:
        Optimal k value
    """
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    
    # Find optimal k (elbow method + silhouette)
    optimal_k = k_range[np.argmax(silhouettes)]
    
    return optimal_k, inertias, silhouettes, k_range


def visualize_clustering_results(X_scaled, labels_kmeans, labels_hierarchical, 
                                labels_dbscan, feature_cols, output_dir="../outputs/plots"):
    """
    Create visualizations for clustering results.
    
    Args:
        X_scaled: Standardized feature matrix
        labels_kmeans: K-means cluster labels
        labels_hierarchical: Hierarchical cluster labels
        labels_dbscan: DBSCAN cluster labels
        feature_cols: List of feature names
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use PCA for 2D visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # K-means visualization
    scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, 
                                  cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].set_title('K-Means Clustering (PCA Projection)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # Hierarchical clustering visualization
    scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_hierarchical, 
                                  cmap='plasma', alpha=0.6, s=20)
    axes[0, 1].set_title('Hierarchical Clustering (PCA Projection)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # DBSCAN visualization
    scatter3 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan, 
                                  cmap='coolwarm', alpha=0.6, s=20)
    axes[1, 0].set_title('DBSCAN Clustering (PCA Projection)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0])
    
    # Comparison: K-means vs Hierarchical
    axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, 
                      cmap='viridis', alpha=0.4, s=15, label='K-means')
    axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_hierarchical, 
                      cmap='plasma', alpha=0.4, s=15, marker='x', label='Hierarchical')
    axes[1, 1].set_title('K-Means vs Hierarchical Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/clustering_results.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Clustering visualizations saved to {output_dir}/clustering_results.png")
    plt.close()


def create_dendrogram(X_scaled, output_dir="../outputs/plots", max_samples=100):
    """
    Create dendrogram for hierarchical clustering.
    
    Args:
        X_scaled: Standardized feature matrix
        output_dir: Output directory
        max_samples: Maximum number of samples to include (for performance)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample data if too large
    if len(X_scaled) > max_samples:
        indices = np.random.choice(len(X_scaled), max_samples, replace=False)
        X_sample = X_scaled[indices]
    else:
        X_sample = X_scaled
        indices = np.arange(len(X_scaled))
    
    # Compute linkage matrix
    linkage_matrix = linkage(X_sample, method='ward')
    
    # Create dendrogram
    plt.figure(figsize=(14, 8))
    dendrogram(linkage_matrix, truncate_mode='level', p=10)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dendrogram.png", dpi=300, bbox_inches='tight')
    print(f"✅ Dendrogram saved to {output_dir}/dendrogram.png")
    plt.close()


def interpret_clusters(df_clean, labels, feature_cols, method_name, output_dir="../outputs"):
    """
    Interpret clusters by analyzing their characteristics.
    
    Args:
        df_clean: Cleaned dataframe
        labels: Cluster labels
        feature_cols: Feature column names
        method_name: Name of clustering method
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df_clustered = df_clean.copy()
    df_clustered['Cluster'] = labels
    
    # Calculate cluster statistics
    cluster_stats = df_clustered.groupby('Cluster')[feature_cols].mean()
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    
    print(f"\n{'='*60}")
    print(f"CLUSTER INTERPRETATION - {method_name.upper()}")
    print(f"{'='*60}")
    
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        if cluster_id == -1 and method_name == 'DBSCAN':
            print(f"\nNoise Points (Cluster -1): {cluster_counts[cluster_id]} samples")
            continue
        
        print(f"\nCluster {cluster_id}: {cluster_counts[cluster_id]} samples ({cluster_counts[cluster_id]/len(df_clustered)*100:.2f}%)")
        cluster_data = cluster_stats.loc[cluster_id]
        
        # Identify key characteristics
        print("Key Characteristics:")
        for col in feature_cols:
            value = cluster_data[col]
            overall_mean = df_clean[col].mean()
            diff_pct = ((value - overall_mean) / overall_mean) * 100
            print(f"  {col}: {value:.2f} (vs overall mean: {diff_pct:+.1f}%)")
    
    # Save cluster statistics
    cluster_stats.to_csv(f"{output_dir}/cluster_stats_{method_name.lower()}.csv")
    print(f"\n✅ Cluster statistics saved to {output_dir}/cluster_stats_{method_name.lower()}.csv")
    
    return cluster_stats


def compare_clustering_methods(results_dict, output_dir="../outputs"):
    """
    Compare results from different clustering methods.
    
    Args:
        results_dict: Dictionary with method names as keys and results as values
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CLUSTERING METHODS COMPARISON")
    print(f"{'='*60}")
    
    comparison_data = []
    for method, results in results_dict.items():
        comparison_data.append({
            'Method': method,
            'Silhouette Score': results.get('silhouette', np.nan),
            'Calinski-Harabasz Score': results.get('calinski', np.nan),
            'N Clusters': results.get('n_clusters', np.nan),
            'N Noise Points': results.get('n_noise', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(f"{output_dir}/clustering_comparison.csv", index=False)
    print(f"\n✅ Comparison saved to {output_dir}/clustering_comparison.csv")
    
    return comparison_df


def main():
    """Main function to run clustering analysis."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuration
    ticker = "S&P500"  # Can be modified to analyze different datasets
    file_path = os.path.join(project_root, "data", "processed", f"{ticker}_merged.csv")
    
    print(f"\n{'='*60}")
    print(f"CLUSTERING ANALYSIS FOR {ticker}")
    print(f"{'='*60}")
    
    # Prepare data
    print("\n📊 Preparing data...")
    X_scaled, X, df_clean, feature_cols, scaler = prepare_data(file_path, drop_na=True)
    print(f"Data shape: {X_scaled.shape}")
    print(f"Features: {', '.join(feature_cols)}")
    
    # Find optimal k for K-means
    print("\n🔍 Finding optimal number of clusters...")
    optimal_k, inertias, silhouettes, k_range = find_optimal_k(X_scaled, max_k=10)
    print(f"Optimal number of clusters (based on silhouette): {optimal_k}")
    
    # Apply clustering methods
    # K-means
    kmeans_model, labels_kmeans, sil_kmeans, cal_kmeans = kmeans_clustering(
        X_scaled, n_clusters=optimal_k, random_state=42
    )
    
    # Hierarchical clustering
    hier_model, labels_hier, sil_hier, cal_hier = hierarchical_clustering(
        X_scaled, n_clusters=optimal_k, linkage_method='ward'
    )
    
    # DBSCAN (tuned parameters)
    # For high-dimensional data, eps needs to be larger
    dbscan_model, labels_dbscan, sil_dbscan, cal_dbscan, n_clusters_dbscan, n_noise = dbscan_clustering(
        X_scaled, eps=1.5, min_samples=10
    )
    
    # Visualize results
    print("\n📈 Creating visualizations...")
    output_plots_dir = os.path.join(project_root, "outputs", "plots")
    output_dir = os.path.join(project_root, "outputs")
    visualize_clustering_results(X_scaled, labels_kmeans, labels_hier, 
                                labels_dbscan, feature_cols, output_dir=output_plots_dir)
    create_dendrogram(X_scaled, output_dir=output_plots_dir, max_samples=200)
    
    # Interpret clusters
    print("\n🔍 Interpreting clusters...")
    interpret_clusters(df_clean.iloc[:len(labels_kmeans)], labels_kmeans, 
                      feature_cols, "K-Means", output_dir=output_dir)
    interpret_clusters(df_clean.iloc[:len(labels_hier)], labels_hier, 
                      feature_cols, "Hierarchical", output_dir=output_dir)
    interpret_clusters(df_clean.iloc[:len(labels_dbscan)], labels_dbscan, 
                      feature_cols, "DBSCAN", output_dir=output_dir)
    
    # Compare methods
    results_dict = {
        'K-Means': {
            'silhouette': sil_kmeans,
            'calinski': cal_kmeans,
            'n_clusters': optimal_k,
            'n_noise': 0
        },
        'Hierarchical': {
            'silhouette': sil_hier,
            'calinski': cal_hier,
            'n_clusters': optimal_k,
            'n_noise': 0
        },
        'DBSCAN': {
            'silhouette': sil_dbscan,
            'calinski': cal_dbscan,
            'n_clusters': n_clusters_dbscan,
            'n_noise': n_noise
        }
    }
    
    compare_clustering_methods(results_dict, output_dir=output_dir)
    
    print(f"\n{'='*60}")
    print("✅ CLUSTERING ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

