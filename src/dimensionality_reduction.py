"""
Dimensionality Reduction Module
Applies PCA, t-SNE, and UMAP to reduce dimensions and visualize data structure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from umap import UMAP
except ImportError:
    # Fallback for older versions
    import umap.umap_ as umap
    UMAP = umap.UMAP
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def prepare_data(file_path, drop_na=True):
    """
    Load and prepare data for dimensionality reduction.
    
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


def apply_pca(X_scaled, n_components=2):
    """
    Apply Principal Component Analysis (PCA).
    
    Args:
        X_scaled: Standardized feature matrix
        n_components: Number of components to keep
    
    Returns:
        PCA model, transformed data, and explained variance
    """
    print(f"\n{'='*60}")
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print(f"{'='*60}")
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"Number of components: {n_components}")
    print(f"Explained variance per component: {explained_variance}")
    print(f"Cumulative explained variance: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.2f}%)")
    
    # Get full PCA to see total variance explained
    pca_full = PCA()
    pca_full.fit(X_scaled)
    total_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    print(f"\nTop 5 components explain {total_variance[4]:.4f} ({total_variance[4]*100:.2f}%) of variance")
    print(f"Top 10 components explain {total_variance[9] if len(total_variance) > 9 else total_variance[-1]:.4f} "
          f"({(total_variance[9] if len(total_variance) > 9 else total_variance[-1])*100:.2f}%) of variance")
    
    return pca, X_pca, explained_variance, cumulative_variance, pca_full


def apply_tsne(X_scaled, n_components=2, perplexity=30, random_state=42):
    """
    Apply t-Distributed Stochastic Neighbor Embedding (t-SNE).
    
    Args:
        X_scaled: Standardized feature matrix
        n_components: Number of dimensions for embedding
        perplexity: Perplexity parameter (typically 5-50)
        random_state: Random seed
    
    Returns:
        t-SNE model and transformed data
    """
    print(f"\n{'='*60}")
    print("t-SNE DIMENSIONALITY REDUCTION")
    print(f"{'='*60}")
    print(f"Number of components: {n_components}")
    print(f"Perplexity: {perplexity}")
    
    # Adjust perplexity if dataset is too small
    n_samples = X_scaled.shape[0]
    if perplexity >= n_samples:
        perplexity = max(5, n_samples - 1)
        print(f"Adjusted perplexity to {perplexity} (dataset size: {n_samples})")
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=random_state, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    
    print("✅ t-SNE transformation complete")
    print(f"Output shape: {X_tsne.shape}")
    
    return tsne, X_tsne


def apply_umap(X_scaled, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Apply Uniform Manifold Approximation and Projection (UMAP).
    
    Args:
        X_scaled: Standardized feature matrix
        n_components: Number of dimensions for embedding
        n_neighbors: Number of neighbors to consider
        min_dist: Minimum distance between points in embedding
        random_state: Random seed
    
    Returns:
        UMAP model and transformed data
    """
    print(f"\n{'='*60}")
    print("UMAP DIMENSIONALITY REDUCTION")
    print(f"{'='*60}")
    print(f"Number of components: {n_components}")
    print(f"n_neighbors: {n_neighbors}")
    print(f"min_dist: {min_dist}")
    
    reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                   min_dist=min_dist, random_state=random_state)
    X_umap = reducer.fit_transform(X_scaled)
    
    print("✅ UMAP transformation complete")
    print(f"Output shape: {X_umap.shape}")
    
    return reducer, X_umap


def visualize_pca_variance(pca_full, output_dir="../outputs/plots"):
    """
    Visualize PCA variance explained.
    
    Args:
        pca_full: Full PCA model (all components)
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scree plot
    axes[0].bar(range(1, min(21, len(explained_variance) + 1)), 
                explained_variance[:20], alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('PCA Scree Plot (Top 20 Components)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative variance plot
    axes[1].plot(range(1, min(21, len(cumulative_variance) + 1)), 
                 cumulative_variance[:20], marker='o', linewidth=2, markersize=6, color='crimson')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    axes[1].axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_variance_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✅ PCA variance plot saved to {output_dir}/pca_variance_analysis.png")
    plt.close()


def visualize_reductions(X_scaled, X_pca, X_tsne, X_umap, df_clean, 
                         feature_cols, output_dir="../outputs/plots"):
    """
    Create before-and-after visualizations of dimensionality reduction.
    
    Args:
        X_scaled: Original standardized data
        X_pca: PCA-transformed data
        X_tsne: t-SNE-transformed data
        X_umap: UMAP-transformed data
        df_clean: Cleaned dataframe
        feature_cols: Feature column names
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create color mapping based on a key feature (e.g., Close price)
    if 'Close' in feature_cols:
        close_idx = feature_cols.index('Close')
        colors = X_scaled[:, close_idx]
        color_label = 'Close Price (Standardized)'
    else:
        colors = np.zeros(len(X_scaled))
        color_label = 'Uniform'
    
    # Before reduction: 2D projection of first two features
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Original data (first 2 features)
    scatter0 = axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                  c=colors, cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].set_title(f'Original Data: {feature_cols[0]} vs {feature_cols[1]}', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel(feature_cols[0])
    axes[0, 0].set_ylabel(feature_cols[1])
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter0, ax=axes[0, 0], label=color_label)
    
    # PCA
    scatter1 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=colors, cmap='viridis', alpha=0.6, s=20)
    axes[0, 1].set_title('PCA Reduction (2D)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 1], label=color_label)
    
    # t-SNE
    scatter2 = axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                 c=colors, cmap='plasma', alpha=0.6, s=20)
    axes[1, 0].set_title('t-SNE Reduction (2D)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1, 0], label=color_label)
    
    # UMAP
    scatter3 = axes[1, 1].scatter(X_umap[:, 0], X_umap[:, 1], 
                                 c=colors, cmap='coolwarm', alpha=0.6, s=20)
    axes[1, 1].set_title('UMAP Reduction (2D)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 1], label=color_label)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dimensionality_reduction_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Dimensionality reduction comparison saved to {output_dir}/dimensionality_reduction_comparison.png")
    plt.close()
    
    # Create 3D visualization if possible
    try:
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X_scaled)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                            c=colors, cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA 3D Visualization', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label=color_label, shrink=0.8)
        plt.savefig(f"{output_dir}/pca_3d_visualization.png", dpi=300, bbox_inches='tight')
        print(f"✅ PCA 3D visualization saved to {output_dir}/pca_3d_visualization.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not create 3D visualization: {e}")


def analyze_pca_components(pca, feature_cols, n_components=5, output_dir="../outputs"):
    """
    Analyze PCA component loadings.
    
    Args:
        pca: Fitted PCA model
        feature_cols: Feature column names
        n_components: Number of top components to analyze
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    components = pca.components_[:n_components]
    explained_variance = pca.explained_variance_ratio_[:n_components]
    
    print(f"\n{'='*60}")
    print("PCA COMPONENT ANALYSIS")
    print(f"{'='*60}")
    
    component_data = []
    for i in range(n_components):
        print(f"\nPrincipal Component {i+1} (Explains {explained_variance[i]:.2%} of variance):")
        component_dict = {'Component': f'PC{i+1}', 'Variance_Explained': explained_variance[i]}
        
        # Get top contributing features
        loadings = components[i]
        feature_contributions = list(zip(feature_cols, loadings))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("Top contributing features:")
        for feature, loading in feature_contributions[:5]:
            print(f"  {feature}: {loading:.4f}")
            component_dict[feature] = loading
        
        component_data.append(component_dict)
    
    # Save component analysis
    component_df = pd.DataFrame(component_data)
    component_df.to_csv(f"{output_dir}/pca_components_analysis.csv", index=False)
    print(f"\n✅ PCA component analysis saved to {output_dir}/pca_components_analysis.csv")
    
    return component_df


def compare_reduction_methods(X_scaled, X_pca, X_tsne, X_umap, output_dir="../outputs"):
    """
    Compare different dimensionality reduction methods.
    
    Args:
        X_scaled: Original standardized data
        X_pca: PCA-transformed data
        X_tsne: t-SNE-transformed data
        X_umap: UMAP-transformed data
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("DIMENSIONALITY REDUCTION METHODS COMPARISON")
    print(f"{'='*60}")
    
    comparison_data = {
        'Method': ['Original', 'PCA', 't-SNE', 'UMAP'],
        'Dimensions': [X_scaled.shape[1], X_pca.shape[1], X_tsne.shape[1], X_umap.shape[1]],
        'Samples': [X_scaled.shape[0], X_pca.shape[0], X_tsne.shape[0], X_umap.shape[0]]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(f"{output_dir}/dimensionality_reduction_comparison.csv", index=False)
    print(f"\n✅ Comparison saved to {output_dir}/dimensionality_reduction_comparison.csv")
    
    return comparison_df


def main():
    """Main function to run dimensionality reduction analysis."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuration
    ticker = "S&P500"  # Can be modified to analyze different datasets
    file_path = os.path.join(project_root, "data", "processed", f"{ticker}_merged.csv")
    
    print(f"\n{'='*60}")
    print(f"DIMENSIONALITY REDUCTION ANALYSIS FOR {ticker}")
    print(f"{'='*60}")
    
    # Prepare data
    print("\n📊 Preparing data...")
    X_scaled, X, df_clean, feature_cols, scaler = prepare_data(file_path, drop_na=True)
    print(f"Original data shape: {X_scaled.shape}")
    print(f"Features: {', '.join(feature_cols)}")
    
    # Apply PCA
    print("\n🔍 Applying PCA...")
    pca, X_pca, explained_variance, cumulative_variance, pca_full = apply_pca(X_scaled, n_components=2)
    
    # Visualize PCA variance
    visualize_pca_variance(pca_full)
    
    # Analyze PCA components
    analyze_pca_components(pca_full, feature_cols, n_components=5)
    
    # Apply t-SNE
    print("\n🔍 Applying t-SNE...")
    # Sample data if too large (t-SNE is computationally expensive)
    n_samples = min(1000, len(X_scaled))
    if len(X_scaled) > n_samples:
        print(f"Sampling {n_samples} points for t-SNE (original: {len(X_scaled)})")
        indices = np.random.choice(len(X_scaled), n_samples, replace=False)
        X_scaled_tsne = X_scaled[indices]
        df_clean_tsne = df_clean.iloc[indices].reset_index(drop=True)
    else:
        X_scaled_tsne = X_scaled
        df_clean_tsne = df_clean.reset_index(drop=True)
    
    tsne, X_tsne = apply_tsne(X_scaled_tsne, n_components=2, perplexity=30, random_state=42)
    
    # Apply UMAP
    print("\n🔍 Applying UMAP...")
    reducer, X_umap = apply_umap(X_scaled, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    # For visualization, use sampled data if t-SNE was sampled
    if len(X_scaled) > n_samples:
        X_scaled_viz = X_scaled[indices]
        X_pca_viz = X_pca[indices]
        X_umap_viz = X_umap[indices]
        df_clean_viz = df_clean_tsne
    else:
        X_scaled_viz = X_scaled
        X_pca_viz = X_pca
        X_umap_viz = X_umap
        df_clean_viz = df_clean
    
    visualize_reductions(X_scaled_viz, X_pca_viz, X_tsne, X_umap_viz, 
                         df_clean_viz, feature_cols)
    
    # Compare methods
    compare_reduction_methods(X_scaled, X_pca, X_tsne, X_umap)
    
    print(f"\n{'='*60}")
    print("✅ DIMENSIONALITY REDUCTION ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    
    return {
        'X_scaled': X_scaled,
        'X_pca': X_pca,
        'X_tsne': X_tsne,
        'X_umap': X_umap,
        'pca': pca,
        'pca_full': pca_full,
        'feature_cols': feature_cols,
        'df_clean': df_clean
    }


if __name__ == "__main__":
    main()

