"""
Integrated Analysis Module
Combines clustering, dimensionality reduction, and outlier detection to provide comprehensive insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import analysis modules
from clustering_analysis import (
    prepare_data as prepare_clustering_data,
    kmeans_clustering,
    hierarchical_clustering,
    dbscan_clustering,
    find_optimal_k
)
from dimensionality_reduction import (
    prepare_data as prepare_dr_data,
    apply_pca,
    apply_tsne,
    apply_umap
)
from outlier_detection import (
    prepare_data as prepare_outlier_data,
    isolation_forest_detection,
    oneclass_svm_detection,
    statistical_outlier_detection
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def analyze_with_outliers_removed(X_scaled, df_clean, feature_cols, outlier_labels, 
                                   method_name="Isolation Forest"):
    """
    Analyze clustering and dimensionality reduction after removing outliers.
    
    Args:
        X_scaled: Standardized feature matrix
        df_clean: Cleaned dataframe
        feature_cols: Feature column names
        outlier_labels: Outlier detection labels (-1 for outliers, 1 for inliers)
        method_name: Name of outlier detection method
    
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS WITH OUTLIERS REMOVED ({method_name})")
    print(f"{'='*60}")
    
    # Remove outliers
    inlier_mask = outlier_labels == 1
    X_clean = X_scaled[inlier_mask]
    df_clean_no_outliers = df_clean[inlier_mask].reset_index(drop=True)
    
    print(f"Original samples: {len(X_scaled)}")
    print(f"Samples after removing outliers: {len(X_clean)}")
    print(f"Outliers removed: {(~inlier_mask).sum()} ({(~inlier_mask).sum()/len(X_scaled)*100:.2f}%)")
    
    # Find optimal k for cleaned data
    optimal_k, _, _, _ = find_optimal_k(X_clean, max_k=10)
    print(f"Optimal number of clusters (cleaned data): {optimal_k}")
    
    # Apply clustering on cleaned data
    kmeans_model, labels_kmeans, sil_kmeans, cal_kmeans = kmeans_clustering(
        X_clean, n_clusters=optimal_k, random_state=42
    )
    
    # Apply PCA on cleaned data
    pca, X_pca, explained_variance, cumulative_variance, pca_full = apply_pca(
        X_clean, n_components=2
    )
    
    return {
        'X_clean': X_clean,
        'df_clean': df_clean_no_outliers,
        'labels_kmeans': labels_kmeans,
        'X_pca': X_pca,
        'pca': pca,
        'optimal_k': optimal_k,
        'silhouette': sil_kmeans,
        'calinski': cal_kmeans
    }


def visualize_integrated_results(X_scaled, X_pca_original, X_pca_cleaned, 
                                labels_kmeans_original, labels_kmeans_cleaned,
                                outlier_labels, output_dir="../outputs/plots"):
    """
    Visualize integrated results showing impact of outlier removal.
    
    Args:
        X_scaled: Original standardized data
        X_pca_original: PCA on original data
        X_pca_cleaned: PCA on cleaned data (outliers removed)
        labels_kmeans_original: K-means labels on original data
        labels_kmeans_cleaned: K-means labels on cleaned data
        outlier_labels: Outlier detection labels
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Original data with outliers highlighted
    inlier_mask = outlier_labels == 1
    outlier_mask = outlier_labels == -1
    
    axes[0, 0].scatter(X_pca_original[inlier_mask, 0], X_pca_original[inlier_mask, 1],
                      c=labels_kmeans_original[inlier_mask], cmap='viridis', 
                      alpha=0.5, s=15, label='Inliers')
    if outlier_mask.sum() > 0:
        axes[0, 0].scatter(X_pca_original[outlier_mask, 0], X_pca_original[outlier_mask, 1],
                          c='red', alpha=0.8, s=50, marker='x', 
                          label='Outliers', linewidths=2)
    axes[0, 0].set_title('Original Data: Clusters + Outliers', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cleaned data (outliers removed)
    scatter = axes[0, 1].scatter(X_pca_cleaned[:, 0], X_pca_cleaned[:, 1],
                                c=labels_kmeans_cleaned, cmap='viridis', 
                                alpha=0.6, s=20)
    axes[0, 1].set_title('Cleaned Data: Clusters (Outliers Removed)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Comparison: cluster assignments
    # Map original inliers to cleaned indices
    original_inlier_indices = np.where(inlier_mask)[0]
    cluster_comparison = pd.DataFrame({
        'Original_Cluster': labels_kmeans_original[inlier_mask],
        'Cleaned_Cluster': labels_kmeans_cleaned
    })
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(cluster_comparison['Original_Cluster'], 
                         cluster_comparison['Cleaned_Cluster'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Cluster Assignment Comparison\n(Original vs Cleaned)', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Cleaned Clusters')
    axes[1, 0].set_ylabel('Original Clusters')
    
    # Outlier impact on clustering quality
    metrics_data = {
        'Metric': ['Silhouette Score', 'Calinski-Harabasz Score'],
        'Original': [0, 0],  # Will be filled
        'Cleaned': [0, 0]    # Will be filled
    }
    
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.5, 'Metrics comparison\nwill be shown in summary', 
                   ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/integrated_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✅ Integrated analysis visualization saved to {output_dir}/integrated_analysis.png")
    plt.close()


def generate_insights_summary(clustering_results, dr_results, outlier_results, 
                             cleaned_results, output_dir="../outputs"):
    """
    Generate comprehensive insights summary.
    
    Args:
        clustering_results: Results from clustering analysis
        dr_results: Results from dimensionality reduction
        outlier_results: Results from outlier detection
        cleaned_results: Results after removing outliers
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS INSIGHTS")
    print(f"{'='*80}")
    
    insights = []
    
    # 1. Clustering Insights
    print("\n1. CLUSTERING ANALYSIS INSIGHTS")
    print("-" * 80)
    
    best_clustering = max(clustering_results.items(), 
                         key=lambda x: x[1].get('silhouette', -1))
    print(f"   • Best performing method: {best_clustering[0]}")
    print(f"     - Silhouette Score: {best_clustering[1].get('silhouette', 0):.4f}")
    print(f"     - Number of clusters: {best_clustering[1].get('n_clusters', 0)}")
    
    insights.append({
        'Category': 'Clustering',
        'Finding': f"Best method: {best_clustering[0]}",
        'Details': f"Silhouette: {best_clustering[1].get('silhouette', 0):.4f}, "
                  f"Clusters: {best_clustering[1].get('n_clusters', 0)}"
    })
    
    # 2. Dimensionality Reduction Insights
    print("\n2. DIMENSIONALITY REDUCTION INSIGHTS")
    print("-" * 80)
    
    if 'pca' in dr_results:
        pca_variance = dr_results['pca'].explained_variance_ratio_.sum()
        print(f"   • PCA (2D) explains {pca_variance:.2%} of total variance")
        print(f"   • This suggests {'strong' if pca_variance > 0.5 else 'moderate'} "
              f"dimensionality reduction potential")
        
        insights.append({
            'Category': 'Dimensionality Reduction',
            'Finding': 'PCA Variance Explained',
            'Details': f"2D PCA explains {pca_variance:.2%} of variance"
        })
    
    # 3. Outlier Detection Insights
    print("\n3. OUTLIER DETECTION INSIGHTS")
    print("-" * 80)
    
    if 'labels_dict' in outlier_results:
        outlier_counts = {method: (labels == -1).sum() 
                         for method, labels in outlier_results['labels_dict'].items()}
        avg_outliers = np.mean(list(outlier_counts.values()))
        print(f"   • Average outliers detected: {avg_outliers:.0f} "
              f"({avg_outliers/len(outlier_results['X_scaled'])*100:.2f}%)")
        print(f"   • Outliers likely represent:")
        print(f"     - Market crashes or extreme volatility events")
        print(f"     - Unusual macroeconomic conditions")
        print(f"     - Data quality issues or recording errors")
        
        insights.append({
            'Category': 'Outlier Detection',
            'Finding': 'Outlier Prevalence',
            'Details': f"Average {avg_outliers:.0f} outliers ({avg_outliers/len(outlier_results['X_scaled'])*100:.2f}%)"
        })
    
    # 4. Impact of Outlier Removal
    print("\n4. IMPACT OF OUTLIER REMOVAL")
    print("-" * 80)
    
    if cleaned_results:
        original_sil = clustering_results.get('K-Means', {}).get('silhouette', 0)
        cleaned_sil = cleaned_results.get('silhouette', 0)
        improvement = cleaned_sil - original_sil
        
        print(f"   • Clustering quality (Silhouette Score):")
        print(f"     - Original: {original_sil:.4f}")
        print(f"     - After outlier removal: {cleaned_sil:.4f}")
        print(f"     - Change: {improvement:+.4f} ({improvement/original_sil*100:+.1f}%)")
        
        if improvement > 0:
            print(f"   • ✅ Outlier removal improved clustering quality")
        else:
            print(f"   • ⚠️ Outlier removal did not improve clustering quality")
        
        insights.append({
            'Category': 'Integration',
            'Finding': 'Outlier Removal Impact',
            'Details': f"Silhouette: {original_sil:.4f} → {cleaned_sil:.4f} ({improvement:+.4f})"
        })
    
    # 5. Strategic Insights
    print("\n5. STRATEGIC AND ACTIONABLE INSIGHTS")
    print("-" * 80)
    
    strategic_insights = [
        "• Market Regimes: Clusters likely represent different market regimes "
        "(bull markets, bear markets, high volatility periods)",
        "• Macroeconomic Sensitivity: Dimensionality reduction reveals which "
        "macro indicators drive most variance in stock performance",
        "• Risk Management: Outliers represent extreme events that should be "
        "considered in risk models and portfolio construction",
        "• Data Quality: Systematic outlier patterns may indicate data quality "
        "issues that need investigation",
        "• Predictive Modeling: Clusters can be used as features in predictive "
        "models to capture regime-dependent relationships"
    ]
    
    for insight in strategic_insights:
        print(f"   {insight}")
        insights.append({
            'Category': 'Strategic Insight',
            'Finding': insight.split(':')[0].replace('•', '').strip(),
            'Details': insight.split(':', 1)[1].strip() if ':' in insight else insight
        })
    
    # 6. Limitations
    print("\n6. LIMITATIONS AND NEXT STEPS")
    print("-" * 80)
    
    limitations = [
        "• Parameter Sensitivity: Clustering and outlier detection results depend "
        "on parameter choices (k, eps, contamination rate)",
        "• Temporal Dynamics: Current analysis treats all time periods equally; "
        "time-series specific methods may reveal additional patterns",
        "• Feature Selection: Current feature set may not capture all relevant "
        "dimensions; domain expertise could guide feature engineering",
        "• Validation: Clusters and outliers should be validated against known "
        "market events and economic conditions",
        "• Scalability: Some methods (t-SNE, DBSCAN) may not scale well to larger "
        "datasets without sampling"
    ]
    
    for limitation in limitations:
        print(f"   {limitation}")
    
    next_steps = [
        "• Apply time-series clustering methods (e.g., Dynamic Time Warping)",
        "• Incorporate external validation using known market events",
        "• Build predictive models using cluster assignments as features",
        "• Extend analysis to multiple stocks/sectors for comparative analysis",
        "• Implement ensemble methods combining multiple clustering algorithms"
    ]
    
    print("\n   Proposed Next Steps:")
    for step in next_steps:
        print(f"   {step}")
    
    # Save insights
    insights_df = pd.DataFrame(insights)
    insights_df.to_csv(f"{output_dir}/comprehensive_insights.csv", index=False)
    print(f"\n✅ Insights saved to {output_dir}/comprehensive_insights.csv")
    
    print(f"\n{'='*80}")
    print("✅ INTEGRATED ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    return insights_df


def main():
    """Main function to run integrated analysis."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuration
    ticker = "S&P500"
    file_path = os.path.join(project_root, "data", "processed", f"{ticker}_merged.csv")
    
    print(f"\n{'='*80}")
    print(f"INTEGRATED ANALYSIS: CLUSTERING + DIMENSIONALITY REDUCTION + OUTLIER DETECTION")
    print(f"Dataset: {ticker}")
    print(f"{'='*80}")
    
    # Step 1: Prepare data (using same preparation for consistency)
    print("\n📊 Step 1: Preparing data...")
    X_scaled, X, df_clean, feature_cols, scaler = prepare_clustering_data(file_path, drop_na=True)
    print(f"Data shape: {X_scaled.shape}")
    
    # Step 2: Dimensionality Reduction
    print("\n📊 Step 2: Applying dimensionality reduction...")
    dr_results = {}
    pca, X_pca, _, _, pca_full = apply_pca(X_scaled, n_components=2)
    dr_results['pca'] = pca
    dr_results['X_pca'] = X_pca
    
    # Step 3: Clustering on original data
    print("\n📊 Step 3: Applying clustering on original data...")
    optimal_k, _, _, _ = find_optimal_k(X_scaled, max_k=10)
    kmeans_model, labels_kmeans_original, sil_kmeans, cal_kmeans = kmeans_clustering(
        X_scaled, n_clusters=optimal_k, random_state=42
    )
    
    clustering_results = {
        'K-Means': {
            'silhouette': sil_kmeans,
            'calinski': cal_kmeans,
            'n_clusters': optimal_k
        }
    }
    
    # Step 4: Outlier Detection
    print("\n📊 Step 4: Detecting outliers...")
    iso_forest, labels_outliers, scores_iso, n_iso = isolation_forest_detection(
        X_scaled, contamination=0.1, random_state=42
    )
    
    outlier_results = {
        'labels_dict': {'Isolation Forest': labels_outliers},
        'X_scaled': X_scaled
    }
    
    # Step 5: Analysis with outliers removed
    print("\n📊 Step 5: Analyzing impact of outlier removal...")
    cleaned_results = analyze_with_outliers_removed(
        X_scaled, df_clean, feature_cols, labels_outliers, "Isolation Forest"
    )
    
    # Step 6: Visualize integrated results
    print("\n📊 Step 6: Creating integrated visualizations...")
    visualize_integrated_results(
        X_scaled, X_pca, cleaned_results['X_pca'],
        labels_kmeans_original, cleaned_results['labels_kmeans'],
        labels_outliers
    )
    
    # Step 7: Generate comprehensive insights
    print("\n📊 Step 7: Generating comprehensive insights...")
    insights_df = generate_insights_summary(
        clustering_results, dr_results, outlier_results, cleaned_results
    )
    
    print(f"\n{'='*80}")
    print("✅ ALL ANALYSES COMPLETE")
    print(f"{'='*80}\n")
    
    return {
        'clustering_results': clustering_results,
        'dr_results': dr_results,
        'outlier_results': outlier_results,
        'cleaned_results': cleaned_results,
        'insights': insights_df
    }


if __name__ == "__main__":
    main()

