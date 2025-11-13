# ============================================================================
# RE-CLUSTER AND VISUALIZE (After Training Complete)
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
import numpy as np
import pandas as pd

sns.set_style("whitegrid")

def cluster_and_visualize(model, passages, tokenizer, device, title="Passage Clusters", n_clusters=50):
    """
    Cluster passages and create visualization
    """
    print(f"\n[Viz] Clustering and visualizing {title}...")
    
    # Step 1: Encode passages
    print(f"  → Encoding {len(passages)} passages...")
    embeddings = model.encode_passages(passages, tokenizer, device, batch_size=64)
    embeddings_np = embeddings.numpy()
    
    # Step 2: Perform clustering
    print(f"  → Clustering into {n_clusters} clusters...")
    n_clusters_actual = min(n_clusters, max(2, len(passages) // 10))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters_actual, random_state=42, 
                             batch_size=1000, max_iter=100, verbose=0)
    cluster_labels = kmeans.fit_predict(embeddings_np)
    
    # Step 3: Create cluster dictionary
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(idx)
    
    # Step 4: PCA for 2D visualization
    print(f"  → Reducing dimensions with PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings_np)
    
    # Step 5: Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Cluster scatter plot
    scatter = axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=cluster_labels, cmap='tab20', s=15, alpha=0.6)
    axes[0, 0].set_title(f'{title}\n({n_clusters_actual} clusters, {len(passages)} passages)', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('PCA Component 1', fontsize=11)
    axes[0, 0].set_ylabel('PCA Component 2', fontsize=11)
    plt.colorbar(scatter, ax=axes[0, 0], label='Cluster ID')
    
    # Plot 2: Cluster size distribution
    sizes = [len(indices) for indices in clusters.values()]
    axes[0, 1].hist(sizes, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Cluster Size', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(sizes):.1f}')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Cluster size bar plot (top 20 clusters)
    top_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    cluster_ids = [str(c[0]) for c in top_clusters]
    cluster_sizes = [len(c[1]) for c in top_clusters]
    
    axes[1, 0].bar(range(len(cluster_ids)), cluster_sizes, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Cluster ID (Top 20)', fontsize=11)
    axes[1, 0].set_ylabel('Size', fontsize=11)
    axes[1, 0].set_title('Top 20 Largest Clusters', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(cluster_ids)))
    axes[1, 0].set_xticklabels(cluster_ids, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Statistics box
    axes[1, 1].axis('off')
    stats_text = f"""
    CLUSTERING STATISTICS
    {'='*40}
    
    Total Passages:        {len(passages):,}
    Number of Clusters:    {n_clusters_actual}
    
    Cluster Size Statistics:
    ──────────────────────
    Mean:                  {np.mean(sizes):.1f}
    Median:                {np.median(sizes):.1f}
    Std Dev:               {np.std(sizes):.2f}
    Min:                   {min(sizes)}
    Max:                   {max(sizes)}
    
    PCA Variance Explained:
    ──────────────────────
    Component 1:           {pca.explained_variance_ratio_[0]:.2%}
    Component 2:           {pca.explained_variance_ratio_[1]:.2%}
    Total:                 {sum(pca.explained_variance_ratio_):.2%}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    filename = f'{CONFIG["base_path"]}/logs/clusters_{title.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Visualization saved to {filename}")
    print(f"  Clusters: {n_clusters_actual} | Passages: {len(passages)} | Avg size: {np.mean(sizes):.1f}")
    
    return clusters, embeddings_np, cluster_labels

# ============================================================================
# VISUALIZE CLUSTERS FOR BOTH TRAINING PHASES
# ============================================================================

print("\n" + "="*80)
print("GENERATING CLUSTER VISUALIZATIONS")
print("="*80)

# For MS MARCO Pre-finetuning phase
if 'passage' in msmarco_data[0]:
    msmarco_passages = [item['passage'] for item in msmarco_data]
elif 'positive' in msmarco_data[0]:
    msmarco_passages = [item['positive'] for item in msmarco_data]

print(f"\n[1/2] Visualizing MS MARCO Pre-finetuning Clusters...")
clusters_pft, emb_pft, labels_pft = cluster_and_visualize(
    model, msmarco_passages[:10000], tokenizer, device,  # Limit to 10k for speed
    title="Pre-finetuning MS MARCO", 
    n_clusters=50
)

# For Mr. TyDi Finetuning phase
if 'passage' in mrtydi_train[0]:
    mrtydi_passages = [item['passage'] for item in mrtydi_train]
elif 'positive' in mrtydi_train[0]:
    mrtydi_passages = [item['positive'] for item in mrtydi_train]

print(f"\n[2/2] Visualizing Mr. TyDi Finetuning Clusters...")
clusters_ft, emb_ft, labels_ft = cluster_and_visualize(
    model, mrtydi_passages, tokenizer, device,
    title="Finetuning Mr TyDi", 
    n_clusters=50
)

print("\n" + "="*80)
print("✓ CLUSTER VISUALIZATIONS COMPLETE")
print("="*80)
print(f"\nMS MARCO Pre-finetuning:")
print(f"  • {len(clusters_pft)} clusters")
print(f"  • {len(emb_pft)} passages")
