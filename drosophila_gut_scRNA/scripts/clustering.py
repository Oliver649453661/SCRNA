#!/usr/bin/env python3
"""
聚类分析和可视化
"""

import os
import sys
import scanpy as sc
import matplotlib.pyplot as plt

# Snakemake 输入输出
input_h5ad = snakemake.input.h5ad
output_h5ad = snakemake.output.h5ad
output_umap_clusters = snakemake.output.umap_clusters
output_umap_groups = snakemake.output.umap_groups
log_file = snakemake.log[0]

# 参数
resolution = snakemake.params.resolution

# 设置日志
os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
os.makedirs(os.path.dirname(output_umap_clusters), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

# 设置 scanpy 参数
sc.settings.figdir = os.path.dirname(output_umap_clusters)
sc.settings.verbosity = 3

print("Loading data...")
adata = sc.read_h5ad(input_h5ad)
print(f"Shape: {adata.shape}")

# Leiden 聚类
print(f"\nRunning Leiden clustering (resolution={resolution})...")
sc.tl.leiden(adata, resolution=resolution, key_added='leiden')

n_clusters = len(adata.obs['leiden'].unique())
print(f"Number of clusters: {n_clusters}")

# 聚类统计
print("\nCluster sizes:")
cluster_counts = adata.obs['leiden'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"  Cluster {cluster}: {count} cells")

# UMAP 可视化：聚类
print("\nGenerating UMAP plots...")
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    adata,
    color='leiden',
    legend_loc='right margin',
    title=f'Leiden clusters (resolution={resolution})',
    show=False,
    ax=ax
)
plt.savefig(output_umap_clusters, dpi=300, bbox_inches='tight')
plt.close()

# UMAP 可视化：样本组
if 'group' in adata.obs.columns:
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(
        adata,
        color='group',
        legend_loc='right margin',
        title='Sample groups',
        show=False,
        ax=ax
    )
    plt.savefig(output_umap_groups, dpi=300, bbox_inches='tight')
    plt.close()
else:
    # 如果没有 group 信息，画 sample
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(
        adata,
        color='sample',
        legend_loc='right margin',
        title='Samples',
        show=False,
        ax=ax
    )
    plt.savefig(output_umap_groups, dpi=300, bbox_inches='tight')
    plt.close()

# 统计
print(f"\nClustering complete:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Cells per cluster: {cluster_counts.to_dict()}")

# 保存
print(f"\nSaving to {output_h5ad}")
adata.write_h5ad(output_h5ad, compression='gzip')

print("Done!")
