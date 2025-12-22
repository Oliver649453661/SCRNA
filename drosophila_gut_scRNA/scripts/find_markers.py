#!/usr/bin/env python3
"""
识别每个聚类的 marker 基因
"""

import os
import sys
import pandas as pd
import scanpy as sc

# Snakemake 输入输出
input_h5ad = snakemake.input.h5ad
output_markers = snakemake.output.markers
log_file = snakemake.log[0]

# 参数
method = snakemake.params.method
n_genes = snakemake.params.n_genes

# 设置日志
os.makedirs(os.path.dirname(output_markers), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("Loading data...")
adata = sc.read_h5ad(input_h5ad)
print(f"Shape: {adata.shape}")

# 检查是否有聚类信息
if 'leiden' not in adata.obs.columns:
    raise ValueError("No 'leiden' clustering found in adata.obs")

n_clusters = len(adata.obs['leiden'].unique())
print(f"Number of clusters: {n_clusters}")

# 识别 marker 基因
print(f"\nFinding marker genes using {method} test...")
sc.tl.rank_genes_groups(
    adata,
    groupby='leiden',
    method=method,
    n_genes=n_genes,
    use_raw=True,
    key_added='rank_genes_groups'
)

# 提取结果
print("\nExtracting marker genes...")
result_df = sc.get.rank_genes_groups_df(adata, group=None)

# 添加基因名称（gene symbol）
print("Adding gene symbols...")
if 'gene_name' in adata.var.columns:
    # 创建gene_id到gene_name的映射
    gene_id_to_name = dict(zip(adata.var.index, adata.var['gene_name']))
    # 添加gene_symbol列
    result_df['gene_symbol'] = result_df['names'].map(gene_id_to_name)
    # 对于没有映射的，使用原始ID
    result_df['gene_symbol'] = result_df['gene_symbol'].fillna(result_df['names'])
    
    # 重新排列列顺序，将gene_symbol放在names后面
    cols = ['group', 'names', 'gene_symbol'] + [c for c in result_df.columns if c not in ['group', 'names', 'gene_symbol']]
    result_df = result_df[cols]
    
    mapped_count = (result_df['gene_symbol'] != result_df['names']).sum()
    print(f"  Mapped {mapped_count}/{len(result_df)} genes to gene symbols")

# 排序
result_df = result_df.sort_values(['group', 'pvals_adj', 'scores'], ascending=[True, True, False])

# 统计
print(f"\nMarker genes identified:")
for cluster in result_df['group'].unique():
    n_sig = (result_df['group'] == cluster) & (result_df['pvals_adj'] < 0.05)
    print(f"  Cluster {cluster}: {n_sig.sum()} significant markers (padj < 0.05)")

# 保存
print(f"\nSaving to {output_markers}")
result_df.to_csv(output_markers, index=False)

# 显示每个聚类的 top 5 marker
print("\nTop 5 markers per cluster:")
for cluster in sorted(result_df['group'].unique()):
    cluster_markers = result_df[result_df['group'] == cluster].head(5)
    print(f"\nCluster {cluster}:")
    for _, row in cluster_markers.iterrows():
        print(f"  {row['names']}: log2FC={row['logfoldchanges']:.2f}, padj={row['pvals_adj']:.2e}")

print("\nDone!")
