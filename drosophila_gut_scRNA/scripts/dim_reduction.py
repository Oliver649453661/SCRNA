#!/usr/bin/env python3
"""
降维分析：PCA、邻居图、UMAP
"""

import os
import sys
import scanpy as sc

# Snakemake 输入输出
input_h5ad = snakemake.input.h5ad
output_h5ad = snakemake.output.h5ad
log_file = snakemake.log[0]

# 参数
n_pcs = snakemake.params.n_pcs
n_neighbors = snakemake.params.n_neighbors
umap_min_dist = snakemake.params.umap_min_dist

# 设置日志
os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("Loading data...")
adata = sc.read_h5ad(input_h5ad)
print(f"Shape: {adata.shape}")

# 只使用高变基因进行降维
print(f"\nSubsetting to highly variable genes...")
n_hvg = adata.var['highly_variable'].sum()
print(f"Using {n_hvg} highly variable genes")

# 缩放数据
print("\nScaling data...")
sc.pp.scale(adata, max_value=10)

# PCA
print(f"\nRunning PCA (n_comps={n_pcs})...")
sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack', use_highly_variable=True)

# 计算邻居图
print(f"\nComputing neighbor graph (n_neighbors={n_neighbors})...")
sc.pp.neighbors(
    adata,
    n_neighbors=n_neighbors,
    n_pcs=n_pcs,
    metric='euclidean'
)

# UMAP
print(f"\nRunning UMAP (min_dist={umap_min_dist})...")
sc.tl.umap(adata, min_dist=umap_min_dist)

# 统计
print(f"\nProcessing complete:")
print(f"  PCA components: {adata.obsm['X_pca'].shape[1]}")
print(f"  UMAP coordinates: {adata.obsm['X_umap'].shape}")

# 保存
print(f"\nSaving to {output_h5ad}")
adata.write_h5ad(output_h5ad, compression='gzip')

print("Done!")
