#!/usr/bin/env python3
"""
标准化和识别高变基因
"""

import os
import sys
import scanpy as sc

# Snakemake 输入输出
input_h5ad = snakemake.input.h5ad
output_h5ad = snakemake.output.h5ad
log_file = snakemake.log[0]

# 参数
n_top_genes = snakemake.params.n_top_genes
target_sum = snakemake.params.target_sum

# 设置日志
os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("Loading data...")
adata = sc.read_h5ad(input_h5ad)
print(f"Shape: {adata.shape}")

# 标准化
print(f"\nNormalizing to {target_sum} counts per cell...")
sc.pp.normalize_total(adata, target_sum=target_sum)

# 对数转换
print("Log-transforming (log1p)...")
sc.pp.log1p(adata)

# 识别高变基因
print(f"\nIdentifying top {n_top_genes} highly variable genes...")
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=n_top_genes,
    flavor='seurat_v3',
    subset=False,
    layer='counts'
)

n_hvg = adata.var['highly_variable'].sum()
print(f"Highly variable genes: {n_hvg}")

# 保存原始标准化数据
adata.raw = adata

# 统计
print(f"\nFinal shape: {adata.shape}")
print(f"Highly variable genes: {n_hvg}")

# 保存
print(f"\nSaving to {output_h5ad}")
adata.write_h5ad(output_h5ad, compression='gzip')

print("Done!")
