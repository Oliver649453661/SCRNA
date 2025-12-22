#!/usr/bin/env python3
"""
合并多个样本的 AnnData 对象
"""

import os
import sys
import pandas as pd
import scanpy as sc

# Snakemake 输入输出
h5ad_files = snakemake.input.h5ads
gene_mapping_file = snakemake.input.gene_mapping
output_h5ad = snakemake.output.h5ad
log_file = snakemake.log[0]

# 设置日志
os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print(f"Merging {len(h5ad_files)} samples...")

# 加载所有 AnnData 对象
adatas = []
for h5ad_file in h5ad_files:
    print(f"Loading {h5ad_file}...")
    adata = sc.read_h5ad(h5ad_file)
    adatas.append(adata)
    print(f"  Shape: {adata.shape}")

# 合并
print("\nConcatenating samples...")
adata_merged = sc.concat(
    adatas,
    axis=0,
    join='outer',
    label='batch',
    keys=[os.path.basename(f).replace('_raw.h5ad', '') for f in h5ad_files],
    merge='unique'
)

# 确保有 counts layer
if 'counts' not in adata_merged.layers:
    adata_merged.layers['counts'] = adata_merged.X.copy()

# 添加基因名称注释
print("\nAdding gene name annotations...")
gene_mapping = pd.read_csv(gene_mapping_file)
id_to_name = dict(zip(gene_mapping['gene_id'], gene_mapping['gene_name']))

# 为adata.var添加gene_name列
# Map FBgn gene IDs to gene names, keep others (FBti transposons) as-is
adata_merged.var['gene_name'] = adata_merged.var.index.map(lambda x: id_to_name.get(x, x))

# Count feature types
fbgn_count = sum(1 for g in adata_merged.var_names if str(g).startswith('FBgn'))
fbti_count = sum(1 for g in adata_merged.var_names if str(g).startswith('FBti'))
mapped_genes = (adata_merged.var['gene_name'] != adata_merged.var.index).sum()

print(f"Feature composition:")
print(f"  FBgn (genes): {fbgn_count}")
print(f"  FBti (transposons): {fbti_count}")
print(f"  Other: {adata_merged.n_vars - fbgn_count - fbti_count}")
print(f"Mapped {mapped_genes}/{adata_merged.n_vars} features to gene symbols")

# 统计信息
print(f"\nMerged shape: {adata_merged.shape}")
print(f"Total cells: {adata_merged.n_obs}")
print(f"Total genes: {adata_merged.n_vars}")
print(f"Samples: {adata_merged.obs['sample'].unique().tolist()}")
print(f"Groups: {adata_merged.obs['group'].unique().tolist()}")

# 保存
print(f"\nSaving to {output_h5ad}")
adata_merged.write_h5ad(output_h5ad, compression='gzip')

print("Done!")
