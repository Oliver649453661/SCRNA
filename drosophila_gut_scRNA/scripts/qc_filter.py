#!/usr/bin/env python3
"""
质控和过滤
"""

import os
import sys
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

# Snakemake 输入输出
input_h5ad = snakemake.input.h5ad
output_h5ad = snakemake.output.h5ad
output_plot = snakemake.output.plot
log_file = snakemake.log[0]

# 参数
min_genes = snakemake.params.min_genes
min_cells = snakemake.params.min_cells
max_genes = snakemake.params.max_genes
max_mito_pct = snakemake.params.max_mito_pct

# 设置日志
os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
os.makedirs(os.path.dirname(output_plot), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("Loading data...")
adata = sc.read_h5ad(input_h5ad)
print(f"Initial shape: {adata.shape}")

# 计算 QC 指标
print("\nCalculating QC metrics...")

# 识别线粒体基因（Drosophila 通常以 mt: 或 Mt: 开头）
adata.var['mt'] = adata.var_names.str.startswith(('mt:', 'Mt:', 'mt-', 'Mt-'))
print(f"Mitochondrial genes found: {adata.var['mt'].sum()}")

# 计算 QC 指标
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=['mt'],
    percent_top=None,
    log1p=False,
    inplace=True
)

# QC 可视化（过滤前）
print("\nGenerating QC plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 每个细胞的基因数
axes[0, 0].hist(adata.obs['n_genes_by_counts'], bins=100, edgecolor='black')
axes[0, 0].axvline(min_genes, color='red', linestyle='--', label=f'min={min_genes}')
axes[0, 0].axvline(max_genes, color='red', linestyle='--', label=f'max={max_genes}')
axes[0, 0].set_xlabel('Number of genes')
axes[0, 0].set_ylabel('Number of cells')
axes[0, 0].set_title('Genes per cell')
axes[0, 0].legend()

# 每个细胞的总 UMI 数
axes[0, 1].hist(adata.obs['total_counts'], bins=100, edgecolor='black')
axes[0, 1].set_xlabel('Total counts')
axes[0, 1].set_ylabel('Number of cells')
axes[0, 1].set_title('UMIs per cell')

# 线粒体比例
axes[1, 0].hist(adata.obs['pct_counts_mt'], bins=100, edgecolor='black')
axes[1, 0].axvline(max_mito_pct, color='red', linestyle='--', label=f'max={max_mito_pct}%')
axes[1, 0].set_xlabel('Mitochondrial %')
axes[1, 0].set_ylabel('Number of cells')
axes[1, 0].set_title('Mitochondrial content')
axes[1, 0].legend()

# 基因表达的细胞数
axes[1, 1].hist(adata.var['n_cells_by_counts'], bins=100, edgecolor='black')
axes[1, 1].axvline(min_cells, color='red', linestyle='--', label=f'min={min_cells}')
axes[1, 1].set_xlabel('Number of cells')
axes[1, 1].set_ylabel('Number of genes')
axes[1, 1].set_title('Cells per gene')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()

# 过滤
print(f"\nFiltering cells with:")
print(f"  min_genes: {min_genes}")
print(f"  max_genes: {max_genes}")
print(f"  max_mito_pct: {max_mito_pct}")

n_cells_before = adata.n_obs
sc.pp.filter_cells(adata, min_genes=min_genes)
adata = adata[adata.obs['n_genes_by_counts'] < max_genes, :].copy()
adata = adata[adata.obs['pct_counts_mt'] < max_mito_pct, :].copy()
n_cells_after = adata.n_obs

print(f"Cells before filtering: {n_cells_before}")
print(f"Cells after filtering: {n_cells_after}")
print(f"Cells removed: {n_cells_before - n_cells_after} ({100*(n_cells_before - n_cells_after)/n_cells_before:.2f}%)")

print(f"\nFiltering genes with min_cells: {min_cells}")
n_genes_before = adata.n_vars
sc.pp.filter_genes(adata, min_cells=min_cells)
n_genes_after = adata.n_vars

print(f"Genes before filtering: {n_genes_before}")
print(f"Genes after filtering: {n_genes_after}")
print(f"Genes removed: {n_genes_before - n_genes_after} ({100*(n_genes_before - n_genes_after)/n_genes_before:.2f}%)")

# 统计
print(f"\nFinal shape: {adata.shape}")

# 保存
print(f"\nSaving to {output_h5ad}")
adata.write_h5ad(output_h5ad, compression='gzip')

print("Done!")
