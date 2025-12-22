#!/usr/bin/env python3
"""
差异表达分析（组间比较）
"""

import os
import sys
import pandas as pd
import scanpy as sc

# Snakemake 输入输出
input_h5ad = snakemake.input.h5ad
input_meta = snakemake.input.meta
output_de = snakemake.output.de_results
log_file = snakemake.log[0]

# 参数
groupby = snakemake.params.groupby
method = snakemake.params.method
reference = snakemake.params.get("reference", "Control")

# 设置日志
os.makedirs(os.path.dirname(output_de), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("Loading data...")
adata = sc.read_h5ad(input_h5ad)
print(f"Shape: {adata.shape}")

# 检查分组变量
if groupby not in adata.obs.columns:
    raise ValueError(f"Grouping variable '{groupby}' not found in adata.obs")

groups = adata.obs[groupby].unique()
print(f"\nGroups in '{groupby}': {groups.tolist()}")

if len(groups) < 2:
    raise ValueError(f"Need at least 2 groups for comparison, found {len(groups)}")

# 检查 Reference 是否存在
if reference not in groups:
    print(f"Warning: Reference group '{reference}' not found in {groupby}. Falling back to 'rest' (One-vs-Rest).")
    reference = 'rest'
else:
    print(f"Using reference group: {reference}")

# 进行差异表达分析
print(f"\nPerforming differential expression analysis using {method} (reference: {reference})...")
sc.tl.rank_genes_groups(
    adata,
    groupby=groupby,
    reference=reference,
    method=method,
    use_raw=True,
    key_added='rank_genes_groups_de'
)

# 提取所有组的结果
print("\nExtracting results...")
de_results = []

# 确定要处理的组 (如果是 pairwise，不需要处理 reference 组本身)
if reference == 'rest':
    groups_to_process = groups
else:
    groups_to_process = [g for g in groups if g != reference]

for group in groups_to_process:
    print(f"  Processing group: {group}")
    result_df = sc.get.rank_genes_groups_df(adata, group=str(group), key='rank_genes_groups_de')
    result_df['comparison_group'] = group
    result_df['reference_group'] = reference
    de_results.append(result_df)

# 合并所有结果
if not de_results:
    print("No results found!")
    de_df = pd.DataFrame(columns=['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj', 'comparison_group', 'reference_group'])
else:
    de_df = pd.concat(de_results, ignore_index=True)

# 添加基因名称（gene symbol）
print("Adding gene symbols...")
if 'gene_name' in adata.var.columns:
    # 创建gene_id到gene_name的映射
    gene_id_to_name = dict(zip(adata.var.index, adata.var['gene_name']))
    # 添加gene_symbol列
    de_df['gene_symbol'] = de_df['names'].map(gene_id_to_name)
    # 对于没有映射的，使用原始ID
    de_df['gene_symbol'] = de_df['gene_symbol'].fillna(de_df['names'])
    
    # 重新排列列顺序，将gene_symbol放在names后面
    cols = ['names', 'gene_symbol'] + [c for c in de_df.columns if c not in ['names', 'gene_symbol']]
    de_df = de_df[cols]
    
    mapped_count = (de_df['gene_symbol'] != de_df['names']).sum()
    print(f"  Mapped {mapped_count}/{len(de_df)} genes to gene symbols")

# 排序
de_df = de_df.sort_values(['comparison_group', 'pvals_adj', 'scores'], ascending=[True, True, False])

# 统计
print(f"\nDifferential expression results (vs {reference}):")
for group in groups_to_process:
    group_results = de_df[de_df['comparison_group'] == group]
    n_sig = (group_results['pvals_adj'] < 0.05).sum()
    n_up = ((group_results['pvals_adj'] < 0.05) & (group_results['logfoldchanges'] > 0)).sum()
    n_down = ((group_results['pvals_adj'] < 0.05) & (group_results['logfoldchanges'] < 0)).sum()
    print(f"  {group}: {n_sig} significant DEGs (padj < 0.05)")
    print(f"    Up-regulated: {n_up}")
    print(f"    Down-regulated: {n_down}")

# 保存
print(f"\nSaving to {output_de}")
de_df.to_csv(output_de, index=False)

# 显示每组的 top 10 DEGs
print("\nTop 10 DEGs per group:")
for group in sorted(groups_to_process):
    group_degs = de_df[(de_df['comparison_group'] == group) & (de_df['pvals_adj'] < 0.05)].head(10)
    print(f"\nGroup {group}:")
    if len(group_degs) == 0:
        print("  No significant DEGs found")
    else:
        for _, row in group_degs.iterrows():
            print(f"  {row['names']}: log2FC={row['logfoldchanges']:.2f}, padj={row['pvals_adj']:.2e}")

print("\nDone!")
