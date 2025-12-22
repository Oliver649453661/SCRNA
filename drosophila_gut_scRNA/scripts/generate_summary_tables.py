#!/usr/bin/env python3
"""
Generate Summary Tables for Publication
生成用于发表的汇总统计表格
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs and outputs
input_h5ad = snakemake.input.h5ad
input_de_global = snakemake.input.de_global
input_de_summary = snakemake.input.de_summary
input_composition = snakemake.input.composition
output_dir = snakemake.output.output_dir
log_file = snakemake.log[0]

# Set up logging
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("Summary Tables Generation")
print("="*80)

# ========== Load data ==========
print("\n1. Loading data...")
adata = sc.read_h5ad(input_h5ad)
print(f"   AnnData: {adata.n_obs} cells, {adata.n_vars} genes")

# ========== Table S1: Sample Statistics ==========
print("\n2. Generating Table S1: Sample Statistics...")

sample_stats = []
for sample in adata.obs['sample'].unique():
    sample_data = adata[adata.obs['sample'] == sample]
    group = sample_data.obs['group'].iloc[0] if 'group' in sample_data.obs.columns else 'Unknown'
    
    stats = {
        'Sample': sample,
        'Group': group,
        'Total_Cells': sample_data.n_obs,
        'Mean_Genes_per_Cell': sample_data.obs['n_genes_by_counts'].mean() if 'n_genes_by_counts' in sample_data.obs.columns else np.nan,
        'Median_Genes_per_Cell': sample_data.obs['n_genes_by_counts'].median() if 'n_genes_by_counts' in sample_data.obs.columns else np.nan,
        'Mean_UMI_per_Cell': sample_data.obs['total_counts'].mean() if 'total_counts' in sample_data.obs.columns else np.nan,
        'Median_UMI_per_Cell': sample_data.obs['total_counts'].median() if 'total_counts' in sample_data.obs.columns else np.nan,
        'Mean_Mito_Pct': sample_data.obs['pct_counts_mt'].mean() if 'pct_counts_mt' in sample_data.obs.columns else np.nan,
    }
    sample_stats.append(stats)

sample_df = pd.DataFrame(sample_stats)
sample_df = sample_df.sort_values(['Group', 'Sample'])
sample_df.to_csv(os.path.join(output_dir, 'Table_S1_Sample_Statistics.csv'), index=False)
print(f"   ✓ Table_S1_Sample_Statistics.csv")

# ========== Table S2: Cell Type Counts ==========
print("\n3. Generating Table S2: Cell Type Counts...")

celltype_col = 'final_cell_type' if 'final_cell_type' in adata.obs.columns else 'cell_type'
if celltype_col in adata.obs.columns:
    # Per sample
    celltype_counts = pd.crosstab(adata.obs['sample'], adata.obs[celltype_col])
    celltype_counts['Total'] = celltype_counts.sum(axis=1)
    
    # Add group info
    sample_to_group = adata.obs.groupby('sample')['group'].first().to_dict()
    celltype_counts['Group'] = celltype_counts.index.map(sample_to_group)
    
    # Reorder columns
    cols = ['Group'] + [c for c in celltype_counts.columns if c not in ['Group', 'Total']] + ['Total']
    celltype_counts = celltype_counts[cols]
    celltype_counts = celltype_counts.sort_values(['Group', celltype_counts.index.name])
    
    celltype_counts.to_csv(os.path.join(output_dir, 'Table_S2_CellType_Counts.csv'))
    print(f"   ✓ Table_S2_CellType_Counts.csv")
    
    # Per group summary
    group_celltype = pd.crosstab(adata.obs['group'], adata.obs[celltype_col])
    group_celltype['Total'] = group_celltype.sum(axis=1)
    group_celltype.to_csv(os.path.join(output_dir, 'Table_S2b_CellType_by_Group.csv'))
    print(f"   ✓ Table_S2b_CellType_by_Group.csv")
    
    # Proportions
    group_props = group_celltype.div(group_celltype['Total'], axis=0) * 100
    group_props = group_props.round(2)
    group_props.to_csv(os.path.join(output_dir, 'Table_S2c_CellType_Proportions.csv'))
    print(f"   ✓ Table_S2c_CellType_Proportions.csv")
else:
    print(f"   Warning: Cell type column not found")

# ========== Table S3: DEG Summary ==========
print("\n4. Generating Table S3: DEG Summary...")

de_global = pd.read_csv(input_de_global)
de_summary = pd.read_csv(input_de_summary)

# Global DEG summary
deg_summary = []
if 'comparison_group' in de_global.columns:
    for group in de_global['comparison_group'].unique():
        if pd.isna(group):
            continue
        group_data = de_global[de_global['comparison_group'] == group]
        sig_data = group_data[group_data['pvals_adj'] < 0.05]
        
        logfc_col = 'logfoldchanges' if 'logfoldchanges' in group_data.columns else 'log2FoldChange'
        
        deg_summary.append({
            'Comparison': f'{group} vs Control',
            'Total_Tested': len(group_data),
            'Total_DEGs': len(sig_data),
            'Up_regulated': len(sig_data[sig_data[logfc_col] > 1]),
            'Down_regulated': len(sig_data[sig_data[logfc_col] < -1]),
            'Up_strong': len(sig_data[sig_data[logfc_col] > 2]),
            'Down_strong': len(sig_data[sig_data[logfc_col] < -2]),
        })

deg_df = pd.DataFrame(deg_summary)
deg_df.to_csv(os.path.join(output_dir, 'Table_S3_DEG_Summary.csv'), index=False)
print(f"   ✓ Table_S3_DEG_Summary.csv")

# ========== Table S4: Per-Cell-Type DEG Summary ==========
print("\n5. Generating Table S4: Per-Cell-Type DEG Summary...")

# Pivot the per-celltype summary
if len(de_summary) > 0 and 'cell_type' in de_summary.columns:
    # Create a cleaner summary
    ct_deg_summary = de_summary[de_summary['status'] == 'completed'].copy()
    
    if len(ct_deg_summary) > 0:
        # Pivot table: cell types as rows, comparisons as columns
        pivot_degs = ct_deg_summary.pivot_table(
            index='cell_type',
            columns='comparison_group',
            values='n_degs',
            aggfunc='sum',
            fill_value=0
        )
        pivot_degs['Total_DEGs'] = pivot_degs.sum(axis=1)
        pivot_degs = pivot_degs.sort_values('Total_DEGs', ascending=False)
        pivot_degs.to_csv(os.path.join(output_dir, 'Table_S4_DEG_per_CellType.csv'))
        print(f"   ✓ Table_S4_DEG_per_CellType.csv")
        
        # Up/Down breakdown
        pivot_up = ct_deg_summary.pivot_table(
            index='cell_type', columns='comparison_group', values='n_up', fill_value=0
        )
        pivot_down = ct_deg_summary.pivot_table(
            index='cell_type', columns='comparison_group', values='n_down', fill_value=0
        )
        
        # Combined with direction
        combined = pd.DataFrame()
        for col in pivot_up.columns:
            combined[f'{col}_Up'] = pivot_up[col]
            combined[f'{col}_Down'] = pivot_down[col]
        combined.to_csv(os.path.join(output_dir, 'Table_S4b_DEG_Direction.csv'))
        print(f"   ✓ Table_S4b_DEG_Direction.csv")
else:
    print("   Warning: Per-celltype DE summary not available")

# ========== Table S5: Cell Composition Statistics ==========
print("\n6. Generating Table S5: Cell Composition Statistics...")

composition = pd.read_csv(input_composition)
if len(composition) > 0:
    # Filter significant changes
    sig_composition = composition[composition['pvalue_adj'] < 0.05].copy()
    sig_composition = sig_composition.sort_values('pvalue_adj')
    sig_composition.to_csv(os.path.join(output_dir, 'Table_S5_Composition_Changes.csv'), index=False)
    print(f"   ✓ Table_S5_Composition_Changes.csv ({len(sig_composition)} significant)")
    
    # Full composition table
    composition.to_csv(os.path.join(output_dir, 'Table_S5b_Composition_Full.csv'), index=False)
    print(f"   ✓ Table_S5b_Composition_Full.csv")
else:
    print("   Warning: Composition data not available")

# ========== Table S6: Top DEGs per Cell Type ==========
print("\n7. Generating Table S6: Top DEGs per Cell Type...")

combined_de_file = os.path.join(os.path.dirname(input_de_summary), "per_celltype", "all_celltype_de_results.csv")
if os.path.exists(combined_de_file):
    all_de = pd.read_csv(combined_de_file)
    
    gene_col = 'gene_symbol' if 'gene_symbol' in all_de.columns else 'names'
    logfc_col = 'logfoldchanges' if 'logfoldchanges' in all_de.columns else 'log2FoldChange'
    
    top_degs = []
    for ct in all_de['cell_type'].unique():
        ct_data = all_de[all_de['cell_type'] == ct]
        for group in ct_data['comparison_group'].unique():
            g_data = ct_data[(ct_data['comparison_group'] == group) & (ct_data['pvals_adj'] < 0.05)]
            
            # Top 10 up
            top_up = g_data.nlargest(10, logfc_col)
            for _, row in top_up.iterrows():
                top_degs.append({
                    'Cell_Type': ct,
                    'Comparison': group,
                    'Gene': row[gene_col],
                    'Log2FC': row[logfc_col],
                    'Adj_Pvalue': row['pvals_adj'],
                    'Direction': 'Up'
                })
            
            # Top 10 down
            top_down = g_data.nsmallest(10, logfc_col)
            for _, row in top_down.iterrows():
                top_degs.append({
                    'Cell_Type': ct,
                    'Comparison': group,
                    'Gene': row[gene_col],
                    'Log2FC': row[logfc_col],
                    'Adj_Pvalue': row['pvals_adj'],
                    'Direction': 'Down'
                })
    
    if top_degs:
        top_degs_df = pd.DataFrame(top_degs)
        top_degs_df.to_csv(os.path.join(output_dir, 'Table_S6_Top_DEGs.csv'), index=False)
        print(f"   ✓ Table_S6_Top_DEGs.csv")
else:
    print("   Warning: Combined DE file not found")

# ========== Summary Report ==========
print("\n8. Generating summary report...")

report = f"""# Single-Cell RNA-seq Analysis Summary

## Dataset Overview
- **Total Cells**: {adata.n_obs:,}
- **Total Genes**: {adata.n_vars:,}
- **Samples**: {adata.obs['sample'].nunique()}
- **Groups**: {', '.join(adata.obs['group'].unique())}

## Cell Types Identified
"""

if celltype_col in adata.obs.columns:
    for ct in adata.obs[celltype_col].value_counts().head(10).index:
        count = (adata.obs[celltype_col] == ct).sum()
        pct = count / adata.n_obs * 100
        report += f"- **{ct}**: {count:,} cells ({pct:.1f}%)\n"

report += f"""
## Differential Expression Summary
"""

if len(deg_summary) > 0:
    for item in deg_summary:
        report += f"- **{item['Comparison']}**: {item['Total_DEGs']} DEGs ({item['Up_regulated']} up, {item['Down_regulated']} down)\n"

report += f"""
## Output Tables
- Table_S1: Sample-level statistics
- Table_S2: Cell type counts and proportions
- Table_S3: Global DEG summary
- Table_S4: Per-cell-type DEG summary
- Table_S5: Cell composition changes
- Table_S6: Top DEGs per cell type

Generated by: drosophila_gut_scRNA pipeline
"""

with open(os.path.join(output_dir, 'Summary_Report.md'), 'w') as f:
    f.write(report)
print(f"   ✓ Summary_Report.md")

print("\n" + "="*80)
print("Summary Tables Generation Completed!")
print(f"Output directory: {output_dir}")
print("="*80)
