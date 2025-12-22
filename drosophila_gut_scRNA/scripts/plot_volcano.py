#!/usr/bin/env python3
"""
Volcano Plot for Differential Expression Results
为差异表达结果生成火山图
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs and outputs
input_de_global = snakemake.input.de_global
input_de_dir = snakemake.input.de_dir
output_plot = snakemake.output.plot
output_per_celltype = snakemake.output.per_celltype_dir
log_file = snakemake.log[0]

# Parameters
logfc_threshold = snakemake.params.get("logfc_threshold", 1.0)
pvalue_threshold = snakemake.params.get("pvalue_threshold", 0.05)

# Set up logging
os.makedirs(output_per_celltype, exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)
os.makedirs(os.path.dirname(output_plot), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("Volcano Plot Generation")
print("="*80)

# ========== Load global DE results ==========
print("\n1. Loading global DE results...")
de_global = pd.read_csv(input_de_global)
print(f"   Loaded {len(de_global)} genes")

# Get comparison groups
if 'comparison_group' in de_global.columns:
    groups = de_global['comparison_group'].unique()
elif 'group' in de_global.columns:
    groups = de_global['group'].unique()
else:
    groups = ['All']
    de_global['comparison_group'] = 'All'

print(f"   Comparison groups: {groups.tolist()}")

# ========== Create volcano plots ==========
print("\n2. Creating volcano plots...")

def create_volcano(df, title, ax, logfc_col='logfoldchanges', pval_col='pvals_adj'):
    """Create a single volcano plot"""
    # Prepare data
    df = df.copy()
    df['neg_log10_pval'] = -np.log10(df[pval_col].clip(lower=1e-300))
    
    # Classify genes
    df['category'] = 'Not Significant'
    df.loc[(df[pval_col] < pvalue_threshold) & (df[logfc_col] > logfc_threshold), 'category'] = 'Up'
    df.loc[(df[pval_col] < pvalue_threshold) & (df[logfc_col] < -logfc_threshold), 'category'] = 'Down'
    
    # Colors
    colors = {'Not Significant': '#cccccc', 'Up': '#e74c3c', 'Down': '#3498db'}
    
    # Plot
    for cat in ['Not Significant', 'Up', 'Down']:
        subset = df[df['category'] == cat]
        ax.scatter(subset[logfc_col], subset['neg_log10_pval'], 
                   c=colors[cat], s=5, alpha=0.6, label=f'{cat} ({len(subset)})')
    
    # Add threshold lines
    ax.axhline(-np.log10(pvalue_threshold), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(logfc_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-logfc_threshold, color='gray', linestyle='--', alpha=0.5)
    
    # Labels
    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10(Adjusted P-value)')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Label top genes
    gene_col = 'gene_symbol' if 'gene_symbol' in df.columns else 'names'
    if gene_col in df.columns:
        top_up = df[df['category'] == 'Up'].nlargest(5, 'neg_log10_pval')
        top_down = df[df['category'] == 'Down'].nlargest(5, 'neg_log10_pval')
        for _, row in pd.concat([top_up, top_down]).iterrows():
            ax.annotate(row[gene_col], (row[logfc_col], row['neg_log10_pval']),
                       fontsize=6, alpha=0.8)
    
    return df['category'].value_counts()

# Global volcano plots
n_groups = len(groups)
n_cols = min(3, n_groups)
n_rows = (n_groups + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
if n_groups == 1:
    axes = np.array([axes])
axes = axes.flatten()

stats_list = []
for i, group in enumerate(groups):
    if 'comparison_group' in de_global.columns:
        group_data = de_global[de_global['comparison_group'] == group]
    else:
        group_data = de_global
    
    if len(group_data) > 0:
        counts = create_volcano(group_data, f'{group} vs Control', axes[i])
        stats_list.append({
            'group': group,
            'total': len(group_data),
            'up': counts.get('Up', 0),
            'down': counts.get('Down', 0),
            'not_sig': counts.get('Not Significant', 0)
        })

# Hide empty axes
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Volcano Plots: Differential Expression Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Global volcano plot: {output_plot}")

# ========== Per-cell-type volcano plots ==========
print("\n3. Creating per-cell-type volcano plots...")

# Load per-celltype DE results
combined_file = os.path.join(input_de_dir, "all_celltype_de_results.csv")
if os.path.exists(combined_file):
    de_celltype = pd.read_csv(combined_file)
    cell_types = de_celltype['cell_type'].unique()
    
    for ct in cell_types:
        ct_data = de_celltype[de_celltype['cell_type'] == ct]
        ct_groups = ct_data['comparison_group'].unique()
        
        if len(ct_groups) == 0:
            continue
        
        n_cols = min(3, len(ct_groups))
        n_rows = (len(ct_groups) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if len(ct_groups) == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, group in enumerate(ct_groups):
            group_data = ct_data[ct_data['comparison_group'] == group]
            if len(group_data) > 0:
                create_volcano(group_data, f'{group} vs Control', axes[i])
        
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        ct_safe = ct.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        plt.suptitle(f'Volcano Plots: {ct}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        ct_plot_path = os.path.join(output_per_celltype, f'volcano_{ct_safe}.pdf')
        plt.savefig(ct_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ {ct}: {ct_plot_path}")
else:
    print("   Warning: No per-celltype DE results found")

print("\n" + "="*80)
print("Volcano Plot Generation Completed!")
print("="*80)
