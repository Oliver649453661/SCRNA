#!/usr/bin/env python3
"""
UpSet Plot for Differential Expression Results
展示不同处理组间差异表达基因的重叠情况
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs and outputs
input_de_global = snakemake.input.de_global
input_de_dir = snakemake.input.de_dir
output_plot = snakemake.output.plot
output_overlap = snakemake.output.overlap_csv
log_file = snakemake.log[0]

# Parameters
logfc_threshold = snakemake.params.get("logfc_threshold", 1.0)
pvalue_threshold = snakemake.params.get("pvalue_threshold", 0.05)

# Set up logging
os.makedirs(os.path.dirname(log_file), exist_ok=True)
os.makedirs(os.path.dirname(output_plot), exist_ok=True)
os.makedirs(os.path.dirname(output_overlap), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("UpSet Plot Generation")
print("="*80)

# Try to import upsetplot
try:
    from upsetplot import UpSet, from_contents
    HAS_UPSETPLOT = True
except ImportError:
    HAS_UPSETPLOT = False
    print("Warning: upsetplot not installed, using alternative visualization")

# ========== Load DE results ==========
print("\n1. Loading DE results...")
de_global = pd.read_csv(input_de_global)
print(f"   Loaded {len(de_global)} genes")

# Get comparison groups
if 'comparison_group' in de_global.columns:
    groups = [g for g in de_global['comparison_group'].unique() if pd.notna(g)]
else:
    groups = []

print(f"   Comparison groups: {groups}")

# ========== Extract significant genes per group ==========
print("\n2. Extracting significant genes...")

gene_col = 'gene_symbol' if 'gene_symbol' in de_global.columns else 'names'
logfc_col = 'logfoldchanges' if 'logfoldchanges' in de_global.columns else 'log2FoldChange'
pval_col = 'pvals_adj' if 'pvals_adj' in de_global.columns else 'padj'

# Separate up and down regulated genes
up_genes = {}
down_genes = {}
all_sig_genes = {}

for group in groups:
    group_data = de_global[de_global['comparison_group'] == group]
    sig_data = group_data[group_data[pval_col] < pvalue_threshold]
    
    up = sig_data[sig_data[logfc_col] > logfc_threshold][gene_col].dropna().tolist()
    down = sig_data[sig_data[logfc_col] < -logfc_threshold][gene_col].dropna().tolist()
    all_sig = sig_data[(sig_data[logfc_col].abs() > logfc_threshold)][gene_col].dropna().tolist()
    
    up_genes[group] = set(up)
    down_genes[group] = set(down)
    all_sig_genes[group] = set(all_sig)
    
    print(f"   {group}: {len(up)} up, {len(down)} down, {len(all_sig)} total")

# ========== Calculate overlaps ==========
print("\n3. Calculating overlaps...")

overlap_data = []

# Pairwise overlaps
for i, g1 in enumerate(groups):
    for g2 in groups[i+1:]:
        # All significant
        overlap_all = len(all_sig_genes[g1] & all_sig_genes[g2])
        union_all = len(all_sig_genes[g1] | all_sig_genes[g2])
        jaccard_all = overlap_all / union_all if union_all > 0 else 0
        
        # Up-regulated
        overlap_up = len(up_genes[g1] & up_genes[g2])
        
        # Down-regulated
        overlap_down = len(down_genes[g1] & down_genes[g2])
        
        overlap_data.append({
            'group1': g1,
            'group2': g2,
            'overlap_all': overlap_all,
            'overlap_up': overlap_up,
            'overlap_down': overlap_down,
            'jaccard_index': jaccard_all,
            'unique_g1': len(all_sig_genes[g1] - all_sig_genes[g2]),
            'unique_g2': len(all_sig_genes[g2] - all_sig_genes[g1])
        })

overlap_df = pd.DataFrame(overlap_data)
overlap_df.to_csv(output_overlap, index=False)
print(f"   ✓ Overlap statistics saved: {output_overlap}")

# ========== Create UpSet plot ==========
print("\n4. Creating UpSet plot...")

fig = plt.figure(figsize=(20, 14))

if HAS_UPSETPLOT and len(groups) >= 2:
    # Create UpSet plot for all significant genes
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # UpSet for all DEGs
    ax1 = fig.add_subplot(gs[0, :])
    try:
        upset_data = from_contents(all_sig_genes)
        upset = UpSet(upset_data, subset_size='count', show_counts=True)
        upset.plot(fig=fig)
        plt.suptitle('UpSet Plot: DEG Overlap Across Treatment Groups', fontsize=14, fontweight='bold')
    except Exception as e:
        print(f"   UpSet plot error: {e}")
        ax1.text(0.5, 0.5, 'Could not generate UpSet plot', ha='center', va='center')
        ax1.axis('off')
else:
    # Alternative: Venn-style bar chart
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Bar chart of DEG counts per group
    ax1 = fig.add_subplot(gs[0, 0])
    group_counts = pd.DataFrame({
        'Group': groups,
        'Up': [len(up_genes[g]) for g in groups],
        'Down': [len(down_genes[g]) for g in groups]
    })
    x = np.arange(len(groups))
    width = 0.35
    ax1.bar(x - width/2, group_counts['Up'], width, label='Up-regulated', color='#e74c3c')
    ax1.bar(x + width/2, group_counts['Down'], width, label='Down-regulated', color='#3498db')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=60, ha='right', fontsize=9)
    ax1.set_ylabel('Number of DEGs')
    ax1.set_title('DEGs per Treatment Group', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Overlap heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    if len(overlap_df) > 0:
        # Create overlap matrix
        overlap_matrix = pd.DataFrame(0, index=groups, columns=groups)
        for _, row in overlap_df.iterrows():
            overlap_matrix.loc[row['group1'], row['group2']] = row['overlap_all']
            overlap_matrix.loc[row['group2'], row['group1']] = row['overlap_all']
        # Diagonal = total DEGs
        for g in groups:
            overlap_matrix.loc[g, g] = len(all_sig_genes[g])
        
        import seaborn as sns
        sns.heatmap(overlap_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax2,
                    cbar_kws={'label': 'Overlap Count'})
        ax2.set_title('DEG Overlap Matrix', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No overlap data', ha='center', va='center')
        ax2.axis('off')
    
    # Plot 3: Jaccard similarity
    ax3 = fig.add_subplot(gs[1, 0])
    if len(overlap_df) > 0:
        jaccard_matrix = pd.DataFrame(1.0, index=groups, columns=groups)
        for _, row in overlap_df.iterrows():
            jaccard_matrix.loc[row['group1'], row['group2']] = row['jaccard_index']
            jaccard_matrix.loc[row['group2'], row['group1']] = row['jaccard_index']
        
        import seaborn as sns
        sns.heatmap(jaccard_matrix, annot=True, fmt='.2f', cmap='viridis', ax=ax3,
                    vmin=0, vmax=1, cbar_kws={'label': 'Jaccard Index'})
        ax3.set_title('Jaccard Similarity', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax3.axis('off')
    
    # Plot 4: Unique vs shared genes
    ax4 = fig.add_subplot(gs[1, 1])
    # Find genes unique to each group and shared by all
    all_genes = set.union(*all_sig_genes.values()) if all_sig_genes else set()
    shared_all = set.intersection(*all_sig_genes.values()) if all_sig_genes else set()
    
    unique_counts = {g: len(all_sig_genes[g] - set.union(*[all_sig_genes[g2] for g2 in groups if g2 != g])) 
                     for g in groups}
    
    categories = list(groups) + ['Shared by All']
    counts = [unique_counts[g] for g in groups] + [len(shared_all)]
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    
    ax4.barh(categories, counts, color=colors)
    ax4.set_xlabel('Number of Genes')
    ax4.set_title('Unique vs Shared DEGs', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, v in enumerate(counts):
        ax4.text(v + max(counts)*0.01, i, str(v), va='center', fontsize=9)

plt.suptitle('DEG Overlap Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ UpSet/Overlap plot: {output_plot}")

# ========== Per-cell-type overlap analysis ==========
print("\n5. Per-cell-type overlap analysis...")

combined_file = os.path.join(input_de_dir, "all_celltype_de_results.csv")
if os.path.exists(combined_file):
    de_celltype = pd.read_csv(combined_file)
    cell_types = de_celltype['cell_type'].unique()
    
    celltype_overlap = []
    for ct in cell_types:
        ct_data = de_celltype[de_celltype['cell_type'] == ct]
        ct_groups = ct_data['comparison_group'].unique()
        
        ct_genes = {}
        for g in ct_groups:
            g_data = ct_data[(ct_data['comparison_group'] == g) & 
                            (ct_data['pvals_adj'] < pvalue_threshold) &
                            (ct_data['logfoldchanges'].abs() > logfc_threshold)]
            ct_genes[g] = set(g_data[gene_col].dropna().tolist())
        
        # Calculate overlaps
        for i, g1 in enumerate(list(ct_groups)):
            for g2 in list(ct_groups)[i+1:]:
                if g1 in ct_genes and g2 in ct_genes:
                    overlap = len(ct_genes[g1] & ct_genes[g2])
                    union = len(ct_genes[g1] | ct_genes[g2])
                    celltype_overlap.append({
                        'cell_type': ct,
                        'group1': g1,
                        'group2': g2,
                        'overlap': overlap,
                        'jaccard': overlap/union if union > 0 else 0
                    })
    
    if celltype_overlap:
        ct_overlap_df = pd.DataFrame(celltype_overlap)
        ct_overlap_path = output_overlap.replace('.csv', '_per_celltype.csv')
        ct_overlap_df.to_csv(ct_overlap_path, index=False)
        print(f"   ✓ Per-celltype overlap: {ct_overlap_path}")
else:
    print("   No per-celltype DE results found")

print("\n" + "="*80)
print("UpSet Plot Generation Completed!")
print("="*80)
