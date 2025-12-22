#!/usr/bin/env python3
"""
Per-cell-type Differential Expression Analysis
对每个细胞类型分别进行处理组间的差异表达分析
"""

import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs and outputs
input_h5ad = snakemake.input.h5ad
output_dir = snakemake.output.output_dir
output_summary = snakemake.output.summary
output_plot = snakemake.output.plot
log_file = snakemake.log[0]

# Parameters
groupby = snakemake.params.get("groupby", "group")
reference = snakemake.params.get("reference", "Control")
method = snakemake.params.get("method", "wilcoxon")
celltype_col = snakemake.params.get("celltype_col", "final_cell_type")

# Set up logging
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("Per-Cell-Type Differential Expression Analysis")
print("="*80)

# Load data
print("\n1. Loading annotated data...")
adata = sc.read_h5ad(input_h5ad)
print(f"   Data shape: {adata.shape}")

# Check required columns
if celltype_col not in adata.obs.columns:
    # Try alternative column names
    for alt_col in ['cell_type', 'predicted_cell_type', 'leiden']:
        if alt_col in adata.obs.columns:
            celltype_col = alt_col
            break
    else:
        raise ValueError(f"Cell type column not found. Available: {adata.obs.columns.tolist()}")

print(f"   Using cell type column: {celltype_col}")
print(f"   Using group column: {groupby}")
print(f"   Reference group: {reference}")

# Get cell types and groups
cell_types = adata.obs[celltype_col].unique()
groups = adata.obs[groupby].unique()

print(f"\n   Cell types: {len(cell_types)}")
for ct in sorted(cell_types):
    n = (adata.obs[celltype_col] == ct).sum()
    print(f"     - {ct}: {n} cells")

print(f"\n   Groups: {groups.tolist()}")

# Check reference group
if reference not in groups:
    print(f"   Warning: Reference '{reference}' not found. Using 'rest' mode.")
    reference = 'rest'

# ========== Per-cell-type DE analysis ==========
print("\n2. Performing per-cell-type differential expression...")

all_results = []
summary_stats = []

for cell_type in sorted(cell_types):
    print(f"\n   Processing: {cell_type}")
    
    # Subset to this cell type
    adata_ct = adata[adata.obs[celltype_col] == cell_type].copy()
    n_cells = adata_ct.n_obs
    
    # Check if we have enough cells in each group
    group_counts = adata_ct.obs[groupby].value_counts()
    print(f"     Total cells: {n_cells}")
    print(f"     Group distribution: {group_counts.to_dict()}")
    
    # Need at least 10 cells per group for meaningful DE
    min_cells_per_group = 10
    valid_groups = group_counts[group_counts >= min_cells_per_group].index.tolist()
    
    if len(valid_groups) < 2:
        print(f"     Skipping: Not enough cells in multiple groups")
        summary_stats.append({
            'cell_type': cell_type,
            'n_cells': n_cells,
            'n_groups': len(valid_groups),
            'status': 'skipped_insufficient_cells',
            'n_degs': 0,
            'n_up': 0,
            'n_down': 0
        })
        continue
    
    # Filter to valid groups
    adata_ct = adata_ct[adata_ct.obs[groupby].isin(valid_groups)].copy()
    
    # Perform DE analysis
    try:
        if reference in valid_groups:
            sc.tl.rank_genes_groups(
                adata_ct,
                groupby=groupby,
                reference=reference,
                method=method,
                use_raw=True if adata_ct.raw is not None else False
            )
            groups_to_process = [g for g in valid_groups if g != reference]
        else:
            sc.tl.rank_genes_groups(
                adata_ct,
                groupby=groupby,
                reference='rest',
                method=method,
                use_raw=True if adata_ct.raw is not None else False
            )
            groups_to_process = valid_groups
        
        # Extract results for each comparison
        for group in groups_to_process:
            try:
                result_df = sc.get.rank_genes_groups_df(adata_ct, group=str(group))
                result_df['cell_type'] = cell_type
                result_df['comparison_group'] = group
                result_df['reference_group'] = reference if reference in valid_groups else 'rest'
                
                # Add gene symbols if available
                if 'gene_name' in adata.var.columns:
                    gene_id_to_name = dict(zip(adata.var.index, adata.var['gene_name']))
                    result_df['gene_symbol'] = result_df['names'].map(gene_id_to_name)
                    result_df['gene_symbol'] = result_df['gene_symbol'].fillna(result_df['names'])
                
                all_results.append(result_df)
                
                # Statistics
                n_sig = (result_df['pvals_adj'] < 0.05).sum()
                n_up = ((result_df['pvals_adj'] < 0.05) & (result_df['logfoldchanges'] > 0)).sum()
                n_down = ((result_df['pvals_adj'] < 0.05) & (result_df['logfoldchanges'] < 0)).sum()
                
                print(f"     {group} vs {reference}: {n_sig} DEGs ({n_up} up, {n_down} down)")
                
                summary_stats.append({
                    'cell_type': cell_type,
                    'comparison_group': group,
                    'reference_group': reference if reference in valid_groups else 'rest',
                    'n_cells': n_cells,
                    'n_cells_comparison': (adata_ct.obs[groupby] == group).sum(),
                    'n_cells_reference': (adata_ct.obs[groupby] == reference).sum() if reference in valid_groups else 'N/A',
                    'status': 'completed',
                    'n_degs': n_sig,
                    'n_up': n_up,
                    'n_down': n_down
                })
                
            except Exception as e:
                print(f"     Error processing {group}: {e}")
                
    except Exception as e:
        print(f"     Error in DE analysis: {e}")
        summary_stats.append({
            'cell_type': cell_type,
            'n_cells': n_cells,
            'n_groups': len(valid_groups),
            'status': f'error: {str(e)[:50]}',
            'n_degs': 0,
            'n_up': 0,
            'n_down': 0
        })

# ========== Save results ==========
print("\n3. Saving results...")

# Combine all results
if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_path = os.path.join(output_dir, "all_celltype_de_results.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"   ✓ Combined results: {combined_path}")
    
    # Save per-cell-type results
    for cell_type in combined_df['cell_type'].unique():
        ct_df = combined_df[combined_df['cell_type'] == cell_type]
        ct_filename = cell_type.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        ct_path = os.path.join(output_dir, f"de_{ct_filename}.csv")
        ct_df.to_csv(ct_path, index=False)
    
    print(f"   ✓ Per-cell-type results saved to {output_dir}")
else:
    print("   Warning: No DE results generated!")
    combined_df = pd.DataFrame()

# Save summary
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(output_summary, index=False)
print(f"   ✓ Summary: {output_summary}")

# ========== Create visualizations ==========
print("\n4. Creating visualizations...")

fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# Plot 1: DEGs per cell type (bar chart)
ax1 = fig.add_subplot(gs[0, :2])
if len(summary_df) > 0 and 'n_degs' in summary_df.columns:
    # Aggregate by cell type
    degs_by_ct = summary_df.groupby('cell_type')['n_degs'].sum().sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(degs_by_ct)))
    bars = ax1.barh(range(len(degs_by_ct)), degs_by_ct.values, color=colors)
    ax1.set_yticks(range(len(degs_by_ct)))
    ax1.set_yticklabels(degs_by_ct.index, fontsize=8)
    ax1.set_xlabel('Number of DEGs (padj < 0.05)')
    ax1.set_title('Differentially Expressed Genes per Cell Type', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
else:
    ax1.text(0.5, 0.5, 'No data available', ha='center', va='center')
    ax1.axis('off')

# Plot 2: Up vs Down regulated genes
ax2 = fig.add_subplot(gs[0, 2])
if len(summary_df) > 0 and 'n_up' in summary_df.columns:
    up_down = summary_df.groupby('cell_type')[['n_up', 'n_down']].sum()
    up_down = up_down.sort_values('n_up', ascending=False).head(10)
    
    x = np.arange(len(up_down))
    width = 0.35
    ax2.bar(x - width/2, up_down['n_up'], width, label='Up-regulated', color='#e74c3c')
    ax2.bar(x + width/2, up_down['n_down'], width, label='Down-regulated', color='#3498db')
    ax2.set_xticks(x)
    ax2.set_xticklabels(up_down.index, rotation=60, ha='right', fontsize=8)
    ax2.set_ylabel('Number of DEGs')
    ax2.set_title('Up vs Down Regulated Genes', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'No data available', ha='center', va='center')
    ax2.axis('off')

# Plot 3: Heatmap of DEG counts by cell type and comparison
ax3 = fig.add_subplot(gs[1, :2])
if len(summary_df) > 0 and 'comparison_group' in summary_df.columns:
    pivot_df = summary_df.pivot_table(
        index='cell_type', 
        columns='comparison_group', 
        values='n_degs', 
        aggfunc='sum',
        fill_value=0
    )
    if pivot_df.shape[0] > 0 and pivot_df.shape[1] > 0:
        sns.heatmap(pivot_df, cmap='YlOrRd', annot=True, fmt='g', ax=ax3,
                    cbar_kws={'label': 'Number of DEGs'}, annot_kws={'fontsize': 7})
        ax3.set_xlabel('Comparison Group', fontsize=10)
        ax3.set_ylabel('Cell Type', fontsize=10)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=8)
        ax3.set_title('DEG Counts: Cell Type × Treatment Group', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center')
        ax3.axis('off')
else:
    ax3.text(0.5, 0.5, 'No comparison data available', ha='center', va='center')
    ax3.axis('off')

# Plot 4: Top DEGs across cell types (if we have results)
ax4 = fig.add_subplot(gs[1, 2])
if len(combined_df) > 0:
    # Get top DEGs by absolute log fold change
    sig_genes = combined_df[combined_df['pvals_adj'] < 0.05].copy()
    if len(sig_genes) > 0:
        sig_genes['abs_lfc'] = sig_genes['logfoldchanges'].abs()
        top_genes = sig_genes.nlargest(15, 'abs_lfc')
        
        gene_col = 'gene_symbol' if 'gene_symbol' in top_genes.columns else 'names'
        colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_genes['logfoldchanges']]
        
        ax4.barh(range(len(top_genes)), top_genes['logfoldchanges'], color=colors)
        ax4.set_yticks(range(len(top_genes)))
        ax4.set_yticklabels(top_genes[gene_col])
        ax4.set_xlabel('Log2 Fold Change')
        ax4.set_title('Top 15 DEGs by |Log2FC|', fontweight='bold')
        ax4.axvline(0, color='black', linewidth=0.5)
        ax4.grid(axis='x', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No significant DEGs', ha='center', va='center')
        ax4.axis('off')
else:
    ax4.text(0.5, 0.5, 'No DE results', ha='center', va='center')
    ax4.axis('off')

# Plot 5: Cell count distribution
ax5 = fig.add_subplot(gs[2, 0])
if len(summary_df) > 0 and 'n_cells' in summary_df.columns:
    cells_by_ct = summary_df.groupby('cell_type')['n_cells'].first().sort_values(ascending=True)
    ax5.barh(range(len(cells_by_ct)), cells_by_ct.values, color='steelblue')
    ax5.set_yticks(range(len(cells_by_ct)))
    ax5.set_yticklabels(cells_by_ct.index, fontsize=8)
    ax5.set_xlabel('Number of Cells')
    ax5.set_title('Cells per Cell Type', fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
else:
    ax5.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax5.axis('off')

# Plot 6: Analysis status
ax6 = fig.add_subplot(gs[2, 1])
if len(summary_df) > 0 and 'status' in summary_df.columns:
    status_counts = summary_df['status'].value_counts()
    colors = {'completed': '#2ecc71', 'skipped_insufficient_cells': '#f39c12'}
    bar_colors = [colors.get(s, '#95a5a6') for s in status_counts.index]
    ax6.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
            colors=bar_colors, startangle=90)
    ax6.set_title('Analysis Status', fontweight='bold')
else:
    ax6.text(0.5, 0.5, 'No status data', ha='center', va='center')
    ax6.axis('off')

# Plot 7: Summary text
ax7 = fig.add_subplot(gs[2, 2])
total_degs = summary_df['n_degs'].sum() if 'n_degs' in summary_df.columns else 0
completed = (summary_df['status'] == 'completed').sum() if 'status' in summary_df.columns else 0
skipped = len(summary_df) - completed if len(summary_df) > 0 else 0

summary_text = (
    f"Per-Cell-Type DE Summary\n"
    f"{'='*35}\n\n"
    f"Total cell types: {len(cell_types)}\n"
    f"Comparisons completed: {completed}\n"
    f"Comparisons skipped: {skipped}\n\n"
    f"Total DEGs identified: {total_degs:,}\n"
    f"Reference group: {reference}\n"
    f"Method: {method}\n\n"
    f"Output directory:\n{output_dir}\n"
)
ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax7.axis('off')

plt.suptitle('Per-Cell-Type Differential Expression Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Plot: {output_plot}")

# ========== Final summary ==========
print("\n" + "="*80)
print("Per-Cell-Type DE Analysis Completed!")
print("="*80)
print(f"\nTotal cell types analyzed: {len(cell_types)}")
print(f"Successful comparisons: {completed}")
print(f"Total DEGs identified: {total_degs:,}")
print("="*80)
