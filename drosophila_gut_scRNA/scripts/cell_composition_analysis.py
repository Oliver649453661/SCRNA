#!/usr/bin/env python3
"""
Cell Composition Analysis
分析各处理组中不同细胞类型的比例变化
"""

import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs and outputs
input_h5ad = snakemake.input.h5ad
output_proportions = snakemake.output.proportions
output_stats = snakemake.output.stats
output_plot = snakemake.output.plot
log_file = snakemake.log[0]

# Parameters
groupby = snakemake.params.get("groupby", "group")
celltype_col = snakemake.params.get("celltype_col", "final_cell_type")
reference = snakemake.params.get("reference", "Control")

# Set up logging
os.makedirs(os.path.dirname(output_proportions), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("Cell Composition Analysis")
print("="*80)

# Load data
print("\n1. Loading annotated data...")
adata = sc.read_h5ad(input_h5ad)
print(f"   Data shape: {adata.shape}")

# Check required columns
if celltype_col not in adata.obs.columns:
    for alt_col in ['cell_type', 'predicted_cell_type', 'leiden']:
        if alt_col in adata.obs.columns:
            celltype_col = alt_col
            break

print(f"   Using cell type column: {celltype_col}")
print(f"   Using group column: {groupby}")

# ========== Calculate cell proportions ==========
print("\n2. Calculating cell type proportions...")

# Get counts per group and cell type
counts = pd.crosstab(adata.obs[groupby], adata.obs[celltype_col])
print(f"   Groups: {counts.index.tolist()}")
print(f"   Cell types: {counts.columns.tolist()}")

# Calculate proportions (within each group)
proportions = counts.div(counts.sum(axis=1), axis=0)

# Also calculate proportions per sample if 'sample' column exists
if 'sample' in adata.obs.columns:
    sample_counts = pd.crosstab(adata.obs['sample'], adata.obs[celltype_col])
    sample_proportions = sample_counts.div(sample_counts.sum(axis=1), axis=0)
    
    # Add group information to samples
    sample_to_group = adata.obs.groupby('sample')[groupby].first().to_dict()
    sample_proportions['group'] = sample_proportions.index.map(sample_to_group)
else:
    sample_proportions = None

print("\n   Cell type proportions by group:")
print(proportions.round(3).to_string())

# ========== Statistical testing ==========
print("\n3. Performing statistical tests...")

stats_results = []

cell_types = counts.columns.tolist()
groups = counts.index.tolist()

# For each cell type, compare proportions between groups
for ct in cell_types:
    print(f"\n   Testing: {ct}")
    
    if sample_proportions is not None:
        # Use sample-level data for proper statistical testing
        ct_data = sample_proportions[[ct, 'group']].copy()
        ct_data.columns = ['proportion', 'group']
        
        # Compare each group to reference
        if reference in groups:
            ref_data = ct_data[ct_data['group'] == reference]['proportion'].values
            
            for group in groups:
                if group == reference:
                    continue
                
                group_data = ct_data[ct_data['group'] == group]['proportion'].values
                
                if len(ref_data) >= 2 and len(group_data) >= 2:
                    # Mann-Whitney U test (non-parametric)
                    try:
                        stat, pval = stats.mannwhitneyu(group_data, ref_data, alternative='two-sided')
                        
                        # Calculate fold change
                        mean_ref = np.mean(ref_data)
                        mean_group = np.mean(group_data)
                        if mean_ref > 0:
                            fc = mean_group / mean_ref
                            log2fc = np.log2(fc) if fc > 0 else 0
                        else:
                            fc = np.inf if mean_group > 0 else 1
                            log2fc = 0
                        
                        stats_results.append({
                            'cell_type': ct,
                            'comparison_group': group,
                            'reference_group': reference,
                            'mean_proportion_group': mean_group,
                            'mean_proportion_reference': mean_ref,
                            'fold_change': fc,
                            'log2_fold_change': log2fc,
                            'statistic': stat,
                            'pvalue': pval,
                            'n_samples_group': len(group_data),
                            'n_samples_reference': len(ref_data)
                        })
                        
                        print(f"     {group} vs {reference}: FC={fc:.2f}, p={pval:.4f}")
                        
                    except Exception as e:
                        print(f"     Error testing {group}: {e}")
                else:
                    print(f"     Skipping {group}: insufficient samples")
        else:
            # Kruskal-Wallis test across all groups
            group_data_list = [ct_data[ct_data['group'] == g]['proportion'].values for g in groups]
            group_data_list = [d for d in group_data_list if len(d) >= 2]
            
            if len(group_data_list) >= 2:
                try:
                    stat, pval = stats.kruskal(*group_data_list)
                    stats_results.append({
                        'cell_type': ct,
                        'comparison_group': 'all',
                        'reference_group': 'kruskal',
                        'statistic': stat,
                        'pvalue': pval
                    })
                    print(f"     Kruskal-Wallis: p={pval:.4f}")
                except Exception as e:
                    print(f"     Error: {e}")
    else:
        # Without replicates, use chi-square test on counts
        print(f"     No sample replicates, using chi-square on counts")
        
        for group in groups:
            if group == reference:
                continue
            
            # 2x2 contingency table
            ct_count_group = counts.loc[group, ct]
            ct_count_ref = counts.loc[reference, ct] if reference in groups else 0
            other_count_group = counts.loc[group].sum() - ct_count_group
            other_count_ref = counts.loc[reference].sum() - ct_count_ref if reference in groups else 0
            
            contingency = [[ct_count_group, other_count_group],
                          [ct_count_ref, other_count_ref]]
            
            try:
                chi2, pval, dof, expected = stats.chi2_contingency(contingency)
                
                prop_group = ct_count_group / counts.loc[group].sum()
                prop_ref = ct_count_ref / counts.loc[reference].sum() if reference in groups else 0
                fc = prop_group / prop_ref if prop_ref > 0 else np.inf
                
                stats_results.append({
                    'cell_type': ct,
                    'comparison_group': group,
                    'reference_group': reference,
                    'proportion_group': prop_group,
                    'proportion_reference': prop_ref,
                    'fold_change': fc,
                    'log2_fold_change': np.log2(fc) if fc > 0 and fc != np.inf else 0,
                    'chi2': chi2,
                    'pvalue': pval
                })
                
                print(f"     {group} vs {reference}: FC={fc:.2f}, p={pval:.4f}")
                
            except Exception as e:
                print(f"     Error: {e}")

# Multiple testing correction
if stats_results:
    stats_df = pd.DataFrame(stats_results)
    
    if 'pvalue' in stats_df.columns and len(stats_df) > 0:
        pvals = stats_df['pvalue'].values
        _, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')
        stats_df['pvalue_adj'] = pvals_adj
        
        # Mark significant results
        stats_df['significant'] = stats_df['pvalue_adj'] < 0.05
        
        n_sig = stats_df['significant'].sum()
        print(f"\n   Significant changes (FDR < 0.05): {n_sig}")
else:
    stats_df = pd.DataFrame()

# ========== Save results ==========
print("\n4. Saving results...")

# Save proportions
proportions_long = proportions.reset_index().melt(
    id_vars=[groupby], 
    var_name='cell_type', 
    value_name='proportion'
)
proportions_long.to_csv(output_proportions, index=False)
print(f"   ✓ Proportions: {output_proportions}")

# Save statistics
if len(stats_df) > 0:
    stats_df = stats_df.sort_values('pvalue_adj')
    stats_df.to_csv(output_stats, index=False)
    print(f"   ✓ Statistics: {output_stats}")
else:
    pd.DataFrame().to_csv(output_stats, index=False)
    print(f"   ✓ Statistics (empty): {output_stats}")

# ========== Create visualizations ==========
print("\n5. Creating visualizations...")

fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# Plot 1: Stacked bar chart of cell type proportions
ax1 = fig.add_subplot(gs[0, :2])
proportions.plot(kind='bar', stacked=True, ax=ax1, colormap='Set3', width=0.8)
ax1.set_xlabel('Treatment Group')
ax1.set_ylabel('Proportion')
ax1.set_title('Cell Type Composition by Treatment Group', fontweight='bold')
ax1.legend(title='Cell Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60, ha='right', fontsize=9)
ax1.set_ylim(0, 1)

# Plot 2: Heatmap of proportions
ax2 = fig.add_subplot(gs[0, 2])
sns.heatmap(proportions.T, cmap='YlOrRd', annot=True, fmt='.2f', ax=ax2,
            cbar_kws={'label': 'Proportion'}, annot_kws={'fontsize': 7})
ax2.set_xlabel('Treatment Group', fontsize=10)
ax2.set_ylabel('Cell Type', fontsize=10)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=7)
ax2.set_title('Proportion Heatmap', fontweight='bold')

# Plot 3: Fold change heatmap (if stats available)
ax3 = fig.add_subplot(gs[1, :2])
if len(stats_df) > 0 and 'log2_fold_change' in stats_df.columns:
    fc_pivot = stats_df.pivot_table(
        index='cell_type',
        columns='comparison_group',
        values='log2_fold_change',
        aggfunc='first'
    )
    
    if fc_pivot.shape[0] > 0 and fc_pivot.shape[1] > 0:
        # Add significance markers
        sig_pivot = stats_df.pivot_table(
            index='cell_type',
            columns='comparison_group',
            values='significant',
            aggfunc='first'
        )
        
        # Create annotation with significance stars
        annot = fc_pivot.round(2).astype(str)
        for ct in annot.index:
            for grp in annot.columns:
                if ct in sig_pivot.index and grp in sig_pivot.columns:
                    if sig_pivot.loc[ct, grp]:
                        annot.loc[ct, grp] = annot.loc[ct, grp] + '*'
        
        sns.heatmap(fc_pivot, cmap='RdBu_r', center=0, annot=annot, fmt='s', ax=ax3,
                    cbar_kws={'label': 'Log2 Fold Change'}, annot_kws={'fontsize': 7})
        ax3.set_xlabel('Comparison Group', fontsize=10)
        ax3.set_ylabel('Cell Type', fontsize=10)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=7)
        ax3.set_title(f'Log2 Fold Change vs {reference} (* = FDR < 0.05)', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        ax3.axis('off')
else:
    ax3.text(0.5, 0.5, 'No fold change data', ha='center', va='center')
    ax3.axis('off')

# Plot 4: Significant changes bar chart
ax4 = fig.add_subplot(gs[1, 2])
if len(stats_df) > 0 and 'significant' in stats_df.columns:
    sig_df = stats_df[stats_df['significant']].copy()
    if len(sig_df) > 0:
        sig_counts = sig_df.groupby('cell_type').size().sort_values(ascending=True)
        ax4.barh(range(len(sig_counts)), sig_counts.values, color='coral')
        ax4.set_yticks(range(len(sig_counts)))
        ax4.set_yticklabels(sig_counts.index, fontsize=8)
        ax4.set_xlabel('Number of Significant Changes')
        ax4.set_title('Significant Composition Changes', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No significant changes', ha='center', va='center')
        ax4.axis('off')
else:
    ax4.text(0.5, 0.5, 'No statistics available', ha='center', va='center')
    ax4.axis('off')

# Plot 5: Box plots for top changing cell types (if sample data available)
ax5 = fig.add_subplot(gs[2, :2])
if sample_proportions is not None and len(stats_df) > 0:
    # Get top 5 most significant cell types
    top_cts = stats_df.nsmallest(5, 'pvalue')['cell_type'].unique()[:5]
    
    if len(top_cts) > 0:
        plot_data = []
        for ct in top_cts:
            for sample in sample_proportions.index:
                plot_data.append({
                    'cell_type': ct,
                    'sample': sample,
                    'group': sample_proportions.loc[sample, 'group'],
                    'proportion': sample_proportions.loc[sample, ct]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        sns.boxplot(data=plot_df, x='cell_type', y='proportion', hue='group', ax=ax5)
        ax5.set_xlabel('Cell Type')
        ax5.set_ylabel('Proportion')
        ax5.set_title('Top Changing Cell Types by Treatment', fontweight='bold')
        ax5.legend(title='Group', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=60, ha='right', fontsize=8)
    else:
        ax5.text(0.5, 0.5, 'No significant cell types', ha='center', va='center')
        ax5.axis('off')
else:
    # Simple grouped bar chart
    proportions.T.plot(kind='bar', ax=ax5, width=0.8)
    ax5.set_xlabel('Cell Type')
    ax5.set_ylabel('Proportion')
    ax5.set_title('Cell Type Proportions by Group', fontweight='bold')
    ax5.legend(title='Group')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')

# Plot 6: Summary
ax6 = fig.add_subplot(gs[2, 2])
n_sig = stats_df['significant'].sum() if 'significant' in stats_df.columns else 0
n_tests = len(stats_df)

summary_text = (
    f"Cell Composition Summary\n"
    f"{'='*35}\n\n"
    f"Total cells: {adata.n_obs:,}\n"
    f"Cell types: {len(cell_types)}\n"
    f"Treatment groups: {len(groups)}\n"
    f"Reference: {reference}\n\n"
    f"Statistical tests: {n_tests}\n"
    f"Significant (FDR<0.05): {n_sig}\n\n"
)

if len(stats_df) > 0 and 'significant' in stats_df.columns:
    sig_cts = stats_df[stats_df['significant']]['cell_type'].unique()
    if len(sig_cts) > 0:
        summary_text += "Significant cell types:\n"
        for ct in sig_cts[:5]:
            summary_text += f"  • {ct}\n"
        if len(sig_cts) > 5:
            summary_text += f"  ... and {len(sig_cts)-5} more\n"

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax6.axis('off')

plt.suptitle('Cell Composition Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Plot: {output_plot}")

print("\n" + "="*80)
print("Cell Composition Analysis Completed!")
print("="*80)
