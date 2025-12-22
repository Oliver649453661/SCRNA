#!/usr/bin/env python3
"""
GO Enrichment Analysis for Per-Cell-Type DE Results
使用本地GO数据库进行富集分析，避免依赖不稳定的API
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs and outputs
input_de_dir = snakemake.input.de_dir
output_dir = snakemake.output.output_dir
output_summary = snakemake.output.summary
output_plot = snakemake.output.plot
log_file = snakemake.log[0]

# Parameters
organism = snakemake.params.get("organism", "drosophila")
pvalue_cutoff = snakemake.params.get("pvalue_cutoff", 0.05)
logfc_threshold = snakemake.params.get("logfc_threshold", 0.5)

# Set up logging
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("Per-Cell-Type Enrichment Analysis (Local GO Database)")
print("="*80)
print(f"Organism: {organism}")
print(f"P-value cutoff: {pvalue_cutoff}")
print(f"Log2FC threshold: {logfc_threshold}")

# ========== Load Local GO Database ==========
print("\nLoading local GO database...")

go_db_dir = "data/reference/go_database"
go_obo_file = os.path.join(go_db_dir, "go-basic.obo")
gene2go_file = os.path.join(go_db_dir, "dmel_gene2go.tsv")

# 加载GO本体
try:
    from goatools.obo_parser import GODag
    godag = GODag(go_obo_file, optional_attrs=['relationship'])
    print(f"   Loaded GO ontology: {len(godag)} terms")
    HAS_GOATOOLS = True
except Exception as e:
    print(f"   Warning: Could not load GO ontology: {e}")
    HAS_GOATOOLS = False
    godag = None

# 加载基因-GO映射
gene2go = {}
go2genes = defaultdict(set)
if os.path.exists(gene2go_file):
    gene2go_df = pd.read_csv(gene2go_file, sep='\t')
    for _, row in gene2go_df.iterrows():
        gene = row['gene_symbol']
        go_ids = row['go_ids'].split(',') if pd.notna(row['go_ids']) else []
        gene2go[gene] = set(go_ids)
        for go_id in go_ids:
            go2genes[go_id].add(gene)
    print(f"   Loaded gene2go mapping: {len(gene2go)} genes, {len(go2genes)} GO terms")
else:
    print(f"   Warning: gene2go file not found: {gene2go_file}")

# 背景基因集
background_genes = set(gene2go.keys())

# ========== Load DE results ==========
print("\n1. Loading per-cell-type DE results...")

# Find the combined results file
combined_file = os.path.join(input_de_dir, "all_celltype_de_results.csv")
if os.path.exists(combined_file):
    de_results = pd.read_csv(combined_file)
    print(f"   Loaded {len(de_results)} DE results")
else:
    # Try to find individual files
    de_files = list(Path(input_de_dir).glob("de_*.csv"))
    if de_files:
        de_results = pd.concat([pd.read_csv(f) for f in de_files], ignore_index=True)
        print(f"   Loaded {len(de_results)} DE results from {len(de_files)} files")
    else:
        raise FileNotFoundError(f"No DE results found in {input_de_dir}")

# Get cell types and comparison groups
cell_types = de_results['cell_type'].unique()
comparison_groups = de_results['comparison_group'].unique()
print(f"   Cell types: {len(cell_types)}")
print(f"   Comparison groups: {comparison_groups.tolist()}")

# ========== Run enrichment analysis ==========
print("\n2. Running enrichment analysis...")

all_enrichment_results = []
summary_stats = []

def run_enrichment_local(gene_list, name, cell_type, direction):
    """使用本地GO数据库进行富集分析（Fisher精确检验）"""
    if len(gene_list) < 5:
        return None
    
    if not gene2go or not go2genes:
        return None
    
    # 过滤有GO注释的基因
    study_genes = set(gene_list) & background_genes
    if len(study_genes) < 3:
        return None
    
    n_study = len(study_genes)
    n_background = len(background_genes)
    
    # 对每个GO term进行Fisher精确检验
    enrichment_results = []
    
    for go_id, go_genes in go2genes.items():
        # 计算2x2列联表
        # 研究集中有该GO term的基因数
        study_in_go = len(study_genes & go_genes)
        if study_in_go < 2:  # 至少2个基因
            continue
        
        study_not_in_go = n_study - study_in_go
        bg_in_go = len(go_genes & background_genes)
        bg_not_in_go = n_background - bg_in_go
        
        # Fisher精确检验
        contingency = [[study_in_go, study_not_in_go], 
                       [bg_in_go - study_in_go, bg_not_in_go - study_not_in_go]]
        try:
            odds_ratio, pvalue = stats.fisher_exact(contingency, alternative='greater')
        except:
            continue
        
        if pvalue < 0.05:  # 初步筛选
            # 获取GO term名称
            go_name = godag[go_id].name if godag and go_id in godag else go_id
            go_namespace = godag[go_id].namespace if godag and go_id in godag else 'unknown'
            
            enrichment_results.append({
                'GO_ID': go_id,
                'Term': go_name,
                'Namespace': go_namespace,
                'P-value': pvalue,
                'Odds_Ratio': odds_ratio,
                'Study_Count': study_in_go,
                'Study_Total': n_study,
                'Background_Count': bg_in_go,
                'Background_Total': n_background,
                'Genes': ','.join(study_genes & go_genes),
                'cell_type': cell_type,
                'comparison': name,
                'direction': direction,
                'method': 'local_fisher'
            })
    
    if not enrichment_results:
        return None
    
    # 转换为DataFrame并进行多重检验校正（BH方法）
    results_df = pd.DataFrame(enrichment_results)
    from statsmodels.stats.multitest import multipletests
    _, adj_pvals, _, _ = multipletests(results_df['P-value'], method='fdr_bh')
    results_df['Adjusted P-value'] = adj_pvals
    
    # 筛选显著结果
    results_df = results_df[results_df['Adjusted P-value'] < 0.1].sort_values('Adjusted P-value')
    
    if len(results_df) > 0:
        print(f"       -> {len(results_df)} GO terms enriched")
    
    return results_df if len(results_df) > 0 else None

for cell_type in cell_types:
    print(f"\n   Processing: {cell_type}")
    ct_data = de_results[de_results['cell_type'] == cell_type]
    
    for group in comparison_groups:
        group_data = ct_data[ct_data['comparison_group'] == group]
        
        if len(group_data) == 0:
            continue
        
        # Get significant genes
        sig_data = group_data[group_data['pvals_adj'] < pvalue_cutoff]
        
        # Separate up and down regulated
        up_genes = sig_data[sig_data['logfoldchanges'] > logfc_threshold]
        down_genes = sig_data[sig_data['logfoldchanges'] < -logfc_threshold]
        
        # Get gene names
        gene_col = 'gene_symbol' if 'gene_symbol' in sig_data.columns else 'names'
        up_gene_list = up_genes[gene_col].dropna().tolist()
        down_gene_list = down_genes[gene_col].dropna().tolist()
        
        print(f"     {group}: {len(up_gene_list)} up, {len(down_gene_list)} down")
        
        # Run enrichment for up-regulated genes
        if len(up_gene_list) >= 5:
            up_results = run_enrichment_local(up_gene_list, group, cell_type, 'up')
            if up_results is not None and len(up_results) > 0:
                all_enrichment_results.append(up_results)
                summary_stats.append({
                    'cell_type': cell_type,
                    'comparison': group,
                    'direction': 'up',
                    'n_genes': len(up_gene_list),
                    'n_terms': len(up_results)
                })
        
        # Run enrichment for down-regulated genes
        if len(down_gene_list) >= 5:
            down_results = run_enrichment_local(down_gene_list, group, cell_type, 'down')
            if down_results is not None and len(down_results) > 0:
                all_enrichment_results.append(down_results)
                summary_stats.append({
                    'cell_type': cell_type,
                    'comparison': group,
                    'direction': 'down',
                    'n_genes': len(down_gene_list),
                    'n_terms': len(down_results)
                })

# ========== Save results ==========
print("\n3. Saving results...")

if all_enrichment_results:
    combined_enrichment = pd.concat(all_enrichment_results, ignore_index=True)
    
    # Save combined results
    combined_path = os.path.join(output_dir, "all_enrichment_results.csv")
    combined_enrichment.to_csv(combined_path, index=False)
    print(f"   ✓ Combined results: {combined_path}")
    
    # Save per-cell-type results
    for cell_type in combined_enrichment['cell_type'].unique():
        ct_results = combined_enrichment[combined_enrichment['cell_type'] == cell_type]
        ct_filename = cell_type.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        ct_path = os.path.join(output_dir, f"enrichment_{ct_filename}.csv")
        ct_results.to_csv(ct_path, index=False)
    
    print(f"   ✓ Per-cell-type results saved")
else:
    combined_enrichment = pd.DataFrame()
    print("   Warning: No enrichment results generated")

# Save summary
summary_df = pd.DataFrame(summary_stats) if summary_stats else pd.DataFrame()
summary_df.to_csv(output_summary, index=False)
print(f"   ✓ Summary: {output_summary}")

# ========== Create visualizations ==========
print("\n4. Creating visualizations...")

fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# Plot 1: Number of enriched terms per cell type
ax1 = fig.add_subplot(gs[0, :2])
if len(summary_df) > 0:
    terms_by_ct = summary_df.groupby('cell_type')['n_terms'].sum().sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(terms_by_ct)))
    ax1.barh(range(len(terms_by_ct)), terms_by_ct.values, color=colors)
    ax1.set_yticks(range(len(terms_by_ct)))
    ax1.set_yticklabels(terms_by_ct.index, fontsize=8)
    ax1.set_xlabel('Number of Enriched Terms')
    ax1.set_title('Enriched Terms per Cell Type', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
else:
    ax1.text(0.5, 0.5, 'No enrichment results', ha='center', va='center')
    ax1.axis('off')

# Plot 2: Up vs Down enrichment
ax2 = fig.add_subplot(gs[0, 2])
if len(summary_df) > 0 and 'direction' in summary_df.columns:
    dir_counts = summary_df.groupby('direction')['n_terms'].sum()
    colors = {'up': '#e74c3c', 'down': '#3498db'}
    ax2.pie(dir_counts.values, labels=dir_counts.index, autopct='%1.1f%%',
            colors=[colors.get(d, 'gray') for d in dir_counts.index])
    ax2.set_title('Up vs Down Regulated\nEnrichment', fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax2.axis('off')

# Plot 3: Top enriched terms (if available)
ax3 = fig.add_subplot(gs[1, :])
if len(combined_enrichment) > 0:
    # Get top terms by significance
    if 'p_value' in combined_enrichment.columns:
        pval_col = 'p_value'
    elif 'Adjusted P-value' in combined_enrichment.columns:
        pval_col = 'Adjusted P-value'
    else:
        pval_col = combined_enrichment.columns[combined_enrichment.columns.str.contains('p', case=False)][0] if any(combined_enrichment.columns.str.contains('p', case=False)) else None
    
    if pval_col:
        top_terms = combined_enrichment.nsmallest(20, pval_col)
        
        # Get term name column
        if 'name' in top_terms.columns:
            term_col = 'name'
        elif 'Term' in top_terms.columns:
            term_col = 'Term'
        else:
            term_col = top_terms.columns[0]
        
        y_pos = range(len(top_terms))
        colors = ['#e74c3c' if d == 'up' else '#3498db' for d in top_terms['direction']]
        
        ax3.barh(y_pos, -np.log10(top_terms[pval_col]), color=colors)
        ax3.set_yticks(y_pos)
        
        # Truncate long term names
        labels = [t[:50] + '...' if len(str(t)) > 50 else str(t) for t in top_terms[term_col]]
        ax3.set_yticklabels(labels, fontsize=7)
        ax3.set_xlabel('-log10(p-value)')
        ax3.set_title('Top 20 Enriched Terms', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#e74c3c', label='Up-regulated'),
                          Patch(facecolor='#3498db', label='Down-regulated')]
        ax3.legend(handles=legend_elements, loc='lower right')
    else:
        ax3.text(0.5, 0.5, 'No p-value column found', ha='center', va='center')
        ax3.axis('off')
else:
    ax3.text(0.5, 0.5, 'No enrichment results', ha='center', va='center')
    ax3.axis('off')

# Plot 4: Heatmap of enrichment by cell type and comparison
ax4 = fig.add_subplot(gs[2, :2])
if len(summary_df) > 0:
    pivot_df = summary_df.pivot_table(
        index='cell_type',
        columns='comparison',
        values='n_terms',
        aggfunc='sum',
        fill_value=0
    )
    if pivot_df.shape[0] > 0 and pivot_df.shape[1] > 0:
        sns.heatmap(pivot_df, cmap='YlOrRd', annot=True, fmt='g', ax=ax4,
                    cbar_kws={'label': 'Number of Terms'}, annot_kws={'fontsize': 7})
        ax4.set_xlabel('Comparison Group', fontsize=10)
        ax4.set_ylabel('Cell Type', fontsize=10)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax4.set_yticklabels(ax4.get_yticklabels(), fontsize=8)
        ax4.set_title('Enriched Terms: Cell Type × Comparison', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        ax4.axis('off')
else:
    ax4.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax4.axis('off')

# Plot 5: Summary
ax5 = fig.add_subplot(gs[2, 2])
total_terms = summary_df['n_terms'].sum() if len(summary_df) > 0 else 0
n_celltypes = len(summary_df['cell_type'].unique()) if len(summary_df) > 0 else 0

summary_text = (
    f"Enrichment Analysis Summary\n"
    f"{'='*35}\n\n"
    f"Cell types analyzed: {n_celltypes}\n"
    f"Total enriched terms: {total_terms:,}\n\n"
    f"Parameters:\n"
    f"  Organism: {organism}\n"
    f"  P-value cutoff: {pvalue_cutoff}\n"
    f"  Log2FC threshold: {logfc_threshold}\n\n"
    f"Output: {output_dir}\n"
)

ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax5.axis('off')

plt.suptitle('Per-Cell-Type Enrichment Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Plot: {output_plot}")

print("\n" + "="*80)
print("Enrichment Analysis Completed!")
print("="*80)
