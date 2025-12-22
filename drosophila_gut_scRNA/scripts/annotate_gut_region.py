#!/usr/bin/env python3
"""
肠道区域表达注释
基于 flygut_marker_genes.xlsx 的区域表达模式为细胞注释肠道区域
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
input_flygut_markers = snakemake.input.flygut_markers
output_h5ad = snakemake.output.h5ad
output_region_scores = snakemake.output.region_scores
output_plot = snakemake.output.plot

# Set up logging
log_file = snakemake.log[0]
os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("Gut Region Annotation based on FlyGut Expression Atlas")
print("="*80)

# Load data
print("\n1. Loading annotated data...")
adata = sc.read_h5ad(input_h5ad)
print(f"   Data shape: {adata.shape}")

# Load FlyGut marker data - read without header first to understand structure
print("\n2. Loading FlyGut marker gene database...")
flygut_raw = pd.read_excel(input_flygut_markers, header=None)
print(f"   FlyGut database raw shape: {flygut_raw.shape}")

# The file has complex multi-row headers:
# Row 0: Main column names (Gene.Title, Gene.Symbol, CATEGORY, etc.)
# Row 1: Region groups (FOREGUT, MIDGUT, HINDGUT)
# Row 2: Specific regions (Crop, prov/R1, R2, R3, R4, R5)
# Row 3: Column identifiers (fc.OnyX.OnyT, OnyX_Mean, etc.)
# Data starts from row 4

# Create proper column names
flygut_df = flygut_raw.iloc[4:].copy()  # Data rows
flygut_df.columns = flygut_raw.iloc[0].tolist()  # Use row 0 as column names
flygut_df = flygut_df.reset_index(drop=True)

print(f"   FlyGut database: {len(flygut_df)} genes")

# Parse the FlyGut data structure
print("\n3. Parsing FlyGut expression data...")

# Get gene symbols
gene_symbols = flygut_df['Gene.Symbol'].dropna().tolist()
print(f"   Found {len(gene_symbols)} genes with symbols")

# Define gut regions and their corresponding column indices
# Based on the file structure analysis
# Note: R0 (proventriculus) is added separately from r0_marker_genes.csv
gut_regions = {
    'Crop': 'Crop',           # Foregut
    'R0': 'R0',               # Proventriculus (foregut-midgut junction)
    'R1': 'prov/R1',          # Anterior midgut
    'R2': 'R2',               # Midgut
    'R3': 'R3',               # Midgut (copper cell region)
    'R4': 'R4',               # Midgut
    'R5': 'R5',               # Posterior midgut
    'Hindgut': 'Hindgut'      # Hindgut
}

# Parse expression data - look for columns with expression values
# The file has fold-change columns (fc.OnyX.OnyT) for each region
print("\n4. Extracting region-specific expression patterns...")

# Create a mapping of genes to their preferred regions
# A gene is considered region-specific if it has high expression in that region

# Find the expression value columns (Affymetrix absolute values)
# These are in columns like 'OnyB_Mean', 'OnyC_Mean', etc.
# OnyB = Crop, OnyC = R1, OnyD = R2, OnyE = R3, OnyF = R4, OnyG = R5, OnyH = Hindgut

# Region to column index mapping based on file structure analysis:
# Col 8: Crop (fc.OnyB.OnyT), Col 9: R1 (fc.OnyC.OnyT), Col 10: R2, Col 11: R3, Col 12: R4, Col 13: R5, Col 14: Hindgut
# Col 24-30: Mean expression values (OnyB_Mean to OnyH_Mean)
# Col 33-39: Expression calls (OnyB to OnyH)

region_col_indices = {
    'Crop': 8,      # fc.OnyB.OnyT - fold change vs total
    # R0 (proventriculus) is not in FlyGut database, will be added from separate file
    'R1': 9,        # fc.OnyC.OnyT (prov/R1 in FlyGut)
    'R2': 10,       # fc.OnyD.OnyT
    'R3': 11,       # fc.OnyE.OnyT
    'R4': 12,       # fc.OnyF.OnyT
    'R5': 13,       # fc.OnyG.OnyT
    'Hindgut': 14   # fc.OnyH.OnyT
}

# Also get mean expression columns
mean_col_indices = {
    'Crop': 24,     # OnyB_Mean
    'R1': 25,       # OnyC_Mean
    'R2': 26,       # OnyD_Mean
    'R3': 27,       # OnyE_Mean
    'R4': 28,       # OnyF_Mean
    'R5': 29,       # OnyG_Mean
    'Hindgut': 30   # OnyH_Mean
}

print(f"   Using fold-change columns (indices 8-14) and mean expression columns (indices 24-30)")

# Build region-specific gene sets (including R0)
region_genes = {region: set() for region in region_col_indices.keys()}
region_genes['R0'] = set()  # Add R0 region

# Load R0 marker genes from separate file
r0_marker_file = os.path.join(os.path.dirname(input_flygut_markers), 'r0_marker_genes.csv')
if os.path.exists(r0_marker_file):
    print(f"\n   Loading R0/Proventriculus marker genes from {r0_marker_file}")
    r0_df = pd.read_csv(r0_marker_file)
    r0_genes = set(r0_df['Gene.Symbol'].dropna().tolist())
    region_genes['R0'] = r0_genes
    print(f"   Loaded {len(r0_genes)} R0 marker genes")
else:
    print(f"   Warning: R0 marker file not found at {r0_marker_file}")

# Process each gene in the database
for idx in range(len(flygut_df)):
    row = flygut_df.iloc[idx]
    gene_symbol = row.iloc[1]  # Gene.Symbol is column 1
    
    if pd.isna(gene_symbol) or not isinstance(gene_symbol, str):
        continue
    
    # Clean gene symbol (handle multiple genes like "CG1234 /// CG5678")
    if '///' in gene_symbol:
        gene_symbol = gene_symbol.split('///')[0].strip()
    
    # Find the region with highest fold-change for this gene
    max_fc = -np.inf
    best_region = None
    
    for region, col_idx in region_col_indices.items():
        try:
            fc_val = float(flygut_raw.iloc[idx + 4, col_idx])  # +4 to skip header rows
            if not np.isnan(fc_val) and fc_val > max_fc:
                max_fc = fc_val
                best_region = region
        except (ValueError, TypeError, IndexError):
            continue
    
    # Also check mean expression to ensure the gene is actually expressed
    # Note: mean expression values are log2 scale (typically 2-12 range)
    if best_region and max_fc > 1.5:  # Threshold for region-specific expression (fold-change)
        try:
            mean_expr = float(flygut_raw.iloc[idx + 4, mean_col_indices[best_region]])
            if mean_expr > 5:  # Minimum log2 expression threshold (~32 in linear scale)
                region_genes[best_region].add(gene_symbol)
        except (ValueError, TypeError, IndexError):
            # If we can't check mean expression, still add if fold-change is high
            if max_fc > 2.0:
                region_genes[best_region].add(gene_symbol)

# Print statistics
print("\n   Region-specific genes:")
for region, genes in region_genes.items():
    print(f"     {region}: {len(genes)} genes")

# Also extract functional categories
print("\n5. Extracting functional categories...")
category_genes = {}

if 'CATEGORY' in flygut_df.columns:
    for idx in range(len(flygut_df)):
        row = flygut_df.iloc[idx]
        gene_symbol = row.iloc[1]  # Gene.Symbol is column 1
        category = row.iloc[2]     # CATEGORY is column 2
        
        if pd.isna(gene_symbol) or pd.isna(category):
            continue
        
        if '///' in str(gene_symbol):
            gene_symbol = str(gene_symbol).split('///')[0].strip()
        
        if category not in category_genes:
            category_genes[category] = set()
        category_genes[category].add(gene_symbol)
    
    print("   Functional categories:")
    for cat, genes in category_genes.items():
        print(f"     {cat}: {len(genes)} genes")

# ========== Score cells for each gut region ==========
print("\n6. Scoring cells for gut region identity...")

# Get gene names in our data
if 'gene_name' in adata.var.columns:
    data_genes = set(adata.var['gene_name'].dropna().tolist())
    gene_id_to_name = dict(zip(adata.var_names, adata.var['gene_name']))
    gene_name_to_id = {v: k for k, v in gene_id_to_name.items() if pd.notna(v)}
else:
    data_genes = set(adata.var_names)
    gene_name_to_id = {g: g for g in adata.var_names}

# Score each region
region_scores = pd.DataFrame(index=adata.obs_names)

for region, genes in region_genes.items():
    # Find overlapping genes
    overlap_genes = genes & data_genes
    
    if len(overlap_genes) < 5:
        print(f"   {region}: Only {len(overlap_genes)} genes found, skipping...")
        region_scores[f'{region}_score'] = 0
        continue
    
    # Convert to gene IDs
    gene_ids = [gene_name_to_id[g] for g in overlap_genes if g in gene_name_to_id]
    gene_ids = [g for g in gene_ids if g in adata.var_names]
    
    if len(gene_ids) < 5:
        print(f"   {region}: Only {len(gene_ids)} gene IDs found, skipping...")
        region_scores[f'{region}_score'] = 0
        continue
    
    print(f"   {region}: Using {len(gene_ids)} genes for scoring")
    
    # Use scanpy's score_genes function
    sc.tl.score_genes(adata, gene_ids, score_name=f'{region}_score')
    region_scores[f'{region}_score'] = adata.obs[f'{region}_score']

# Assign cells to regions based on highest score
print("\n7. Assigning cells to gut regions...")

score_cols = [c for c in region_scores.columns if c.endswith('_score')]
if score_cols:
    # Normalize scores
    scores_matrix = region_scores[score_cols].values
    
    # Assign to region with highest score
    region_names = [c.replace('_score', '') for c in score_cols]
    best_region_idx = np.argmax(scores_matrix, axis=1)
    best_scores = np.max(scores_matrix, axis=1)
    
    # Assign region labels
    adata.obs['gut_region'] = [region_names[i] for i in best_region_idx]
    adata.obs['gut_region_score'] = best_scores
    
    # Mark low-confidence assignments
    score_threshold = np.percentile(best_scores, 25)  # Bottom 25% are uncertain
    adata.obs.loc[adata.obs['gut_region_score'] < score_threshold, 'gut_region'] = 'Uncertain'
    
    print("\n   Gut region distribution:")
    for region in adata.obs['gut_region'].value_counts().index:
        count = (adata.obs['gut_region'] == region).sum()
        pct = count / adata.n_obs * 100
        print(f"     {region}: {count} cells ({pct:.1f}%)")
else:
    print("   Warning: No region scores computed, using default assignment")
    adata.obs['gut_region'] = 'Unknown'
    adata.obs['gut_region_score'] = 0

# ========== Score functional categories ==========
print("\n8. Scoring functional categories...")

for category, genes in category_genes.items():
    if category in ['-', 'UNKNOWN', 'Gene expression']:
        continue
    
    overlap_genes = genes & data_genes
    if len(overlap_genes) < 5:
        continue
    
    gene_ids = [gene_name_to_id[g] for g in overlap_genes if g in gene_name_to_id]
    gene_ids = [g for g in gene_ids if g in adata.var_names]
    
    if len(gene_ids) >= 5:
        score_name = f'func_{category.replace(" ", "_").replace("/", "_")}'
        sc.tl.score_genes(adata, gene_ids, score_name=score_name)
        print(f"   {category}: {len(gene_ids)} genes")

# ========== Create visualizations ==========
print("\n9. Creating visualizations...")

fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

# Plot 1: UMAP with gut regions
ax1 = fig.add_subplot(gs[0, 0])
if 'X_umap' in adata.obsm:
    sc.pl.umap(adata, color='gut_region', ax=ax1, show=False,
               title='Gut Region Assignment', legend_loc='right margin',
               frameon=False, size=30)
else:
    ax1.text(0.5, 0.5, 'UMAP not available', ha='center', va='center')
    ax1.axis('off')

# Plot 2: UMAP with cell types (if available)
ax2 = fig.add_subplot(gs[0, 1])
cell_type_col = None
for col in ['final_cell_type', 'cell_type', 'predicted_cell_type']:
    if col in adata.obs.columns:
        cell_type_col = col
        break

if cell_type_col and 'X_umap' in adata.obsm:
    sc.pl.umap(adata, color=cell_type_col, ax=ax2, show=False,
               title='Cell Type', legend_loc='right margin',
               frameon=False, size=30)
else:
    ax2.text(0.5, 0.5, 'Cell type not available', ha='center', va='center')
    ax2.axis('off')

# Plot 3: Region score heatmap
ax3 = fig.add_subplot(gs[0, 2])
if score_cols:
    # Sample cells for visualization
    n_sample = min(1000, adata.n_obs)
    sample_idx = np.random.choice(adata.n_obs, n_sample, replace=False)
    
    scores_sample = region_scores.iloc[sample_idx][score_cols]
    sns.heatmap(scores_sample.T, cmap='RdYlBu_r', ax=ax3, 
                xticklabels=False, cbar_kws={'label': 'Score'})
    ax3.set_ylabel('Gut Region')
    ax3.set_xlabel(f'Cells (n={n_sample})')
    ax3.set_title('Region Scores per Cell')
else:
    ax3.text(0.5, 0.5, 'No scores available', ha='center', va='center')
    ax3.axis('off')

# Plot 4: Region distribution bar chart
ax4 = fig.add_subplot(gs[1, 0])
region_counts = adata.obs['gut_region'].value_counts()
colors = plt.cm.Set2(np.linspace(0, 1, len(region_counts)))
bars = ax4.bar(range(len(region_counts)), region_counts.values, color=colors)
ax4.set_xticks(range(len(region_counts)))
ax4.set_xticklabels(region_counts.index, rotation=60, ha='right', fontsize=9)
ax4.set_ylabel('Number of Cells')
ax4.set_title('Gut Region Distribution')
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Region by cell type (if available)
ax5 = fig.add_subplot(gs[1, 1:])
if cell_type_col:
    ct_region = pd.crosstab(adata.obs[cell_type_col], adata.obs['gut_region'], 
                            normalize='index')
    sns.heatmap(ct_region, cmap='YlOrRd', ax=ax5, annot=True, fmt='.2f',
                cbar_kws={'label': 'Fraction'}, annot_kws={'fontsize': 7})
    ax5.set_xlabel('Gut Region', fontsize=10)
    ax5.set_ylabel('Cell Type', fontsize=10)
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax5.set_yticklabels(ax5.get_yticklabels(), fontsize=8)
    ax5.set_title('Cell Type Distribution across Gut Regions', fontweight='bold')
else:
    ax5.text(0.5, 0.5, 'Cell type not available', ha='center', va='center')
    ax5.axis('off')

# Plot 6: Functional category scores (top categories)
ax6 = fig.add_subplot(gs[2, :2])
func_cols = [c for c in adata.obs.columns if c.startswith('func_')]
if func_cols and cell_type_col:
    # Mean functional scores by cell type
    func_scores = adata.obs.groupby(cell_type_col)[func_cols].mean()
    func_scores.columns = [c.replace('func_', '').replace('_', ' ') for c in func_scores.columns]
    
    sns.heatmap(func_scores.T, cmap='RdYlBu_r', ax=ax6, 
                cbar_kws={'label': 'Mean Score'}, center=0)
    ax6.set_xlabel('Cell Type', fontsize=10)
    ax6.set_ylabel('Functional Category', fontsize=10)
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=60, ha='right', fontsize=7)
    ax6.set_yticklabels(ax6.get_yticklabels(), fontsize=7)
    ax6.set_title('Functional Category Enrichment by Cell Type', fontweight='bold')
else:
    ax6.text(0.5, 0.5, 'Functional scores not available', ha='center', va='center')
    ax6.axis('off')

# Plot 7: Summary
ax7 = fig.add_subplot(gs[2, 2])
summary_text = (
    f"Gut Region Annotation Summary\n"
    f"{'='*35}\n\n"
    f"Total cells: {adata.n_obs:,}\n"
    f"Regions identified: {adata.obs['gut_region'].nunique()}\n\n"
    f"Region distribution:\n"
)
for region in region_counts.index[:7]:
    count = region_counts[region]
    pct = count / adata.n_obs * 100
    summary_text += f"  {region}: {count:,} ({pct:.1f}%)\n"

if func_cols:
    summary_text += f"\nFunctional categories: {len(func_cols)}\n"

ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax7.axis('off')

plt.suptitle('Gut Region Expression Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Plot saved: {output_plot}")

# ========== Save results ==========
print("\n10. Saving results...")

# Save annotated data
adata.write_h5ad(output_h5ad, compression='gzip')
print(f"   ✓ Annotated data: {output_h5ad}")

# Save region scores
region_scores['gut_region'] = adata.obs['gut_region']
region_scores['gut_region_score'] = adata.obs['gut_region_score']
if cell_type_col:
    region_scores['cell_type'] = adata.obs[cell_type_col]
region_scores.to_csv(output_region_scores)
print(f"   ✓ Region scores: {output_region_scores}")

print("\n" + "="*80)
print("Gut Region Annotation Completed Successfully!")
print("="*80)
