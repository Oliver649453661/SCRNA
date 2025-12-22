#!/usr/bin/env python3
"""
Literature-based Cell Type Annotation for Drosophila Midgut
基于文献marker基因的果蝇中肠细胞类型注释

References:
1. Hung et al. 2020 PNAS - "A cell atlas of the adult Drosophila midgut"
2. Dutta et al. 2015 Development - "Regional Cell-Specific Transcriptome Mapping"
3. Zeng & Hou 2015 Genetics - "Intestinal stem cells in the adult Drosophila midgut"
4. Guo & Ohlstein 2015 Nature - "Stem cell regulation. Bidirectional Notch signaling"
5. Fly Cell Atlas (Li et al. 2022) - Single-cell transcriptomic atlas
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
input_reference = snakemake.input.reference
output_h5ad = snakemake.output.annotated_h5ad
output_predictions = snakemake.output.predictions
output_confidence = snakemake.output.confidence
output_umap = snakemake.output.umap_plot
output_summary = snakemake.output.summary_plot

# Set up logging
log_file = snakemake.log[0]
os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("Literature-based Cell Type Annotation for Drosophila Midgut")
print("="*80)

# ========== Define Marker Genes from Literature ==========
# Based on Hung et al. 2020 PNAS, Fly Cell Atlas, and other key publications

CELL_TYPE_MARKERS = {
    # Intestinal Stem Cells (ISC)
    # Markers: esg, Delta (Dl), escargot maintains stemness
    # Ref: Micchelli & Perrimon 2006, Ohlstein & Spradling 2006
    'ISC': {
        'markers': ['esg', 'Dl', 'N', 'Tet', 'hdc', 'zfh2', 'Lrch', 'apt', 'robo2'],
        'description': 'Intestinal Stem Cells - multipotent progenitors'
    },
    
    # Enteroblasts (EB)
    # Markers: esg+, klu, Sox21a, E(spl) complex genes
    # Ref: Zeng et al. 2010, Korzelius et al. 2014
    'EB': {
        'markers': ['klu', 'Sox21a', 'E(spl)mbeta-HLH', 'E(spl)m3-HLH', 'Tet', 'Nop60B', 'zfh2'],
        'description': 'Enteroblasts - transit-amplifying progenitors'
    },
    
    # Enterocytes - Anterior (aEC)
    # Markers: Maltase genes (Mal-A), carbohydrate digestion
    # Ref: Buchon et al. 2013, Marianes & Spradling 2013
    'aEC': {
        'markers': ['Mal-A6', 'Mal-A1', 'Mal-A2', 'Mal-A3', 'Mal-A4', 'Men-b', 'betaTry', 'Amy-p', 'Amy-d'],
        'description': 'Anterior Enterocytes - carbohydrate digestion'
    },
    
    # Enterocytes - Middle (mEC)
    # Markers: Trypsin genes, protein digestion
    # Ref: Buchon et al. 2013
    'mEC': {
        'markers': ['betaTry', 'alphaTry', 'lambdaTry', 'epsilonTry', 'Ser6', 'Jon65Aiii', 'Jon99Ci'],
        'description': 'Middle Enterocytes - protein digestion'
    },
    
    # Enterocytes - Posterior (pEC)
    # Markers: Npc2 genes, lipid absorption
    # Ref: Buchon et al. 2013, Hung et al. 2020
    'pEC': {
        'markers': ['Npc2e', 'Npc2g', 'Npc2b', 'yip7', 'CG9568', 'CG10911', 'CG13492'],
        'description': 'Posterior Enterocytes - lipid absorption'
    },
    
    # General Enterocytes (EC)
    # Markers: Myo31DF, Vha16-1, general absorptive markers
    # Ref: Fly Cell Atlas
    'EC': {
        'markers': ['Myo31DF', 'Vha16-1', 'nub', 'Tg', 'cv-2', 'cora', 'Dgp-1'],
        'description': 'General Enterocytes - absorptive epithelial cells'
    },
    
    # Enteroendocrine cells - AstA subtype
    # Markers: prospero (pros), Allatostatin A
    # Ref: Hung et al. 2020, Guo et al. 2019
    'EE-AstA': {
        'markers': ['pros', 'AstA', 'AstC'],
        'description': 'Enteroendocrine cells - Allatostatin A producing'
    },
    
    # Enteroendocrine cells - DH31 subtype
    # Markers: prospero, Diuretic hormone 31
    # Ref: Hung et al. 2020
    'EE-DH31': {
        'markers': ['pros', 'Dh31', 'CCHa2'],
        'description': 'Enteroendocrine cells - DH31 producing'
    },
    
    # Enteroendocrine cells - Tk subtype
    # Markers: prospero, Tachykinin
    # Ref: Hung et al. 2020, Guo et al. 2019
    'EE-Tk': {
        'markers': ['pros', 'Tk', 'NPF', 'sNPF'],
        'description': 'Enteroendocrine cells - Tachykinin producing'
    },
    
    # General Enteroendocrine (EE)
    # Markers: prospero is the master regulator
    # Ref: Micchelli & Perrimon 2006
    'EE': {
        'markers': ['pros', 'Rim', 'brp', 'IA-2', 'dysc', 'nrv3', 'cac'],
        'description': 'Enteroendocrine cells - hormone-secreting'
    },
    
    # Copper Cells
    # Markers: labial (lab), Uro, acid-secreting cells in middle midgut
    # Ref: Dubreuil 2004, Strand & Bhuin 2014
    'Copper Cell': {
        'markers': ['lab', 'Uro', 'Ptx1', 'CAH1', 'Nhe2', 'CG15423', 'CG10912'],
        'description': 'Copper Cells - acid-secreting, metal homeostasis'
    },
    
    # Iron Cells
    # Markers: Ferritin genes (Fer1HCH, Fer2LCH)
    # Ref: Tang & Zhou 2013
    'Iron Cell': {
        'markers': ['Fer1HCH', 'Fer2LCH', 'Tsf1', 'Tsf3', 'Mvl'],
        'description': 'Iron Cells - iron storage and homeostasis'
    },
    
    # Large Flat Cells (LFC)
    # Markers: mesh, Tsp2A
    # Ref: Hung et al. 2020, Fly Cell Atlas
    'LFC': {
        'markers': ['mesh', 'Tsp2A', 'hth', 'CG10472', 'CG18493', 'yip7'],
        'description': 'Large Flat Cells - structural support'
    },
    
    # Visceral Muscle (VM)
    # Markers: Mhc, Mef2, muscle-specific genes
    # Ref: Fly Cell Atlas
    'VM': {
        'markers': ['Mhc', 'Mef2', 'sls', 'bt', 'up', 'Mf', 'wupA', 'Zasp66'],
        'description': 'Visceral Muscle - gut motility'
    },
    
    # Cardia (Proventriculus)
    # Markers: Muc68D, specific to cardia region
    # Ref: Fly Cell Atlas
    'Cardia': {
        'markers': ['Muc68D', 'Pgant5', 'GlcAT-P', 'CG11672', 'CG3906', 'CG7720'],
        'description': 'Cardia - proventriculus region'
    },
    
    # Hemocytes
    # Markers: Hemolectin (Hml), serpent (srp)
    # Ref: Banerjee et al. 2019
    'Hemocyte': {
        'markers': ['Hml', 'He', 'srp', 'PPO1', 'PPO2', 'eater', 'NimC1'],
        'description': 'Hemocytes - immune cells'
    },
    
    # Trachea
    # Markers: breathless (btl), trachealess (trh)
    # Ref: Ghabrial et al. 2003
    'Trachea': {
        'markers': ['btl', 'trh', 'bnl', 'DSRF', 'sty'],
        'description': 'Tracheal cells - oxygen delivery'
    },
    
    # Interstitial Cells
    # Markers: Collagen, basement membrane genes
    # Ref: Fly Cell Atlas
    'Interstitial Cell': {
        'markers': ['vkg', 'Col4a1', 'Pxn', 'mys', 'LanA', 'LanB1'],
        'description': 'Interstitial Cells - connective tissue'
    },
}

# ========== Load Data ==========
print("\n1. Loading query data...")
adata_query = sc.read_h5ad(input_h5ad)
print(f"   Query data shape: {adata_query.shape}")
print(f"   Cells: {adata_query.n_obs}")
print(f"   Genes: {adata_query.n_vars}")

# Determine cluster key
cluster_key = 'leiden' if 'leiden' in adata_query.obs.columns else None
if cluster_key:
    print(f"   Cluster column: {cluster_key}")

# ========== Build Gene Name Mapping ==========
print("\n2. Building gene name mapping...")
gene_name_col = 'gene_name' if 'gene_name' in adata_query.var.columns else None
if gene_name_col:
    gene_to_idx = dict(zip(adata_query.var[gene_name_col], adata_query.var_names))
    print(f"   Using '{gene_name_col}' column for gene symbol mapping")
    print(f"   Total gene symbols: {len(gene_to_idx)}")
else:
    gene_to_idx = dict(zip(adata_query.var_names, adata_query.var_names))
    print(f"   Using var_names directly as gene symbols")

# ========== Score Marker Genes ==========
print("\n3. Computing marker gene scores for each cell type...")
print("   " + "-"*60)

marker_scores = {}
marker_info = []

for cell_type, info in CELL_TYPE_MARKERS.items():
    markers = info['markers']
    
    # Find available markers
    available_markers = []
    for m in markers:
        if m in gene_to_idx:
            available_markers.append(gene_to_idx[m])
        elif m in adata_query.var_names:
            available_markers.append(m)
    
    if available_markers:
        try:
            sc.tl.score_genes(adata_query, available_markers, score_name=f'score_{cell_type}')
            marker_scores[cell_type] = f'score_{cell_type}'
            status = "✓"
        except Exception as e:
            status = f"✗ ({e})"
    else:
        status = "✗ (no markers found)"
    
    marker_info.append({
        'cell_type': cell_type,
        'total_markers': len(markers),
        'found_markers': len(available_markers),
        'markers_found': available_markers[:5],  # First 5
        'status': status
    })
    
    print(f"   {cell_type:20s}: {len(available_markers):2d}/{len(markers):2d} markers {status}")

print("   " + "-"*60)
print(f"   Successfully scored: {len(marker_scores)} cell types")

# ========== Assign Cell Types Based on Marker Scores ==========
print("\n4. Assigning cell types based on marker scores...")

if marker_scores:
    score_cols = [f'score_{ct}' for ct in marker_scores.keys() if f'score_{ct}' in adata_query.obs.columns]
    
    if score_cols:
        # Get score matrix
        score_matrix = adata_query.obs[score_cols].values
        marker_types = [col.replace('score_', '') for col in score_cols]
        
        # Assign cell type based on highest marker score
        max_score_idx = np.argmax(score_matrix, axis=1)
        max_scores = np.max(score_matrix, axis=1)
        
        # Assign cell types
        assigned_types = np.array([marker_types[idx] for idx in max_score_idx])
        
        # Calculate confidence using softmax normalization
        # This gives probability-like confidence scores
        from scipy.special import softmax
        
        # Apply softmax with temperature scaling for better discrimination
        temperature = 0.5  # Lower temperature = sharper distribution
        score_probs = softmax(score_matrix / temperature, axis=1)
        
        # Confidence = probability of assigned cell type
        confidence = np.max(score_probs, axis=1)
        
        # Store all probabilities for reference
        for i, ct in enumerate(marker_types):
            adata_query.obs[f'prob_{ct}'] = score_probs[:, i]
        
        adata_query.obs['final_cell_type'] = assigned_types
        adata_query.obs['predicted_cell_type'] = assigned_types  # For compatibility
        adata_query.obs['prediction_confidence'] = confidence
        
        print(f"   ✓ Assigned cell types based on marker gene scores")
        print(f"   Total cell types identified: {len(np.unique(assigned_types))}")
        print(f"   Mean confidence: {confidence.mean():.3f}")
        print(f"   High confidence (>0.7): {(confidence > 0.7).sum()} cells ({(confidence > 0.7).sum()/len(confidence)*100:.1f}%)")
else:
    raise ValueError("No marker scores computed. Cannot assign cell types.")

# ========== Print Cell Type Distribution ==========
print("\n5. Cell type distribution:")
print("   " + "-"*60)
cell_type_counts = pd.Series(assigned_types).value_counts()
for ct, count in cell_type_counts.items():
    pct = count / len(assigned_types) * 100
    desc = CELL_TYPE_MARKERS.get(ct, {}).get('description', '')
    print(f"   {ct:20s}: {count:7,} cells ({pct:5.1f}%)")
print("   " + "-"*60)

# ========== Save Results ==========
print("\n6. Saving results...")

# Save annotated data
adata_query.write_h5ad(output_h5ad, compression='gzip')
print(f"   ✓ Annotated data: {output_h5ad}")

# Save predictions
predictions_df = pd.DataFrame({
    'cell_id': adata_query.obs_names,
    'cluster': adata_query.obs.get(cluster_key, 'N/A') if cluster_key else 'N/A',
    'predicted_cell_type': adata_query.obs['predicted_cell_type'],
    'final_cell_type': adata_query.obs['final_cell_type'],
    'prediction_confidence': adata_query.obs['prediction_confidence'],
})
predictions_df.to_csv(output_predictions, index=False)
print(f"   ✓ Predictions: {output_predictions}")

# Save confidence statistics
confidence_stats = []
for cell_type in adata_query.obs['final_cell_type'].unique():
    ct_cells = adata_query.obs['final_cell_type'] == cell_type
    n_cells = ct_cells.sum()
    mean_conf = adata_query.obs.loc[ct_cells, 'prediction_confidence'].mean()
    
    if cluster_key and cluster_key in adata_query.obs.columns:
        n_clusters = adata_query.obs.loc[ct_cells, cluster_key].nunique()
    else:
        n_clusters = 0
    
    confidence_stats.append({
        'cell_type': cell_type,
        'n_cells': n_cells,
        'fraction': n_cells / adata_query.n_obs,
        'mean_confidence': mean_conf,
        'n_clusters': n_clusters
    })

confidence_df = pd.DataFrame(confidence_stats)
confidence_df = confidence_df.sort_values('n_cells', ascending=False)
confidence_df.to_csv(output_confidence, index=False)
print(f"   ✓ Confidence stats: {output_confidence}")

# ========== Create Visualizations ==========
print("\n7. Creating publication-quality visualizations...")

# Figure 1: UMAP with annotations
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Plot 1: Final cell type annotations
if 'X_umap' in adata_query.obsm:
    sc.pl.umap(adata_query, color='final_cell_type', ax=axes[0], show=False,
               title='Cell Type Annotation (Literature-based)',
               legend_loc='right margin', frameon=False, size=30)
else:
    axes[0].text(0.5, 0.5, 'UMAP not available', ha='center', va='center')
    axes[0].axis('off')

# Plot 2: Prediction confidence
if 'X_umap' in adata_query.obsm:
    sc.pl.umap(adata_query, color='prediction_confidence', ax=axes[1], show=False,
               title='Prediction Confidence', frameon=False, size=30,
               cmap='YlOrRd', vmin=0, vmax=1)
else:
    axes[1].text(0.5, 0.5, 'UMAP not available', ha='center', va='center')
    axes[1].axis('off')

# Plot 3: Original clusters
if cluster_key and cluster_key in adata_query.obs.columns and 'X_umap' in adata_query.obsm:
    sc.pl.umap(adata_query, color=cluster_key, ax=axes[2], show=False,
               title='Original Clusters', frameon=False, size=30)
else:
    axes[2].text(0.5, 0.5, 'Clusters not available', ha='center', va='center')
    axes[2].axis('off')

plt.suptitle('Literature-based Cell Type Annotation - Drosophila Midgut', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_umap, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ UMAP plot: {output_umap}")

# Figure 2: Summary statistics
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Cell type proportions (pie) - only show top 10 for readability
ax1 = fig.add_subplot(gs[0, 0])
n_types = len(cell_type_counts)
colors = plt.cm.tab20(np.linspace(0, 1, n_types))

# For pie chart, only show labels for types with >2% to avoid overlap
def autopct_func(pct):
    return f'{pct:.1f}%' if pct > 2 else ''

wedges, texts, autotexts = ax1.pie(
    cell_type_counts.values, 
    labels=None,  # Don't show labels on pie
    autopct=autopct_func,
    colors=colors, 
    startangle=90, 
    textprops={'fontsize': 7}
)
# Add legend instead of labels to avoid overlap
ax1.legend(wedges, cell_type_counts.index, loc='center left', bbox_to_anchor=(1, 0.5), 
           fontsize=7, title='Cell Types')
ax1.set_title('Cell Type Distribution', fontweight='bold', fontsize=12)

# Plot 2: Cell type counts (bar)
ax2 = fig.add_subplot(gs[0, 1:])
confidence_df_sorted = confidence_df.sort_values('n_cells', ascending=True)
bars = ax2.barh(range(len(confidence_df_sorted)), confidence_df_sorted['n_cells'], 
                color=colors[:len(confidence_df_sorted)])
ax2.set_yticks(range(len(confidence_df_sorted)))
ax2.set_yticklabels(confidence_df_sorted['cell_type'])
ax2.set_xlabel('Number of Cells', fontsize=10)
ax2.set_title('Cell Type Abundance', fontweight='bold', fontsize=12)
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Confidence distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(adata_query.obs['prediction_confidence'], bins=50, color='steelblue', 
         edgecolor='black', alpha=0.7)
ax3.axvline(0.7, color='red', linestyle='--', label='High confidence')
ax3.axvline(0.5, color='orange', linestyle='--', label='Medium confidence')
ax3.set_xlabel('Prediction Confidence', fontsize=10)
ax3.set_ylabel('Number of Cells', fontsize=10)
ax3.set_title('Confidence Score Distribution', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Confidence by cell type
ax4 = fig.add_subplot(gs[1, 1:])
confidence_by_type = []
for ct in cell_type_counts.index:
    ct_cells = adata_query.obs['final_cell_type'] == ct
    conf_values = adata_query.obs.loc[ct_cells, 'prediction_confidence']
    confidence_by_type.append(conf_values)

bp = ax4.boxplot(confidence_by_type, labels=cell_type_counts.index, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax4.set_xticklabels(cell_type_counts.index, rotation=60, ha='right', fontsize=8)
ax4.set_ylabel('Prediction Confidence', fontsize=10)
ax4.set_title('Confidence Distribution by Cell Type', fontweight='bold', fontsize=12)
ax4.axhline(0.7, color='red', linestyle='--', alpha=0.3)
ax4.grid(axis='y', alpha=0.3)
plt.setp(ax4.get_xticklabels(), rotation=60, ha='right')

# Plot 5: Cluster-Cell Type mapping
if cluster_key and cluster_key in adata_query.obs.columns:
    ax5 = fig.add_subplot(gs[2, :2])
    cluster_celltype = pd.crosstab(
        adata_query.obs[cluster_key],
        adata_query.obs['final_cell_type'],
        normalize='index'
    )
    sns.heatmap(cluster_celltype, cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Fraction'},
                xticklabels=True, yticklabels=True)
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=60, ha='right', fontsize=7)
    ax5.set_yticklabels(ax5.get_yticklabels(), fontsize=7)
    ax5.set_xlabel('Cell Type', fontsize=10)
    ax5.set_ylabel('Cluster', fontsize=10)
    ax5.set_title('Cluster-to-Cell Type Mapping', fontweight='bold', fontsize=12)
else:
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.text(0.5, 0.5, 'Cluster information not available', ha='center', va='center')
    ax5.axis('off')

# Plot 6: Summary text
ax6 = fig.add_subplot(gs[2, 2])
summary_text = (
    f"Annotation Summary\n"
    f"{'='*40}\n\n"
    f"Method: Literature-based marker scoring\n\n"
    f"References:\n"
    f"  - Hung et al. 2020 PNAS\n"
    f"  - Fly Cell Atlas (Li et al. 2022)\n"
    f"  - Buchon et al. 2013\n\n"
    f"Total cells: {adata_query.n_obs:,}\n"
    f"Cell types: {adata_query.obs['final_cell_type'].nunique()}\n\n"
    f"Confidence Statistics:\n"
    f"  Mean: {adata_query.obs['prediction_confidence'].mean():.3f}\n"
    f"  Median: {adata_query.obs['prediction_confidence'].median():.3f}\n"
    f"  High (>0.7): {(adata_query.obs['prediction_confidence'] > 0.7).sum():,}\n"
    f"  Low (<0.5): {(adata_query.obs['prediction_confidence'] < 0.5).sum():,}\n"
)
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax6.axis('off')

plt.suptitle('Cell Type Annotation Statistics and Quality Control', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(output_summary, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Summary plot: {output_summary}")

# ========== Print Final Summary ==========
print("\n" + "="*80)
print("Cell Type Annotation Completed Successfully!")
print("="*80)
print(f"\nMethod: Literature-based marker gene scoring")
print(f"References: Hung et al. 2020 PNAS, Fly Cell Atlas, Buchon et al. 2013")
print(f"\nFinal cell type distribution:")
for cell_type, count in cell_type_counts.items():
    fraction = count / adata_query.n_obs * 100
    print(f"  {cell_type:20s}: {count:7,} cells ({fraction:5.2f}%)")

print(f"\nMean prediction confidence: {adata_query.obs['prediction_confidence'].mean():.3f}")
print(f"Cells with high confidence (>0.7): {(adata_query.obs['prediction_confidence'] > 0.7).sum():,}")

print("\n✓ All results saved successfully!")
print("="*80)
