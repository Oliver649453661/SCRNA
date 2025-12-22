#!/usr/bin/env python3
"""
Batch effect correction using Harmony, BBKNN, or Scanorama
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Snakemake inputs and outputs
input_h5ad = snakemake.input.h5ad
output_h5ad = snakemake.output.h5ad
output_plot = snakemake.output.plot

# Parameters
batch_key = snakemake.params.batch_key
method = snakemake.params.method

# Set up logging
import sys
sys.stderr = open(snakemake.log[0], 'w')
sys.stdout = sys.stderr

print("Loading data...")
adata = sc.read_h5ad(input_h5ad)

print(f"Data shape: {adata.shape}")
print(f"Batch key: {batch_key}")
print(f"Batch correction method: {method}")

# Check if batch key exists
if batch_key not in adata.obs.columns:
    print(f"Warning: Batch key '{batch_key}' not found in obs. Available keys: {adata.obs.columns.tolist()}")
    print("Skipping batch correction...")
    adata.write_h5ad(output_h5ad, compression='gzip')
    sys.exit(0)

# Store original for comparison
adata.obsm['X_pca_uncorrected'] = adata.obsm['X_pca'].copy() if 'X_pca' in adata.obsm else None

# Perform normalization and PCA if not already done
if 'X_pca' not in adata.obsm or adata.obsm['X_pca'] is None:
    print("Normalizing and computing PCA...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50, svd_solver='arpack')

print(f"\nApplying {method} batch correction...")

if method.lower() == 'harmony':
    try:
        import harmonypy as hm
        # Run Harmony
        ho = hm.run_harmony(
            adata.obsm['X_pca'],
            adata.obs,
            batch_key,
            max_iter_harmony=20
        )
        adata.obsm['X_pca_harmony'] = ho.Z_corr.T
        adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
        print("Harmony correction completed")
    except ImportError:
        print("Harmony not installed, falling back to combat")
        method = 'combat'

if method.lower() == 'combat':
    # Use Combat for batch correction
    sc.pp.combat(adata, key=batch_key)
    print("Combat correction completed")

elif method.lower() == 'bbknn':
    # Use BBKNN for batch correction
    import bbknn
    bbknn.bbknn(adata, batch_key=batch_key, n_pcs=50)
    print("BBKNN correction completed")

elif method.lower() == 'scanorama':
    # Use Scanorama for batch correction
    import scanorama
    # Split by batch
    batches = []
    batch_names = adata.obs[batch_key].unique()
    for batch in batch_names:
        batch_data = adata[adata.obs[batch_key] == batch].copy()
        batches.append(batch_data)
    
    # Integrate
    integrated = scanorama.assemble(batches, verbose=1)
    adata.obsm['X_scanorama'] = integrated
    adata.obsm['X_pca'] = integrated
    print("Scanorama correction completed")

# Recompute neighbors and UMAP with corrected data
print("\nRecomputing neighbors and UMAP...")
if method.lower() != 'bbknn':  # BBKNN already computes neighbors
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

sc.tl.umap(adata, min_dist=0.5)

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original PCA
if adata.obsm['X_pca_uncorrected'] is not None:
    sc.pl.pca(adata, color=batch_key, 
              use_raw=False,
              components=['1,2'],
              ax=axes[0, 0], show=False)
    axes[0, 0].set_title('PCA - Before Correction')
else:
    axes[0, 0].text(0.5, 0.5, 'PCA not available', ha='center', va='center')
    axes[0, 0].axis('off')

# Corrected PCA
sc.pl.pca(adata, color=batch_key, 
          use_raw=False,
          components=['1,2'],
          ax=axes[0, 1], show=False)
axes[0, 1].set_title(f'PCA - After {method.upper()} Correction')

# UMAP colored by batch
sc.pl.umap(adata, color=batch_key, ax=axes[0, 2], show=False)
axes[0, 2].set_title('UMAP - Batch')

# UMAP colored by n_genes_by_counts (如果存在)
if 'n_genes_by_counts' in adata.obs.columns:
    sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[1, 0], show=False, cmap='viridis')
    axes[1, 0].set_title('UMAP - n_genes_by_counts')
else:
    axes[1, 0].text(0.5, 0.5, 'n_genes_by_counts\nnot available', 
                   ha='center', va='center', fontsize=12)
    axes[1, 0].set_title('UMAP - n_genes')

# UMAP colored by total_counts (如果存在)
if 'total_counts' in adata.obs.columns:
    sc.pl.umap(adata, color='total_counts', ax=axes[1, 1], show=False, cmap='viridis')
    axes[1, 1].set_title('UMAP - total_counts')
else:
    axes[1, 1].text(0.5, 0.5, 'total_counts\nnot available', 
                   ha='center', va='center', fontsize=12)
    axes[1, 1].set_title('UMAP - n_counts')

# Summary text
n_batches = adata.obs[batch_key].nunique()
batch_sizes = adata.obs[batch_key].value_counts()
summary_text = (
    f"Batch Correction: {method.upper()}\n\n"
    f"Number of batches: {n_batches}\n\n"
    f"Batch sizes:\n"
)
for batch, size in batch_sizes.items():
    summary_text += f"  {batch}: {size}\n"

axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center', 
                family='monospace')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlot saved to: {output_plot}")

# Save corrected data
print(f"Saving batch-corrected data to: {output_h5ad}")
adata.write_h5ad(output_h5ad, compression='gzip')

print("\nBatch correction completed successfully!")
