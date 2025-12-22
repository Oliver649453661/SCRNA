#!/usr/bin/env python3
"""
Prepare data for CellxGene interactive visualization
准备用于 CellxGene 交互式可视化的数据
"""

import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs and outputs
input_h5ad = snakemake.input.h5ad
input_de_summary = snakemake.input.get("de_summary", None)
input_composition = snakemake.input.get("composition", None)
output_h5ad = snakemake.output.h5ad
output_info = snakemake.output.info
log_file = snakemake.log[0]

# Set up logging
os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("Preparing Data for CellxGene Visualization")
print("="*80)

# Load data
print("\n1. Loading annotated data...")
adata = sc.read_h5ad(input_h5ad)
print(f"   Shape: {adata.shape}")

# ========== Clean up and optimize for CellxGene ==========
print("\n2. Optimizing data structure...")

# Ensure we have the key embeddings
if 'X_umap' not in adata.obsm:
    print("   Computing UMAP...")
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)

# Rename embeddings for CellxGene compatibility
if 'X_umap' in adata.obsm:
    print("   ✓ UMAP embedding available")

if 'X_pca' in adata.obsm:
    print("   ✓ PCA embedding available")

# ========== Organize observation metadata ==========
print("\n3. Organizing cell metadata...")

# Define key columns to keep and their display names
key_columns = {
    # Sample/Group info
    'sample': 'Sample',
    'group': 'Treatment Group',
    'replicate': 'Replicate',
    
    # Cell type annotations
    'final_cell_type': 'Cell Type',
    'predicted_cell_type': 'Predicted Cell Type',
    'cell_type': 'Cell Type (Original)',
    'refined_cell_type': 'Refined Cell Type',
    
    # Gut region
    'gut_region': 'Gut Region',
    'gut_region_score': 'Gut Region Score',
    
    # Clustering
    'leiden': 'Leiden Cluster',
    'louvain': 'Louvain Cluster',
    
    # QC metrics
    'n_genes': 'Genes Detected',
    'n_counts': 'Total Counts',
    'total_counts': 'Total Counts',
    'n_genes_by_counts': 'Genes by Counts',
    'pct_counts_mt': 'Mitochondrial %',
    'percent_mito': 'Mitochondrial %',
    
    # Confidence scores
    'prediction_confidence': 'Annotation Confidence',
    'cluster_consensus': 'Cluster Consensus',
    
    # Batch info
    'batch': 'Batch',
}

# Keep only existing columns
obs_to_keep = []
for col in adata.obs.columns:
    if col in key_columns or col.startswith('func_') or col.endswith('_score'):
        obs_to_keep.append(col)

# Also keep any columns that look important
for col in adata.obs.columns:
    if col not in obs_to_keep:
        # Keep categorical columns with reasonable number of categories
        if adata.obs[col].dtype == 'category' or adata.obs[col].dtype == 'object':
            n_unique = adata.obs[col].nunique()
            if 2 <= n_unique <= 50:
                obs_to_keep.append(col)

print(f"   Keeping {len(obs_to_keep)} metadata columns")

# ========== Ensure proper data types ==========
print("\n4. Fixing data types...")

for col in obs_to_keep:
    if col in adata.obs.columns:
        # Convert object to category for efficiency
        if adata.obs[col].dtype == 'object':
            adata.obs[col] = adata.obs[col].astype('category')
        
        # Ensure numeric columns are float32
        elif adata.obs[col].dtype in ['float64', 'int64']:
            adata.obs[col] = adata.obs[col].astype('float32')

# ========== Add marker gene information ==========
print("\n5. Adding marker gene information...")

# Compute marker genes if not present
cluster_key = None
for key in ['final_cell_type', 'cell_type', 'leiden', 'louvain']:
    if key in adata.obs.columns:
        cluster_key = key
        break

if cluster_key and 'rank_genes_groups' not in adata.uns:
    print(f"   Computing marker genes for {cluster_key}...")
    try:
        sc.tl.rank_genes_groups(adata, cluster_key, method='wilcoxon', 
                                use_raw=True if adata.raw is not None else False)
    except Exception as e:
        print(f"   Warning: Could not compute markers: {e}")

# ========== Clean up var (gene) metadata ==========
print("\n6. Cleaning gene metadata...")

# Keep essential gene columns
var_cols_to_keep = ['gene_name', 'gene_id', 'highly_variable', 'means', 'dispersions']
var_cols_to_keep = [c for c in var_cols_to_keep if c in adata.var.columns]

# Add gene symbols as index if available
if 'gene_name' in adata.var.columns:
    # Create a mapping for display - handle Categorical type
    gene_names = adata.var['gene_name'].copy()
    # Convert to string type if categorical
    if hasattr(gene_names, 'cat'):
        gene_names = gene_names.astype(str)
    # Replace NaN/empty with var_names
    mask = gene_names.isna() | (gene_names == '') | (gene_names == 'nan')
    gene_names[mask] = pd.Series(adata.var_names, index=adata.var.index)[mask]
    adata.var['display_name'] = gene_names
    print(f"   Added display names for {(~mask).sum()} genes")

# ========== Optimize storage ==========
print("\n7. Optimizing storage...")

# Convert sparse matrix if needed
if hasattr(adata.X, 'toarray'):
    print("   Matrix is sparse (good for storage)")
else:
    print("   Matrix is dense")

# Remove unnecessary uns entries that might cause issues
keys_to_remove = []
for key in adata.uns.keys():
    # Remove large or problematic entries
    if 'neighbors' in key.lower() and 'connectivities' not in str(type(adata.uns[key])):
        pass  # Keep neighbor info
    elif isinstance(adata.uns[key], dict) and len(str(adata.uns[key])) > 100000:
        keys_to_remove.append(key)

for key in keys_to_remove:
    del adata.uns[key]
    print(f"   Removed large uns entry: {key}")

# ========== Add analysis summary to uns ==========
print("\n8. Adding analysis summary...")

analysis_summary = {
    'n_cells': adata.n_obs,
    'n_genes': adata.n_vars,
    'cell_types': adata.obs[cluster_key].nunique() if cluster_key else 0,
    'samples': adata.obs['sample'].nunique() if 'sample' in adata.obs.columns else 0,
    'groups': adata.obs['group'].unique().tolist() if 'group' in adata.obs.columns else [],
}

if 'gut_region' in adata.obs.columns:
    analysis_summary['gut_regions'] = adata.obs['gut_region'].unique().tolist()

adata.uns['analysis_summary'] = analysis_summary

# ========== Save for CellxGene ==========
print("\n9. Saving CellxGene-compatible file...")

# CellxGene requires specific format
adata.write_h5ad(output_h5ad, compression='gzip')
print(f"   ✓ Saved: {output_h5ad}")

# Calculate file size
file_size = os.path.getsize(output_h5ad) / (1024 * 1024)
print(f"   File size: {file_size:.1f} MB")

# ========== Generate info file ==========
print("\n10. Generating info file...")

info_content = f"""# CellxGene Visualization Data

## Dataset Summary
- **Cells**: {adata.n_obs:,}
- **Genes**: {adata.n_vars:,}
- **File**: {output_h5ad}
- **Size**: {file_size:.1f} MB

## Available Annotations

### Cell Metadata
"""

for col in sorted(obs_to_keep):
    if col in adata.obs.columns:
        n_unique = adata.obs[col].nunique()
        dtype = str(adata.obs[col].dtype)
        info_content += f"- **{col}**: {n_unique} unique values ({dtype})\n"

info_content += """
### Embeddings
"""
for key in adata.obsm.keys():
    shape = adata.obsm[key].shape
    info_content += f"- **{key}**: {shape}\n"

info_content += f"""
## How to View

### Option 1: CellxGene (Recommended)
```bash
# Install cellxgene
pip install cellxgene

# Launch viewer
cellxgene launch {output_h5ad} --open
```

### Option 2: Python
```python
import scanpy as sc
adata = sc.read_h5ad("{output_h5ad}")
sc.pl.umap(adata, color=['final_cell_type', 'group', 'gut_region'])
```

### Option 3: R (Seurat)
```r
library(Seurat)
library(SeuratDisk)
Convert("{output_h5ad}", dest="h5seurat")
sobj <- LoadH5Seurat("{output_h5ad.replace('.h5ad', '.h5seurat')}")
```

## Key Visualizations

1. **Cell Types**: Color by `final_cell_type` or `Cell Type`
2. **Treatment Groups**: Color by `group` or `Treatment Group`
3. **Gut Regions**: Color by `gut_region` or `Gut Region`
4. **QC Metrics**: Color by `n_genes`, `pct_counts_mt`
5. **Confidence**: Color by `prediction_confidence`

## Notes
- Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- This file is optimized for CellxGene interactive visualization
"""

with open(output_info, 'w') as f:
    f.write(info_content)

print(f"   ✓ Info file: {output_info}")

# ========== Print summary ==========
print("\n" + "="*80)
print("CellxGene Data Preparation Complete!")
print("="*80)
print(f"\nTo launch CellxGene:")
print(f"  cellxgene launch {output_h5ad} --open")
print("\nOr with custom port:")
print(f"  cellxgene launch {output_h5ad} --port 5005 --host 0.0.0.0")
print("="*80)
