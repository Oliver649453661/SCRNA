#!/usr/bin/env python3
"""
Pathway Network Visualization
展示富集通路间的关系网络
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs and outputs
input_enrichment = snakemake.input.enrichment_dir
output_plot = snakemake.output.plot
output_network = snakemake.output.network_csv
log_file = snakemake.log[0]

# Parameters
top_n = snakemake.params.get("top_n", 30)
pvalue_threshold = snakemake.params.get("pvalue_threshold", 0.05)

# Set up logging
os.makedirs(os.path.dirname(log_file), exist_ok=True)
os.makedirs(os.path.dirname(output_plot), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("Pathway Network Visualization")
print("="*80)

# Try to import networkx
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed")

# ========== Load enrichment results ==========
print("\n1. Loading enrichment results...")

enrichment_files = list(Path(input_enrichment).glob("*.csv")) + list(Path(input_enrichment).glob("*.xlsx"))
print(f"   Found {len(enrichment_files)} enrichment files")

all_enrichment = []
for f in enrichment_files:
    try:
        if str(f).endswith('.xlsx'):
            df = pd.read_excel(f)
        else:
            df = pd.read_csv(f)
        df['source_file'] = f.stem
        all_enrichment.append(df)
    except Exception as e:
        print(f"   Warning: Could not load {f}: {e}")

if not all_enrichment:
    print("   No enrichment results found, creating placeholder output")
    # Create placeholder outputs
    pd.DataFrame().to_csv(output_network, index=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.5, 'No enrichment data available', ha='center', va='center', fontsize=14)
    ax.axis('off')
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    sys.exit(0)

enrichment_df = pd.concat(all_enrichment, ignore_index=True)
print(f"   Total enrichment entries: {len(enrichment_df)}")

# Identify columns
term_col = None
pval_col = None
genes_col = None

for col in enrichment_df.columns:
    col_lower = col.lower()
    if 'term' in col_lower or 'name' in col_lower or 'description' in col_lower:
        term_col = col
    if 'p_value' in col_lower or 'pvalue' in col_lower or 'p-value' in col_lower:
        pval_col = col
    if 'gene' in col_lower and ('list' in col_lower or 'name' in col_lower or 'overlap' in col_lower):
        genes_col = col

print(f"   Term column: {term_col}")
print(f"   P-value column: {pval_col}")
print(f"   Genes column: {genes_col}")

# ========== Process pathways ==========
print("\n2. Processing pathways...")

# Filter significant pathways
if pval_col:
    sig_enrichment = enrichment_df[enrichment_df[pval_col] < pvalue_threshold].copy()
else:
    sig_enrichment = enrichment_df.copy()

print(f"   Significant pathways: {len(sig_enrichment)}")

# Get top pathways
if pval_col and len(sig_enrichment) > top_n:
    top_pathways = sig_enrichment.nsmallest(top_n, pval_col)
else:
    top_pathways = sig_enrichment.head(top_n)

print(f"   Top pathways selected: {len(top_pathways)}")

# ========== Build pathway-gene network ==========
print("\n3. Building pathway network...")

pathway_genes = {}
if genes_col and term_col:
    for _, row in top_pathways.iterrows():
        term = str(row[term_col])[:50]  # Truncate long names
        genes_str = str(row[genes_col])
        
        # Parse genes (handle different formats)
        if ';' in genes_str:
            genes = [g.strip() for g in genes_str.split(';')]
        elif ',' in genes_str:
            genes = [g.strip() for g in genes_str.split(',')]
        else:
            genes = genes_str.split()
        
        pathway_genes[term] = set(genes)

# Calculate pathway similarity (Jaccard)
pathways = list(pathway_genes.keys())
n_pathways = len(pathways)

similarity_matrix = np.zeros((n_pathways, n_pathways))
edges = []

for i in range(n_pathways):
    for j in range(i+1, n_pathways):
        genes_i = pathway_genes[pathways[i]]
        genes_j = pathway_genes[pathways[j]]
        
        intersection = len(genes_i & genes_j)
        union = len(genes_i | genes_j)
        
        if union > 0:
            jaccard = intersection / union
            similarity_matrix[i, j] = jaccard
            similarity_matrix[j, i] = jaccard
            
            if jaccard > 0.1:  # Only keep edges with some overlap
                edges.append({
                    'pathway1': pathways[i],
                    'pathway2': pathways[j],
                    'shared_genes': intersection,
                    'jaccard': jaccard
                })

edges_df = pd.DataFrame(edges)
edges_df.to_csv(output_network, index=False)
print(f"   ✓ Network edges saved: {len(edges_df)} edges")

# ========== Create visualization ==========
print("\n4. Creating visualization...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: Pathway similarity heatmap
ax1 = fig.add_subplot(gs[0, :])
if n_pathways > 0:
    # Truncate pathway names for display
    short_names = [p[:30] + '...' if len(p) > 30 else p for p in pathways]
    
    sim_df = pd.DataFrame(similarity_matrix, index=short_names, columns=short_names)
    
    # Cluster if possible
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Convert similarity to distance
        dist_matrix = 1 - similarity_matrix
        np.fill_diagonal(dist_matrix, 0)
        
        if n_pathways > 2:
            condensed = squareform(dist_matrix)
            linkage_matrix = linkage(condensed, method='average')
            dendro = dendrogram(linkage_matrix, no_plot=True)
            order = dendro['leaves']
            sim_df = sim_df.iloc[order, order]
    except:
        pass
    
    sns.heatmap(sim_df, cmap='YlOrRd', ax=ax1, 
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Jaccard Similarity'})
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=7)
    ax1.set_title('Pathway Similarity Matrix', fontweight='bold', fontsize=12)
else:
    ax1.text(0.5, 0.5, 'No pathways to display', ha='center', va='center')
    ax1.axis('off')

# Plot 2: Network graph (if networkx available)
ax2 = fig.add_subplot(gs[1, 0])
if HAS_NETWORKX and len(edges_df) > 0:
    G = nx.Graph()
    
    # Add nodes
    for p in pathways:
        G.add_node(p, size=len(pathway_genes.get(p, [])))
    
    # Add edges
    for _, row in edges_df.iterrows():
        G.add_edge(row['pathway1'], row['pathway2'], weight=row['jaccard'])
    
    # Layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Draw
    node_sizes = [G.nodes[n].get('size', 10) * 20 for n in G.nodes()]
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=node_sizes, 
                           node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax2, width=edge_weights, 
                           alpha=0.5, edge_color='gray')
    
    # Labels for top nodes
    labels = {n: n[:20] + '...' if len(n) > 20 else n for n in list(G.nodes())[:15]}
    nx.draw_networkx_labels(G, pos, labels, ax=ax2, font_size=6)
    
    ax2.set_title('Pathway Network', fontweight='bold')
    ax2.axis('off')
else:
    ax2.text(0.5, 0.5, 'Network visualization not available', ha='center', va='center')
    ax2.axis('off')

# Plot 3: Top pathways bar chart
ax3 = fig.add_subplot(gs[1, 1])
if term_col and pval_col and len(top_pathways) > 0:
    plot_data = top_pathways.nsmallest(15, pval_col).copy()
    plot_data['neg_log_p'] = -np.log10(plot_data[pval_col].clip(lower=1e-50))
    plot_data['short_term'] = plot_data[term_col].apply(lambda x: str(x)[:40] + '...' if len(str(x)) > 40 else str(x))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_data)))
    
    y_pos = range(len(plot_data))
    ax3.barh(y_pos, plot_data['neg_log_p'], color=colors)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(plot_data['short_term'], fontsize=8)
    ax3.set_xlabel('-log10(p-value)')
    ax3.set_title('Top Enriched Pathways', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'No pathway data', ha='center', va='center')
    ax3.axis('off')

plt.suptitle('Pathway Enrichment Network Analysis', fontsize=14, fontweight='bold')
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Plot saved: {output_plot}")

print("\n" + "="*80)
print("Pathway Network Visualization Completed!")
print("="*80)
