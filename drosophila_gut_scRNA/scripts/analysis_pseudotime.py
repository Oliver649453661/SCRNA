#!/usr/bin/env python3
"""
补充分析3: 伪时序分析
追踪 ISC → EB → EC 分化轨迹在不同处理组中的变化
"""

import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs/outputs
input_h5ad = snakemake.input.h5ad
output_plot = snakemake.output.plot
output_h5ad = snakemake.output.h5ad
log_file = snakemake.log[0]
root_celltype = snakemake.params.root_celltype
celltype_col = snakemake.params.celltype_col

# 重定向输出到日志
sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("补充分析3: 伪时序分析 (Pseudotime Analysis)")
print("="*80)

os.makedirs(os.path.dirname(output_plot), exist_ok=True)

# 加载数据
print("\n1. 加载数据...")
adata = sc.read_h5ad(input_h5ad)
print(f"   Shape: {adata.shape}")

# 检查是否有扩散图组件
print("\n2. 计算扩散图...")

# 确保有邻居图
if 'neighbors' not in adata.uns:
    print("   计算邻居图...")
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')

# 计算扩散图
try:
    sc.tl.diffmap(adata)
    print("   ✓ 扩散图计算完成")
except Exception as e:
    print(f"   警告: 扩散图计算失败 - {e}")
    # 使用UMAP作为替代
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)

# 计算伪时序
print("\n3. 计算伪时序...")

# 找到根细胞 (ISC)
if celltype_col in adata.obs.columns:
    root_mask = adata.obs[celltype_col] == root_celltype
    if root_mask.sum() > 0:
        # 使用ISC细胞的中心作为根
        if 'X_diffmap' in adata.obsm:
            root_idx = np.where(root_mask)[0]
            # 选择扩散图第一维最小的ISC作为根
            dc1_values = adata.obsm['X_diffmap'][root_idx, 0]
            root_cell = root_idx[np.argmin(dc1_values)]
        else:
            root_cell = np.where(root_mask)[0][0]
        
        adata.uns['iroot'] = root_cell
        print(f"   根细胞设置为: {root_celltype} (index: {root_cell})")
        
        # 计算扩散伪时序
        try:
            sc.tl.dpt(adata)
            print("   ✓ DPT伪时序计算完成")
            has_dpt = True
        except Exception as e:
            print(f"   警告: DPT计算失败 - {e}")
            has_dpt = False
            # 使用简单的扩散图距离作为伪时序
            if 'X_diffmap' in adata.obsm:
                adata.obs['dpt_pseudotime'] = adata.obsm['X_diffmap'][:, 0]
                adata.obs['dpt_pseudotime'] = (adata.obs['dpt_pseudotime'] - adata.obs['dpt_pseudotime'].min()) / \
                                              (adata.obs['dpt_pseudotime'].max() - adata.obs['dpt_pseudotime'].min())
                has_dpt = True
    else:
        print(f"   警告: 未找到 {root_celltype} 细胞")
        has_dpt = False
else:
    print(f"   警告: 未找到 {celltype_col} 列")
    has_dpt = False

# 创建可视化
print("\n4. 创建可视化...")

with PdfPages(output_plot) as pdf:
    
    # 第1页: 伪时序概览
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Pseudotime Trajectory Analysis', fontsize=14, fontweight='bold')
    
    # 1.1 UMAP - 细胞类型
    ax1 = axes[0, 0]
    if 'X_umap' in adata.obsm:
        cell_types = adata.obs[celltype_col].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types)))
        color_map = dict(zip(cell_types, colors))
        
        for ct in cell_types:
            mask = adata.obs[celltype_col] == ct
            ax1.scatter(adata.obsm['X_umap'][mask, 0], adata.obsm['X_umap'][mask, 1],
                       c=[color_map[ct]], label=ct, s=1, alpha=0.5)
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        ax1.set_title('Cell Types')
        ax1.legend(loc='upper right', markerscale=5, fontsize=7)
    
    # 1.2 UMAP - 伪时序
    ax2 = axes[0, 1]
    if has_dpt and 'X_umap' in adata.obsm:
        scatter = ax2.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                             c=adata.obs['dpt_pseudotime'], cmap='viridis', s=1, alpha=0.5)
        plt.colorbar(scatter, ax=ax2, label='Pseudotime')
        ax2.set_xlabel('UMAP1')
        ax2.set_ylabel('UMAP2')
        ax2.set_title('Pseudotime')
    else:
        ax2.text(0.5, 0.5, 'Pseudotime not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
    
    # 1.3 伪时序分布 - 按细胞类型
    ax3 = axes[1, 0]
    if has_dpt:
        # 选择分化相关细胞类型
        diff_types = ['ISC', 'EB', 'EC', 'EE-Tk', 'EE-AstA', 'EE-DH31']
        diff_types = [ct for ct in diff_types if ct in adata.obs[celltype_col].values]
        
        if diff_types:
            data_to_plot = []
            for ct in diff_types:
                mask = adata.obs[celltype_col] == ct
                data_to_plot.append(adata.obs.loc[mask, 'dpt_pseudotime'].values)
            
            parts = ax3.violinplot(data_to_plot, showmeans=True, showmedians=True)
            ax3.set_xticks(range(1, len(diff_types) + 1))
            ax3.set_xticklabels(diff_types, rotation=45, ha='right')
            ax3.set_ylabel('Pseudotime')
            ax3.set_title('Pseudotime Distribution by Cell Type')
    else:
        ax3.text(0.5, 0.5, 'Pseudotime not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
    
    # 1.4 伪时序分布 - 按处理组
    ax4 = axes[1, 1]
    if has_dpt and 'group' in adata.obs.columns:
        groups = ['Control', 'PS-NPs', 'Cd', 'Cd-PS-NPs']
        groups = [g for g in groups if g in adata.obs['group'].values]
        
        data_to_plot = []
        for group in groups:
            mask = adata.obs['group'] == group
            data_to_plot.append(adata.obs.loc[mask, 'dpt_pseudotime'].values)
        
        parts = ax4.violinplot(data_to_plot, showmeans=True, showmedians=True)
        ax4.set_xticks(range(1, len(groups) + 1))
        ax4.set_xticklabels(groups, rotation=45, ha='right')
        ax4.set_ylabel('Pseudotime')
        ax4.set_title('Pseudotime Distribution by Treatment')
    else:
        ax4.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=300)
    plt.close()
    
    # 第2页: 处理组间伪时序比较
    if has_dpt and 'group' in adata.obs.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Pseudotime Comparison Across Treatments', fontsize=14, fontweight='bold')
        
        groups = ['Control', 'PS-NPs', 'Cd', 'Cd-PS-NPs']
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        
        for idx, group in enumerate(groups):
            if group not in adata.obs['group'].values:
                continue
            ax = axes[idx // 2, idx % 2]
            mask = adata.obs['group'] == group
            
            if 'X_umap' in adata.obsm:
                scatter = ax.scatter(adata.obsm['X_umap'][mask, 0], adata.obsm['X_umap'][mask, 1],
                                    c=adata.obs.loc[mask, 'dpt_pseudotime'], cmap='viridis', s=1, alpha=0.5)
                plt.colorbar(scatter, ax=ax, label='Pseudotime')
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.set_title(f'{group} (n={mask.sum():,})')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=300)
        plt.close()
    
    # 第3页: 分化轨迹统计
    fig = plt.figure(figsize=(14, 10))
    
    if has_dpt:
        # 计算统计
        stats_text = "Pseudotime Analysis Summary\n" + "="*50 + "\n\n"
        
        # 各细胞类型的平均伪时序
        stats_text += "Mean pseudotime by cell type:\n"
        ct_pseudotime = adata.obs.groupby(celltype_col)['dpt_pseudotime'].mean().sort_values()
        for ct, pt in ct_pseudotime.items():
            stats_text += f"  {ct}: {pt:.3f}\n"
        
        stats_text += "\n" + "-"*50 + "\n\n"
        
        # 各处理组的平均伪时序
        if 'group' in adata.obs.columns:
            stats_text += "Mean pseudotime by treatment:\n"
            group_pseudotime = adata.obs.groupby('group')['dpt_pseudotime'].mean()
            for group in ['Control', 'PS-NPs', 'Cd', 'Cd-PS-NPs']:
                if group in group_pseudotime.index:
                    stats_text += f"  {group}: {group_pseudotime[group]:.3f}\n"
        
        stats_text += "\n" + "-"*50 + "\n\n"
        
        stats_text += "Biological interpretation:\n"
        stats_text += "• ISC (stem cells) should have lowest pseudotime\n"
        stats_text += "• EB (progenitors) should have intermediate pseudotime\n"
        stats_text += "• EC (differentiated) should have highest pseudotime\n"
        stats_text += "• Treatment effects on differentiation can be assessed\n"
        stats_text += "  by comparing pseudotime distributions\n"
    else:
        stats_text = "Pseudotime analysis could not be completed.\n"
        stats_text += "This may be due to:\n"
        stats_text += "• Insufficient ISC cells for root definition\n"
        stats_text += "• Data structure not suitable for trajectory analysis\n"
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    plt.axis('off')
    pdf.savefig(fig, dpi=300)
    plt.close()

print(f"   ✓ 保存图表: {output_plot}")

# 保存带伪时序的数据
print("\n5. 保存数据...")
adata.write_h5ad(output_h5ad, compression='gzip')
print(f"   ✓ 保存数据: {output_h5ad}")

print("\n" + "="*80)
print("伪时序分析完成!")
print("="*80)
