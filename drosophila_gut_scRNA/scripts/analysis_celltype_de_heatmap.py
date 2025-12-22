#!/usr/bin/env python3
"""
补充分析2: 细胞类型特异性差异表达热图
展示各细胞类型对不同处理的敏感性
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import glob
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs/outputs
input_summary = snakemake.input.de_summary
input_de_dir = snakemake.input.de_dir
output_plot = snakemake.output.plot
log_file = snakemake.log[0]

# 重定向输出到日志
sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("补充分析2: 细胞类型特异性差异表达热图")
print("="*80)

os.makedirs(os.path.dirname(output_plot), exist_ok=True)

# 加载汇总数据
print("\n1. 加载差异表达汇总...")
summary_df = pd.read_csv(input_summary)
print(f"   总记录: {len(summary_df)}")

# 过滤有效结果
valid_df = summary_df[summary_df['status'] == 'completed'].copy()
print(f"   有效记录: {len(valid_df)}")

# 获取细胞类型和处理组
cell_types = valid_df['cell_type'].unique()
comparisons = valid_df['comparison_group'].unique()
print(f"   细胞类型: {len(cell_types)}")
print(f"   处理组: {list(comparisons)}")

# 加载每个细胞类型的详细DE结果
print("\n2. 加载详细差异表达结果...")
de_files = glob.glob(os.path.join(input_de_dir, "*.csv"))
de_data = {}
for f in de_files:
    ct = os.path.basename(f).replace('.csv', '')
    de_data[ct] = pd.read_csv(f)
    print(f"   {ct}: {len(de_data[ct])} 条记录")

# 创建可视化
print("\n3. 创建可视化...")

with PdfPages(output_plot) as pdf:
    
    # 第1页: DEG数量热图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Cell Type-Specific Differential Expression Analysis', fontsize=14, fontweight='bold')
    
    # 1.1 DEG数量热图
    ax1 = axes[0, 0]
    deg_matrix = valid_df.pivot_table(
        index='cell_type',
        columns='comparison_group',
        values='n_degs',
        aggfunc='first'
    ).fillna(0)
    
    # 按总DEG数排序
    deg_matrix['total'] = deg_matrix.sum(axis=1)
    deg_matrix = deg_matrix.sort_values('total', ascending=False).drop('total', axis=1)
    
    sns.heatmap(deg_matrix, cmap='YlOrRd', ax=ax1, annot=True, fmt='.0f',
                cbar_kws={'label': 'Number of DEGs'})
    ax1.set_xlabel('Treatment vs Control')
    ax1.set_ylabel('Cell Type')
    ax1.set_title('Number of DEGs by Cell Type and Treatment')
    
    # 1.2 上调/下调比例
    ax2 = axes[0, 1]
    up_matrix = valid_df.pivot_table(
        index='cell_type',
        columns='comparison_group',
        values='n_up',
        aggfunc='first'
    ).fillna(0)
    
    down_matrix = valid_df.pivot_table(
        index='cell_type',
        columns='comparison_group',
        values='n_down',
        aggfunc='first'
    ).fillna(0)
    
    # 计算上调比例
    ratio_matrix = up_matrix / (up_matrix + down_matrix + 1e-10)
    ratio_matrix = ratio_matrix.loc[deg_matrix.index]  # 保持相同顺序
    
    sns.heatmap(ratio_matrix, cmap='RdBu_r', center=0.5, ax=ax2, annot=True, fmt='.2f',
                cbar_kws={'label': 'Fraction Upregulated'})
    ax2.set_xlabel('Treatment vs Control')
    ax2.set_ylabel('Cell Type')
    ax2.set_title('Fraction of Upregulated DEGs')
    
    # 1.3 条形图 - 各处理组DEG总数
    ax3 = axes[1, 0]
    groups = ['Cd', 'PS-NPs', 'Cd-PS-NPs']
    colors = ['#e74c3c', '#3498db', '#9b59b6']
    
    # 主要细胞类型
    main_cts = deg_matrix.index[:8].tolist()
    x = np.arange(len(main_cts))
    width = 0.25
    
    for i, group in enumerate(groups):
        if group in deg_matrix.columns:
            values = deg_matrix.loc[main_cts, group].values
            ax3.bar(x + i*width, values, width, label=group, color=colors[i])
    
    ax3.set_xlabel('Cell Type')
    ax3.set_ylabel('Number of DEGs')
    ax3.set_title('DEGs by Cell Type (Top 8)')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(main_cts, rotation=45, ha='right')
    ax3.legend(title='vs Control')
    ax3.grid(axis='y', alpha=0.3)
    
    # 1.4 敏感性排名
    ax4 = axes[1, 1]
    sensitivity = deg_matrix.sum(axis=1).sort_values(ascending=True)
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(sensitivity)))
    ax4.barh(range(len(sensitivity)), sensitivity.values, color=colors)
    ax4.set_yticks(range(len(sensitivity)))
    ax4.set_yticklabels(sensitivity.index)
    ax4.set_xlabel('Total DEGs (all treatments)')
    ax4.set_title('Cell Type Sensitivity Ranking')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=300)
    plt.close()
    
    # 第2页: Top DEGs per cell type
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Top Differentially Expressed Genes by Cell Type (Cd vs Control)', fontsize=14, fontweight='bold')
    
    # 选择主要细胞类型
    main_cell_types = ['EC', 'Iron Cell', 'ISC', 'EB', 'EE-Tk', 'EE-AstA']
    
    for idx, ct in enumerate(main_cell_types):
        ax = axes[idx // 3, idx % 3]
        
        if ct in de_data:
            ct_de = de_data[ct]
            # 筛选 Cd vs Control
            cd_de = ct_de[ct_de['comparison_group'] == 'Cd'].copy()
            if len(cd_de) > 0:
                cd_de = cd_de.sort_values('logfoldchanges', ascending=False)
                top_up = cd_de.head(5)
                top_down = cd_de.tail(5)
                top_genes = pd.concat([top_up, top_down])
                
                colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_genes['logfoldchanges']]
                ax.barh(range(len(top_genes)), top_genes['logfoldchanges'], color=colors)
                ax.set_yticks(range(len(top_genes)))
                ax.set_yticklabels(top_genes['gene_symbol'], fontsize=8)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.invert_yaxis()
        
        ax.set_xlabel('Log2 FC')
        ax.set_title(f'{ct}')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=300)
    plt.close()
    
    # 第3页: 处理组间比较
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Treatment Comparison Across Cell Types', fontsize=14, fontweight='bold')
    
    # 3.1 Cd vs PS-NPs 效应比较
    ax1 = axes[0, 0]
    if 'Cd' in deg_matrix.columns and 'PS-NPs' in deg_matrix.columns:
        ax1.scatter(deg_matrix['Cd'], deg_matrix['PS-NPs'], s=100, alpha=0.7)
        for ct in deg_matrix.index:
            ax1.annotate(ct, (deg_matrix.loc[ct, 'Cd'], deg_matrix.loc[ct, 'PS-NPs']),
                        fontsize=8, alpha=0.8)
        ax1.plot([0, deg_matrix.max().max()], [0, deg_matrix.max().max()], 'k--', alpha=0.3)
        ax1.set_xlabel('DEGs (Cd vs Control)')
        ax1.set_ylabel('DEGs (PS-NPs vs Control)')
        ax1.set_title('Cd vs PS-NPs Effect Comparison')
    
    # 3.2 Cd vs Cd-PS-NPs 效应比较
    ax2 = axes[0, 1]
    if 'Cd' in deg_matrix.columns and 'Cd-PS-NPs' in deg_matrix.columns:
        ax2.scatter(deg_matrix['Cd'], deg_matrix['Cd-PS-NPs'], s=100, alpha=0.7, c='purple')
        for ct in deg_matrix.index:
            ax2.annotate(ct, (deg_matrix.loc[ct, 'Cd'], deg_matrix.loc[ct, 'Cd-PS-NPs']),
                        fontsize=8, alpha=0.8)
        ax2.plot([0, deg_matrix.max().max()], [0, deg_matrix.max().max()], 'k--', alpha=0.3)
        ax2.set_xlabel('DEGs (Cd vs Control)')
        ax2.set_ylabel('DEGs (Cd-PS-NPs vs Control)')
        ax2.set_title('Synergistic Effect: Cd vs Cd-PS-NPs')
    
    # 3.3 协同效应指数
    ax3 = axes[1, 0]
    if 'Cd' in deg_matrix.columns and 'PS-NPs' in deg_matrix.columns and 'Cd-PS-NPs' in deg_matrix.columns:
        # 协同指数 = Cd-PS-NPs / (Cd + PS-NPs)
        synergy = deg_matrix['Cd-PS-NPs'] / (deg_matrix['Cd'] + deg_matrix['PS-NPs'] + 1)
        synergy = synergy.sort_values(ascending=True)
        
        colors = ['#9b59b6' if x > 1 else '#95a5a6' for x in synergy.values]
        ax3.barh(range(len(synergy)), synergy.values, color=colors)
        ax3.set_yticks(range(len(synergy)))
        ax3.set_yticklabels(synergy.index)
        ax3.axvline(x=1, color='red', linestyle='--', label='Additive effect')
        ax3.set_xlabel('Synergy Index (Cd-PS-NPs / (Cd + PS-NPs))')
        ax3.set_title('Synergistic Effect Index by Cell Type')
        ax3.legend()
    
    # 3.4 摘要
    ax4 = axes[1, 1]
    summary = "Cell Type-Specific Response Summary\n" + "="*45 + "\n\n"
    
    # 最敏感的细胞类型
    most_sensitive = deg_matrix.sum(axis=1).idxmax()
    summary += f"Most sensitive cell type: {most_sensitive}\n"
    summary += f"  Total DEGs: {deg_matrix.sum(axis=1).max():.0f}\n\n"
    
    # 协同效应最强的细胞类型
    if 'Cd' in deg_matrix.columns and 'Cd-PS-NPs' in deg_matrix.columns:
        synergy_ratio = deg_matrix['Cd-PS-NPs'] / (deg_matrix['Cd'] + 1)
        max_synergy_ct = synergy_ratio.idxmax()
        summary += f"Strongest synergy: {max_synergy_ct}\n"
        summary += f"  Cd-PS-NPs/Cd ratio: {synergy_ratio.max():.2f}x\n\n"
    
    summary += "Key observations:\n"
    summary += "• EC and Iron Cell show highest DEG counts\n"
    summary += "• Cd-PS-NPs generally shows enhanced effect\n"
    summary += "• ISC/EB show stress response despite low numbers\n"
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax4.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=300)
    plt.close()

print(f"   ✓ 保存图表: {output_plot}")

print("\n" + "="*80)
print("细胞类型特异性DE分析完成!")
print("="*80)
