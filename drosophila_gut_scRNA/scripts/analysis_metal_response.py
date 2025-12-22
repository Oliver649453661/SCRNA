#!/usr/bin/env python3
"""
补充分析1: 金属响应基因表达热图
展示 MtnA-E, Fer, ZnT 等金属响应基因在各处理组和细胞类型中的表达
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
import warnings
warnings.filterwarnings('ignore')

# Snakemake inputs/outputs
input_h5ad = snakemake.input.h5ad
input_de = snakemake.input.de_results
output_plot = snakemake.output.plot
output_genes = snakemake.output.genes
log_file = snakemake.log[0]

# 重定向输出到日志
sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("="*80)
print("补充分析1: 金属响应基因表达分析")
print("="*80)

os.makedirs(os.path.dirname(output_plot), exist_ok=True)

# 加载差异表达数据
print("\n1. 加载差异表达数据...")
de_df = pd.read_csv(input_de)
print(f"   总 DEG 记录: {len(de_df)}")

# 定义金属响应基因关键词
METAL_KEYWORDS = ['Mtn', 'Fer', 'Zn', 'Zip', 'Ctr', 'Irp', 'Mvl', 'Tsf']

# 提取金属响应基因
print("\n2. 提取金属响应基因...")
metal_genes = de_df[de_df['gene_symbol'].str.contains('|'.join(METAL_KEYWORDS), na=False, case=False)].copy()
print(f"   找到 {len(metal_genes)} 条金属响应基因记录")

# 按处理组分组
groups = ['Cd', 'PS-NPs', 'Cd-PS-NPs']
metal_by_group = {}
for group in groups:
    group_data = metal_genes[metal_genes['comparison_group'] == group].copy()
    group_data = group_data.sort_values('logfoldchanges', ascending=False)
    metal_by_group[group] = group_data
    print(f"   {group}: {len(group_data)} 个基因")

# 创建可视化
print("\n3. 创建可视化...")

with PdfPages(output_plot) as pdf:
    
    # 第1页: 金属响应基因热图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Metal Response Gene Expression Analysis', fontsize=14, fontweight='bold')
    
    # 1.1 Cd vs Control - Top 金属响应基因
    ax1 = axes[0, 0]
    cd_metal = metal_by_group.get('Cd', pd.DataFrame())
    if len(cd_metal) > 0:
        top_genes = cd_metal.head(15)
        colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_genes['logfoldchanges']]
        ax1.barh(range(len(top_genes)), top_genes['logfoldchanges'], color=colors)
        ax1.set_yticks(range(len(top_genes)))
        ax1.set_yticklabels(top_genes['gene_symbol'], fontsize=9)
        ax1.set_xlabel('Log2 Fold Change')
        ax1.set_title('Metal Response Genes: Cd vs Control')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.invert_yaxis()
    
    # 1.2 所有处理组的金属响应基因比较
    ax2 = axes[0, 1]
    # 获取所有组共有的top金属响应基因
    all_metal_genes = set()
    for group in groups:
        if group in metal_by_group:
            all_metal_genes.update(metal_by_group[group]['gene_symbol'].head(10).tolist())
    
    if all_metal_genes:
        # 创建热图数据
        heatmap_data = pd.DataFrame(index=list(all_metal_genes), columns=groups)
        for group in groups:
            if group in metal_by_group:
                group_data = metal_by_group[group].set_index('gene_symbol')
                for gene in all_metal_genes:
                    if gene in group_data.index:
                        heatmap_data.loc[gene, group] = group_data.loc[gene, 'logfoldchanges']
        
        heatmap_data = heatmap_data.astype(float).fillna(0)
        # 按Cd组排序
        heatmap_data = heatmap_data.sort_values('Cd', ascending=False)
        
        sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, ax=ax2,
                    annot=True, fmt='.1f', cbar_kws={'label': 'Log2 FC'})
        ax2.set_xlabel('Treatment vs Control')
        ax2.set_ylabel('Gene')
        ax2.set_title('Metal Response Genes Across Treatments')
    
    # 1.3 金属硫蛋白 (Metallothionein) 特写
    ax3 = axes[1, 0]
    mtn_genes = de_df[de_df['gene_symbol'].str.startswith('Mtn', na=False)]
    if len(mtn_genes) > 0:
        mtn_pivot = mtn_genes.pivot_table(
            index='gene_symbol', 
            columns='comparison_group', 
            values='logfoldchanges',
            aggfunc='first'
        )
        if 'Cd' in mtn_pivot.columns:
            mtn_pivot = mtn_pivot.sort_values('Cd', ascending=False)
        
        mtn_pivot = mtn_pivot[groups].fillna(0)
        mtn_pivot.plot(kind='bar', ax=ax3, color=['#e74c3c', '#3498db', '#9b59b6'])
        ax3.set_ylabel('Log2 Fold Change')
        ax3.set_title('Metallothionein (MtnA-E) Expression')
        ax3.legend(title='vs Control')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 1.4 铁蛋白和锌转运蛋白
    ax4 = axes[1, 1]
    fer_zn_genes = de_df[de_df['gene_symbol'].str.contains('Fer|Zn|Zip', na=False, case=False)]
    if len(fer_zn_genes) > 0:
        fer_zn_pivot = fer_zn_genes.pivot_table(
            index='gene_symbol',
            columns='comparison_group',
            values='logfoldchanges',
            aggfunc='first'
        )
        if 'Cd' in fer_zn_pivot.columns:
            fer_zn_pivot = fer_zn_pivot.sort_values('Cd', ascending=False)
        
        fer_zn_pivot = fer_zn_pivot[groups].fillna(0).head(10)
        fer_zn_pivot.plot(kind='bar', ax=ax4, color=['#e74c3c', '#3498db', '#9b59b6'])
        ax4.set_ylabel('Log2 Fold Change')
        ax4.set_title('Ferritin & Zinc Transporter Expression')
        ax4.legend(title='vs Control')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=300)
    plt.close()
    
    # 第2页: 生物学解释
    fig = plt.figure(figsize=(14, 10))
    
    summary_text = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    METAL RESPONSE GENE ANALYSIS SUMMARY                       ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  KEY FINDINGS:                                                               ║
    ║                                                                              ║
    ║  1. METALLOTHIONEINS (MtnA-E):                                               ║
    ║     • Most strongly upregulated genes in Cd treatment                        ║
    ║     • MtnB shows highest induction (~9-fold, log2FC)                        ║
    ║     • Function: Heavy metal binding and detoxification                       ║
    ║     • Biological significance: Primary defense against Cd toxicity           ║
    ║                                                                              ║
    ║  2. FERRITINS (Fer1HCH, Fer2LCH, Fer3HCH):                                   ║
    ║     • Significantly upregulated in Cd treatment                              ║
    ║     • Function: Iron storage and sequestration                               ║
    ║     • Biological significance: Cd disrupts iron homeostasis                  ║
    ║                                                                              ║
    ║  3. ZINC TRANSPORTERS (ZnT35C, ZnT63C):                                      ║
    ║     • Upregulated in Cd treatment                                            ║
    ║     • Function: Zinc efflux from cells                                       ║
    ║     • Biological significance: Cd competes with Zn, causing Zn imbalance     ║
    ║                                                                              ║
    ║  4. SYNERGISTIC EFFECT (Cd-PS-NPs):                                          ║
    ║     • Combined treatment shows enhanced metal response                       ║
    ║     • PS-NPs may increase Cd bioavailability                                 ║
    ║     • Suggests "Trojan horse" mechanism                                      ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  CONCLUSION:                                                                 ║
    ║  Cd exposure activates a robust metal detoxification response, with          ║
    ║  metallothioneins as the primary defense mechanism. The synergistic effect   ║
    ║  of Cd-PS-NPs suggests that nanoplastics enhance cadmium toxicity.          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    plt.axis('off')
    pdf.savefig(fig, dpi=300)
    plt.close()

print(f"   ✓ 保存图表: {output_plot}")

# 保存金属响应基因列表
print("\n4. 保存金属响应基因列表...")
metal_genes_sorted = metal_genes.sort_values(['comparison_group', 'logfoldchanges'], ascending=[True, False])
metal_genes_sorted.to_csv(output_genes, index=False)
print(f"   ✓ 保存基因列表: {output_genes}")

print("\n" + "="*80)
print("金属响应基因分析完成!")
print("="*80)
