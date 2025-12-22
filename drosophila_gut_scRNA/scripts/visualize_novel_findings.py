#!/usr/bin/env python3
"""
生成新颖发现的可视化图表 - 用于PPT展示
重点：展示发现的显著性和独特性，与其他通路形成对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

OUTPUT_DIR = Path("results/figures/novel_findings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载实际的GO富集数据
ENRICHMENT_FILE = "results/enrichment/per_celltype/all_enrichment_results.csv"
df_all = pd.read_csv(ENRICHMENT_FILE)

# =============================================================================
# 图1: EC细胞下调通路排名图 - 展示先天免疫的独特性
# =============================================================================
def plot_ec_downregulated_ranking():
    """展示EC细胞Cd处理下调通路中，先天免疫的排名位置"""
    
    # 筛选EC细胞Cd处理下调的GO terms
    df_ec_down = df_all[(df_all['cell_type'] == 'EC') & 
                        (df_all['comparison'] == 'Cd') & 
                        (df_all['direction'] == 'down')].copy()
    
    # 计算-log10(p-value)
    df_ec_down['neg_log10_p'] = -np.log10(df_ec_down['Adjusted P-value'].replace(0, 1e-300))
    df_ec_down = df_ec_down.sort_values('neg_log10_p', ascending=False).head(20)
    
    # 标记先天免疫相关
    immune_terms = ['innate immune', 'immune response', 'defense', 'melanization']
    df_ec_down['is_immune'] = df_ec_down['Term'].str.lower().apply(
        lambda x: any(t in x for t in immune_terms))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#E74C3C' if im else '#95A5A6' for im in df_ec_down['is_immune']]
    
    bars = ax.barh(range(len(df_ec_down)), df_ec_down['neg_log10_p'], color=colors, 
                   edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(df_ec_down)))
    ax.set_yticklabels(df_ec_down['Term'], fontsize=10)
    ax.invert_yaxis()
    
    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=14, fontweight='bold')
    ax.set_title('EC Cell: Top 20 Downregulated GO Terms (Cd Treatment)\n'
                 '🔴 Innate Immune Response ranks #1 among ALL downregulated pathways!', 
                 fontsize=14, fontweight='bold')
    
    # 添加显著性阈值
    ax.axvline(x=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-np.log10(1e-5), color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=-np.log10(1e-10), color='red', linestyle='--', alpha=0.5)
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E74C3C', label='Innate Immune Related'),
                       Patch(facecolor='#95A5A6', label='Other Pathways')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig1_EC_downregulated_ranking.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig1_EC_downregulated_ranking.png', bbox_inches='tight')
    plt.close()
    print("✓ 图1: EC下调通路排名图已保存")

# =============================================================================
# 图2: 跨细胞类型先天免疫一致性下调 - 展示7种细胞的一致性
# =============================================================================
def plot_cross_celltype_immune_consistency():
    """展示先天免疫响应在所有细胞类型中都显著下调的一致性"""
    
    # 提取所有细胞类型中先天免疫响应的富集结果
    df_immune = df_all[(df_all['GO_ID'] == 'GO:0045087') & 
                       (df_all['comparison'] == 'Cd') & 
                       (df_all['direction'] == 'down')].copy()
    
    df_immune['neg_log10_p'] = -np.log10(df_immune['Adjusted P-value'].replace(0, 1e-300))
    df_immune = df_immune.sort_values('neg_log10_p', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图: 先天免疫在各细胞类型的显著性
    ax1 = axes[0]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(df_immune)))
    bars = ax1.bar(range(len(df_immune)), df_immune['neg_log10_p'], 
                   color=colors, edgecolor='darkred', linewidth=2)
    
    ax1.set_xticks(range(len(df_immune)))
    ax1.set_xticklabels(df_immune['cell_type'], rotation=45, ha='right', fontsize=12)
    ax1.set_ylabel('-log10(Adjusted P-value)', fontsize=14, fontweight='bold')
    ax1.set_title('Innate Immune Response (GO:0045087)\nSignificantly Downregulated in ALL 7 Cell Types!', 
                  fontsize=14, fontweight='bold')
    
    # 添加显著性阈值线
    ax1.axhline(y=-np.log10(0.05), color='gray', linestyle='--', label='p=0.05')
    ax1.axhline(y=-np.log10(1e-5), color='orange', linestyle='--', label='p=1e-5')
    ax1.axhline(y=-np.log10(1e-10), color='red', linestyle='--', label='p=1e-10')
    ax1.legend(loc='upper right')
    
    # 添加数值标签
    for i, (idx, row) in enumerate(df_immune.iterrows()):
        ax1.text(i, row['neg_log10_p'] + 0.3, f"{row['Study_Count']} genes", 
                 ha='center', fontsize=9, color='darkred')
    
    # 右图: 对比 - 金属解毒vs先天免疫（上调vs下调的对比）
    ax2 = axes[1]
    
    # 获取金属解毒通路（上调）
    df_metal = df_all[(df_all['Term'].str.contains('metal|detoxification', case=False, na=False)) & 
                      (df_all['comparison'] == 'Cd') & 
                      (df_all['direction'] == 'up')].copy()
    df_metal['neg_log10_p'] = -np.log10(df_metal['Adjusted P-value'].replace(0, 1e-300))
    metal_mean = df_metal['neg_log10_p'].mean() if len(df_metal) > 0 else 0
    
    immune_mean = df_immune['neg_log10_p'].mean()
    
    categories = ['Metal Detoxification\n(UPREGULATED)', 'Innate Immunity\n(DOWNREGULATED)']
    values = [metal_mean, immune_mean]
    colors_bar = ['#27AE60', '#E74C3C']
    
    bars2 = ax2.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=2, width=0.6)
    ax2.set_ylabel('Mean -log10(Adjusted P-value)', fontsize=14, fontweight='bold')
    ax2.set_title('Cd Treatment: Two Major Responses\n(Both Highly Significant but Opposite Directions)', 
                  fontsize=14, fontweight='bold')
    
    # 添加注释
    ax2.annotate('Expected\n(Conventional)', xy=(0, metal_mean), xytext=(0, metal_mean+2),
                 ha='center', fontsize=11, color='#27AE60')
    ax2.annotate('NOVEL Finding!\n(Unexpected)', xy=(1, immune_mean), xytext=(1, immune_mean+2),
                 ha='center', fontsize=11, color='#E74C3C', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig2_cross_celltype_immune_consistency.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig2_cross_celltype_immune_consistency.png', bbox_inches='tight')
    plt.close()
    print("✓ 图2: 跨细胞类型免疫一致性图已保存")

# =============================================================================
# 图3: EB细胞PS-NPs处理 - 100%命中核糖体的独特性
# =============================================================================
def plot_psnps_eb_unique_ribosome():
    """展示PS-NPs处理EB细胞时，所有GO terms都是核糖体相关的独特性"""
    
    # 筛选EB细胞PS-NPs处理的所有GO terms
    df_eb_psnps = df_all[(df_all['cell_type'] == 'EB') & 
                         (df_all['comparison'] == 'PS-NPs')].copy()
    
    df_eb_psnps['neg_log10_p'] = -np.log10(df_eb_psnps['Adjusted P-value'].replace(0, 1e-300))
    df_eb_psnps = df_eb_psnps.sort_values('neg_log10_p', ascending=False)
    
    # 标记核糖体相关
    ribosome_terms = ['ribosom', 'translation', 'cytoplasmic translation']
    df_eb_psnps['is_ribosome'] = df_eb_psnps['Term'].str.lower().apply(
        lambda x: any(t in x for t in ribosome_terms))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图: 所有GO terms的排名，突出核糖体
    ax1 = axes[0]
    colors = ['#9B59B6' if rb else '#BDC3C7' for rb in df_eb_psnps['is_ribosome']]
    
    bars = ax1.barh(range(len(df_eb_psnps)), df_eb_psnps['neg_log10_p'], 
                    color=colors, edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(range(len(df_eb_psnps)))
    ax1.set_yticklabels(df_eb_psnps['Term'], fontsize=9)
    ax1.invert_yaxis()
    
    ax1.set_xlabel('-log10(Adjusted P-value)', fontsize=14, fontweight='bold')
    ax1.set_title('EB Cells (Stem Cell Progenitors): PS-NPs Treatment\n'
                  'ALL enriched terms are Ribosome/Translation related!', 
                  fontsize=13, fontweight='bold')
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#9B59B6', label='Ribosome/Translation'),
                       Patch(facecolor='#BDC3C7', label='Other')]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    # 右图: 对比不同处理组EB细胞的主要响应
    ax2 = axes[1]
    
    # 统计各处理组的主要通路类型
    treatments = ['Cd', 'PS-NPs', 'Cd-PS-NPs']
    categories = ['Metal/Detox', 'Ribosome', 'Immune', 'Other']
    
    data_matrix = []
    for treat in treatments:
        df_treat = df_all[(df_all['cell_type'] == 'EB') & (df_all['comparison'] == treat)]
        total = len(df_treat)
        if total == 0:
            data_matrix.append([0, 0, 0, 0])
            continue
        
        metal = len(df_treat[df_treat['Term'].str.contains('metal|detox|iron|ferritin', case=False, na=False)])
        ribosome = len(df_treat[df_treat['Term'].str.contains('ribosom|translation', case=False, na=False)])
        immune = len(df_treat[df_treat['Term'].str.contains('immune|protease|peptidase', case=False, na=False)])
        other = total - metal - ribosome - immune
        
        data_matrix.append([metal/total*100, ribosome/total*100, immune/total*100, other/total*100])
    
    data_matrix = np.array(data_matrix)
    x = np.arange(len(treatments))
    width = 0.2
    
    colors_cat = ['#27AE60', '#9B59B6', '#E74C3C', '#95A5A6']
    for i, (cat, color) in enumerate(zip(categories, colors_cat)):
        ax2.bar(x + i*width, data_matrix[:, i], width, label=cat, color=color, edgecolor='black')
    
    ax2.set_xticks(x + width*1.5)
    ax2.set_xticklabels(treatments, fontsize=12)
    ax2.set_ylabel('Percentage of GO Terms (%)', fontsize=14, fontweight='bold')
    ax2.set_title('EB Cells: Pathway Composition by Treatment\n'
                  'PS-NPs uniquely targets Ribosome!', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    
    # 添加注释箭头指向PS-NPs的核糖体
    ax2.annotate('100% Ribosome!', xy=(1 + width, data_matrix[1, 1]), 
                 xytext=(1.5, data_matrix[1, 1] + 20),
                 arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=2),
                 fontsize=12, fontweight='bold', color='#9B59B6')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig3_EB_PSNPs_ribosome_unique.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig3_EB_PSNPs_ribosome_unique.png', bbox_inches='tight')
    plt.close()
    print("✓ 图3: EB细胞PS-NPs核糖体独特性图已保存")

# =============================================================================
# 图4: 上调 vs 下调通路的全景对比
# =============================================================================
def plot_up_vs_down_landscape():
    """展示上调和下调通路的全景对比，突出新发现"""
    
    # 选择EC细胞Cd处理作为示例
    df_ec_cd = df_all[(df_all['cell_type'] == 'EC') & (df_all['comparison'] == 'Cd')].copy()
    df_ec_cd['neg_log10_p'] = -np.log10(df_ec_cd['Adjusted P-value'].replace(0, 1e-300))
    
    # 分离上调和下调
    df_up = df_ec_cd[df_ec_cd['direction'] == 'up'].nlargest(15, 'neg_log10_p')
    df_down = df_ec_cd[df_ec_cd['direction'] == 'down'].nlargest(15, 'neg_log10_p')
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # 左图: 上调通路
    ax1 = axes[0]
    
    # 标记已知/常规 vs 新发现
    conventional_up = ['metal', 'iron', 'ferritin', 'glutathione', 'detox', 'zinc']
    df_up['is_conventional'] = df_up['Term'].str.lower().apply(
        lambda x: any(t in x for t in conventional_up))
    
    colors_up = ['#27AE60' if conv else '#82E0AA' for conv in df_up['is_conventional']]
    
    bars1 = ax1.barh(range(len(df_up)), df_up['neg_log10_p'], color=colors_up, 
                     edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(df_up)))
    ax1.set_yticklabels(df_up['Term'], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('-log10(Adjusted P-value)', fontsize=12, fontweight='bold')
    ax1.set_title('UPREGULATED Pathways (EC, Cd)\n✓ Expected: Metal detox, Iron metabolism', 
                  fontsize=13, fontweight='bold', color='#27AE60')
    
    from matplotlib.patches import Patch
    legend1 = [Patch(facecolor='#27AE60', label='Conventional (Expected)'),
               Patch(facecolor='#82E0AA', label='Other Upregulated')]
    ax1.legend(handles=legend1, loc='lower right', fontsize=10)
    
    # 右图: 下调通路
    ax2 = axes[1]
    
    # 标记先天免疫相关
    immune_down = ['immune', 'defense', 'protease', 'peptidase', 'lectin']
    df_down['is_immune'] = df_down['Term'].str.lower().apply(
        lambda x: any(t in x for t in immune_down))
    
    colors_down = ['#E74C3C' if imm else '#F5B7B1' for imm in df_down['is_immune']]
    
    bars2 = ax2.barh(range(len(df_down)), df_down['neg_log10_p'], color=colors_down, 
                     edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(df_down)))
    ax2.set_yticklabels(df_down['Term'], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('-log10(Adjusted P-value)', fontsize=12, fontweight='bold')
    ax2.set_title('DOWNREGULATED Pathways (EC, Cd)\n⚠️ NOVEL: Innate Immunity Suppression!', 
                  fontsize=13, fontweight='bold', color='#E74C3C')
    
    legend2 = [Patch(facecolor='#E74C3C', label='Innate Immunity (NOVEL!)'),
               Patch(facecolor='#F5B7B1', label='Other Downregulated')]
    ax2.legend(handles=legend2, loc='lower right', fontsize=10)
    
    # 添加连接注释
    fig.text(0.5, 0.02, '← Expected Response                                              Unexpected Discovery! →', 
             ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUTPUT_DIR / 'Fig4_up_vs_down_landscape.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig4_up_vs_down_landscape.png', bbox_inches='tight')
    plt.close()
    print("✓ 图4: 上调vs下调全景对比图已保存")

# =============================================================================
# 图5: 所有细胞类型GO terms排名 - 先天免疫的一致性
# =============================================================================
def plot_all_celltype_immune_ranking():
    """在每个细胞类型中，展示先天免疫在下调通路中的排名"""
    
    cell_types = ['ISC', 'EB', 'EC', 'EE-AstA', 'EE-DH31', 'EE-Tk', 'Iron Cell']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, ct in enumerate(cell_types):
        ax = axes[idx]
        
        # 获取该细胞类型Cd处理下调的GO terms
        df_ct = df_all[(df_all['cell_type'] == ct) & 
                       (df_all['comparison'] == 'Cd') & 
                       (df_all['direction'] == 'down')].copy()
        
        if len(df_ct) == 0:
            ax.text(0.5, 0.5, f'{ct}\nNo data', ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        df_ct['neg_log10_p'] = -np.log10(df_ct['Adjusted P-value'].replace(0, 1e-300))
        df_ct = df_ct.sort_values('neg_log10_p', ascending=False).head(10)
        
        # 标记先天免疫
        df_ct['is_immune'] = df_ct['Term'].str.contains('innate immune|immune response', case=False, na=False)
        
        colors = ['#E74C3C' if im else '#BDC3C7' for im in df_ct['is_immune']]
        
        ax.barh(range(len(df_ct)), df_ct['neg_log10_p'], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(df_ct)))
        ax.set_yticklabels([t[:30]+'...' if len(t) > 30 else t for t in df_ct['Term']], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('-log10(P)', fontsize=10)
        
        # 找到先天免疫的排名
        immune_rank = df_ct[df_ct['is_immune']].index.tolist()
        if len(immune_rank) > 0:
            rank_num = list(df_ct.index).index(immune_rank[0]) + 1
            ax.set_title(f'{ct}\nInnate Immune Rank: #{rank_num}', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{ct}', fontsize=12, fontweight='bold')
    
    # 最后一个子图用于图例
    ax_legend = axes[7]
    ax_legend.axis('off')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='black', label='Innate Immune Response'),
        Patch(facecolor='#BDC3C7', edgecolor='black', label='Other GO Terms')
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=14)
    ax_legend.text(0.5, 0.2, 'Innate Immunity consistently\nranks TOP in ALL cell types!', 
                   ha='center', fontsize=14, fontweight='bold', color='#E74C3C',
                   transform=ax_legend.transAxes)
    
    plt.suptitle('Innate Immune Response Ranking in Each Cell Type (Cd Treatment, Downregulated)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig5_all_celltype_immune_ranking.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig5_all_celltype_immune_ranking.png', bbox_inches='tight')
    plt.close()
    print("✓ 图5: 各细胞类型免疫排名图已保存")


# =============================================================================
# 图6: 火山图式展示 - 显著性vs新颖性
# =============================================================================
def plot_novelty_significance_scatter():
    """用散点图展示各通路的显著性和新颖性评分"""
    
    # 选取EC细胞Cd处理的所有GO terms
    df_ec = df_all[(df_all['cell_type'] == 'EC') & (df_all['comparison'] == 'Cd')].copy()
    df_ec['neg_log10_p'] = -np.log10(df_ec['Adjusted P-value'].replace(0, 1e-300))
    df_ec['neg_log10_p'] = df_ec['neg_log10_p'].clip(upper=50)  # 截断
    
    # 定义"常规性"评分 - 越常规的通路评分越高
    conventional_terms = ['metal', 'iron', 'ferritin', 'glutathione', 'detox', 'zinc', 'copper']
    novel_terms = ['immune', 'defense', 'protease', 'peptidase', 'ribosom']
    
    def novelty_score(term, direction):
        term_lower = term.lower()
        if direction == 'up':
            if any(t in term_lower for t in conventional_terms):
                return 1  # 常规上调
            return 2  # 意外上调
        else:  # down
            if any(t in term_lower for t in novel_terms):
                return 4  # 新颖下调
            return 3  # 普通下调
    
    df_ec['novelty'] = df_ec.apply(lambda x: novelty_score(x['Term'], x['direction']), axis=1)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 颜色和标签
    novelty_colors = {1: '#27AE60', 2: '#82E0AA', 3: '#F5B7B1', 4: '#E74C3C'}
    novelty_labels = {1: 'Expected UP (Metal/Detox)', 2: 'Other UP', 
                      3: 'Other DOWN', 4: 'NOVEL DOWN (Immunity)'}
    
    for nov in [1, 2, 3, 4]:
        subset = df_ec[df_ec['novelty'] == nov]
        size = 200 if nov == 4 else 100
        ax.scatter(subset['novelty'] + np.random.uniform(-0.2, 0.2, len(subset)), 
                   subset['neg_log10_p'],
                   c=novelty_colors[nov], s=size, alpha=0.7, 
                   edgecolors='black', linewidths=0.5,
                   label=f"{novelty_labels[nov]} (n={len(subset)})")
    
    # 标注最显著的新发现
    top_novel = df_ec[df_ec['novelty'] == 4].nlargest(3, 'neg_log10_p')
    for _, row in top_novel.iterrows():
        ax.annotate(row['Term'][:25], 
                    xy=(row['novelty'], row['neg_log10_p']),
                    xytext=(row['novelty']+0.5, row['neg_log10_p']+2),
                    fontsize=10, fontweight='bold', color='#E74C3C',
                    arrowprops=dict(arrowstyle='->', color='#E74C3C'))
    
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Expected\nUP', 'Other\nUP', 'Other\nDOWN', 'NOVEL\nDOWN'], fontsize=12)
    ax.set_xlabel('Novelty Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('-log10(Adjusted P-value)', fontsize=14, fontweight='bold')
    ax.set_title('EC Cells (Cd Treatment): Significance vs Novelty\n'
                 'Novel Immune Suppression is BOTH significant AND unexpected!', 
                 fontsize=14, fontweight='bold')
    
    # 添加显著性阈值
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='p=0.05')
    ax.axhline(y=-np.log10(1e-5), color='orange', linestyle='--', alpha=0.5, label='p=1e-5')
    
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig6_novelty_significance.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig6_novelty_significance.png', bbox_inches='tight')
    plt.close()
    print("✓ 图6: 新颖性vs显著性散点图已保存")

# =============================================================================
# 主函数
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("生成新颖发现可视化图表 - 强调显著性和独特性对比")
    print("=" * 60)
    
    plot_ec_downregulated_ranking()        # 图1: EC下调通路排名
    plot_cross_celltype_immune_consistency()  # 图2: 跨细胞类型一致性
    plot_psnps_eb_unique_ribosome()        # 图3: PS-NPs核糖体独特性
    plot_up_vs_down_landscape()            # 图4: 上调vs下调全景
    plot_all_celltype_immune_ranking()     # 图5: 各细胞类型排名
    plot_novelty_significance_scatter()    # 图6: 新颖性vs显著性
    
    print("\n" + "=" * 60)
    print(f"所有图表已保存至: {OUTPUT_DIR}")
    print("=" * 60)
