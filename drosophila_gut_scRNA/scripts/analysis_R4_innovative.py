#!/usr/bin/env python3
"""
R4区域创新性分析脚本
分析方向:
1. PS-NPs的"特洛伊木马"效应 - Cd-PS-NPs vs Cd的独特差异
2. 肠道干细胞龛微环境 - VM与肠上皮互作、干细胞增殖分化信号
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置绘图参数
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sc.settings.verbosity = 2

# ============================================================================
# 配置
# ============================================================================
R4_DIR = Path("results/R4_analysis")
OUTPUT_DIR = R4_DIR / "innovative_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 果蝇肠道相关基因集定义
# ============================================================================

# 干细胞与分化相关
ISC_MARKERS = ['esg', 'Dl', 'N', 'Su(H)', 'klu', 'Ras85D', 'Stat92E']
EB_MARKERS = ['esg', 'klu', 'E(spl)m3-HLH', 'E(spl)mbeta-HLH']
EC_DIFFERENTIATION = ['Pdp1', 'nub', 'Myo31DF', 'vri']

# 干细胞龛信号通路
WNT_PATHWAY = ['wg', 'Wnt4', 'Wnt2', 'arm', 'dsh', 'fz', 'fz2', 'arr', 'Axn', 'sgg', 'pan']
JAK_STAT_PATHWAY = ['upd', 'upd2', 'upd3', 'dome', 'hop', 'Stat92E', 'Socs36E']
EGFR_PATHWAY = ['vn', 'spi', 'Egfr', 'aos', 'pnt', 'yan', 'rho', 'S']
HIPPO_PATHWAY = ['yki', 'sd', 'wts', 'hpo', 'sav', 'mats', 'ex', 'mer', 'kibra']
NOTCH_PATHWAY = ['Dl', 'N', 'Su(H)', 'E(spl)m3-HLH', 'E(spl)mbeta-HLH', 'kuz', 'mam']
BMP_DPP_PATHWAY = ['dpp', 'gbb', 'tkv', 'put', 'Mad', 'Med', 'brk', 'Dad']
INSULIN_PATHWAY = ['InR', 'chico', 'Pi3K92E', 'Akt1', 'foxo', 'Thor', 'dilp2', 'dilp3', 'dilp5']

# 细胞连接与屏障
SEPTATE_JUNCTION = ['Nrx-IV', 'cont', 'Nrg', 'cora', 'sinu', 'Lac', 'Gli', 'Mcr']
ADHERENS_JUNCTION = ['shg', 'arm', 'alpha-Cat', 'p120ctn', 'Vinc']
CYTOSKELETON = ['Act5C', 'Act42A', 'Act57B', 'Myo61F', 'Myo31DF', 'zip', 'sqh']

# 内吞与囊泡运输 (可能与纳米颗粒摄取相关)
ENDOCYTOSIS = ['Rab5', 'Rab7', 'Rab11', 'Hrs', 'Stam', 'Vps25', 'Vps28', 'ESCRT-III']
AUTOPHAGY = ['Atg1', 'Atg6', 'Atg8a', 'Atg8b', 'Atg12', 'Atg17', 'ref(2)P', 'blue']
LYSOSOME = ['Lamp1', 'Vha16-1', 'Vha68-2', 'cathD', 'Cp1']

# 应激与损伤响应
ER_STRESS_UPR = ['Xbp1', 'PEK', 'Ire1', 'Hsc70-3', 'BiP', 'crc', 'ATF4']
OXIDATIVE_STRESS = ['Sod1', 'Sod2', 'Cat', 'Prx5', 'GstD1', 'GstE1', 'Trx-2', 'TrxR-1', 'Keap1', 'cnc']
DNA_DAMAGE = ['p53', 'tefu', 'mei-41', 'grp', 'lok', 'mus304', 'Rad51']
APOPTOSIS = ['rpr', 'hid', 'grim', 'Dcp-1', 'Drice', 'Dark', 'Debcl', 'Buffy']

# 代谢相关
MITOCHONDRIA = ['CoVa', 'CoVb', 'ATPsynB', 'ATPsynC', 'blw', 'sesB', 'Cyt-c-p']
GLYCOLYSIS = ['Hex-A', 'Pfk', 'Ald', 'Gapdh1', 'Pgk', 'Pyk', 'Ldh']
LIPID_METABOLISM = ['bmm', 'Lsd-1', 'Lsd-2', 'FASN1', 'ACC', 'mdy', 'Dgat2']

# VM (内脏肌肉) 相关
VM_MARKERS = ['Mhc', 'Mlc2', 'up', 'bt', 'Tm1', 'Act57B', 'sls']
VM_SIGNALING = ['vn', 'wg', 'dpp', 'hh']  # VM分泌的信号分子

# ============================================================================
# 辅助函数
# ============================================================================

def get_gene_name_map(adata):
    """获取基因ID到基因名的映射"""
    return dict(zip(adata.var_names, adata.var['gene_name']))

def find_genes_in_adata(adata, gene_symbols, gene_name_map=None):
    """在adata中查找基因（支持symbol和ID）"""
    if gene_name_map is None:
        gene_name_map = get_gene_name_map(adata)
    
    # 反向映射：symbol -> ID
    symbol_to_id = {v: k for k, v in gene_name_map.items()}
    
    found_genes = []
    for gene in gene_symbols:
        if gene in adata.var_names:
            found_genes.append(gene)
        elif gene in symbol_to_id and symbol_to_id[gene] in adata.var_names:
            found_genes.append(symbol_to_id[gene])
    
    return found_genes

def calculate_pathway_score(adata, genes, gene_name_map=None):
    """计算通路得分（平均表达）"""
    found_genes = find_genes_in_adata(adata, genes, gene_name_map)
    if len(found_genes) == 0:
        return None
    
    # 计算平均表达
    gene_indices = [list(adata.var_names).index(g) for g in found_genes]
    scores = np.array(adata.X[:, gene_indices].mean(axis=1)).flatten()
    return scores, found_genes

def compare_groups_pathway(adata, pathway_genes, group_col='group', 
                           groups_to_compare=None, gene_name_map=None):
    """比较不同组之间的通路活性"""
    result = calculate_pathway_score(adata, pathway_genes, gene_name_map)
    if result is None:
        return None
    
    scores, found_genes = result
    adata.obs['_pathway_score'] = scores
    
    if groups_to_compare is None:
        groups_to_compare = adata.obs[group_col].unique()
    
    group_scores = {}
    for group in groups_to_compare:
        mask = adata.obs[group_col] == group
        group_scores[group] = adata.obs.loc[mask, '_pathway_score'].values
    
    return group_scores, found_genes


# ============================================================================
# 分析1: PS-NPs "特洛伊木马"效应
# ============================================================================

def analyze_trojan_horse_effect(adata_r4):
    """
    分析Cd-PS-NPs相比单独Cd的独特效应
    核心问题：纳米塑料是否增强了Cd的毒性？通过什么机制？
    """
    print("\n" + "="*70)
    print("分析1: PS-NPs '特洛伊木马'效应")
    print("="*70)
    
    output_dir = OUTPUT_DIR / "trojan_horse"
    output_dir.mkdir(exist_ok=True)
    
    gene_name_map = get_gene_name_map(adata_r4)
    
    # 1.1 直接比较 Cd-PS-NPs vs Cd 的差异基因
    print("\n1.1 Cd-PS-NPs vs Cd 差异表达分析...")
    
    # 提取这两组
    mask = adata_r4.obs['group'].isin(['Cd', 'Cd-PS-NPs'])
    adata_compare = adata_r4[mask].copy()
    adata_compare.obs_names_make_unique()
    
    # 差异表达
    sc.tl.rank_genes_groups(adata_compare, groupby='group', 
                            groups=['Cd-PS-NPs'], reference='Cd',
                            method='wilcoxon')
    
    de_results = sc.get.rank_genes_groups_df(adata_compare, group='Cd-PS-NPs')
    de_results['gene_name'] = de_results['names'].map(gene_name_map)
    de_results.to_csv(output_dir / 'Cd-PS-NPs_vs_Cd_DE.csv', index=False)
    
    # 筛选显著差异基因
    sig_up = de_results[(de_results['pvals_adj'] < 0.05) & (de_results['logfoldchanges'] > 0.5)]
    sig_down = de_results[(de_results['pvals_adj'] < 0.05) & (de_results['logfoldchanges'] < -0.5)]
    
    print(f"  Cd-PS-NPs相比Cd上调基因: {len(sig_up)}")
    print(f"  Cd-PS-NPs相比Cd下调基因: {len(sig_down)}")
    
    # 1.2 分析内吞/囊泡运输通路 - 纳米颗粒摄取机制
    print("\n1.2 内吞与囊泡运输通路分析...")
    
    pathways_to_analyze = {
        'Endocytosis': ENDOCYTOSIS,
        'Autophagy': AUTOPHAGY,
        'Lysosome': LYSOSOME,
        'ER_Stress': ER_STRESS_UPR,
    }
    
    pathway_results = []
    
    for pathway_name, genes in pathways_to_analyze.items():
        result = compare_groups_pathway(adata_r4, genes, gene_name_map=gene_name_map)
        if result is None:
            continue
        
        group_scores, found_genes = result
        
        for group, scores in group_scores.items():
            pathway_results.append({
                'pathway': pathway_name,
                'group': group,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'n_cells': len(scores),
                'genes_found': len(found_genes)
            })
        
        # 统计检验: Cd-PS-NPs vs Cd
        if 'Cd-PS-NPs' in group_scores and 'Cd' in group_scores:
            stat, pval = stats.mannwhitneyu(group_scores['Cd-PS-NPs'], group_scores['Cd'])
            effect = np.mean(group_scores['Cd-PS-NPs']) - np.mean(group_scores['Cd'])
            print(f"  {pathway_name}: Cd-PS-NPs vs Cd, effect={effect:.3f}, p={pval:.2e}")
    
    pathway_df = pd.DataFrame(pathway_results)
    pathway_df.to_csv(output_dir / 'pathway_scores_by_group.csv', index=False)
    
    # 绘制通路得分热图
    if len(pathway_df) > 0:
        pivot = pathway_df.pivot(index='pathway', columns='group', values='mean_score')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, cmap='RdYlBu_r', center=pivot.values.mean(), 
                    annot=True, fmt='.3f', ax=ax)
        ax.set_title('Pathway Activity Scores by Treatment Group')
        plt.tight_layout()
        plt.savefig(output_dir / 'pathway_heatmap.png', bbox_inches='tight')
        plt.close()
    
    # 1.3 分析"协同毒性"效应
    print("\n1.3 协同毒性效应分析...")
    
    # 计算各组相对于Control的变化
    # 如果 Cd-PS-NPs效应 > Cd效应 + PS-NPs效应，则存在协同作用
    
    synergy_results = analyze_synergy_effect(adata_r4, gene_name_map, output_dir)
    
    # 1.4 特异性上调基因的功能分析
    print("\n1.4 Cd-PS-NPs特异性上调基因分析...")
    
    # 找到只在Cd-PS-NPs中上调，但在单独Cd或PS-NPs中不上调的基因
    specific_genes = find_cdpsnps_specific_genes(adata_r4, gene_name_map, output_dir)
    
    # 1.5 绘制关键发现
    plot_trojan_horse_summary(adata_r4, de_results, pathway_df, output_dir, gene_name_map)
    
    return de_results, pathway_df, synergy_results


def analyze_synergy_effect(adata, gene_name_map, output_dir):
    """分析Cd和PS-NPs的协同效应"""
    
    # 对每个基因计算：
    # Control作为baseline
    # Cd效应 = Cd - Control
    # PS-NPs效应 = PS-NPs - Control  
    # Cd-PS-NPs效应 = Cd-PS-NPs - Control
    # 协同指数 = Cd-PS-NPs效应 / (Cd效应 + PS-NPs效应)
    
    groups = ['Control', 'Cd', 'PS-NPs', 'Cd-PS-NPs']
    
    # 计算每个基因在每组的平均表达
    gene_means = {}
    for group in groups:
        mask = adata.obs['group'] == group
        if mask.sum() > 0:
            gene_means[group] = np.array(adata[mask].X.mean(axis=0)).flatten()
    
    if not all(g in gene_means for g in groups):
        return None
    
    # 计算效应
    cd_effect = gene_means['Cd'] - gene_means['Control']
    psnps_effect = gene_means['PS-NPs'] - gene_means['Control']
    cdpsnps_effect = gene_means['Cd-PS-NPs'] - gene_means['Control']
    
    # 计算协同指数（避免除零）
    additive_effect = cd_effect + psnps_effect
    synergy_index = np.zeros_like(cdpsnps_effect)
    nonzero_mask = np.abs(additive_effect) > 0.01
    synergy_index[nonzero_mask] = cdpsnps_effect[nonzero_mask] / additive_effect[nonzero_mask]
    
    # 创建结果DataFrame
    synergy_df = pd.DataFrame({
        'gene_id': adata.var_names,
        'gene_name': [gene_name_map.get(g, g) for g in adata.var_names],
        'control_expr': gene_means['Control'],
        'cd_effect': cd_effect,
        'psnps_effect': psnps_effect,
        'cdpsnps_effect': cdpsnps_effect,
        'additive_expected': additive_effect,
        'synergy_index': synergy_index
    })
    
    # 筛选有意义的协同基因
    # 协同: synergy_index > 1.5 且 cdpsnps_effect > 0.5
    # 拮抗: synergy_index < 0.5 且 additive_effect > 0.5
    
    synergistic = synergy_df[
        (synergy_df['synergy_index'] > 1.5) & 
        (synergy_df['cdpsnps_effect'] > 0.5) &
        (synergy_df['additive_expected'] > 0.1)
    ].sort_values('synergy_index', ascending=False)
    
    antagonistic = synergy_df[
        (synergy_df['synergy_index'] < 0.5) & 
        (synergy_df['synergy_index'] > 0) &
        (synergy_df['additive_expected'] > 0.5)
    ].sort_values('synergy_index')
    
    print(f"  协同上调基因 (synergy > 1.5): {len(synergistic)}")
    print(f"  拮抗基因 (synergy < 0.5): {len(antagonistic)}")
    
    if len(synergistic) > 0:
        print("\n  Top 10 协同上调基因:")
        for _, row in synergistic.head(10).iterrows():
            print(f"    {row['gene_name']}: synergy={row['synergy_index']:.2f}, "
                  f"Cd-PS-NPs effect={row['cdpsnps_effect']:.2f}")
    
    synergy_df.to_csv(output_dir / 'synergy_analysis_all_genes.csv', index=False)
    synergistic.to_csv(output_dir / 'synergistic_genes.csv', index=False)
    antagonistic.to_csv(output_dir / 'antagonistic_genes.csv', index=False)
    
    # 绘制协同效应散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 只绘制有效应的基因
    plot_mask = (np.abs(synergy_df['additive_expected']) > 0.1) | (np.abs(synergy_df['cdpsnps_effect']) > 0.1)
    plot_df = synergy_df[plot_mask]
    
    ax.scatter(plot_df['additive_expected'], plot_df['cdpsnps_effect'], 
               alpha=0.3, s=5, c='gray')
    
    # 标记协同基因
    if len(synergistic) > 0:
        ax.scatter(synergistic['additive_expected'], synergistic['cdpsnps_effect'],
                   alpha=0.8, s=20, c='red', label=f'Synergistic ({len(synergistic)})')
    
    # 添加对角线（加性效应线）
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Additive (y=x)')
    ax.plot(lims, [1.5*x for x in lims], 'r--', alpha=0.5, label='Synergy threshold (y=1.5x)')
    
    ax.set_xlabel('Expected Additive Effect (Cd + PS-NPs)')
    ax.set_ylabel('Observed Cd-PS-NPs Effect')
    ax.set_title('Synergy Analysis: Cd-PS-NPs vs Expected Additive Effect')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'synergy_scatter.png', bbox_inches='tight')
    plt.close()
    
    return synergy_df


def find_cdpsnps_specific_genes(adata, gene_name_map, output_dir):
    """找到Cd-PS-NPs特异性响应基因"""
    
    # 对每组vs Control做差异分析
    de_results = {}
    
    for group in ['Cd', 'PS-NPs', 'Cd-PS-NPs']:
        mask = adata.obs['group'].isin([group, 'Control'])
        adata_sub = adata[mask].copy()
        adata_sub.obs_names_make_unique()
        
        sc.tl.rank_genes_groups(adata_sub, groupby='group',
                                groups=[group], reference='Control',
                                method='wilcoxon')
        de_results[group] = sc.get.rank_genes_groups_df(adata_sub, group=group)
    
    # 找Cd-PS-NPs特异性上调基因
    # 条件: 在Cd-PS-NPs中显著上调，但在Cd和PS-NPs中不显著上调
    
    cdpsnps_up = set(de_results['Cd-PS-NPs'][
        (de_results['Cd-PS-NPs']['pvals_adj'] < 0.05) & 
        (de_results['Cd-PS-NPs']['logfoldchanges'] > 0.5)
    ]['names'])
    
    cd_up = set(de_results['Cd'][
        (de_results['Cd']['pvals_adj'] < 0.05) & 
        (de_results['Cd']['logfoldchanges'] > 0.3)
    ]['names'])
    
    psnps_up = set(de_results['PS-NPs'][
        (de_results['PS-NPs']['pvals_adj'] < 0.05) & 
        (de_results['PS-NPs']['logfoldchanges'] > 0.3)
    ]['names'])
    
    # Cd-PS-NPs特异性 = 在Cd-PS-NPs上调，但不在Cd或PS-NPs中上调
    specific_up = cdpsnps_up - cd_up - psnps_up
    
    print(f"  Cd-PS-NPs特异性上调基因: {len(specific_up)}")
    
    # 获取这些基因的详细信息
    specific_df = de_results['Cd-PS-NPs'][de_results['Cd-PS-NPs']['names'].isin(specific_up)].copy()
    specific_df['gene_name'] = specific_df['names'].map(gene_name_map)
    specific_df = specific_df.sort_values('logfoldchanges', ascending=False)
    
    specific_df.to_csv(output_dir / 'cdpsnps_specific_genes.csv', index=False)
    
    if len(specific_df) > 0:
        print("\n  Top 15 Cd-PS-NPs特异性上调基因:")
        for _, row in specific_df.head(15).iterrows():
            print(f"    {row['gene_name']}: logFC={row['logfoldchanges']:.2f}")
    
    # 绘制Venn图概念
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 简单条形图展示
    categories = ['Cd only', 'PS-NPs only', 'Cd-PS-NPs only', 
                  'Cd ∩ Cd-PS-NPs', 'PS-NPs ∩ Cd-PS-NPs', 'All three']
    
    cd_only = cd_up - psnps_up - cdpsnps_up
    psnps_only = psnps_up - cd_up - cdpsnps_up
    cdpsnps_only = specific_up
    cd_cdpsnps = (cd_up & cdpsnps_up) - psnps_up
    psnps_cdpsnps = (psnps_up & cdpsnps_up) - cd_up
    all_three = cd_up & psnps_up & cdpsnps_up
    
    counts = [len(cd_only), len(psnps_only), len(cdpsnps_only),
              len(cd_cdpsnps), len(psnps_cdpsnps), len(all_three)]
    
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71', '#f39c12', '#1abc9c']
    bars = ax.bar(categories, counts, color=colors)
    ax.set_ylabel('Number of Upregulated Genes')
    ax.set_title('Overlap of Upregulated Genes Across Treatment Groups')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gene_overlap_barplot.png', bbox_inches='tight')
    plt.close()
    
    return specific_df


def plot_trojan_horse_summary(adata, de_results, pathway_df, output_dir, gene_name_map):
    """绘制特洛伊木马效应总结图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 火山图: Cd-PS-NPs vs Cd
    ax = axes[0, 0]
    de_results['neg_log_pval'] = -np.log10(de_results['pvals_adj'] + 1e-300)
    
    # 分类
    de_results['category'] = 'NS'
    de_results.loc[(de_results['logfoldchanges'] > 0.5) & (de_results['pvals_adj'] < 0.05), 'category'] = 'Up'
    de_results.loc[(de_results['logfoldchanges'] < -0.5) & (de_results['pvals_adj'] < 0.05), 'category'] = 'Down'
    
    colors = {'Up': '#e74c3c', 'Down': '#3498db', 'NS': '#95a5a6'}
    for cat, color in colors.items():
        mask = de_results['category'] == cat
        ax.scatter(de_results.loc[mask, 'logfoldchanges'],
                   de_results.loc[mask, 'neg_log_pval'],
                   c=color, alpha=0.5, s=10, label=f'{cat} ({mask.sum()})')
    
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Log2 Fold Change (Cd-PS-NPs vs Cd)')
    ax.set_ylabel('-Log10 Adjusted P-value')
    ax.set_title('Differential Expression: Cd-PS-NPs vs Cd')
    ax.legend()
    
    # 2. 通路活性比较
    ax = axes[0, 1]
    if len(pathway_df) > 0:
        pivot = pathway_df.pivot(index='pathway', columns='group', values='mean_score')
        # 只保留关键组
        cols_to_plot = [c for c in ['Control', 'Cd', 'PS-NPs', 'Cd-PS-NPs'] if c in pivot.columns]
        if len(cols_to_plot) > 0:
            pivot[cols_to_plot].plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Pathway Activity by Treatment')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Group', bbox_to_anchor=(1.02, 1))
    
    # 3. 细胞比例变化
    ax = axes[1, 0]
    cell_counts = adata.obs.groupby(['group', 'final_cell_type']).size().unstack(fill_value=0)
    cell_props = cell_counts.div(cell_counts.sum(axis=1), axis=0) * 100
    
    # 只显示主要细胞类型
    main_types = cell_props.sum().nlargest(5).index
    cell_props[main_types].plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Cell Type Composition by Treatment')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Cell Type', bbox_to_anchor=(1.02, 1))
    
    # 4. 关键基因表达热图
    ax = axes[1, 1]
    
    # 选择一些关键基因
    key_pathways = {
        'Endocytosis': ['Rab5', 'Rab7', 'Hrs'],
        'Autophagy': ['Atg8a', 'ref(2)P'],
        'ER_Stress': ['Xbp1', 'BiP'],
    }
    
    key_genes = []
    for genes in key_pathways.values():
        key_genes.extend(find_genes_in_adata(adata, genes, gene_name_map))
    
    if len(key_genes) > 0:
        # 计算每组平均表达
        expr_data = []
        for group in ['Control', 'Cd', 'PS-NPs', 'Cd-PS-NPs']:
            mask = adata.obs['group'] == group
            if mask.sum() > 0:
                for gene in key_genes:
                    gene_idx = list(adata.var_names).index(gene)
                    mean_expr = float(adata[mask].X[:, gene_idx].mean())
                    expr_data.append({
                        'group': group,
                        'gene': gene_name_map.get(gene, gene),
                        'expression': mean_expr
                    })
        
        if expr_data:
            try:
                expr_df = pd.DataFrame(expr_data)
                pivot = expr_df.pivot(index='gene', columns='group', values='expression')
                pivot = pivot.astype(float)
                sns.heatmap(pivot, cmap='YlOrRd', ax=ax, annot=True, fmt='.2f')
                ax.set_title('Key Pathway Genes Expression')
            except Exception as e:
                ax.text(0.5, 0.5, f'Plot error: {str(e)[:50]}', ha='center', va='center')
                ax.set_title('Key Pathway Genes (error)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trojan_horse_summary.png', bbox_inches='tight')
    plt.close()


# ============================================================================
# 分析2: 肠道干细胞龛微环境
# ============================================================================

def analyze_stem_cell_niche(adata_r4):
    """
    分析肠道干细胞龛微环境的变化
    重点：VM与肠上皮的互作、干细胞增殖分化信号
    """
    print("\n" + "="*70)
    print("分析2: 肠道干细胞龛微环境")
    print("="*70)
    
    output_dir = OUTPUT_DIR / "stem_cell_niche"
    output_dir.mkdir(exist_ok=True)
    
    gene_name_map = get_gene_name_map(adata_r4)
    
    # 2.1 干细胞相关信号通路分析
    print("\n2.1 干细胞信号通路分析...")
    
    signaling_pathways = {
        'JAK-STAT': JAK_STAT_PATHWAY,
        'EGFR': EGFR_PATHWAY,
        'Wnt': WNT_PATHWAY,
        'Hippo': HIPPO_PATHWAY,
        'Notch': NOTCH_PATHWAY,
        'BMP/Dpp': BMP_DPP_PATHWAY,
        'Insulin': INSULIN_PATHWAY,
    }
    
    pathway_scores = []
    
    for pathway_name, genes in signaling_pathways.items():
        result = compare_groups_pathway(adata_r4, genes, gene_name_map=gene_name_map)
        if result is None:
            continue
        
        group_scores, found_genes = result
        
        print(f"\n  {pathway_name} (found {len(found_genes)} genes):")
        
        for group in ['Control', 'Cd', 'PS-NPs', 'Cd-PS-NPs']:
            if group in group_scores:
                mean_score = np.mean(group_scores[group])
                pathway_scores.append({
                    'pathway': pathway_name,
                    'group': group,
                    'mean_score': mean_score,
                    'n_genes': len(found_genes)
                })
                print(f"    {group}: {mean_score:.3f}")
    
    pathway_df = pd.DataFrame(pathway_scores)
    pathway_df.to_csv(output_dir / 'signaling_pathway_scores.csv', index=False)
    
    # 2.2 VM细胞分析
    print("\n2.2 VM (内脏肌肉) 细胞分析...")
    
    vm_mask = adata_r4.obs['final_cell_type'] == 'VM'
    print(f"  VM细胞数: {vm_mask.sum()}")
    
    if vm_mask.sum() > 50:
        adata_vm = adata_r4[vm_mask].copy()
        adata_vm.obs_names_make_unique()
        
        # VM分泌的信号分子分析
        vm_signals = ['vn', 'wg', 'dpp', 'hh', 'upd', 'upd2', 'upd3']
        
        vm_signal_expr = []
        for gene in vm_signals:
            found = find_genes_in_adata(adata_vm, [gene], gene_name_map)
            if found:
                gene_id = found[0]
                gene_idx = list(adata_vm.var_names).index(gene_id)
                
                for group in adata_vm.obs['group'].unique():
                    mask = adata_vm.obs['group'] == group
                    if mask.sum() > 0:
                        mean_expr = adata_vm[mask].X[:, gene_idx].mean()
                        vm_signal_expr.append({
                            'gene': gene,
                            'group': group,
                            'mean_expression': mean_expr
                        })
        
        if vm_signal_expr:
            vm_signal_df = pd.DataFrame(vm_signal_expr)
            vm_signal_df.to_csv(output_dir / 'vm_signaling_molecules.csv', index=False)
            
            # 绘制VM信号分子表达
            try:
                pivot = vm_signal_df.pivot(index='gene', columns='group', values='mean_expression')
                pivot = pivot.astype(float)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.2f', ax=ax)
                ax.set_title('VM-secreted Signaling Molecules')
                plt.tight_layout()
                plt.savefig(output_dir / 'vm_signaling_heatmap.png', bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  VM信号分子热图绑制失败: {e}")
    
    # 2.3 ISC/EB细胞分析
    print("\n2.3 ISC/EB (干细胞/祖细胞) 分析...")
    
    # 检查是否有ISC/EB细胞
    isc_eb_types = ['ISC', 'EB']
    isc_eb_mask = adata_r4.obs['final_cell_type'].isin(isc_eb_types)
    print(f"  ISC/EB细胞数: {isc_eb_mask.sum()}")
    
    # 即使ISC/EB数量少，也可以分析EC中的干细胞标记基因
    print("\n  分析EC中的干细胞相关基因表达...")
    
    ec_mask = adata_r4.obs['final_cell_type'] == 'EC'
    adata_ec = adata_r4[ec_mask].copy()
    adata_ec.obs_names_make_unique()
    
    stem_genes = ISC_MARKERS + EB_MARKERS + EC_DIFFERENTIATION
    stem_genes = list(set(stem_genes))  # 去重
    
    stem_expr = []
    for gene in stem_genes:
        found = find_genes_in_adata(adata_ec, [gene], gene_name_map)
        if found:
            gene_id = found[0]
            gene_idx = list(adata_ec.var_names).index(gene_id)
            
            for group in adata_ec.obs['group'].unique():
                mask = adata_ec.obs['group'] == group
                if mask.sum() > 0:
                    mean_expr = adata_ec[mask].X[:, gene_idx].mean()
                    pct_expr = (adata_ec[mask].X[:, gene_idx] > 0).mean() * 100
                    stem_expr.append({
                        'gene': gene,
                        'group': group,
                        'mean_expression': mean_expr,
                        'pct_expressing': pct_expr
                    })
    
    if stem_expr:
        stem_df = pd.DataFrame(stem_expr)
        stem_df.to_csv(output_dir / 'stem_cell_markers_in_EC.csv', index=False)
    
    # 2.4 细胞连接与屏障完整性
    print("\n2.4 细胞连接与屏障完整性分析...")
    
    junction_pathways = {
        'Septate_Junction': SEPTATE_JUNCTION,
        'Adherens_Junction': ADHERENS_JUNCTION,
        'Cytoskeleton': CYTOSKELETON,
    }
    
    junction_scores = []
    
    for pathway_name, genes in junction_pathways.items():
        result = compare_groups_pathway(adata_r4, genes, gene_name_map=gene_name_map)
        if result is None:
            continue
        
        group_scores, found_genes = result
        
        print(f"\n  {pathway_name} (found {len(found_genes)} genes):")
        
        for group in ['Control', 'Cd', 'PS-NPs', 'Cd-PS-NPs']:
            if group in group_scores:
                mean_score = np.mean(group_scores[group])
                junction_scores.append({
                    'pathway': pathway_name,
                    'group': group,
                    'mean_score': mean_score
                })
                print(f"    {group}: {mean_score:.3f}")
    
    junction_df = pd.DataFrame(junction_scores)
    junction_df.to_csv(output_dir / 'junction_pathway_scores.csv', index=False)
    
    # 2.5 绘制综合图
    plot_stem_cell_niche_summary(adata_r4, pathway_df, junction_df, output_dir, gene_name_map)
    
    return pathway_df, junction_df


def plot_stem_cell_niche_summary(adata, pathway_df, junction_df, output_dir, gene_name_map):
    """绘制干细胞龛分析总结图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 信号通路热图
    ax = axes[0, 0]
    if len(pathway_df) > 0:
        pivot = pathway_df.pivot(index='pathway', columns='group', values='mean_score')
        cols_order = [c for c in ['Control', 'Cd', 'PS-NPs', 'Cd-PS-NPs'] if c in pivot.columns]
        if cols_order:
            # 计算相对于Control的变化
            if 'Control' in pivot.columns:
                pivot_change = pivot[cols_order].subtract(pivot['Control'], axis=0)
                sns.heatmap(pivot_change, cmap='RdBu_r', center=0, annot=True, fmt='.2f', ax=ax)
                ax.set_title('Signaling Pathway Changes (vs Control)')
            else:
                sns.heatmap(pivot[cols_order], cmap='YlOrRd', annot=True, fmt='.2f', ax=ax)
                ax.set_title('Signaling Pathway Activity')
    
    # 2. 细胞连接通路
    ax = axes[0, 1]
    if len(junction_df) > 0:
        pivot = junction_df.pivot(index='pathway', columns='group', values='mean_score')
        cols_order = [c for c in ['Control', 'Cd', 'PS-NPs', 'Cd-PS-NPs'] if c in pivot.columns]
        if cols_order:
            if 'Control' in pivot.columns:
                pivot_change = pivot[cols_order].subtract(pivot['Control'], axis=0)
                sns.heatmap(pivot_change, cmap='RdBu_r', center=0, annot=True, fmt='.2f', ax=ax)
                ax.set_title('Cell Junction Changes (vs Control)')
            else:
                sns.heatmap(pivot[cols_order], cmap='YlOrRd', annot=True, fmt='.2f', ax=ax)
                ax.set_title('Cell Junction Activity')
    
    # 3. 细胞类型比例变化
    ax = axes[1, 0]
    
    # 计算各组细胞类型比例
    cell_counts = adata.obs.groupby(['group', 'final_cell_type']).size().unstack(fill_value=0)
    cell_props = cell_counts.div(cell_counts.sum(axis=1), axis=0) * 100
    
    # 计算相对于Control的变化
    if 'Control' in cell_props.index:
        prop_change = cell_props.subtract(cell_props.loc['Control'], axis=1)
        prop_change = prop_change.drop('Control', errors='ignore')
        
        # 只显示变化较大的细胞类型
        significant_types = prop_change.abs().max().nlargest(6).index
        prop_change[significant_types].T.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Cell Type Proportion Change (vs Control)')
        ax.set_ylabel('Percentage Point Change')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Treatment', bbox_to_anchor=(1.02, 1))
    
    # 4. 关键发现文字总结
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = """
    Key Findings Summary:
    
    1. Signaling Pathway Changes:
       - JAK-STAT: Stress response activation
       - EGFR: Regeneration signaling
       - Wnt: Stem cell maintenance
       - Hippo: Growth control
    
    2. Cell Junction Integrity:
       - Septate junctions: Barrier function
       - Adherens junctions: Cell adhesion
    
    3. Stem Cell Niche:
       - VM-derived signals
       - ISC/EB proliferation status
       - EC differentiation markers
    
    4. Treatment-specific Effects:
       - Cd: Direct metal toxicity
       - PS-NPs: Physical disruption
       - Cd-PS-NPs: Synergistic effects
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stem_cell_niche_summary.png', bbox_inches='tight')
    plt.close()


# ============================================================================
# 生成综合报告
# ============================================================================

def generate_innovative_report(trojan_results, niche_results, output_dir):
    """生成创新性分析报告"""
    
    report_path = output_dir / 'innovative_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("R4区域创新性分析报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("## 分析1: PS-NPs '特洛伊木马'效应\n\n")
        f.write("核心问题: 纳米塑料是否增强了Cd的毒性？通过什么机制？\n\n")
        
        if trojan_results:
            de_results, pathway_df, synergy_df = trojan_results
            
            f.write("### 1.1 Cd-PS-NPs vs Cd 差异表达\n")
            sig_up = de_results[(de_results['pvals_adj'] < 0.05) & (de_results['logfoldchanges'] > 0.5)]
            sig_down = de_results[(de_results['pvals_adj'] < 0.05) & (de_results['logfoldchanges'] < -0.5)]
            f.write(f"- 上调基因: {len(sig_up)}\n")
            f.write(f"- 下调基因: {len(sig_down)}\n\n")
            
            if synergy_df is not None:
                synergistic = synergy_df[synergy_df['synergy_index'] > 1.5]
                f.write("### 1.2 协同效应基因\n")
                f.write(f"- 协同上调基因数: {len(synergistic)}\n")
                if len(synergistic) > 0:
                    f.write("- Top 10 协同基因:\n")
                    for _, row in synergistic.head(10).iterrows():
                        f.write(f"  - {row['gene_name']}: synergy={row['synergy_index']:.2f}\n")
        
        f.write("\n\n## 分析2: 肠道干细胞龛微环境\n\n")
        f.write("核心问题: 金属暴露如何影响干细胞龛信号和肠道屏障？\n\n")
        
        if niche_results:
            pathway_df, junction_df = niche_results
            
            f.write("### 2.1 信号通路变化\n")
            if len(pathway_df) > 0:
                for pathway in pathway_df['pathway'].unique():
                    f.write(f"\n{pathway}:\n")
                    pw_data = pathway_df[pathway_df['pathway'] == pathway]
                    for _, row in pw_data.iterrows():
                        f.write(f"  - {row['group']}: {row['mean_score']:.3f}\n")
            
            f.write("\n### 2.2 细胞连接变化\n")
            if len(junction_df) > 0:
                for pathway in junction_df['pathway'].unique():
                    f.write(f"\n{pathway}:\n")
                    pw_data = junction_df[junction_df['pathway'] == pathway]
                    for _, row in pw_data.iterrows():
                        f.write(f"  - {row['group']}: {row['mean_score']:.3f}\n")
        
        f.write("\n\n" + "="*70 + "\n")
        f.write("分析完成\n")
        f.write("="*70 + "\n")
    
    print(f"\n报告已保存: {report_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*70)
    print("R4区域创新性分析")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    adata_r4 = sc.read_h5ad(R4_DIR / 'R4_cells.h5ad')
    print(f"R4细胞数: {adata_r4.n_obs}")
    print(f"处理组: {adata_r4.obs['group'].value_counts().to_dict()}")
    
    # 分析1: 特洛伊木马效应
    trojan_results = analyze_trojan_horse_effect(adata_r4)
    
    # 分析2: 干细胞龛微环境
    niche_results = analyze_stem_cell_niche(adata_r4)
    
    # 生成报告
    generate_innovative_report(trojan_results, niche_results, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("创新性分析完成!")
    print(f"结果保存在: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
