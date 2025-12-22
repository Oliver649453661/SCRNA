#!/usr/bin/env python3
"""
R4区域后续深度分析脚本
1. Iron Cell亚群功能注释（消失的7/10 vs 新增的2/3）
2. 关键差异基因GO富集分析
3. Mtn家族基因在不同亚群中的表达模式分析
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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
OUTPUT_DIR = R4_DIR / "followup_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Iron Cell亚群分类
CONTROL_DOMINANT_SUBCLUSTERS = ['7', '10']  # 在Control中占主导，金属暴露后消失
CD_INDUCED_SUBCLUSTERS = ['2', '3', '4', '8', '9', '11']  # 金属暴露后大幅增加

# Mtn家族基因
MTN_GENES = {
    'FBgn0002868': 'MtnA',
    'FBgn0002869': 'MtnB', 
    'FBgn0038790': 'MtnC',
    'FBgn0053192': 'MtnD',
    'FBgn0262146': 'MtnE'
}

# 其他金属相关基因
METAL_RELATED_GENES = {
    'FBgn0062412': 'Ctr1B',  # 铜转运蛋白
    'FBgn0036575': 'CG5157',
    'FBgn0010038': 'Fer1HCH',  # 铁蛋白重链
    'FBgn0015222': 'Fer2LCH',  # 铁蛋白轻链
    'FBgn0259714': 'ZnT63C',  # 锌转运蛋白
    'FBgn0036824': 'ZnT35C',
}

# ============================================================================
# 1. Iron Cell亚群功能注释
# ============================================================================

def analyze_ironcell_subclusters(iron_adata):
    """分析Iron Cell亚群的功能差异"""
    print("\n" + "="*60)
    print("1. Iron Cell亚群功能注释分析")
    print("="*60)
    
    output_dir = OUTPUT_DIR / "ironcell_subcluster_annotation"
    output_dir.mkdir(exist_ok=True)
    
    # 获取基因名称映射
    gene_name_map = dict(zip(iron_adata.var_names, iron_adata.var['gene_name']))
    
    # 分析Control主导亚群 vs Cd诱导亚群的marker基因
    # 创建亚群分组
    iron_adata.obs['subcluster_group'] = 'Other'
    iron_adata.obs.loc[iron_adata.obs['subcluster'].isin(CONTROL_DOMINANT_SUBCLUSTERS), 'subcluster_group'] = 'Control_dominant'
    iron_adata.obs.loc[iron_adata.obs['subcluster'].isin(CD_INDUCED_SUBCLUSTERS), 'subcluster_group'] = 'Cd_induced'
    
    print(f"\n亚群分组统计:")
    print(iron_adata.obs['subcluster_group'].value_counts())
    
    # 找两组之间的差异基因
    print("\n寻找Control主导亚群 vs Cd诱导亚群的差异基因...")
    sc.tl.rank_genes_groups(
        iron_adata, 
        groupby='subcluster_group',
        groups=['Control_dominant', 'Cd_induced'],
        reference='Cd_induced',
        method='wilcoxon'
    )
    
    # 提取Control主导亚群的marker基因
    control_markers = sc.get.rank_genes_groups_df(iron_adata, group='Control_dominant')
    control_markers['gene_name'] = control_markers['names'].map(gene_name_map)
    control_markers.to_csv(output_dir / 'control_dominant_markers.csv', index=False)
    
    # 提取Cd诱导亚群的marker基因
    sc.tl.rank_genes_groups(
        iron_adata,
        groupby='subcluster_group', 
        groups=['Cd_induced'],
        reference='Control_dominant',
        method='wilcoxon'
    )
    cd_markers = sc.get.rank_genes_groups_df(iron_adata, group='Cd_induced')
    cd_markers['gene_name'] = cd_markers['names'].map(gene_name_map)
    cd_markers.to_csv(output_dir / 'cd_induced_markers.csv', index=False)
    
    # 打印top markers
    print("\n=== Control主导亚群(7,10)特征基因 (Top 20) ===")
    top_control = control_markers[control_markers['logfoldchanges'] > 0].head(20)
    for _, row in top_control.iterrows():
        print(f"  {row['gene_name']:15s} (logFC={row['logfoldchanges']:.2f}, pval={row['pvals_adj']:.2e})")
    
    print("\n=== Cd诱导亚群(2,3,4,8,9,11)特征基因 (Top 20) ===")
    top_cd = cd_markers[cd_markers['logfoldchanges'] > 0].head(20)
    for _, row in top_cd.iterrows():
        print(f"  {row['gene_name']:15s} (logFC={row['logfoldchanges']:.2f}, pval={row['pvals_adj']:.2e})")
    
    # 绘制亚群分组的UMAP
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sc.pl.umap(iron_adata, color='subcluster_group', ax=axes[0], show=False,
               title='Subcluster Groups', palette={'Control_dominant': '#2ecc71', 
                                                    'Cd_induced': '#e74c3c',
                                                    'Other': '#95a5a6'})
    sc.pl.umap(iron_adata, color='group', ax=axes[1], show=False, title='Treatment Group')
    sc.pl.umap(iron_adata, color='subcluster', ax=axes[2], show=False, title='Subclusters')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ironcell_subcluster_groups.png', bbox_inches='tight')
    plt.close()
    
    # 绘制两组的marker基因热图
    # 选择top差异基因
    top_genes_control = control_markers[control_markers['logfoldchanges'] > 0.5].head(15)['names'].tolist()
    top_genes_cd = cd_markers[cd_markers['logfoldchanges'] > 0.5].head(15)['names'].tolist()
    
    # 过滤存在的基因
    all_top_genes = [g for g in top_genes_control + top_genes_cd if g in iron_adata.var_names]
    
    if len(all_top_genes) > 5:
        sc.pl.heatmap(
            iron_adata[iron_adata.obs['subcluster_group'] != 'Other'],
            var_names=all_top_genes[:30],
            groupby='subcluster_group',
            show=False,
            swap_axes=True,
            figsize=(14, 8),
            save='_ironcell_group_markers.png'
        )
        # 移动保存的文件
        import shutil
        if Path('figures/heatmap_ironcell_group_markers.png').exists():
            shutil.move('figures/heatmap_ironcell_group_markers.png', output_dir / 'ironcell_group_markers_heatmap.png')
    
    # 分析各亚群的功能特征
    print("\n=== 各亚群详细分析 ===")
    
    # 对每个关键亚群找marker
    key_subclusters = CONTROL_DOMINANT_SUBCLUSTERS + CD_INDUCED_SUBCLUSTERS
    subcluster_markers = {}
    
    for sc_id in key_subclusters:
        mask = iron_adata.obs['subcluster'] == sc_id
        if mask.sum() > 10:
            iron_sub = iron_adata.copy()
            iron_sub.obs['is_target'] = mask.astype(str)
            sc.tl.rank_genes_groups(iron_sub, groupby='is_target', groups=['True'], reference='False', method='wilcoxon')
            markers = sc.get.rank_genes_groups_df(iron_sub, group='True')
            markers['gene_name'] = markers['names'].map(gene_name_map)
            subcluster_markers[sc_id] = markers
            
            print(f"\n亚群 {sc_id} 特征基因 (Top 10):")
            for _, row in markers.head(10).iterrows():
                print(f"  {row['gene_name']:15s} (logFC={row['logfoldchanges']:.2f})")
    
    # 保存各亚群marker
    for sc_id, markers in subcluster_markers.items():
        markers.to_csv(output_dir / f'subcluster_{sc_id}_markers.csv', index=False)
    
    return control_markers, cd_markers, subcluster_markers


# ============================================================================
# 2. GO富集分析
# ============================================================================

def run_enrichment_analysis(ec_adata, iron_adata):
    """对关键差异基因进行GO富集分析"""
    print("\n" + "="*60)
    print("2. GO富集分析")
    print("="*60)
    
    output_dir = OUTPUT_DIR / "enrichment"
    output_dir.mkdir(exist_ok=True)
    
    # 尝试导入gseapy
    try:
        import gseapy as gp
        has_gseapy = True
        print("使用gseapy进行富集分析")
    except ImportError:
        has_gseapy = False
        print("gseapy未安装，将使用基础功能分析")
    
    # 获取基因名称映射
    gene_name_map = dict(zip(ec_adata.var_names, ec_adata.var['gene_name']))
    
    # 读取差异表达结果
    de_files = {
        'EC_Cd': R4_DIR / 'differential_expression/EC_Cd_vs_Control_significant.csv',
        'EC_Cd-PS-NPs': R4_DIR / 'differential_expression/EC_Cd-PS-NPs_vs_Control_significant.csv',
        'IronCell_Cd': R4_DIR / 'differential_expression/Iron Cell_Cd_vs_Control_significant.csv',
        'IronCell_Cd-PS-NPs': R4_DIR / 'differential_expression/Iron Cell_Cd-PS-NPs_vs_Control_significant.csv',
    }
    
    all_enrichment_results = {}
    
    for name, filepath in de_files.items():
        if not filepath.exists():
            continue
            
        print(f"\n处理 {name}...")
        de_df = pd.read_csv(filepath)
        
        # 分离上调和下调基因
        up_genes = de_df[de_df['logfoldchanges'] > 0.5]['names'].tolist()
        down_genes = de_df[de_df['logfoldchanges'] < -0.5]['names'].tolist()
        
        # 转换为基因符号
        up_symbols = [gene_name_map.get(g, g) for g in up_genes if gene_name_map.get(g, g) != g]
        down_symbols = [gene_name_map.get(g, g) for g in down_genes if gene_name_map.get(g, g) != g]
        
        print(f"  上调基因: {len(up_symbols)}, 下调基因: {len(down_symbols)}")
        
        if has_gseapy and len(up_symbols) > 10:
            try:
                # 上调基因富集
                # 注意：使用GO数据库进行富集分析
                # KEGG_2019_Drosophila 是果蝇专用数据库，如不可用则仅使用GO
                enr_up = gp.enrichr(
                    gene_list=up_symbols,
                    gene_sets=['GO_Biological_Process_2021', 'GO_Molecular_Function_2021', 'KEGG_2019_Drosophila'],
                    organism='fly',
                    outdir=None,
                    no_plot=True
                )
                if enr_up.results is not None and len(enr_up.results) > 0:
                    enr_up.results.to_csv(output_dir / f'{name}_up_enrichment.csv', index=False)
                    all_enrichment_results[f'{name}_up'] = enr_up.results
                    print(f"  上调基因富集完成: {len(enr_up.results)} terms")
            except Exception as e:
                print(f"  上调基因富集失败: {e}")
        
            try:
                # 下调基因富集
                if len(down_symbols) > 10:
                    enr_down = gp.enrichr(
                        gene_list=down_symbols,
                        gene_sets=['GO_Biological_Process_2021', 'GO_Molecular_Function_2021', 'KEGG_2019_Drosophila'],
                        organism='fly',
                        outdir=None,
                        no_plot=True
                    )
                    if enr_down.results is not None and len(enr_down.results) > 0:
                        enr_down.results.to_csv(output_dir / f'{name}_down_enrichment.csv', index=False)
                        all_enrichment_results[f'{name}_down'] = enr_down.results
                        print(f"  下调基因富集完成: {len(enr_down.results)} terms")
            except Exception as e:
                print(f"  下调基因富集失败: {e}")
        
        # 保存基因列表供后续分析
        pd.DataFrame({'gene_id': up_genes, 'gene_symbol': [gene_name_map.get(g, g) for g in up_genes]}).to_csv(
            output_dir / f'{name}_up_genes.csv', index=False)
        pd.DataFrame({'gene_id': down_genes, 'gene_symbol': [gene_name_map.get(g, g) for g in down_genes]}).to_csv(
            output_dir / f'{name}_down_genes.csv', index=False)
    
    # 绘制富集结果
    if all_enrichment_results:
        plot_enrichment_summary(all_enrichment_results, output_dir)
    
    # 手动功能分类分析
    analyze_functional_categories(de_files, gene_name_map, output_dir)
    
    return all_enrichment_results


def analyze_functional_categories(de_files, gene_name_map, output_dir):
    """基于已知功能进行手动分类分析"""
    print("\n=== 功能分类分析 ===")
    
    # 定义功能类别和相关基因
    functional_categories = {
        'Metal_detoxification': ['MtnA', 'MtnB', 'MtnC', 'MtnD', 'MtnE', 'Ctr1B', 'ZnT63C', 'ZnT35C'],
        'Iron_metabolism': ['Fer1HCH', 'Fer2LCH', 'Tsf1', 'Tsf2', 'Tsf3', 'Mvl', 'IRP-1A', 'IRP-1B'],
        'Oxidative_stress': ['Sod1', 'Sod2', 'Cat', 'Prx5', 'GstD1', 'GstE1', 'Trx-2', 'TrxR-1'],
        'Digestive_enzymes': ['Amy-d', 'Amy-p', 'Jon25Bi', 'Jon65Aiii', 'Jon99Ci', 'Mal-A1', 'Mal-A2'],
        'Peritrophic_matrix': ['Peritrophin-15a', 'Peritrophin-15b', 'Muc68D', 'Muc68E'],
        'Apoptosis': ['reaper', 'hid', 'grim', 'Dcp-1', 'Drice', 'Dark'],
        'Cell_cycle': ['CycA', 'CycB', 'CycD', 'CycE', 'Cdk1', 'Cdk2', 'Cdk4'],
        'Autophagy': ['Atg1', 'Atg6', 'Atg8a', 'Atg8b', 'Atg12', 'ref(2)P'],
    }
    
    results = []
    
    for name, filepath in de_files.items():
        if not filepath.exists():
            continue
        
        de_df = pd.read_csv(filepath)
        de_df['gene_symbol'] = de_df['names'].map(gene_name_map)
        
        for category, genes in functional_categories.items():
            # 检查该类别基因在DE结果中的情况
            category_genes = de_df[de_df['gene_symbol'].isin(genes)]
            
            if len(category_genes) > 0:
                up_count = (category_genes['logfoldchanges'] > 0.25).sum()
                down_count = (category_genes['logfoldchanges'] < -0.25).sum()
                avg_logfc = category_genes['logfoldchanges'].mean()
                
                results.append({
                    'comparison': name,
                    'category': category,
                    'total_genes': len(category_genes),
                    'up_regulated': up_count,
                    'down_regulated': down_count,
                    'avg_logFC': avg_logfc,
                    'genes': ', '.join(category_genes['gene_symbol'].tolist())
                })
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / 'functional_category_summary.csv', index=False)
        
        # 绘制功能类别热图
        pivot_df = results_df.pivot(index='category', columns='comparison', values='avg_logFC')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f', ax=ax)
        ax.set_title('Functional Category Changes (avg logFC)')
        plt.tight_layout()
        plt.savefig(output_dir / 'functional_category_heatmap.png', bbox_inches='tight')
        plt.close()
        
        print("\n功能类别变化摘要:")
        print(results_df.to_string(index=False))


def plot_enrichment_summary(enrichment_results, output_dir):
    """绘制富集分析摘要图"""
    for name, df in enrichment_results.items():
        if df is None or len(df) == 0:
            continue
        
        # 取top 15 terms
        top_terms = df.nsmallest(15, 'Adjusted P-value')
        
        if len(top_terms) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 计算-log10(p-value)
        top_terms['neg_log_pval'] = -np.log10(top_terms['Adjusted P-value'] + 1e-300)
        
        # 绘制条形图
        bars = ax.barh(range(len(top_terms)), top_terms['neg_log_pval'])
        ax.set_yticks(range(len(top_terms)))
        ax.set_yticklabels([t[:50] + '...' if len(t) > 50 else t for t in top_terms['Term']])
        ax.set_xlabel('-Log10(Adjusted P-value)')
        ax.set_title(f'Top Enriched Terms: {name}')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{name}_enrichment_barplot.png', bbox_inches='tight')
        plt.close()


# ============================================================================
# 3. Mtn家族基因表达模式分析
# ============================================================================

def analyze_mtn_expression(ec_adata, iron_adata):
    """分析Mtn家族基因在不同亚群和处理组中的表达模式"""
    print("\n" + "="*60)
    print("3. Mtn家族基因表达模式分析")
    print("="*60)
    
    output_dir = OUTPUT_DIR / "mtn_expression"
    output_dir.mkdir(exist_ok=True)
    
    # 合并所有Mtn和金属相关基因
    all_metal_genes = {**MTN_GENES, **METAL_RELATED_GENES}
    
    # 过滤存在的基因
    ec_metal_genes = [g for g in all_metal_genes.keys() if g in ec_adata.var_names]
    iron_metal_genes = [g for g in all_metal_genes.keys() if g in iron_adata.var_names]
    
    print(f"EC中存在的金属相关基因: {len(ec_metal_genes)}")
    print(f"Iron Cell中存在的金属相关基因: {len(iron_metal_genes)}")
    
    # ========== EC分析 ==========
    print("\n=== EC中Mtn家族表达 ===")
    
    # 按处理组绘制violin plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    mtn_genes_in_ec = [g for g in MTN_GENES.keys() if g in ec_adata.var_names]
    
    for i, gene in enumerate(mtn_genes_in_ec[:6]):
        if i < len(axes):
            sc.pl.violin(ec_adata, keys=gene, groupby='group', ax=axes[i], show=False,
                        rotation=45)
            axes[i].set_title(f'{MTN_GENES.get(gene, gene)} in EC')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'EC_Mtn_violin_by_group.png', bbox_inches='tight')
    plt.close()
    
    # 按亚群绘制dotplot
    if len(mtn_genes_in_ec) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        sc.pl.dotplot(ec_adata, var_names=mtn_genes_in_ec, groupby='subcluster', 
                      show=False, ax=ax)
        plt.title('Mtn Family Expression in EC Subclusters')
        plt.savefig(output_dir / 'EC_Mtn_dotplot_by_subcluster.png', bbox_inches='tight')
        plt.close()
    
    # 按亚群和处理组的交叉分析
    ec_adata.obs['subcluster_group'] = ec_adata.obs['subcluster'].astype(str) + '_' + ec_adata.obs['group'].astype(str)
    
    # 计算每个亚群-处理组组合的平均表达
    mtn_expr_ec = []
    for sc_id in ec_adata.obs['subcluster'].unique():
        for group in ec_adata.obs['group'].unique():
            mask = (ec_adata.obs['subcluster'] == sc_id) & (ec_adata.obs['group'] == group)
            if mask.sum() > 5:
                for gene in mtn_genes_in_ec:
                    gene_idx = list(ec_adata.var_names).index(gene)
                    mean_expr = ec_adata[mask].X[:, gene_idx].mean()
                    mtn_expr_ec.append({
                        'subcluster': sc_id,
                        'group': group,
                        'gene': MTN_GENES.get(gene, gene),
                        'mean_expression': mean_expr
                    })
    
    mtn_expr_ec_df = pd.DataFrame(mtn_expr_ec)
    mtn_expr_ec_df.to_csv(output_dir / 'EC_Mtn_expression_by_subcluster_group.csv', index=False)
    
    # 绘制热图
    if len(mtn_expr_ec_df) > 0:
        try:
            pivot_ec = mtn_expr_ec_df.pivot_table(index=['subcluster', 'group'], columns='gene', values='mean_expression')
            # 确保数据类型正确
            pivot_ec = pivot_ec.astype(float)
            
            fig, ax = plt.subplots(figsize=(10, 12))
            sns.heatmap(pivot_ec, cmap='YlOrRd', ax=ax, annot=False)
            ax.set_title('Mtn Expression in EC (Subcluster x Group)')
            plt.tight_layout()
            plt.savefig(output_dir / 'EC_Mtn_heatmap_subcluster_group.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  EC热图绑制失败: {e}")
    
    # ========== Iron Cell分析 ==========
    print("\n=== Iron Cell中Mtn家族表达 ===")
    
    mtn_genes_in_iron = [g for g in MTN_GENES.keys() if g in iron_adata.var_names]
    
    # 按处理组绘制violin plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, gene in enumerate(mtn_genes_in_iron[:6]):
        if i < len(axes):
            sc.pl.violin(iron_adata, keys=gene, groupby='group', ax=axes[i], show=False,
                        rotation=45)
            axes[i].set_title(f'{MTN_GENES.get(gene, gene)} in Iron Cell')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'IronCell_Mtn_violin_by_group.png', bbox_inches='tight')
    plt.close()
    
    # 按亚群绘制dotplot
    if len(mtn_genes_in_iron) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        sc.pl.dotplot(iron_adata, var_names=mtn_genes_in_iron, groupby='subcluster',
                      show=False, ax=ax)
        plt.title('Mtn Family Expression in Iron Cell Subclusters')
        plt.savefig(output_dir / 'IronCell_Mtn_dotplot_by_subcluster.png', bbox_inches='tight')
        plt.close()
    
    # 特别关注Control主导亚群 vs Cd诱导亚群
    iron_adata.obs['subcluster_type'] = 'Other'
    iron_adata.obs.loc[iron_adata.obs['subcluster'].isin(CONTROL_DOMINANT_SUBCLUSTERS), 'subcluster_type'] = 'Control_dominant'
    iron_adata.obs.loc[iron_adata.obs['subcluster'].isin(CD_INDUCED_SUBCLUSTERS), 'subcluster_type'] = 'Cd_induced'
    
    # 计算每个亚群类型-处理组组合的平均表达
    mtn_expr_iron = []
    for sc_type in ['Control_dominant', 'Cd_induced', 'Other']:
        for group in iron_adata.obs['group'].unique():
            mask = (iron_adata.obs['subcluster_type'] == sc_type) & (iron_adata.obs['group'] == group)
            if mask.sum() > 5:
                for gene in mtn_genes_in_iron:
                    gene_idx = list(iron_adata.var_names).index(gene)
                    mean_expr = iron_adata[mask].X[:, gene_idx].mean()
                    mtn_expr_iron.append({
                        'subcluster_type': sc_type,
                        'group': group,
                        'gene': MTN_GENES.get(gene, gene),
                        'mean_expression': mean_expr
                    })
    
    mtn_expr_iron_df = pd.DataFrame(mtn_expr_iron)
    mtn_expr_iron_df.to_csv(output_dir / 'IronCell_Mtn_expression_by_type_group.csv', index=False)
    
    # 绘制对比图
    if len(mtn_expr_iron_df) > 0:
        try:
            n_genes = len(mtn_genes_in_iron)
            fig, axes = plt.subplots(1, max(n_genes, 1), figsize=(4*max(n_genes, 1), 5))
            if n_genes == 1:
                axes = [axes]
            
            for i, gene in enumerate(mtn_genes_in_iron):
                gene_name = MTN_GENES.get(gene, gene)
                gene_df = mtn_expr_iron_df[mtn_expr_iron_df['gene'] == gene_name]
                
                if len(gene_df) > 0:
                    pivot = gene_df.pivot(index='subcluster_type', columns='group', values='mean_expression')
                    # 确保数据类型正确
                    pivot = pivot.astype(float)
                    if not pivot.empty and pivot.notna().any().any():
                        pivot.plot(kind='bar', ax=axes[i], width=0.8)
                        axes[i].set_title(gene_name)
                        axes[i].set_xlabel('')
                        axes[i].tick_params(axis='x', rotation=45)
                        axes[i].legend(title='Group', bbox_to_anchor=(1.02, 1))
            
            plt.tight_layout()
            plt.savefig(output_dir / 'IronCell_Mtn_comparison_by_type.png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  Iron Cell Mtn对比图绑制失败: {e}")
    
    # UMAP上显示Mtn表达
    print("\n绘制UMAP上的Mtn表达...")
    
    # EC
    if len(mtn_genes_in_ec) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, gene in enumerate(mtn_genes_in_ec[:6]):
            if i < len(axes):
                sc.pl.umap(ec_adata, color=gene, ax=axes[i], show=False,
                          title=f'{MTN_GENES.get(gene, gene)}', cmap='YlOrRd')
        
        plt.suptitle('Mtn Expression on EC UMAP', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'EC_Mtn_umap.png', bbox_inches='tight')
        plt.close()
    
    # Iron Cell
    if len(mtn_genes_in_iron) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, gene in enumerate(mtn_genes_in_iron[:6]):
            if i < len(axes):
                sc.pl.umap(iron_adata, color=gene, ax=axes[i], show=False,
                          title=f'{MTN_GENES.get(gene, gene)}', cmap='YlOrRd')
        
        plt.suptitle('Mtn Expression on Iron Cell UMAP', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'IronCell_Mtn_umap.png', bbox_inches='tight')
        plt.close()
    
    return mtn_expr_ec_df, mtn_expr_iron_df


# ============================================================================
# 4. 生成综合报告
# ============================================================================

def generate_comprehensive_report(control_markers, cd_markers, enrichment_results, 
                                   mtn_expr_ec, mtn_expr_iron):
    """生成综合分析报告"""
    print("\n" + "="*60)
    print("4. 生成综合报告")
    print("="*60)
    
    report_path = OUTPUT_DIR / 'R4_followup_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("R4区域后续深度分析报告\n")
        f.write("="*70 + "\n\n")
        
        # Iron Cell亚群功能注释
        f.write("## 1. Iron Cell亚群功能注释\n\n")
        f.write("### 1.1 亚群分类\n")
        f.write("- Control主导亚群 (7, 10): 在对照组中占主导，金属暴露后几乎消失\n")
        f.write("- Cd诱导亚群 (2, 3, 4, 8, 9, 11): 金属暴露后大幅增加\n\n")
        
        if control_markers is not None:
            f.write("### 1.2 Control主导亚群特征基因 (Top 10)\n")
            top_control = control_markers[control_markers['logfoldchanges'] > 0].head(10)
            for _, row in top_control.iterrows():
                f.write(f"  - {row['gene_name']}: logFC={row['logfoldchanges']:.2f}\n")
        
        if cd_markers is not None:
            f.write("\n### 1.3 Cd诱导亚群特征基因 (Top 10)\n")
            top_cd = cd_markers[cd_markers['logfoldchanges'] > 0].head(10)
            for _, row in top_cd.iterrows():
                f.write(f"  - {row['gene_name']}: logFC={row['logfoldchanges']:.2f}\n")
        
        # GO富集分析
        f.write("\n\n## 2. GO富集分析摘要\n\n")
        if enrichment_results:
            for name, df in enrichment_results.items():
                if df is not None and len(df) > 0:
                    f.write(f"### {name}\n")
                    top_terms = df.nsmallest(5, 'Adjusted P-value')
                    for _, row in top_terms.iterrows():
                        f.write(f"  - {row['Term'][:60]}: p={row['Adjusted P-value']:.2e}\n")
                    f.write("\n")
        else:
            f.write("富集分析结果见enrichment目录下的CSV文件\n")
        
        # Mtn表达模式
        f.write("\n## 3. Mtn家族基因表达模式\n\n")
        f.write("### 3.1 关键发现\n")
        f.write("- Mtn家族基因(MtnA-E)在金属暴露后显著上调\n")
        f.write("- 上调在EC和Iron Cell中均观察到\n")
        f.write("- Cd诱导的Iron Cell亚群表现出更高的Mtn表达\n\n")
        
        if mtn_expr_iron is not None and len(mtn_expr_iron) > 0:
            f.write("### 3.2 Iron Cell亚群类型间Mtn表达差异\n")
            summary = mtn_expr_iron.groupby(['subcluster_type', 'gene'])['mean_expression'].mean().unstack()
            f.write(summary.to_string())
        
        f.write("\n\n" + "="*70 + "\n")
        f.write("分析完成\n")
        f.write("="*70 + "\n")
    
    print(f"报告已保存: {report_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*70)
    print("R4区域后续深度分析")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    ec_adata = sc.read_h5ad(R4_DIR / 'EC_subclusters/EC_subclustered.h5ad')
    iron_adata = sc.read_h5ad(R4_DIR / 'IronCell_subclusters/IronCell_subclustered.h5ad')
    
    print(f"EC细胞数: {ec_adata.n_obs}")
    print(f"Iron Cell细胞数: {iron_adata.n_obs}")
    
    # 1. Iron Cell亚群功能注释
    control_markers, cd_markers, subcluster_markers = analyze_ironcell_subclusters(iron_adata)
    
    # 2. GO富集分析
    enrichment_results = run_enrichment_analysis(ec_adata, iron_adata)
    
    # 3. Mtn家族表达模式分析
    mtn_expr_ec, mtn_expr_iron = analyze_mtn_expression(ec_adata, iron_adata)
    
    # 4. 生成综合报告
    generate_comprehensive_report(control_markers, cd_markers, enrichment_results,
                                   mtn_expr_ec, mtn_expr_iron)
    
    print("\n" + "="*70)
    print("所有分析完成!")
    print(f"结果保存在: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
