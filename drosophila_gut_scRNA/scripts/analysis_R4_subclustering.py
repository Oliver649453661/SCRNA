#!/usr/bin/env python3
"""
R4区域亚群分析脚本
对R4区域的EC和Iron Cell进行精细亚群分析，并进行差异表达分析
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
# 配置参数
# ============================================================================
INPUT_FILE = "results/annotation/gut_region_annotated.h5ad"
OUTPUT_DIR = Path("results/R4_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 亚群分析参数
EC_RESOLUTION = 0.5  # EC亚群聚类分辨率
IRON_CELL_RESOLUTION = 0.3  # Iron Cell亚群聚类分辨率
MIN_CELLS_FOR_SUBCLUSTER = 100  # 最少细胞数才进行亚群分析

# 差异表达参数
DE_METHOD = 'wilcoxon'
DE_MIN_CELLS = 10
DE_LOGFC_THRESHOLD = 0.25

# ============================================================================
# 辅助函数
# ============================================================================

def subset_and_recluster(adata, cell_mask, resolution, n_pcs=30, n_neighbors=15):
    """
    提取子集并重新进行降维聚类
    """
    adata_sub = adata[cell_mask].copy()
    
    # 修复索引问题 - 重置index name避免与列名冲突
    adata_sub.obs.index.name = None
    adata_sub.obs_names = pd.Index(adata_sub.obs_names.astype(str))
    adata_sub.obs_names_make_unique()
    
    print(f"  子集细胞数: {adata_sub.n_obs}")
    
    if adata_sub.n_obs < MIN_CELLS_FOR_SUBCLUSTER:
        print(f"  细胞数不足 {MIN_CELLS_FOR_SUBCLUSTER}，跳过亚群分析")
        return None
    
    # 重新计算HVG和PCA
    sc.pp.highly_variable_genes(adata_sub, n_top_genes=2000, batch_key='batch')
    sc.tl.pca(adata_sub, n_comps=min(n_pcs, adata_sub.n_obs - 1))
    
    # 邻域图和UMAP
    sc.pp.neighbors(adata_sub, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata_sub.n_obs - 1))
    sc.tl.umap(adata_sub)
    
    # 聚类
    sc.tl.leiden(adata_sub, resolution=resolution, key_added='subcluster')
    
    return adata_sub


def run_differential_expression(adata, groupby, groups=None, reference='Control'):
    """
    运行差异表达分析
    """
    if groups is None:
        groups = [g for g in adata.obs[groupby].unique() if g != reference]
    
    results = {}
    for group in groups:
        if group == reference:
            continue
        
        # 检查两组细胞数
        n_group = (adata.obs[groupby] == group).sum()
        n_ref = (adata.obs[groupby] == reference).sum()
        
        if n_group < DE_MIN_CELLS or n_ref < DE_MIN_CELLS:
            print(f"  跳过 {group} vs {reference}: 细胞数不足")
            continue
        
        try:
            sc.tl.rank_genes_groups(
                adata, 
                groupby=groupby, 
                groups=[group], 
                reference=reference,
                method=DE_METHOD,
                pts=True
            )
            
            # 提取结果
            de_df = sc.get.rank_genes_groups_df(adata, group=group)
            de_df['comparison'] = f"{group}_vs_{reference}"
            results[group] = de_df
            
        except Exception as e:
            print(f"  差异表达分析失败 {group}: {e}")
    
    return results


def plot_subcluster_overview(adata_sub, cell_type_name, output_dir):
    """
    绘制亚群分析概览图
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # UMAP by subcluster
    sc.pl.umap(adata_sub, color='subcluster', ax=axes[0, 0], show=False, 
               title=f'{cell_type_name} Subclusters')
    
    # UMAP by condition
    sc.pl.umap(adata_sub, color='group', ax=axes[0, 1], show=False,
               title='By Treatment Group')
    
    # UMAP by sample
    sc.pl.umap(adata_sub, color='sample', ax=axes[0, 2], show=False,
               title='By Sample', legend_loc='none')
    
    # Subcluster composition by group
    ct = pd.crosstab(adata_sub.obs['subcluster'], adata_sub.obs['group'], normalize='columns') * 100
    ct.plot(kind='bar', ax=axes[1, 0], width=0.8)
    axes[1, 0].set_title('Subcluster Composition by Group (%)')
    axes[1, 0].set_xlabel('Subcluster')
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].legend(title='Group', bbox_to_anchor=(1.02, 1))
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Cell counts per subcluster per group
    ct_counts = pd.crosstab(adata_sub.obs['subcluster'], adata_sub.obs['group'])
    ct_counts.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Cell Counts by Subcluster and Group')
    axes[1, 1].set_xlabel('Subcluster')
    axes[1, 1].set_ylabel('Cell Count')
    axes[1, 1].legend(title='Group', bbox_to_anchor=(1.02, 1))
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    # Proportion change (treatment vs control)
    control_prop = ct['Control'] if 'Control' in ct.columns else ct.iloc[:, 0]
    for group in ['Cd', 'Cd-PS-NPs', 'PS-NPs']:
        if group in ct.columns:
            # 使用更规范的方法处理零值：设置最小阈值0.1%
            fold_change = np.where(
                control_prop > 0.001,  # 对照组比例>0.1%才计算fold change
                ct[group] / control_prop,
                np.nan  # 对照组为0或极低时不计算
            )
            axes[1, 2].bar(ct.index.astype(str), fold_change, alpha=0.7, label=group)
    axes[1, 2].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('Fold Change vs Control')
    axes[1, 2].set_xlabel('Subcluster')
    axes[1, 2].set_ylabel('Fold Change')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{cell_type_name}_subcluster_overview.png', bbox_inches='tight')
    plt.close()


def plot_subcluster_markers(adata_sub, cell_type_name, output_dir, n_genes=5):
    """
    绘制亚群marker基因
    """
    # 找marker基因
    sc.tl.rank_genes_groups(adata_sub, 'subcluster', method='wilcoxon')
    
    # Dotplot
    fig, ax = plt.subplots(figsize=(12, 8))
    sc.pl.rank_genes_groups_dotplot(
        adata_sub, n_genes=n_genes, 
        show=False, ax=ax
    )
    plt.savefig(output_dir / f'{cell_type_name}_subcluster_markers_dotplot.png', bbox_inches='tight')
    plt.close()
    
    # Heatmap
    sc.pl.rank_genes_groups_heatmap(
        adata_sub, n_genes=n_genes,
        show=False, figsize=(12, 10)
    )
    plt.savefig(output_dir / f'{cell_type_name}_subcluster_markers_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # 保存marker基因表
    marker_df = sc.get.rank_genes_groups_df(adata_sub, group=None)
    marker_df.to_csv(output_dir / f'{cell_type_name}_subcluster_markers.csv', index=False)
    
    return marker_df


def analyze_subcluster_de(adata_sub, cell_type_name, output_dir):
    """
    对每个亚群进行处理组间差异表达分析
    """
    all_de_results = []
    
    for subcluster in adata_sub.obs['subcluster'].unique():
        print(f"  分析亚群 {subcluster}...")
        adata_sc = adata_sub[adata_sub.obs['subcluster'] == subcluster].copy()
        
        if adata_sc.n_obs < DE_MIN_CELLS * 2:
            print(f"    细胞数不足，跳过")
            continue
        
        de_results = run_differential_expression(adata_sc, 'group')
        
        for group, df in de_results.items():
            df['subcluster'] = subcluster
            df['cell_type'] = cell_type_name
            all_de_results.append(df)
    
    if all_de_results:
        combined_df = pd.concat(all_de_results, ignore_index=True)
        combined_df.to_csv(output_dir / f'{cell_type_name}_subcluster_DE.csv', index=False)
        return combined_df
    
    return None


def plot_de_volcano(de_df, title, output_path, logfc_thresh=0.5, pval_thresh=0.05):
    """
    绘制火山图
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    de_df = de_df.copy()
    de_df['neg_log10_pval'] = -np.log10(de_df['pvals_adj'] + 1e-300)
    
    # 分类
    de_df['significance'] = 'Not Significant'
    de_df.loc[(de_df['logfoldchanges'] > logfc_thresh) & (de_df['pvals_adj'] < pval_thresh), 'significance'] = 'Up'
    de_df.loc[(de_df['logfoldchanges'] < -logfc_thresh) & (de_df['pvals_adj'] < pval_thresh), 'significance'] = 'Down'
    
    colors = {'Up': '#e74c3c', 'Down': '#3498db', 'Not Significant': '#95a5a6'}
    
    for sig, color in colors.items():
        mask = de_df['significance'] == sig
        ax.scatter(de_df.loc[mask, 'logfoldchanges'], 
                   de_df.loc[mask, 'neg_log10_pval'],
                   c=color, alpha=0.6, s=10, label=f'{sig} ({mask.sum()})')
    
    ax.axhline(y=-np.log10(pval_thresh), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=logfc_thresh, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-logfc_thresh, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10 Adjusted P-value')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def analyze_r4_cell_type_de(adata_r4, output_dir):
    """
    对R4区域各细胞类型进行处理组间差异表达分析
    """
    print("\n" + "="*60)
    print("R4区域细胞类型差异表达分析")
    print("="*60)
    
    de_output_dir = output_dir / 'differential_expression'
    de_output_dir.mkdir(exist_ok=True)
    
    all_de_results = []
    
    # 主要分析EC和Iron Cell
    for cell_type in ['EC', 'Iron Cell']:
        print(f"\n分析 {cell_type}...")
        
        mask = adata_r4.obs['final_cell_type'] == cell_type
        if mask.sum() < DE_MIN_CELLS * 2:
            print(f"  细胞数不足，跳过")
            continue
        
        adata_ct = adata_r4[mask].copy()
        adata_ct.obs_names_make_unique()
        
        de_results = run_differential_expression(adata_ct, 'group')
        
        for group, df in de_results.items():
            df['cell_type'] = cell_type
            all_de_results.append(df)
            
            # 绘制火山图
            plot_de_volcano(
                df, 
                f'{cell_type}: {group} vs Control',
                de_output_dir / f'{cell_type}_{group}_vs_Control_volcano.png'
            )
            
            # 保存显著基因
            sig_df = df[(df['pvals_adj'] < 0.05) & (abs(df['logfoldchanges']) > DE_LOGFC_THRESHOLD)]
            sig_df.to_csv(de_output_dir / f'{cell_type}_{group}_vs_Control_significant.csv', index=False)
            
            print(f"  {group} vs Control: {len(sig_df)} 显著差异基因")
    
    if all_de_results:
        combined_df = pd.concat(all_de_results, ignore_index=True)
        combined_df.to_csv(de_output_dir / 'R4_all_DE_results.csv', index=False)
        return combined_df
    
    return None


def generate_summary_report(adata_r4, ec_sub, iron_sub, de_results, output_dir):
    """
    生成分析摘要报告
    """
    report_path = output_dir / 'R4_analysis_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("R4区域亚群分析报告\n")
        f.write("="*70 + "\n\n")
        
        # 基本统计
        f.write("## 1. R4区域细胞组成\n\n")
        f.write(f"总细胞数: {adata_r4.n_obs}\n\n")
        
        f.write("各处理组细胞数:\n")
        for group, count in adata_r4.obs['group'].value_counts().items():
            f.write(f"  - {group}: {count}\n")
        
        f.write("\n细胞类型分布:\n")
        for ct, count in adata_r4.obs['final_cell_type'].value_counts().items():
            pct = count / adata_r4.n_obs * 100
            f.write(f"  - {ct}: {count} ({pct:.1f}%)\n")
        
        # EC亚群分析
        f.write("\n\n## 2. EC亚群分析\n\n")
        if ec_sub is not None:
            f.write(f"EC细胞总数: {ec_sub.n_obs}\n")
            f.write(f"亚群数量: {ec_sub.obs['subcluster'].nunique()}\n\n")
            
            f.write("各亚群细胞数:\n")
            for sc, count in ec_sub.obs['subcluster'].value_counts().sort_index().items():
                f.write(f"  - Subcluster {sc}: {count}\n")
            
            f.write("\n各处理组亚群分布 (%):\n")
            ct = pd.crosstab(ec_sub.obs['subcluster'], ec_sub.obs['group'], normalize='columns') * 100
            f.write(ct.round(2).to_string())
        else:
            f.write("EC细胞数不足，未进行亚群分析\n")
        
        # Iron Cell亚群分析
        f.write("\n\n## 3. Iron Cell亚群分析\n\n")
        if iron_sub is not None:
            f.write(f"Iron Cell细胞总数: {iron_sub.n_obs}\n")
            f.write(f"亚群数量: {iron_sub.obs['subcluster'].nunique()}\n\n")
            
            f.write("各亚群细胞数:\n")
            for sc, count in iron_sub.obs['subcluster'].value_counts().sort_index().items():
                f.write(f"  - Subcluster {sc}: {count}\n")
            
            f.write("\n各处理组亚群分布 (%):\n")
            ct = pd.crosstab(iron_sub.obs['subcluster'], iron_sub.obs['group'], normalize='columns') * 100
            f.write(ct.round(2).to_string())
        else:
            f.write("Iron Cell细胞数不足，未进行亚群分析\n")
        
        # 差异表达摘要
        f.write("\n\n## 4. 差异表达分析摘要\n\n")
        if de_results is not None:
            for cell_type in de_results['cell_type'].unique():
                ct_df = de_results[de_results['cell_type'] == cell_type]
                f.write(f"\n{cell_type}:\n")
                for comp in ct_df['comparison'].unique():
                    comp_df = ct_df[ct_df['comparison'] == comp]
                    sig = comp_df[(comp_df['pvals_adj'] < 0.05) & (abs(comp_df['logfoldchanges']) > DE_LOGFC_THRESHOLD)]
                    up = (sig['logfoldchanges'] > 0).sum()
                    down = (sig['logfoldchanges'] < 0).sum()
                    f.write(f"  - {comp}: {len(sig)} DEGs (↑{up}, ↓{down})\n")
        
        f.write("\n\n" + "="*70 + "\n")
        f.write("分析完成\n")
        f.write("="*70 + "\n")
    
    print(f"\n报告已保存: {report_path}")


# ============================================================================
# 主分析流程
# ============================================================================

def main():
    print("="*70)
    print("R4区域亚群深度分析")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    adata = sc.read_h5ad(INPUT_FILE)
    # 处理重复的obs_names - 先转换为字符串类型
    adata.obs_names = pd.Index(adata.obs_names.astype(str))
    adata.obs_names_make_unique()
    print(f"总细胞数: {adata.n_obs}")
    
    # 提取R4区域
    print("\n提取R4区域细胞...")
    r4_mask = adata.obs['gut_region'] == 'R4'
    adata_r4 = adata[r4_mask].copy()
    adata_r4.obs.index.name = None
    adata_r4.obs_names_make_unique()
    print(f"R4细胞数: {adata_r4.n_obs}")
    
    # 创建输出目录
    ec_dir = OUTPUT_DIR / 'EC_subclusters'
    iron_dir = OUTPUT_DIR / 'IronCell_subclusters'
    ec_dir.mkdir(exist_ok=True)
    iron_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # 1. EC亚群分析
    # ========================================================================
    print("\n" + "="*60)
    print("1. EC亚群分析")
    print("="*60)
    
    ec_mask = adata_r4.obs['final_cell_type'] == 'EC'
    print(f"EC细胞数: {ec_mask.sum()}")
    
    ec_sub = subset_and_recluster(adata_r4, ec_mask, EC_RESOLUTION)
    
    if ec_sub is not None:
        print(f"  发现 {ec_sub.obs['subcluster'].nunique()} 个亚群")
        
        # 绘制概览图
        plot_subcluster_overview(ec_sub, 'EC', ec_dir)
        
        # 找marker基因
        print("  寻找亚群marker基因...")
        plot_subcluster_markers(ec_sub, 'EC', ec_dir)
        
        # 亚群差异表达
        print("  亚群差异表达分析...")
        analyze_subcluster_de(ec_sub, 'EC', ec_dir)
        
        # 保存数据
        ec_sub.write_h5ad(ec_dir / 'EC_subclustered.h5ad')
        print(f"  EC亚群数据已保存")
    
    # ========================================================================
    # 2. Iron Cell亚群分析
    # ========================================================================
    print("\n" + "="*60)
    print("2. Iron Cell亚群分析")
    print("="*60)
    
    iron_mask = adata_r4.obs['final_cell_type'] == 'Iron Cell'
    print(f"Iron Cell细胞数: {iron_mask.sum()}")
    
    iron_sub = subset_and_recluster(adata_r4, iron_mask, IRON_CELL_RESOLUTION)
    
    if iron_sub is not None:
        print(f"  发现 {iron_sub.obs['subcluster'].nunique()} 个亚群")
        
        # 绘制概览图
        plot_subcluster_overview(iron_sub, 'IronCell', iron_dir)
        
        # 找marker基因
        print("  寻找亚群marker基因...")
        plot_subcluster_markers(iron_sub, 'IronCell', iron_dir)
        
        # 亚群差异表达
        print("  亚群差异表达分析...")
        analyze_subcluster_de(iron_sub, 'IronCell', iron_dir)
        
        # 保存数据
        iron_sub.write_h5ad(iron_dir / 'IronCell_subclustered.h5ad')
        print(f"  Iron Cell亚群数据已保存")
    
    # ========================================================================
    # 3. R4区域细胞类型差异表达分析
    # ========================================================================
    de_results = analyze_r4_cell_type_de(adata_r4, OUTPUT_DIR)
    
    # ========================================================================
    # 4. 生成摘要报告
    # ========================================================================
    print("\n生成分析报告...")
    generate_summary_report(adata_r4, ec_sub, iron_sub, de_results, OUTPUT_DIR)
    
    # 保存R4数据
    adata_r4.write_h5ad(OUTPUT_DIR / 'R4_cells.h5ad')
    
    print("\n" + "="*70)
    print("分析完成!")
    print(f"结果保存在: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
