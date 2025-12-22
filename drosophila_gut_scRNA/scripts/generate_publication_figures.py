#!/usr/bin/env python
"""
出版级图表生成脚本
整合所有分析结果生成高质量的出版级图表
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# 设置出版级图表样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['figure.dpi'] = 300


def create_umap_overview(adata, celltype_col, output_path):
    """创建UMAP概览图"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 检查celltype_col是否存在
    if celltype_col not in adata.obs.columns:
        if 'leiden' in adata.obs.columns:
            celltype_col = 'leiden'
        else:
            celltype_col = adata.obs.columns[0]
    
    # UMAP by cell type
    try:
        sc.pl.umap(adata, color=celltype_col, ax=axes[0], show=False, 
                   title='Cell Types', frameon=True, legend_loc='right margin')
    except Exception as e:
        axes[0].text(0.5, 0.5, f'UMAP error: {str(e)[:30]}', ha='center', va='center')
        axes[0].axis('off')
    
    # UMAP by group
    if 'group' in adata.obs.columns:
        try:
            sc.pl.umap(adata, color='group', ax=axes[1], show=False,
                       title='Treatment Groups', frameon=True, legend_loc='right margin')
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
            axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No group info', ha='center', va='center')
        axes[1].axis('off')
    
    # UMAP by sample
    if 'sample' in adata.obs.columns:
        try:
            sc.pl.umap(adata, color='sample', ax=axes[2], show=False,
                       title='Samples', frameon=True, legend_loc='right margin')
        except Exception as e:
            axes[2].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
            axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'No sample info', ha='center', va='center')
        axes[2].axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_composition_figure(composition_df, output_path, groupby='group'):
    """创建细胞组成图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 堆叠条形图
    if 'cell_type' in composition_df.columns and groupby in composition_df.columns:
        pivot_df = composition_df.pivot_table(
            index=groupby, 
            columns='cell_type', 
            values='proportion' if 'proportion' in composition_df.columns else 'count',
            aggfunc='mean'
        )
        pivot_df.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab20')
        axes[0].set_xlabel('Treatment Group')
        axes[0].set_ylabel('Proportion')
        axes[0].set_title('Cell Type Composition by Group')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        axes[0].tick_params(axis='x', rotation=60)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=9)
    else:
        axes[0].text(0.5, 0.5, 'Composition data not available', ha='center', va='center')
        axes[0].axis('off')
    
    # 热图
    if 'cell_type' in composition_df.columns and groupby in composition_df.columns:
        pivot_df = composition_df.pivot_table(
            index='cell_type',
            columns=groupby,
            values='proportion' if 'proportion' in composition_df.columns else 'count',
            aggfunc='mean'
        )
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1], annot_kws={'fontsize': 7})
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)
        axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=7)
        axes[1].set_title('Cell Type Proportion Heatmap', fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Composition data not available', ha='center', va='center')
        axes[1].axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_de_summary_figure(de_summary_df, output_path):
    """创建差异表达汇总图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # DEG数量条形图
    if 'cell_type' in de_summary_df.columns:
        # 计算每个细胞类型的上调和下调基因数
        if 'n_up' in de_summary_df.columns and 'n_down' in de_summary_df.columns:
            de_counts = de_summary_df.groupby('cell_type')[['n_up', 'n_down']].sum()
            de_counts.plot(kind='barh', ax=axes[0], color=['#e74c3c', '#3498db'])
            axes[0].set_xlabel('Number of DEGs')
            axes[0].set_ylabel('Cell Type')
            axes[0].set_title('Differentially Expressed Genes by Cell Type')
            axes[0].legend(['Upregulated', 'Downregulated'])
        else:
            axes[0].text(0.5, 0.5, 'DEG count data not available', ha='center', va='center')
            axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'DE summary not available', ha='center', va='center')
        axes[0].axis('off')
    
    # 热图显示每个细胞类型在不同处理组的DEG数
    if 'comparison' in de_summary_df.columns and 'cell_type' in de_summary_df.columns:
        if 'n_deg' in de_summary_df.columns:
            pivot_df = de_summary_df.pivot_table(
                index='cell_type',
                columns='comparison',
                values='n_deg',
                aggfunc='sum'
            ).fillna(0)
            sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='Reds', ax=axes[1], annot_kws={'fontsize': 7})
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)
            axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=7)
            axes[1].set_title('DEG Count by Cell Type and Comparison', fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'DEG count not available', ha='center', va='center')
            axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'DE summary not available', ha='center', va='center')
        axes[1].axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_gut_region_figure(adata, region_scores_df, output_path, celltype_col='final_cell_type'):
    """创建肠道区域注释图"""
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. UMAP by gut region
    ax1 = fig.add_subplot(gs[0, 0])
    if 'gut_region' in adata.obs.columns:
        try:
            # 定义区域颜色
            region_colors = {
                'Crop': '#e41a1c', 'R0': '#ff7f00', 'R1': '#ffff33',
                'R2': '#4daf4a', 'R3': '#377eb8', 'R4': '#984ea3',
                'R5': '#f781bf', 'Hindgut': '#a65628', 'Uncertain': '#999999'
            }
            sc.pl.umap(adata, color='gut_region', ax=ax1, show=False,
                       title='Gut Region', frameon=True, legend_loc='right margin',
                       palette=region_colors)
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
            ax1.axis('off')
    else:
        ax1.text(0.5, 0.5, 'Gut region not available', ha='center', va='center')
        ax1.axis('off')
    
    # 2. UMAP by cell type
    ax2 = fig.add_subplot(gs[0, 1])
    if celltype_col in adata.obs.columns:
        try:
            sc.pl.umap(adata, color=celltype_col, ax=ax2, show=False,
                       title='Cell Types', frameon=True, legend_loc='right margin')
        except Exception as e:
            ax2.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
            ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, 'Cell type not available', ha='center', va='center')
        ax2.axis('off')
    
    # 3. UMAP by treatment group
    ax3 = fig.add_subplot(gs[0, 2])
    if 'group' in adata.obs.columns:
        try:
            sc.pl.umap(adata, color='group', ax=ax3, show=False,
                       title='Treatment Groups', frameon=True, legend_loc='right margin')
        except Exception as e:
            ax3.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
            ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, 'Group not available', ha='center', va='center')
        ax3.axis('off')
    
    # 4. 区域分布条形图
    ax4 = fig.add_subplot(gs[1, 0])
    if 'gut_region' in adata.obs.columns:
        region_counts = adata.obs['gut_region'].value_counts()
        colors = [region_colors.get(r, '#999999') for r in region_counts.index]
        region_counts.plot(kind='bar', ax=ax4, color=colors, edgecolor='black')
        ax4.set_xlabel('Gut Region')
        ax4.set_ylabel('Cell Count')
        ax4.set_title('Cell Distribution by Gut Region')
        ax4.tick_params(axis='x', rotation=45)
        for i, v in enumerate(region_counts.values):
            ax4.text(i, v + 100, str(v), ha='center', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'Region data not available', ha='center', va='center')
        ax4.axis('off')
    
    # 5. 细胞类型在各区域的分布热图
    ax5 = fig.add_subplot(gs[1, 1:3])
    if 'gut_region' in adata.obs.columns and celltype_col in adata.obs.columns:
        ct_region = pd.crosstab(adata.obs[celltype_col], adata.obs['gut_region'], normalize='index')
        # 按区域顺序排列
        region_order = ['Crop', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'Hindgut', 'Uncertain']
        ct_region = ct_region.reindex(columns=[c for c in region_order if c in ct_region.columns])
        sns.heatmap(ct_region, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax5,
                    annot_kws={'fontsize': 7}, cbar_kws={'shrink': 0.8})
        ax5.set_title('Cell Type Distribution Across Gut Regions', fontweight='bold')
        ax5.set_xlabel('Gut Region')
        ax5.set_ylabel('Cell Type')
    else:
        ax5.text(0.5, 0.5, 'Data not available', ha='center', va='center')
        ax5.axis('off')
    
    # 6. 区域得分热图
    ax6 = fig.add_subplot(gs[2, 0:2])
    if region_scores_df is not None and len(region_scores_df) > 0:
        score_cols = [c for c in region_scores_df.columns if c.endswith('_score') and c != 'gut_region_score']
        if score_cols and 'cell_type' in region_scores_df.columns:
            mean_scores = region_scores_df.groupby('cell_type')[score_cols].mean()
            # 重命名列
            mean_scores.columns = [c.replace('_score', '') for c in mean_scores.columns]
            sns.heatmap(mean_scores, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                        ax=ax6, annot_kws={'fontsize': 6}, cbar_kws={'shrink': 0.8})
            ax6.set_title('Mean Region Scores by Cell Type', fontweight='bold')
            ax6.set_xlabel('Gut Region')
            ax6.set_ylabel('Cell Type')
        else:
            ax6.text(0.5, 0.5, 'Score data not available', ha='center', va='center')
            ax6.axis('off')
    else:
        ax6.text(0.5, 0.5, 'Region scores not available', ha='center', va='center')
        ax6.axis('off')
    
    # 7. 各处理组的区域分布
    ax7 = fig.add_subplot(gs[2, 2])
    if 'gut_region' in adata.obs.columns and 'group' in adata.obs.columns:
        group_region = pd.crosstab(adata.obs['group'], adata.obs['gut_region'], normalize='index')
        group_region = group_region.reindex(columns=[c for c in region_order if c in group_region.columns])
        group_region.plot(kind='bar', stacked=True, ax=ax7, colormap='tab10')
        ax7.set_xlabel('Treatment Group')
        ax7.set_ylabel('Proportion')
        ax7.set_title('Gut Region Distribution by Treatment')
        ax7.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        ax7.tick_params(axis='x', rotation=45)
    else:
        ax7.text(0.5, 0.5, 'Data not available', ha='center', va='center')
        ax7.axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_enrichment_summary_figure(enrichment_df, output_path):
    """创建富集分析汇总图"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    if enrichment_df is not None and len(enrichment_df) > 0:
        # 1. Top enriched terms bubble plot
        ax1 = axes[0]
        if 'Term' in enrichment_df.columns and 'cell_type' in enrichment_df.columns:
            # 获取每个细胞类型的top terms
            top_terms = enrichment_df.groupby('cell_type').head(3)
            if 'Combined Score' in top_terms.columns:
                score_col = 'Combined Score'
            elif 'NES' in top_terms.columns:
                score_col = 'NES'
            else:
                score_col = top_terms.select_dtypes(include=[np.number]).columns[0] if len(top_terms.select_dtypes(include=[np.number]).columns) > 0 else None
            
            if score_col and len(top_terms) > 0:
                # 简化term名称
                top_terms = top_terms.copy()
                top_terms['Term_short'] = top_terms['Term'].str[:40]
                
                scatter = ax1.scatter(
                    range(len(top_terms)),
                    top_terms['cell_type'],
                    s=np.abs(top_terms[score_col]) * 10,
                    c=top_terms[score_col],
                    cmap='RdBu_r',
                    alpha=0.7
                )
                ax1.set_xlabel('Enriched Terms')
                ax1.set_ylabel('Cell Type')
                ax1.set_title('Top Enriched Terms by Cell Type')
                plt.colorbar(scatter, ax=ax1, label=score_col)
            else:
                ax1.text(0.5, 0.5, 'Enrichment scores not available', ha='center', va='center')
                ax1.axis('off')
        else:
            ax1.text(0.5, 0.5, 'Enrichment data format error', ha='center', va='center')
            ax1.axis('off')
        
        # 2. Enrichment count by cell type
        ax2 = axes[1]
        if 'cell_type' in enrichment_df.columns:
            term_counts = enrichment_df.groupby('cell_type').size().sort_values(ascending=True)
            term_counts.plot(kind='barh', ax=ax2, color='steelblue', edgecolor='black')
            ax2.set_xlabel('Number of Enriched Terms')
            ax2.set_ylabel('Cell Type')
            ax2.set_title('Enrichment Analysis Summary')
        else:
            ax2.text(0.5, 0.5, 'Cell type info not available', ha='center', va='center')
            ax2.axis('off')
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'Enrichment data not available', ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_marker_expression_figure(adata, output_path, celltype_col='final_cell_type'):
    """创建Marker基因表达图"""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 定义果蝇肠道细胞类型的典型marker基因
    marker_genes = {
        'ISC': ['esg', 'Dl', 'N'],
        'EB': ['esg', 'klu', 'E(spl)m3-HLH'],
        'EC': ['Myo31DF', 'nub', 'Pdp1'],
        'EE': ['pros', 'AstA', 'Tk', 'DH31'],
        'Copper Cell': ['Mvl', 'Fer1HCH', 'Fer2LCH'],
        'Iron Cell': ['Fer1HCH', 'Fer2LCH', 'Tsf1'],
        'LFC': ['Mex1', 'Npc2b'],
        'VM': ['Mhc', 'up', 'Act57B']
    }
    
    # 获取数据中存在的marker基因
    all_markers = []
    for genes in marker_genes.values():
        all_markers.extend(genes)
    available_markers = [g for g in set(all_markers) if g in adata.var_names]
    
    if len(available_markers) > 0:
        # 1. Dotplot
        ax1 = fig.add_subplot(gs[0, :])
        try:
            sc.pl.dotplot(adata, var_names=available_markers[:15], groupby=celltype_col,
                         ax=ax1, show=False, title='Marker Gene Expression')
        except Exception as e:
            ax1.text(0.5, 0.5, f'Dotplot error: {str(e)[:40]}', ha='center', va='center')
            ax1.axis('off')
        
        # 2. Violin plot for top markers
        ax2 = fig.add_subplot(gs[1, 0])
        try:
            top_markers = available_markers[:4]
            if len(top_markers) > 0:
                sc.pl.violin(adata, keys=top_markers, groupby=celltype_col,
                            ax=ax2, show=False, rotation=45)
        except Exception as e:
            ax2.text(0.5, 0.5, f'Violin error: {str(e)[:40]}', ha='center', va='center')
            ax2.axis('off')
        
        # 3. UMAP with marker expression
        ax3 = fig.add_subplot(gs[1, 1])
        try:
            if len(available_markers) > 0:
                sc.pl.umap(adata, color=available_markers[0], ax=ax3, show=False,
                          title=f'{available_markers[0]} Expression', cmap='Reds')
        except Exception as e:
            ax3.text(0.5, 0.5, f'UMAP error: {str(e)[:40]}', ha='center', va='center')
            ax3.axis('off')
    else:
        for i in range(4):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            ax.text(0.5, 0.5, 'Marker genes not found in data', ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_treatment_comparison_figure(adata, de_summary_df, output_path, celltype_col='final_cell_type'):
    """创建处理组比较图"""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    groups = adata.obs['group'].unique() if 'group' in adata.obs.columns else []
    
    # 1-3. 每个处理组的UMAP（分面）
    for i, group in enumerate(groups[:3]):
        ax = fig.add_subplot(gs[0, i])
        try:
            mask = adata.obs['group'] == group
            adata_sub = adata[mask].copy()
            sc.pl.umap(adata_sub, color=celltype_col, ax=ax, show=False,
                      title=f'{group} (n={mask.sum():,})', frameon=True)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
            ax.axis('off')
    
    # 4. DEG数量比较
    ax4 = fig.add_subplot(gs[1, 0])
    if de_summary_df is not None and 'comparison_group' in de_summary_df.columns:
        deg_by_group = de_summary_df.groupby('comparison_group')[['n_up', 'n_down']].sum()
        deg_by_group.plot(kind='bar', ax=ax4, color=['#e74c3c', '#3498db'])
        ax4.set_xlabel('Treatment Group')
        ax4.set_ylabel('Number of DEGs')
        ax4.set_title('DEGs by Treatment Group')
        ax4.legend(['Upregulated', 'Downregulated'])
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'DE data not available', ha='center', va='center')
        ax4.axis('off')
    
    # 5. 细胞类型比例变化
    ax5 = fig.add_subplot(gs[1, 1])
    if 'group' in adata.obs.columns and celltype_col in adata.obs.columns:
        ct_prop = pd.crosstab(adata.obs['group'], adata.obs[celltype_col], normalize='index')
        ct_prop.plot(kind='bar', stacked=True, ax=ax5, colormap='tab20')
        ax5.set_xlabel('Treatment Group')
        ax5.set_ylabel('Proportion')
        ax5.set_title('Cell Type Proportions by Treatment')
        ax5.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)
        ax5.tick_params(axis='x', rotation=45)
    else:
        ax5.text(0.5, 0.5, 'Data not available', ha='center', va='center')
        ax5.axis('off')
    
    # 6. 响应细胞类型热图
    ax6 = fig.add_subplot(gs[1, 2])
    if de_summary_df is not None and 'cell_type' in de_summary_df.columns and 'comparison_group' in de_summary_df.columns:
        if 'n_degs' in de_summary_df.columns:
            deg_col = 'n_degs'
        elif 'n_deg' in de_summary_df.columns:
            deg_col = 'n_deg'
        else:
            deg_col = None
        
        if deg_col:
            pivot = de_summary_df.pivot_table(index='cell_type', columns='comparison_group',
                                              values=deg_col, aggfunc='sum').fillna(0)
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Reds', ax=ax6,
                       annot_kws={'fontsize': 7})
            ax6.set_title('DEG Count by Cell Type and Treatment')
        else:
            ax6.text(0.5, 0.5, 'DEG count not available', ha='center', va='center')
            ax6.axis('off')
    else:
        ax6.text(0.5, 0.5, 'DE summary not available', ha='center', va='center')
        ax6.axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_pseudotime_figure(adata, output_path, celltype_col='final_cell_type'):
    """创建伪时序分析图"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 检查celltype_col是否存在
    if celltype_col not in adata.obs.columns:
        if 'leiden' in adata.obs.columns:
            celltype_col = 'leiden'
        else:
            celltype_col = adata.obs.columns[0]
    
    # 伪时序UMAP
    if 'dpt_pseudotime' in adata.obs.columns:
        try:
            sc.pl.umap(adata, color='dpt_pseudotime', ax=axes[0], show=False,
                       title='Pseudotime', cmap='viridis', frameon=True)
        except Exception as e:
            axes[0].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
            axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'Pseudotime not available', ha='center', va='center')
        axes[0].axis('off')
    
    # 按细胞类型着色
    try:
        sc.pl.umap(adata, color=celltype_col, ax=axes[1], show=False,
                   title='Cell Types', frameon=True)
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
        axes[1].axis('off')
    
    # 伪时序分布
    if 'dpt_pseudotime' in adata.obs.columns:
        for ct in adata.obs[celltype_col].unique():
            mask = adata.obs[celltype_col] == ct
            if mask.sum() > 0:
                axes[2].hist(adata.obs.loc[mask, 'dpt_pseudotime'].dropna(), 
                           alpha=0.5, label=ct, bins=30)
        axes[2].set_xlabel('Pseudotime')
        axes[2].set_ylabel('Cell Count')
        axes[2].set_title('Pseudotime Distribution by Cell Type')
        axes[2].legend(fontsize=8, loc='upper right')
    else:
        axes[2].text(0.5, 0.5, 'Pseudotime not available', ha='center', va='center')
        axes[2].axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def main():
    # 获取snakemake参数
    h5ad_path = snakemake.input.h5ad
    pseudotime_h5ad_path = snakemake.input.pseudotime_h5ad
    de_summary_path = snakemake.input.de_summary
    composition_path = snakemake.input.composition
    enrichment_summary_path = snakemake.input.enrichment_summary
    region_scores_path = snakemake.input.region_scores if hasattr(snakemake.input, 'region_scores') else None
    output_dir = snakemake.output.output_dir
    log_file = snakemake.log[0]
    
    celltype_col = snakemake.params.get('celltype_col', 'final_cell_type')
    groupby = snakemake.params.get('groupby', 'group')
    
    # 设置日志
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as log:
        sys.stdout = log
        sys.stderr = log
        
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Loading AnnData from {h5ad_path}")
            adata = sc.read_h5ad(h5ad_path)
            
            print(f"Loading pseudotime AnnData from {pseudotime_h5ad_path}")
            adata_pt = sc.read_h5ad(pseudotime_h5ad_path)
            
            print(f"Loading DE summary from {de_summary_path}")
            de_summary = pd.read_csv(de_summary_path)
            
            print(f"Loading composition from {composition_path}")
            composition = pd.read_csv(composition_path)
            
            print(f"Loading enrichment summary from {enrichment_summary_path}")
            try:
                enrichment_summary = pd.read_csv(enrichment_summary_path)
            except Exception as e:
                print(f"Warning: Could not load enrichment summary: {e}")
                enrichment_summary = None
            
            # 加载区域得分
            region_scores = None
            if region_scores_path:
                print(f"Loading region scores from {region_scores_path}")
                try:
                    region_scores = pd.read_csv(region_scores_path)
                except Exception as e:
                    print(f"Warning: Could not load region scores: {e}")
            
            # ========== 生成各个图表 ==========
            
            # Fig1: UMAP概览
            print("Generating Fig1: UMAP overview...")
            create_umap_overview(adata, celltype_col, 
                               os.path.join(output_dir, 'Fig1_UMAP_Overview.pdf'))
            
            # Fig2: 细胞组成
            print("Generating Fig2: Cell composition...")
            create_composition_figure(composition, 
                                     os.path.join(output_dir, 'Fig2_Cell_Composition.pdf'),
                                     groupby=groupby)
            
            # Fig3: 差异表达汇总
            print("Generating Fig3: DE summary...")
            create_de_summary_figure(de_summary,
                                    os.path.join(output_dir, 'Fig3_DE_Summary.pdf'))
            
            # Fig4: 伪时序分析
            print("Generating Fig4: Pseudotime...")
            create_pseudotime_figure(adata_pt,
                                    os.path.join(output_dir, 'Fig4_Pseudotime.pdf'),
                                    celltype_col=celltype_col)
            
            # Fig5: 肠道区域注释（新增）
            print("Generating Fig5: Gut region annotation...")
            create_gut_region_figure(adata, region_scores,
                                    os.path.join(output_dir, 'Fig5_Gut_Region.pdf'),
                                    celltype_col=celltype_col)
            
            # Fig6: 富集分析汇总（新增）
            print("Generating Fig6: Enrichment summary...")
            create_enrichment_summary_figure(enrichment_summary,
                                            os.path.join(output_dir, 'Fig6_Enrichment.pdf'))
            
            # Fig7: Marker基因表达（新增）
            print("Generating Fig7: Marker expression...")
            create_marker_expression_figure(adata,
                                           os.path.join(output_dir, 'Fig7_Marker_Expression.pdf'),
                                           celltype_col=celltype_col)
            
            # Fig8: 处理组比较（新增）
            print("Generating Fig8: Treatment comparison...")
            create_treatment_comparison_figure(adata, de_summary,
                                              os.path.join(output_dir, 'Fig8_Treatment_Comparison.pdf'),
                                              celltype_col=celltype_col)
            
            # 生成综合PDF
            print("Generating combined publication figures PDF...")
            all_figs = [
                'Fig1_UMAP_Overview.pdf', 'Fig2_Cell_Composition.pdf',
                'Fig3_DE_Summary.pdf', 'Fig4_Pseudotime.pdf',
                'Fig5_Gut_Region.pdf', 'Fig6_Enrichment.pdf',
                'Fig7_Marker_Expression.pdf', 'Fig8_Treatment_Comparison.pdf'
            ]
            
            with PdfPages(os.path.join(output_dir, 'All_Publication_Figures.pdf')) as pdf:
                # 封面页
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.7, 'Drosophila Gut scRNA-seq Analysis', 
                       ha='center', va='center', fontsize=24, fontweight='bold')
                ax.text(0.5, 0.55, 'Publication Figures', 
                       ha='center', va='center', fontsize=18)
                ax.text(0.5, 0.4, f'Total cells: {adata.n_obs:,}', 
                       ha='center', va='center', fontsize=14)
                ax.text(0.5, 0.3, f'Cell types: {adata.obs[celltype_col].nunique() if celltype_col in adata.obs.columns else "N/A"}',
                       ha='center', va='center', fontsize=14)
                ax.text(0.5, 0.2, f'Treatment groups: {adata.obs["group"].nunique() if "group" in adata.obs.columns else "N/A"}',
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            # 生成图表索引文件
            with open(os.path.join(output_dir, 'figure_index.txt'), 'w') as f:
                f.write("Publication Figures Index\n")
                f.write("=" * 50 + "\n\n")
                f.write("Fig1_UMAP_Overview.pdf - UMAP visualization by cell type, group, and sample\n")
                f.write("Fig2_Cell_Composition.pdf - Cell type composition analysis\n")
                f.write("Fig3_DE_Summary.pdf - Differential expression summary\n")
                f.write("Fig4_Pseudotime.pdf - Pseudotime trajectory analysis\n")
                f.write("Fig5_Gut_Region.pdf - Gut region annotation and distribution\n")
                f.write("Fig6_Enrichment.pdf - Functional enrichment analysis summary\n")
                f.write("Fig7_Marker_Expression.pdf - Marker gene expression patterns\n")
                f.write("Fig8_Treatment_Comparison.pdf - Treatment group comparisons\n")
                f.write("\nAll_Publication_Figures.pdf - Combined cover page\n")
            
            print(f"\nPublication figures saved to {output_dir}")
            print(f"Generated {len(all_figs)} publication-quality figures")
            print("Publication figure generation completed successfully!")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # 创建空白输出目录
            os.makedirs(output_dir, exist_ok=True)
            # 创建一个占位文件
            with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
                f.write(f"Publication figure generation encountered an error: {e}\n")
            raise

if __name__ == "__main__":
    main()
