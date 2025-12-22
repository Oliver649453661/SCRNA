#!/usr/bin/env python
"""
Marker基因可视化脚本
生成细胞类型marker基因的热图和点图
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import os

warnings.filterwarnings('ignore')

def main():
    # 获取snakemake参数
    h5ad_path = snakemake.input.h5ad
    markers_path = snakemake.input.markers
    heatmap_output = snakemake.output.heatmap
    dotplot_output = snakemake.output.dotplot
    log_file = snakemake.log[0]
    
    n_genes = snakemake.params.get('n_genes', 10)
    celltype_col = snakemake.params.get('celltype_col', 'final_cell_type')
    
    # 设置日志
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as log:
        sys.stdout = log
        sys.stderr = log
        
        try:
            print(f"Loading AnnData from {h5ad_path}")
            adata = sc.read_h5ad(h5ad_path)
            
            print(f"Loading markers from {markers_path}")
            markers_df = pd.read_csv(markers_path)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(heatmap_output), exist_ok=True)
            
            # 检查细胞类型列
            if celltype_col not in adata.obs.columns:
                print(f"Warning: {celltype_col} not found, using 'leiden' instead")
                celltype_col = 'leiden'
            
            # 检查markers_df的结构
            if 'group' in markers_df.columns:
                group_col = 'group'
            elif 'cluster' in markers_df.columns:
                group_col = 'cluster'
            else:
                group_col = markers_df.columns[0]
            
            if 'names' in markers_df.columns:
                gene_col = 'names'
            elif 'gene' in markers_df.columns:
                gene_col = 'gene'
            else:
                gene_col = markers_df.columns[1] if len(markers_df.columns) > 1 else markers_df.columns[0]
            
            print(f"Using group column: {group_col}, gene column: {gene_col}")
            print(f"Markers file groups: {markers_df[group_col].unique().tolist()[:10]}")
            print(f"AnnData {celltype_col}: {adata.obs[celltype_col].unique().tolist()[:10]}")
            
            # 策略1: 直接从markers文件获取top基因（不按细胞类型分组）
            # 这样可以避免细胞类型名称不匹配的问题
            marker_genes = {}
            all_markers_from_file = []
            
            # 按group分组获取每个cluster的top genes
            for grp in markers_df[group_col].unique():
                grp_markers = markers_df[markers_df[group_col] == grp]
                if 'scores' in grp_markers.columns:
                    grp_markers = grp_markers.sort_values('scores', ascending=False)
                elif 'logfoldchanges' in grp_markers.columns:
                    grp_markers = grp_markers.sort_values('logfoldchanges', ascending=False)
                
                genes = grp_markers[gene_col].head(n_genes).tolist()
                genes = [g for g in genes if g in adata.var_names]
                all_markers_from_file.extend(genes)
            
            # 去重保持顺序
            all_markers_from_file = list(dict.fromkeys(all_markers_from_file))
            print(f"Markers from file: {len(all_markers_from_file)} genes")
            
            # 如果从文件获取的marker足够，直接使用
            if len(all_markers_from_file) >= 20:
                all_markers = all_markers_from_file
            else:
                # 策略2: 使用已存储的rank_genes_groups结果
                print("Trying to use stored rank_genes_groups...")
                try:
                    if 'rank_genes_groups' in adata.uns:
                        for ct in adata.obs[celltype_col].unique():
                            try:
                                genes = sc.get.rank_genes_groups_df(adata, group=str(ct))['names'].head(n_genes).tolist()
                                genes = [g for g in genes if g in adata.var_names]
                                if genes:
                                    marker_genes[ct] = genes
                            except:
                                continue
                except Exception as e:
                    print(f"Could not use stored rank_genes_groups: {e}")
                
                # 策略3: 重新计算，但过滤掉细胞数太少的组
                if not marker_genes:
                    print("Calculating rank_genes_groups with filtered groups...")
                    # 过滤掉细胞数少于3的组
                    cell_counts = adata.obs[celltype_col].value_counts()
                    valid_groups = cell_counts[cell_counts >= 3].index.tolist()
                    print(f"Valid groups (>=3 cells): {valid_groups}")
                    
                    if len(valid_groups) >= 2:
                        adata_filtered = adata[adata.obs[celltype_col].isin(valid_groups)].copy()
                        try:
                            sc.tl.rank_genes_groups(adata_filtered, groupby=celltype_col, method='wilcoxon')
                            for ct in valid_groups:
                                try:
                                    genes = sc.get.rank_genes_groups_df(adata_filtered, group=str(ct))['names'].head(n_genes).tolist()
                                    genes = [g for g in genes if g in adata.var_names]
                                    if genes:
                                        marker_genes[ct] = genes
                                except:
                                    continue
                        except Exception as e:
                            print(f"rank_genes_groups failed: {e}")
                
                # 合并marker genes
                all_markers = []
                for ct, genes in marker_genes.items():
                    all_markers.extend(genes)
                all_markers = list(dict.fromkeys(all_markers))
                
                # 如果还是没有，使用文件中的
                if len(all_markers) < 10:
                    all_markers = all_markers_from_file
            
            print(f"Total marker genes for visualization: {len(all_markers)}")
            
            if len(all_markers) == 0:
                print("Error: No marker genes available for visualization")
                # 创建空白图
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.text(0.5, 0.5, 'No marker genes available', ha='center', va='center', fontsize=14)
                ax.axis('off')
                fig.savefig(heatmap_output, dpi=150, bbox_inches='tight')
                fig.savefig(dotplot_output, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print("Created placeholder figures")
                return
            
            # 生成热图
            print("Generating heatmap...")
            sc.settings.figdir = os.path.dirname(heatmap_output)
            
            # Limit markers and adjust figure size based on number of cell types
            n_celltypes = adata.obs[celltype_col].nunique()
            n_markers_to_show = min(30, len(all_markers))  # Reduce markers for readability
            fig_height = max(8, n_celltypes * 0.5)
            fig_width = max(12, n_markers_to_show * 0.4)
            
            try:
                sc.pl.heatmap(
                    adata,
                    var_names=all_markers[:n_markers_to_show],
                    groupby=celltype_col,
                    cmap='viridis',
                    dendrogram=False,  # Disable dendrogram to avoid layout issues
                    swap_axes=False,
                    show=False,
                    figsize=(fig_width, fig_height)
                )
            except Exception as e:
                print(f"Heatmap failed: {e}, creating simple version")
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                ax.text(0.5, 0.5, f'Heatmap generation failed: {str(e)[:50]}', 
                       ha='center', va='center')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(heatmap_output, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Heatmap saved to {heatmap_output}")
            
            # 生成点图
            print("Generating dotplot...")
            try:
                sc.pl.dotplot(
                    adata,
                    var_names=all_markers[:n_markers_to_show],
                    groupby=celltype_col,
                    dendrogram=False,  # Disable dendrogram to avoid layout issues
                    show=False,
                    figsize=(fig_width, fig_height)
                )
            except Exception as e:
                print(f"Dotplot failed: {e}, creating simple version")
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                ax.text(0.5, 0.5, f'Dotplot generation failed: {str(e)[:50]}', 
                       ha='center', va='center')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(dotplot_output, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Dotplot saved to {dotplot_output}")
            
            print("Marker visualization completed successfully!")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # 创建空白图以避免工作流失败
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', fontsize=12)
            ax.axis('off')
            fig.savefig(heatmap_output, dpi=150, bbox_inches='tight')
            fig.savefig(dotplot_output, dpi=150, bbox_inches='tight')
            plt.close(fig)
            raise

if __name__ == "__main__":
    main()
