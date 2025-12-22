#!/usr/bin/env python3
"""
导出Python分析结果为R可读格式

用法：
    python export_for_R.py input.h5ad output_prefix
    
输出：
    - output_prefix_counts.csv: 表达矩阵
    - output_prefix_metadata.csv: 细胞元数据
    - output_prefix_genes.csv: 基因信息
"""

import sys
import scanpy as sc
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def export_for_r(h5ad_path, output_prefix):
    """
    导出AnnData对象为R友好的格式
    """
    logging.info(f"读取数据: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    
    # 1. 导出表达矩阵
    logging.info("导出表达矩阵...")
    if hasattr(adata.X, 'toarray'):
        counts = pd.DataFrame(
            adata.X.toarray(),
            index=adata.obs_names,
            columns=adata.var_names
        )
    else:
        counts = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )
    counts.to_csv(f"{output_prefix}_counts.csv")
    logging.info(f"  保存: {output_prefix}_counts.csv")
    
    # 2. 导出细胞元数据
    logging.info("导出细胞元数据...")
    adata.obs.to_csv(f"{output_prefix}_metadata.csv")
    logging.info(f"  保存: {output_prefix}_metadata.csv")
    
    # 3. 导出基因信息
    logging.info("导出基因信息...")
    adata.var.to_csv(f"{output_prefix}_genes.csv")
    logging.info(f"  保存: {output_prefix}_genes.csv")
    
    # 4. 如果有降维结果，也导出
    if 'X_pca' in adata.obsm:
        logging.info("导出PCA...")
        pca = pd.DataFrame(
            adata.obsm['X_pca'],
            index=adata.obs_names,
            columns=[f"PC{i+1}" for i in range(adata.obsm['X_pca'].shape[1])]
        )
        pca.to_csv(f"{output_prefix}_pca.csv")
        logging.info(f"  保存: {output_prefix}_pca.csv")
    
    if 'X_umap' in adata.obsm:
        logging.info("导出UMAP...")
        umap = pd.DataFrame(
            adata.obsm['X_umap'],
            index=adata.obs_names,
            columns=['UMAP1', 'UMAP2']
        )
        umap.to_csv(f"{output_prefix}_umap.csv")
        logging.info(f"  保存: {output_prefix}_umap.csv")
    
    # 5. 创建R脚本模板
    r_script = f"""
# R脚本：读取Python分析结果
# R版本: 4.3.3 (base环境)

library(Seurat)

# 读取数据
counts <- read.csv("{output_prefix}_counts.csv", row.names=1)
metadata <- read.csv("{output_prefix}_metadata.csv", row.names=1)
genes <- read.csv("{output_prefix}_genes.csv", row.names=1)

# 创建Seurat对象
seurat_obj <- CreateSeuratObject(
    counts = t(counts),
    meta.data = metadata
)

# 如果有PCA和UMAP，也添加进去
if (file.exists("{output_prefix}_pca.csv")) {{
    pca <- read.csv("{output_prefix}_pca.csv", row.names=1)
    seurat_obj[["pca"]] <- CreateDimReducObject(
        embeddings = as.matrix(pca),
        key = "PC_",
        assay = "RNA"
    )
}}

if (file.exists("{output_prefix}_umap.csv")) {{
    umap <- read.csv("{output_prefix}_umap.csv", row.names=1)
    seurat_obj[["umap"]] <- CreateDimReducObject(
        embeddings = as.matrix(umap),
        key = "UMAP_",
        assay = "RNA"
    )
}}

# 现在可以使用Seurat的功能了
print(seurat_obj)

# 示例：使用Seurat的可视化
# DimPlot(seurat_obj, reduction = "umap", group.by = "leiden")

# 示例：使用Seurat的差异分析
# markers <- FindAllMarkers(seurat_obj, only.pos = TRUE)

# 保存Seurat对象
saveRDS(seurat_obj, file = "{output_prefix}_seurat.rds")
cat("Seurat对象已保存至: {output_prefix}_seurat.rds\\n")
"""
    
    with open(f"{output_prefix}_load_in_R.R", 'w') as f:
        f.write(r_script)
    
    logging.info(f"\n✅ 导出完成！")
    logging.info(f"\n在R中使用（base环境，R 4.3.3）:")
    logging.info(f"  conda activate base")
    logging.info(f"  Rscript {output_prefix}_load_in_R.R")
    logging.info(f"\n或交互式使用:")
    logging.info(f"  R")
    logging.info(f"  > source('{output_prefix}_load_in_R.R')")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    
    h5ad_path = sys.argv[1]
    output_prefix = sys.argv[2]
    
    export_for_r(h5ad_path, output_prefix)
