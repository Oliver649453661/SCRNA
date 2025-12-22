#!/usr/bin/env python3
"""
从 alevin-fry 输出创建 AnnData 对象
"""

import os
import sys
import pandas as pd
import scanpy as sc
import pyroe

# Snakemake 输入输出
quant_dir = os.path.dirname(snakemake.input.quant)
sample_name = snakemake.wildcards.sample
meta_file = snakemake.input.meta
output_h5ad = snakemake.output.h5ad
log_file = snakemake.log[0]

# 设置日志
os.makedirs(os.path.dirname(output_h5ad), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# 重定向输出到日志
sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print(f"Processing sample: {sample_name}")
print(f"Quantification directory: {quant_dir}")

# 读取元数据
meta_df = pd.read_csv(meta_file, sep='\t')
sample_meta = meta_df[meta_df['sample'] == sample_name].iloc[0].to_dict()

# 使用 pyroe 加载 alevin-fry 输出
print("Loading alevin-fry quantification...")
adata = pyroe.load_fry(
    quant_dir,
    output_format='velocity'
)

# 添加样本元数据
adata.obs['sample'] = sample_name
adata.obs['group'] = sample_meta['group']
if sample_meta['replicate'] is not None:
    adata.obs['replicate'] = sample_meta['replicate']

# 保存原始计数
adata.layers['counts'] = adata.X.copy()

# 基本统计
print(f"Shape: {adata.shape}")
print(f"Genes: {adata.n_vars}")
print(f"Cells: {adata.n_obs}")

# 保存
print(f"Saving to {output_h5ad}")
adata.write_h5ad(output_h5ad, compression='gzip')

print("Done!")
