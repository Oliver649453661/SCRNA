#!/usr/bin/env python3
"""
从GTF文件中提取基因ID到基因symbol的映射
"""

import os
import sys
import pandas as pd
import re

# Snakemake 输入输出
gtf_file = snakemake.input.gtf
output_file = snakemake.output.mapping
log_file = snakemake.log[0]

# 设置日志
os.makedirs(os.path.dirname(output_file), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print(f"Extracting gene mapping from {gtf_file}")

gene_mapping = {}

with open(gtf_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        
        fields = line.strip().split('\t')
        if len(fields) < 9:
            continue
        
        feature_type = fields[2]
        if feature_type != 'gene':
            continue
        
        # 解析属性
        attributes = {}
        for item in fields[8].strip().split(';'):
            item = item.strip()
            if item:
                match = re.match(r'(\S+)\s+"([^"]+)"', item)
                if match:
                    key, value = match.groups()
                    attributes[key] = value
        
        gene_id = attributes.get('gene_id', '')
        gene_name = attributes.get('gene_name', gene_id)
        
        if gene_id:
            gene_mapping[gene_id] = gene_name

print(f"Extracted {len(gene_mapping)} gene mappings")

# 保存为CSV
df = pd.DataFrame([
    {'gene_id': gid, 'gene_name': gname}
    for gid, gname in sorted(gene_mapping.items())
])

df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")

print("Done!")
