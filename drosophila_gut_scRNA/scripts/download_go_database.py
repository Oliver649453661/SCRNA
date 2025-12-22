#!/usr/bin/env python3
"""
下载GO数据库和果蝇基因注释文件
用于本地富集分析，避免依赖不稳定的API
"""

import os
import urllib.request
import gzip
import shutil

# 输出目录
output_dir = "data/reference/go_database"
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("Downloading GO Database for Local Enrichment Analysis")
print("="*60)

# 1. 下载GO本体文件 (OBO格式)
go_obo_url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
go_obo_file = os.path.join(output_dir, "go-basic.obo")

if not os.path.exists(go_obo_file):
    print(f"\n1. Downloading GO ontology (go-basic.obo)...")
    try:
        urllib.request.urlretrieve(go_obo_url, go_obo_file)
        print(f"   ✓ Saved to {go_obo_file}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print(f"\n1. GO ontology already exists: {go_obo_file}")

# 2. 下载果蝇GO注释文件 (GAF格式)
# 从Gene Ontology官方下载
fly_gaf_url = "http://current.geneontology.org/annotations/fb.gaf.gz"
fly_gaf_gz = os.path.join(output_dir, "fb.gaf.gz")
fly_gaf_file = os.path.join(output_dir, "fb.gaf")

if not os.path.exists(fly_gaf_file):
    print(f"\n2. Downloading Drosophila GO annotations (fb.gaf)...")
    try:
        urllib.request.urlretrieve(fly_gaf_url, fly_gaf_gz)
        # 解压
        with gzip.open(fly_gaf_gz, 'rb') as f_in:
            with open(fly_gaf_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(fly_gaf_gz)
        print(f"   ✓ Saved to {fly_gaf_file}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print(f"\n2. Drosophila GO annotations already exist: {fly_gaf_file}")

# 3. 解析GAF文件，创建gene2go映射
gene2go_file = os.path.join(output_dir, "dmel_gene2go.tsv")

if not os.path.exists(gene2go_file):
    print(f"\n3. Creating gene2go mapping...")
    try:
        gene2go = {}
        with open(fly_gaf_file, 'r') as f:
            for line in f:
                if line.startswith('!'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    # GAF格式: DB, DB_Object_ID, DB_Object_Symbol, ..., GO_ID, ...
                    gene_symbol = parts[2]
                    go_id = parts[4]
                    if gene_symbol not in gene2go:
                        gene2go[gene_symbol] = set()
                    gene2go[gene_symbol].add(go_id)
        
        # 保存为TSV
        with open(gene2go_file, 'w') as f:
            f.write("gene_symbol\tgo_ids\n")
            for gene, go_ids in gene2go.items():
                f.write(f"{gene}\t{','.join(go_ids)}\n")
        
        print(f"   ✓ Created {gene2go_file} with {len(gene2go)} genes")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print(f"\n3. Gene2GO mapping already exists: {gene2go_file}")

print("\n" + "="*60)
print("Download complete!")
print("="*60)
print(f"\nFiles in {output_dir}:")
for f in os.listdir(output_dir):
    size = os.path.getsize(os.path.join(output_dir, f)) / 1024 / 1024
    print(f"  - {f} ({size:.1f} MB)")
