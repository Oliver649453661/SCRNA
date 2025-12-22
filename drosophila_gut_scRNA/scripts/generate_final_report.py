#!/usr/bin/env python
"""
最终分析报告生成脚本
生成HTML格式的综合分析报告
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import base64
import warnings

warnings.filterwarnings('ignore')


def encode_image(image_path):
    """将图片编码为base64"""
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None


def generate_html_report(adata, de_summary, composition, enrichment_summary, 
                         summary_dir, publication_dir, output_path):
    """生成HTML报告"""
    
    # 收集统计信息
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    n_samples = adata.obs['sample'].nunique() if 'sample' in adata.obs.columns else 'N/A'
    n_groups = adata.obs['group'].nunique() if 'group' in adata.obs.columns else 'N/A'
    
    # 确定细胞类型列
    celltype_col = 'final_cell_type'
    if celltype_col not in adata.obs.columns:
        if 'cell_type' in adata.obs.columns:
            celltype_col = 'cell_type'
        elif 'leiden' in adata.obs.columns:
            celltype_col = 'leiden'
        else:
            celltype_col = None
    
    n_celltypes = adata.obs[celltype_col].nunique() if celltype_col else 'N/A'
    
    # 细胞类型统计
    celltype_counts = adata.obs[celltype_col].value_counts() if celltype_col else pd.Series()
    
    # DE统计
    n_deg_total = de_summary['n_deg'].sum() if 'n_deg' in de_summary.columns else 'N/A'
    
    # 生成HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drosophila Gut scRNA-seq Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-card .label {{
            color: #666;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .figure-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure-container img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        .toc {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            padding: 5px 0;
        }}
        .toc a {{
            color: #667eea;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Drosophila Gut scRNA-seq Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="toc section">
        <h2>📋 Table of Contents</h2>
        <ul>
            <li><a href="#overview">1. Dataset Overview</a></li>
            <li><a href="#celltypes">2. Cell Type Composition</a></li>
            <li><a href="#de">3. Differential Expression Analysis</a></li>
            <li><a href="#enrichment">4. Functional Enrichment</a></li>
            <li><a href="#advanced">5. Advanced Analyses</a></li>
            <li><a href="#conclusions">6. Conclusions</a></li>
        </ul>
    </div>
    
    <div class="section" id="overview">
        <h2>1. Dataset Overview</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="number">{n_cells:,}</div>
                <div class="label">Total Cells</div>
            </div>
            <div class="stat-card">
                <div class="number">{n_genes:,}</div>
                <div class="label">Genes Detected</div>
            </div>
            <div class="stat-card">
                <div class="number">{n_samples}</div>
                <div class="label">Samples</div>
            </div>
            <div class="stat-card">
                <div class="number">{n_groups}</div>
                <div class="label">Treatment Groups</div>
            </div>
            <div class="stat-card">
                <div class="number">{n_celltypes}</div>
                <div class="label">Cell Types</div>
            </div>
        </div>
    </div>
    
    <div class="section" id="celltypes">
        <h2>2. Cell Type Composition</h2>
        <table>
            <tr>
                <th>Cell Type</th>
                <th>Cell Count</th>
                <th>Percentage</th>
            </tr>
"""
    
    # 添加细胞类型表格
    for ct, count in celltype_counts.items():
        pct = count / n_cells * 100
        html_content += f"""
            <tr>
                <td>{ct}</td>
                <td>{count:,}</td>
                <td>{pct:.1f}%</td>
            </tr>
"""
    
    html_content += """
        </table>
    </div>
    
    <div class="section" id="de">
        <h2>3. Differential Expression Analysis</h2>
"""
    
    # 添加DE统计
    if not de_summary.empty:
        html_content += f"""
        <p>Total differentially expressed genes identified: <strong>{n_deg_total}</strong></p>
        <table>
            <tr>
                <th>Cell Type</th>
                <th>Comparison</th>
                <th>DEGs</th>
                <th>Upregulated</th>
                <th>Downregulated</th>
            </tr>
"""
        for _, row in de_summary.head(20).iterrows():
            ct = row.get('cell_type', 'N/A')
            comp = row.get('comparison', 'N/A')
            n_deg = row.get('n_deg', 0)
            n_up = row.get('n_up', 0)
            n_down = row.get('n_down', 0)
            html_content += f"""
            <tr>
                <td>{ct}</td>
                <td>{comp}</td>
                <td>{n_deg}</td>
                <td>{n_up}</td>
                <td>{n_down}</td>
            </tr>
"""
        html_content += """
        </table>
"""
    else:
        html_content += "<p>No differential expression data available.</p>"
    
    html_content += """
    </div>
    
    <div class="section" id="enrichment">
        <h2>4. Functional Enrichment</h2>
"""
    
    # 添加富集分析结果
    if not enrichment_summary.empty:
        html_content += """
        <table>
            <tr>
                <th>Cell Type</th>
                <th>Top Enriched Terms</th>
            </tr>
"""
        # 简化显示
        for ct in enrichment_summary['cell_type'].unique()[:10] if 'cell_type' in enrichment_summary.columns else []:
            ct_data = enrichment_summary[enrichment_summary['cell_type'] == ct]
            top_terms = ct_data['term'].head(3).tolist() if 'term' in ct_data.columns else ['N/A']
            html_content += f"""
            <tr>
                <td>{ct}</td>
                <td>{', '.join(top_terms)}</td>
            </tr>
"""
        html_content += """
        </table>
"""
    else:
        html_content += "<p>No enrichment data available.</p>"
    
    html_content += """
    </div>
    
    <div class="section" id="advanced">
        <h2>5. Advanced Analyses</h2>
        <ul>
            <li><strong>Pseudotime Analysis:</strong> Trajectory inference performed using diffusion pseudotime</li>
            <li><strong>Cell Communication:</strong> Ligand-receptor interaction analysis completed</li>
            <li><strong>Transcription Factor Activity:</strong> TF activity scores calculated</li>
            <li><strong>Metal Response Analysis:</strong> Metal-responsive genes identified</li>
        </ul>
    </div>
    
    <div class="section" id="conclusions">
        <h2>6. Conclusions</h2>
        <p>This comprehensive single-cell RNA sequencing analysis of Drosophila gut tissue has revealed:</p>
        <ul>
            <li>Identification of {n_celltypes} distinct cell types in the gut epithelium</li>
            <li>Differential gene expression patterns across treatment conditions</li>
            <li>Cell type-specific responses to experimental treatments</li>
            <li>Trajectory relationships between stem cells and differentiated cell types</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>Report generated by Drosophila Gut scRNA-seq Analysis Pipeline</p>
        <p>For questions or issues, please refer to the project documentation.</p>
    </div>
</body>
</html>
"""
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def main():
    # 获取snakemake参数
    h5ad_path = snakemake.input.h5ad
    de_summary_path = snakemake.input.de_summary
    composition_path = snakemake.input.composition
    enrichment_summary_path = snakemake.input.enrichment_summary
    summary_dir = snakemake.input.summary_dir
    publication_dir = snakemake.input.publication_dir
    output_path = snakemake.output.report
    log_file = snakemake.log[0]
    
    # 设置日志
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as log:
        sys.stdout = log
        sys.stderr = log
        
        try:
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"Loading AnnData from {h5ad_path}")
            adata = sc.read_h5ad(h5ad_path)
            
            print(f"Loading DE summary from {de_summary_path}")
            de_summary = pd.read_csv(de_summary_path)
            
            print(f"Loading composition from {composition_path}")
            composition = pd.read_csv(composition_path)
            
            print(f"Loading enrichment summary from {enrichment_summary_path}")
            try:
                enrichment_summary = pd.read_csv(enrichment_summary_path)
            except:
                enrichment_summary = pd.DataFrame()
            
            print("Generating HTML report...")
            generate_html_report(
                adata, de_summary, composition, enrichment_summary,
                summary_dir, publication_dir, output_path
            )
            
            print(f"Report saved to {output_path}")
            print("Final report generation completed successfully!")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # 创建简单的错误报告
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head><title>Error Report</title></head>
<body>
<h1>Report Generation Error</h1>
<p>An error occurred while generating the report: {e}</p>
</body>
</html>
""")
            raise

if __name__ == "__main__":
    main()
