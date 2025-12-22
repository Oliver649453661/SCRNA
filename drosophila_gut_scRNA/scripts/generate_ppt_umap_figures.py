#!/usr/bin/env python3
"""
Generate a publication-ready composite figure for PPT/report usage:
- Panel A: UMAP colored by gut region annotation
- Panel B: UMAP colored by final cell type annotation
Outputs both PDF and PNG into results/PPT (created if missing).
This script is standalone and does not integrate with the Snakemake workflow.
"""

import argparse
import os
import scanpy as sc
import matplotlib.pyplot as plt


def set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "figure.dpi": 300,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PPT-ready UMAP figure.")
    parser.add_argument(
        "--h5ad",
        required=True,
        help="Path to the annotated AnnData (should include UMAP, gut_region, final cell type).",
    )
    parser.add_argument(
        "--celltype-col",
        default="final_cell_type",
        help="Column name in adata.obs for cell type annotation (default: final_cell_type).",
    )
    parser.add_argument(
        "--region-col",
        default="gut_region",
        help="Column name in adata.obs for gut region annotation (default: gut_region).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/PPT",
        help="Directory to save the figure (default: results/PPT).",
    )
    parser.add_argument(
        "--basename",
        default="umap_region_celltype",
        help="Base filename for outputs (default: umap_region_celltype).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, f"{args.basename}.pdf")
    png_path = os.path.join(args.output_dir, f"{args.basename}.png")

    print(f"Loading AnnData from {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"AnnData shape: {adata.shape}")

    if "X_umap" not in adata.obsm:
        raise ValueError("AnnData missing UMAP coordinates (X_umap). Please run sc.tl.umap first.")

    if args.celltype_col not in adata.obs:
        raise KeyError(f"Cell type column '{args.celltype_col}' not found in adata.obs.")

    if args.region_col not in adata.obs:
        raise KeyError(f"Gut region column '{args.region_col}' not found in adata.obs.")

    set_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    region_palette = {
        "Crop": "#e41a1c",
        "R0": "#ff7f00",
        "R1": "#8dd3c7",
        "R2": "#4daf4a",
        "R3": "#377eb8",
        "R4": "#984ea3",
        "R5": "#f781bf",
        "Hindgut": "#a65628",
        "Uncertain": "#999999",
    }

    print("Plotting UMAP colored by gut region...")
    sc.pl.umap(
        adata,
        color=args.region_col,
        ax=axes[0],
        show=False,
        frameon=False,
        title="UMAP · Gut Region Annotation",
        palette=region_palette,
    )

    print("Plotting UMAP colored by cell type...")
    sc.pl.umap(
        adata,
        color=args.celltype_col,
        ax=axes[1],
        show=False,
        frameon=False,
        title="UMAP · Cell Type Annotation",
    )

    for ax in axes:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    plt.tight_layout()

    print(f"Saving figure to {pdf_path} and {png_path}")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print("PPT UMAP figure generation completed.")


if __name__ == "__main__":
    main()
