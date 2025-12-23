#!/usr/bin/env python3
"""
Generate publication-ready UMAP figures for PPT/report usage:
- One image colored by gut region annotation
- One image colored by final cell type annotation
Each image is saved separately as PDF and PNG under results/PPT (created if missing).
This script is standalone and does not integrate with the Snakemake workflow.
"""

import argparse
import os
import scanpy as sc
import matplotlib.pyplot as plt

DEFAULT_FIGSIZE = (10, 8)


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


def save_single_umap(
    adata,
    *,
    color,
    title,
    pdf_path,
    png_path,
    palette=None,
    figure_size=DEFAULT_FIGSIZE,
    legend_kwargs=None,
):
    if legend_kwargs is None:
        legend_kwargs = {}

    fig, ax = plt.subplots(figsize=figure_size)

    sc.pl.umap(
        adata,
        color=color,
        ax=ax,
        show=False,
        frameon=False,
        title=title,
        palette=palette,
        legend_loc="right margin",
    )

    legend = ax.legend_
    if legend is not None:
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()
        ax.legend(
            handles,
            labels,
            loc=legend_kwargs.get("loc", "center left"),
            bbox_to_anchor=legend_kwargs.get("bbox_to_anchor", (1.02, 0.5)),
            borderaxespad=legend_kwargs.get("borderaxespad", 0.4),
            frameon=False,
            ncol=legend_kwargs.get("ncol", 1),
            fontsize=legend_kwargs.get("fontsize", 11),
            columnspacing=legend_kwargs.get("columnspacing", 0.8),
            handlelength=legend_kwargs.get("handlelength", 1.4),
            handletextpad=legend_kwargs.get("handletextpad", 0.4),
            labelspacing=legend_kwargs.get("labelspacing", 0.8),
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()

    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


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
    region_pdf_path = os.path.join(args.output_dir, f"{args.basename}_region.pdf")
    region_png_path = os.path.join(args.output_dir, f"{args.basename}_region.png")
    celltype_pdf_path = os.path.join(args.output_dir, f"{args.basename}_celltype.pdf")
    celltype_png_path = os.path.join(args.output_dir, f"{args.basename}_celltype.png")

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
    save_single_umap(
        adata,
        color=args.region_col,
        title="UMAP · Gut Region Annotation",
        pdf_path=region_pdf_path,
        png_path=region_png_path,
        palette=region_palette,
        legend_kwargs={
            "loc": "center left",
            "bbox_to_anchor": (1.02, 0.5),
            "fontsize": 11,
            "columnspacing": 0.7,
            "handlelength": 1.2,
            "handletextpad": 0.35,
            "labelspacing": 0.5,
        },
    )

    print("Plotting UMAP colored by cell type...")
    save_single_umap(
        adata,
        color=args.celltype_col,
        title="UMAP · Cell Type Annotation",
        pdf_path=celltype_pdf_path,
        png_path=celltype_png_path,
        legend_kwargs={
            "loc": "center left",
            "bbox_to_anchor": (1.02, 0.5),
            "fontsize": 10,
            "columnspacing": 0.5,
            "handlelength": 1.0,
            "handletextpad": 0.3,
            "labelspacing": 0.45,
            "ncol": 1,
        },
    )

    print(
        "Saved separate figures to:\n"
        f"  Region:  {region_pdf_path} / {region_png_path}\n"
        f"  Celltype:{celltype_pdf_path} / {celltype_png_path}"
    )

    print("PPT UMAP figure generation completed.")


if __name__ == "__main__":
    main()
