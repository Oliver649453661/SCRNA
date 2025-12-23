#!/usr/bin/env python3
"""
Generate a publication-ready pie chart summarizing the overall cluster
composition (percentage of all cells in each cluster). Saves both PDF and PNG
under results/PPT by default, matching the styling of other PPT outputs.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot overall cluster composition as a pie chart."
    )
    parser.add_argument(
        "--h5ad",
        default="results/anndata/merged_clustered.h5ad",
        help="AnnData file containing cluster annotations (default: results/anndata/merged_clustered.h5ad).",
    )
    parser.add_argument(
        "--cluster-col",
        default="leiden",
        help="Column in adata.obs that stores cluster labels (default: leiden).",
    )
    parser.add_argument(
        "--min-proportion",
        type=float,
        default=0.0,
        help="Optional threshold (0-1). Clusters below this fraction are grouped into 'Other' (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/PPT",
        help="Directory to save the resulting figures (default: results/PPT).",
    )
    parser.add_argument(
        "--basename",
        default="cluster_composition_pie",
        help="Base filename for outputs (default: cluster_composition_pie).",
    )
    return parser.parse_args()


PALETTE = [
    "#264653",
    "#287271",
    "#2a9d8f",
    "#8ab17d",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#b8336a",
    "#6a4c93",
    "#4361ee",
    "#3a0ca3",
    "#14213d",
    "#720026",
    "#9381ff",
    "#ff686b",
    "#ffb703",
]


def build_color_cycle(n):
    repeats = (n + len(PALETTE) - 1) // len(PALETTE)
    return (PALETTE * repeats)[:n]


def load_cluster_counts(adata, cluster_col, min_prop):
    if cluster_col not in adata.obs:
        raise KeyError(f"Column '{cluster_col}' not found in AnnData.obs.")

    counts = adata.obs[cluster_col].astype(str).value_counts()
    total = counts.sum()
    props = counts / total

    if min_prop > 0:
        small_mask = props < min_prop
        if small_mask.any():
            other_count = counts[small_mask].sum()
            counts = counts[~small_mask]
            if other_count > 0:
                counts.loc["Other"] = other_count

    counts = counts.sort_values(ascending=False)
    props = counts / counts.sum()
    return counts, props


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, f"{args.basename}.pdf")
    png_path = os.path.join(args.output_dir, f"{args.basename}.png")

    print(f"Loading AnnData from {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    counts, props = load_cluster_counts(adata, args.cluster_col, args.min_proportion)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "figure.dpi": 300,
        }
    )

    colors = build_color_cycle(len(counts))

    fig = plt.figure(figsize=(10.2, 6.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.6, 0.8], wspace=0.18)
    ax_donut = fig.add_subplot(grid[0, 0])
    ax_legend = fig.add_subplot(grid[0, 1])

    wedges, _ = ax_donut.pie(
        counts.values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white", "width": 0.55},
    )
    ax_donut.set_title("Cluster Composition (All Cells)")
    ax_donut.set(aspect="equal")
    ax_donut.text(
        0,
        0,
        f"n = {counts.sum():,}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#333333",
    )

    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")

    n_clusters = len(counts)
    if n_clusters <= 10:
        n_cols = 1
    elif n_clusters <= 20:
        n_cols = 2
    else:
        n_cols = 3
    n_rows = int(np.ceil(n_clusters / n_cols))
    row_span = 0.82
    col_width = 0.82 / n_cols
    column_gap = 0.06
    row_gap = row_span / max(n_rows - 1, 1) if n_rows > 1 else 0

    for idx, (label, prop, count, color) in enumerate(
        zip(counts.index, props.values, counts.values, colors)
    ):
        col_idx = idx // n_rows
        row_idx = idx % n_rows
        y = 0.91 - row_idx * row_gap if n_rows > 1 else 0.5
        x = 0.05 + col_idx * (col_width + column_gap)

        ax_legend.scatter(
            x,
            y,
            s=230,
            color=color,
            marker="o",
            edgecolor="white",
            linewidth=1.2,
        )
        ax_legend.text(
            x + 0.07,
            y + 0.02,
            label,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="semibold",
            color="#1f1f1f",
        )
        ax_legend.text(
            x + 0.07,
            y - 0.02,
            f"{prop * 100:.1f}% ({count:,})",
            ha="left",
            va="center",
            fontsize=7.6,
            color="#5c5c5c",
        )
        ax_legend.axhline(
            y - 0.055,
            color="#d0d0d0",
            linewidth=0.4,
            xmin=x + 0.07,
            xmax=min(x + col_width - 0.05, 0.92),
        )

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(
        "Saved cluster composition pie chart to:\n"
        f"  PDF: {pdf_path}\n"
        f"  PNG: {png_path}"
    )


if __name__ == "__main__":
    main()
