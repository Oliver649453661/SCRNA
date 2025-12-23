#!/usr/bin/env python3
"""
Standalone script to create a publication-ready stacked bar chart showing
cell-type composition differences and gut-region composition differences
across four groups (or a user-specified subset and order).

Inputs:
    - Cell composition CSV produced by the workflow (columns: group, cell_type, proportion)
    - Annotated AnnData (h5ad) containing gut region annotations

Outputs:
    - PDF and PNG saved to results/PPT (created automatically)
      filenames default to composition_barplot.(pdf|png)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


def set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "figure.dpi": 300,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate PPT-ready stacked bar charts for cell type and gut region composition across groups."
        )
    )
    parser.add_argument(
        "--composition",
        default="results/composition/cell_proportions.csv",
        help="Path to cell composition CSV (default: results/composition/cell_proportions.csv).",
    )
    parser.add_argument(
        "--h5ad",
        default="results/annotation/gut_region_annotated.h5ad",
        help="Annotated h5ad file that contains gut region annotations (default: results/annotation/gut_region_annotated.h5ad).",
    )
    parser.add_argument(
        "--group-col",
        default="group",
        help="Column name in AnnData.obs describing treatment group (default: group).",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Specific groups to include (max 4). Defaults to Control, PS-NPs, Cd, Cd-PS-NPs if present, else top four by abundance.",
    )
    parser.add_argument(
        "--region-col",
        default="gut_region",
        help="Column name in AnnData.obs for gut region annotation (default: gut_region).",
    )
    parser.add_argument(
        "--top-celltypes",
        type=int,
        default=12,
        help="Number of most abundant cell types to display (default: 12). Remaining cell types will be grouped as 'Others'.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/PPT",
        help="Directory to store the resulting figure (default: results/PPT).",
    )
    parser.add_argument(
        "--basename",
        default="composition_barplot",
        help="Base filename for outputs (default: composition_barplot).",
    )
    return parser.parse_args()


def determine_group_order(df_groups, preferred=None, max_groups=4):
    if preferred:
        ordered = [g for g in preferred if g in df_groups]
        if ordered:
            return ordered[:max_groups]
    default_order = ["Control", "PS-NPs", "Cd", "Cd-PS-NPs"]
    ordered = [g for g in default_order if g in df_groups]
    if len(ordered) < max_groups:
        remaining = [g for g in df_groups if g not in ordered]
        ordered.extend(remaining)
    return ordered[:max_groups]


def load_and_prepare_celltype_data(path, groups=None, top_celltypes=12):
    df = pd.read_csv(path)
    required_cols = {"group", "cell_type", "proportion"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in composition file: {missing}")

    all_groups = df["group"].unique().tolist()
    group_order = determine_group_order(all_groups, groups)
    df = df[df["group"].isin(group_order)]
    if df.empty:
        raise ValueError("Filtered cell-type composition dataframe is empty. Check group names.")

    top_ct = (
        df.groupby("cell_type")["proportion"]
        .mean()
        .sort_values(ascending=False)
        .head(top_celltypes)
        .index.tolist()
    )
    df["cell_type_plot"] = df["cell_type"].where(df["cell_type"].isin(top_ct), "Others")

    collapsed = (
        df.groupby(["group", "cell_type_plot"])["proportion"]
        .mean()
        .reset_index()
    )

    pivot = (
        collapsed.pivot(
            index="group", columns="cell_type_plot", values="proportion"
        )
        .fillna(0)
        .reindex(group_order)
    )

    pivot = pivot[
        pivot.mean().sort_values(ascending=False).index.tolist()
    ]

    return pivot, group_order


def load_and_prepare_region_data(h5ad_path, group_col, region_col, group_order):
    adata = sc.read_h5ad(h5ad_path)
    for col in (group_col, region_col):
        if col not in adata.obs:
            raise KeyError(f"Column '{col}' not found in AnnData.obs.")

    df = adata.obs[[group_col, region_col]].copy()
    df = df[df[group_col].isin(group_order)]
    if df.empty:
        raise ValueError("Filtered AnnData obs is empty for selected groups.")

    counts = (
        df.groupby([group_col, region_col])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(group_col)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals

    pivot = counts.pivot(
        index=group_col, columns=region_col, values="proportion"
    ).fillna(0)

    region_order = [
        "Crop",
        "R0",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "Hindgut",
        "Uncertain",
    ]
    cols = [c for c in region_order if c in pivot.columns] + [
        c for c in pivot.columns if c not in region_order
    ]

    return pivot.reindex(group_order).loc[:, cols]


def plot_stacked_bar(ax, pivot_df, title):
    set_publication_style()
    colors = plt.cm.tab20.colors
    color_cycle = colors * ((len(pivot_df.columns) // len(colors)) + 1)

    x = np.arange(len(pivot_df.index))
    bottom = np.zeros(len(pivot_df.index))

    for idx, col in enumerate(pivot_df.columns):
        values = pivot_df[col].values
        color = "#b5b5b5" if col.lower() == "uncertain" else color_cycle[idx]
        ax.bar(x, values, bottom=bottom, color=color, edgecolor="white", linewidth=0.7, label=col)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=20, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Category")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    celltype_pdf_path = os.path.join(args.output_dir, f"{args.basename}_celltype.pdf")
    celltype_png_path = os.path.join(args.output_dir, f"{args.basename}_celltype.png")
    region_pdf_path = os.path.join(args.output_dir, f"{args.basename}_region.pdf")
    region_png_path = os.path.join(args.output_dir, f"{args.basename}_region.png")

    celltype_df, group_order = load_and_prepare_celltype_data(
        args.composition, args.groups, args.top_celltypes
    )
    region_df = load_and_prepare_region_data(
        args.h5ad, args.group_col, args.region_col, group_order
    )

    print(f"Groups included (ordered): {group_order}")
    print(f"Cell types displayed: {list(celltype_df.columns)}")
    print(f"Regions displayed: {list(region_df.columns)}")

    set_publication_style()
    fig_celltype, ax_celltype = plt.subplots(figsize=(8, 7))
    plot_stacked_bar(ax_celltype, celltype_df, "Cell-Type Composition")
    fig_celltype.tight_layout()
    fig_celltype.savefig(celltype_pdf_path, bbox_inches="tight", dpi=300)
    fig_celltype.savefig(celltype_png_path, bbox_inches="tight", dpi=300)
    plt.close(fig_celltype)
    print(f"Saved cell-type stacked bar chart to: {celltype_pdf_path} and {celltype_png_path}")

    fig_region, ax_region = plt.subplots(figsize=(8, 7))
    plot_stacked_bar(ax_region, region_df, "Gut Region Composition")
    fig_region.tight_layout()
    fig_region.savefig(region_pdf_path, bbox_inches="tight", dpi=300)
    fig_region.savefig(region_png_path, bbox_inches="tight", dpi=300)
    plt.close(fig_region)
    print(f"Saved gut-region stacked bar chart to: {region_pdf_path} and {region_png_path}")


if __name__ == "__main__":
    main()

