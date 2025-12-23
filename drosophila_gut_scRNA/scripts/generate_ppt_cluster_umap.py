UMAP_WIDTH_IN = 6.5
UMAP_HEIGHT_IN = 6.0
MARGIN_LEFT_IN = 0.5
MARGIN_RIGHT_IN = 0.4
MARGIN_TOP_IN = 0.5
MARGIN_BOTTOM_IN = 0.6
LEGEND_GAP_IN = 0.35
LEGEND_BASE_WIDTH_IN = 1.9
LEGEND_WIDTH_PER_CHAR = 0.03

#!/usr/bin/env python3
"""
Generate high-quality UMAP cluster visuals for PPT/report usage.
Outputs two figures (embedding + annotation legend) with matching dimensions
under results/PPT (created if missing).
"""

import argparse
import os
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import scanpy as sc

DEFAULT_FIGSIZE = (9, 8)
DEFAULT_GROUPS = ["Control", "Cd", "PS-NPs", "Cd-PS-NPs"]
DEFAULT_TOP_MARKERS = 3
DEFAULT_LEGEND_STYLE = {
    "bbox_to_anchor": (1.01, 0.5),
    "fontsize": 10,
    "columnspacing": 0.55,
    "handlelength": 0.9,
    "handletextpad": 0.3,
    "labelspacing": 0.5,
    "borderaxespad": 0.35,
    "ncol": 1,
}


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
    parser = argparse.ArgumentParser(
        description="Generate PPT-ready UMAP colored by cluster annotation."
    )
    parser.add_argument(
        "--h5ad",
        default="results/anndata/merged_clustered.h5ad",
        help="Path to the clustered AnnData (default: results/anndata/merged_clustered.h5ad).",
    )
    parser.add_argument(
        "--cluster-col",
        default="leiden",
        help="Column in adata.obs containing cluster labels (default: leiden).",
    )
    parser.add_argument(
        "--group-col",
        default="group",
        help="Column in adata.obs containing treatment group labels (default: group).",
    )
    parser.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUPS),
        help=(
            "Comma-separated list of groups to plot individually "
            '(default: "Control,Cd,PS-NPs,Cd-PS-NPs").'
        ),
    )
    parser.add_argument(
        "--markers-csv",
        default="results/markers/cluster_markers.csv",
        help="CSV containing cluster marker genes (default: results/markers/cluster_markers.csv).",
    )
    parser.add_argument(
        "--top-markers",
        type=int,
        default=DEFAULT_TOP_MARKERS,
        help=f"Number of marker genes to show per cluster (default: {DEFAULT_TOP_MARKERS}).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/PPT",
        help="Directory to save figures (default: results/PPT).",
    )
    parser.add_argument(
        "--basename",
        default="umap_clusters",
        help="Base filename for outputs (default: umap_clusters).",
    )
    return parser.parse_args()


def ensure_cluster_column(adata, cluster_col):
    if cluster_col not in adata.obs:
        raise KeyError(f"Cluster column '{cluster_col}' not found in adata.obs.")
    adata.obs[cluster_col] = adata.obs[cluster_col].astype("category")


def ensure_group_column(adata, group_col):
    if group_col not in adata.obs:
        raise KeyError(f"Group column '{group_col}' not found in adata.obs.")
    return adata.obs[group_col].astype(str).unique().tolist()


def load_palette(adata, cluster_col, categories: List[str]):
    uns_key = f"{cluster_col}_colors"
    palette = adata.uns.get(uns_key)
    if palette is not None and len(palette) >= len(categories):
        return list(palette)
    base_palette = sc.pl.palettes.default_102
    if len(categories) > len(base_palette):
        raise ValueError(
            f"Cluster count ({len(categories)}) exceeds available palette colors ({len(base_palette)})."
        )
    return base_palette[: len(categories)]


def load_marker_annotations(csv_path: str, top_n: int) -> Dict[str, List[str]]:
    if not csv_path or not os.path.exists(csv_path):
        print(f"Marker file not found at {csv_path}, skipping marker annotations.")
        return {}

    df = pd.read_csv(csv_path)
    if "group" not in df.columns:
        print(f"Marker file {csv_path} missing 'group' column, skipping markers.")
        return {}

    if "gene_symbol" not in df.columns:
        df["gene_symbol"] = df.get("names", "")

    marker_dict: Dict[str, List[str]] = {}
    for cluster_label, sub_df in df.groupby("group"):
        top = (
            sub_df.sort_values("scores", ascending=False)
            .head(top_n)
            .copy()
        )
        genes = [str(g) for g in top["gene_symbol"].fillna(top["names"]).tolist()]
        marker_dict[str(cluster_label)] = genes
    return marker_dict


def build_legend_entries(
    categories: List[str],
    palette: List[str],
    marker_dict: Dict[str, List[str]],
    available: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    entries = []
    available_set = set(available) if available is not None else None
    for idx, cat in enumerate(categories):
        if available_set is not None and cat not in available_set:
            continue
        color = palette[idx]
        markers = marker_dict.get(str(cat)) or marker_dict.get(cat) or []
        label = f"{cat}"
        if markers:
            label = f"{label} · {', '.join(markers)}"
        entries.append({"label": label, "color": color})
    return entries


def compute_layout(legend_entries: List[Dict[str, str]]):
    max_label_len = max((len(entry["label"]) for entry in legend_entries), default=8)
    legend_width_in = LEGEND_BASE_WIDTH_IN + LEGEND_WIDTH_PER_CHAR * max_label_len
    legend_width_in = max(LEGEND_BASE_WIDTH_IN, min(legend_width_in, 4.5))

    fig_width = (
        MARGIN_LEFT_IN
        + UMAP_WIDTH_IN
        + LEGEND_GAP_IN
        + legend_width_in
        + MARGIN_RIGHT_IN
    )
    fig_height = MARGIN_TOP_IN + UMAP_HEIGHT_IN + MARGIN_BOTTOM_IN

    umap_left = MARGIN_LEFT_IN / fig_width
    umap_bottom = MARGIN_BOTTOM_IN / fig_height
    umap_width = UMAP_WIDTH_IN / fig_width
    umap_height = UMAP_HEIGHT_IN / fig_height

    legend_left = (MARGIN_LEFT_IN + UMAP_WIDTH_IN + LEGEND_GAP_IN) / fig_width
    legend_width = legend_width_in / fig_width
    legend_height = umap_height
    legend_bottom = umap_bottom

    return {
        "fig_size": (fig_width, fig_height),
        "umap_rect": [umap_left, umap_bottom, umap_width, umap_height],
        "legend_rect": [legend_left, legend_bottom, legend_width, legend_height],
    }


def save_cluster_umap(
    adata,
    cluster_col,
    categories,
    palette_map,
    legend_entries,
    pdf_path,
    png_path,
    title="UMAP · Cluster Embedding",
    legend_style: Optional[Dict] = None,
):
    legend_style = legend_style or DEFAULT_LEGEND_STYLE
    adata_plot = adata.copy()
    palette_for_plot = [
        palette_map.get(cat, "#999999")
        for cat in adata_plot.obs[cluster_col].cat.categories
    ]

    layout = compute_layout(legend_entries)
    fig = plt.figure(figsize=layout["fig_size"])
    ax = fig.add_axes(layout["umap_rect"])
    legend_ax = fig.add_axes(layout["legend_rect"])
    legend_ax.axis("off")

    sc.pl.umap(
        adata_plot,
        color=cluster_col,
        ax=ax,
        show=False,
        frameon=False,
        title=title,
        palette=palette_for_plot,
        legend_loc=None,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=entry["color"],
            markeredgecolor="white",
            label=entry["label"],
        )
        for entry in legend_entries
    ]

    if not handles:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor="#555555",
                markeredgecolor="white",
                label="No clusters detected",
            )
        ]

    legend_ax.legend(
        handles,
        [h.get_label() for h in handles],
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        frameon=False,
        ncol=legend_style.get("ncol", 1),
        fontsize=legend_style.get("fontsize", 10),
        columnspacing=legend_style.get("columnspacing", 0.6),
        handlelength=legend_style.get("handlelength", 1.0),
        handletextpad=legend_style.get("handletextpad", 0.35),
        labelspacing=legend_style.get("labelspacing", 0.6),
        borderaxespad=legend_style.get("borderaxespad", 0.35),
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    embed_pdf = os.path.join(args.output_dir, f"{args.basename}_embedding.pdf")
    embed_png = os.path.join(args.output_dir, f"{args.basename}_embedding.png")

    print(f"Loading AnnData from {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"AnnData shape: {adata.shape}")

    if "X_umap" not in adata.obsm:
        raise ValueError("AnnData missing UMAP coordinates (X_umap). Please run sc.tl.umap first.")

    ensure_cluster_column(adata, args.cluster_col)
    unique_groups = ensure_group_column(adata, args.group_col)
    requested_groups = [
        g.strip() for g in args.groups.split(",") if g.strip()
    ] or unique_groups
    marker_annotations = load_marker_annotations(args.markers_csv, args.top_markers)

    categories = adata.obs[args.cluster_col].cat.categories.tolist()
    palette = load_palette(adata, args.cluster_col, categories)
    palette_map = dict(zip(categories, palette))

    set_publication_style()

    print("Saving UMAP embedding...")
    legend_entries_all = build_legend_entries(categories, palette, marker_annotations)
    save_cluster_umap(
        adata,
        args.cluster_col,
        categories,
        palette_map,
        legend_entries_all,
        embed_pdf,
        embed_png,
        legend_style=DEFAULT_LEGEND_STYLE,
    )

    per_group_outputs = []
    for group in requested_groups:
        if group not in unique_groups:
            print(f"  Skipping group '{group}' (not found in '{args.group_col}').")
            continue

        mask = adata.obs[args.group_col] == group
        if mask.sum() == 0:
            print(f"  Skipping group '{group}' (no cells).")
            continue

        adata_group = adata[mask].copy()
        present_categories = [
            cat for cat in categories if (adata_group.obs[args.cluster_col] == cat).any()
        ]
        if not present_categories:
            print(f"  Skipping group '{group}' (no clusters).")
            continue

        legend_entries_group = build_legend_entries(
            categories,
            palette,
            marker_dict={},
            available=present_categories,
        )

        group_pdf = os.path.join(
            args.output_dir, f"{args.basename}_{group}_embedding.pdf"
        )
        group_png = os.path.join(
            args.output_dir, f"{args.basename}_{group}_embedding.png"
        )

        print(f"Saving group-specific UMAP for {group}...")
        save_cluster_umap(
            adata_group,
            args.cluster_col,
            categories,
            palette_map,
            legend_entries_group,
            group_pdf,
            group_png,
            title=f"UMAP · {group}",
            legend_style=DEFAULT_LEGEND_STYLE,
        )
        per_group_outputs.append((group, group_pdf, group_png))

    print(
        "Saved cluster figures to:\n"
        f"  Combined UMAP: {embed_pdf} / {embed_png}"
    )
    if per_group_outputs:
        print("  Group-specific UMAPs:")
        for group, pdf_path, png_path in per_group_outputs:
            print(f"    {group}: {pdf_path} / {png_path}")


if __name__ == "__main__":
    main()
