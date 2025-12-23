#!/usr/bin/env python3
"""
Create publication-ready bar plots showing (1) gut region composition changes
and (2) cell-type composition changes of multiple treatment groups relative to
a control group, with significance annotations directly on the figure.

Outputs a single PDF and PNG (stacked panels) saved to results/PPT.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.stats.multitest import multipletests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot region & cell-type composition changes vs control with significance."
    )
    parser.add_argument(
        "--h5ad",
        default="results/annotation/gut_region_annotated.h5ad",
        help="Annotated AnnData file containing gut region annotations.",
    )
    parser.add_argument(
        "--group-col",
        default="group",
        help="Column name in adata.obs for treatment groups.",
    )
    parser.add_argument(
        "--sample-col",
        default="sample",
        help="Column name in adata.obs for biological sample / replicate.",
    )
    parser.add_argument(
        "--region-col",
        default="gut_region",
        help="Column name in adata.obs for gut region annotations.",
    )
    parser.add_argument(
        "--celltype-col",
        default="final_cell_type",
        help="Column name in adata.obs for cell type annotations.",
    )
    parser.add_argument(
        "--control-group",
        default="Control",
        help="Control group label.",
    )
    parser.add_argument(
        "--comparison-groups",
        nargs="*",
        default=["PS-NPs", "Cd", "Cd-PS-NPs"],
        help="Treatment groups to compare against control.",
    )
    parser.add_argument(
        "--top-celltypes",
        type=int,
        default=4,
        help="Number of top cell types (by mean proportion) to display (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/PPT",
        help="Directory to store the resulting figure.",
    )
    parser.add_argument(
        "--basename",
        default="region_change_vs_control",
        help="Base filename for the figure outputs.",
    )
    return parser.parse_args()


def compute_category_proportions(adata, group_col, category_col):
    df = adata.obs[[group_col, category_col]].copy()
    counts = (
        df.groupby([group_col, category_col])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(group_col)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals

    summary = counts.pivot(
        index=group_col, columns=category_col, values="proportion"
    ).fillna(0)
    return summary


def aggregate_counts_by_sample(adata, sample_col, group_col, category_col):
    df = adata.obs[[sample_col, group_col, category_col]].copy()
    totals = df.groupby([sample_col, group_col]).size().reset_index(name="total")
    counts = (
        df.groupby([sample_col, group_col, category_col])
        .size()
        .reset_index(name="count")
        .rename(columns={category_col: "category"})
    )
    return counts, totals


def build_glm_data(counts_df, totals_df, category, sample_col, group_col):
    cat_counts = counts_df[counts_df["category"] == category][[sample_col, group_col, "count"]]
    data = totals_df.merge(cat_counts, on=[sample_col, group_col], how="left")
    data["count"] = data["count"].fillna(0)
    data = data[data["total"] > 0]
    return data


def compute_glm_pvals(counts_df, totals_df, categories, sample_col, group_col, control_group, fdr=True):
    pvals = {category: {} for category in categories}
    formula = f"success_rate ~ C(group_for_glm, Treatment(reference='{control_group}'))"

    for category in categories:
        data = build_glm_data(counts_df, totals_df, category, sample_col, group_col)
        if data.empty or data[group_col].nunique() < 2:
            continue

        glm_data = data.rename(columns={group_col: "group_for_glm"}).copy()
        glm_data["success_rate"] = glm_data["count"] / glm_data["total"]
        if glm_data["group_for_glm"].nunique() < 2:
            continue

        try:
            model = smf.glm(
                formula=formula,
                data=glm_data,
                family=Binomial(),
                freq_weights=glm_data["total"],
            )
            result = model.fit()
            for term, pval in result.pvalues.items():
                if term.startswith("C(group_for_glm"):
                    group_name = term.split("T.")[-1].rstrip("]")
                    pvals[category][group_name] = pval
        except Exception:
            continue

    if not fdr:
        return pvals

    records = []
    for category, group_dict in pvals.items():
        for group, pval in group_dict.items():
            records.append({"category": category, "group": group, "pval": pval})

    if not records:
        return pvals

    df = pd.DataFrame(records)
    _, adj_pvals, _, _ = multipletests(df["pval"], method="fdr_bh")
    df["pval_adj"] = adj_pvals

    adj_dict = {category: {} for category in categories}
    for _, row in df.iterrows():
        adj_dict[row["category"]][row["group"]] = row["pval_adj"]
    return adj_dict


def significance_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    region_pdf_path = os.path.join(args.output_dir, f"{args.basename}_region.pdf")
    region_png_path = os.path.join(args.output_dir, f"{args.basename}_region.png")
    celltype_pdf_path = os.path.join(args.output_dir, f"{args.basename}_celltype.pdf")
    celltype_png_path = os.path.join(args.output_dir, f"{args.basename}_celltype.png")

    print(f"Loading AnnData from {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)

    for col in (args.group_col, args.region_col):
        if col not in adata.obs:
            raise KeyError(f"Column '{col}' not found in AnnData.obs.")

    for col in (args.celltype_col,):
        if col not in adata.obs:
            raise KeyError(f"Column '{col}' not found in AnnData.obs.")

    region_summary = compute_category_proportions(
        adata, args.group_col, args.region_col
    )
    celltype_summary = compute_category_proportions(
        adata, args.group_col, args.celltype_col
    )
    region_counts_sample, region_totals_sample = aggregate_counts_by_sample(
        adata, args.sample_col, args.group_col, args.region_col
    )
    celltype_counts_sample, celltype_totals_sample = aggregate_counts_by_sample(
        adata, args.sample_col, args.group_col, args.celltype_col
    )

    control = args.control_group
    if control not in region_summary.index:
        raise ValueError(f"Control group '{control}' not found in data.")

    comparison_groups = [g for g in args.comparison_groups if g in region_summary.index]
    if not comparison_groups:
        raise ValueError("No comparison groups found in data.")

    region_order = ["R0", "R1", "R2", "R3", "R4", "R5"]
    regions = [r for r in region_order if r in region_summary.columns]

    def compute_diffs(summary, categories):
        ctrl_props = summary.loc[control, categories].fillna(0)

        diffs_list = []
        for group in comparison_groups:
            grp_props = summary.loc[group, categories].fillna(0)
            diff_pct = (grp_props - ctrl_props) * 100
            diffs_list.append(diff_pct)

        return np.array(diffs_list)

    region_diffs = compute_diffs(region_summary, regions)
    region_glm_pvals = compute_glm_pvals(
        region_counts_sample, region_totals_sample, regions,
        args.sample_col, args.group_col, control
    )

    # Select top cell types
    mean_celltype = celltype_summary.mean(axis=0)
    top_celltypes = (
        mean_celltype.sort_values(ascending=False)
        .head(args.top_celltypes)
        .index.tolist()
    )
    celltype_diffs = compute_diffs(celltype_summary, top_celltypes)
    celltype_glm_pvals = compute_glm_pvals(
        celltype_counts_sample, celltype_totals_sample, top_celltypes,
        args.sample_col, args.group_col, control
    )

    colors = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c"]

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

    def plot_panel(ax, categories, diffs, pvals_dict, title, show_legend=True):
        n_cat = len(categories)
        n_groups = len(comparison_groups)
        indices = np.arange(n_cat)
        bar_width = min(0.2, 0.6 / n_groups)
        offsets = np.linspace(
            -(n_groups - 1) * bar_width / 2,
            (n_groups - 1) * bar_width / 2,
            n_groups,
        )

        for idx, group in enumerate(comparison_groups):
            bars = ax.bar(
                indices + offsets[idx],
                diffs[idx],
                width=bar_width,
                color=colors[idx % len(colors)],
                label=group if show_legend else None,
            )
            for j, cat in enumerate(categories):
                star = significance_label(pvals_dict.get(cat, {}).get(group, 1.0))
                y = diffs[idx, j]
                va = "bottom" if y >= 0 else "top"
                offset = 0.1 if y >= 0 else -0.1
                ax.text(
                    indices[j] + offsets[idx],
                    y + offset,
                    star,
                    ha="center",
                    va=va,
                    fontsize=9,
                )

        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(indices)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.set_ylabel("Change vs Control (%)")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if show_legend:
            ax.legend(loc="upper right")

    fig_region, ax_region = plt.subplots(figsize=(10, 6))
    plot_panel(
        ax_region,
        regions,
        region_diffs,
        region_glm_pvals,
        "Gut Region Distribution Change vs Control",
        show_legend=True,
    )
    fig_region.tight_layout()
    fig_region.savefig(region_pdf_path, bbox_inches="tight", dpi=300)
    fig_region.savefig(region_png_path, bbox_inches="tight", dpi=300)
    plt.close(fig_region)
    print(f"Saved gut region change plot to: {region_pdf_path} and {region_png_path}")

    fig_celltype, ax_celltype = plt.subplots(figsize=(10, 6))
    plot_panel(
        ax_celltype,
        top_celltypes,
        celltype_diffs,
        celltype_glm_pvals,
        "Cell-Type Composition Change vs Control",
        show_legend=True,
    )
    fig_celltype.tight_layout()
    fig_celltype.savefig(celltype_pdf_path, bbox_inches="tight", dpi=300)
    fig_celltype.savefig(celltype_png_path, bbox_inches="tight", dpi=300)
    plt.close(fig_celltype)
    print(f"Saved cell-type change plot to: {celltype_pdf_path} and {celltype_png_path}")


if __name__ == "__main__":
    main()
