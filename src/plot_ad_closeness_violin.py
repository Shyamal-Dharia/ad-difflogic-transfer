"""AD-closeness distributions per genetic-risk group.

Panel B of the PSD feature-space figure. Plots the continuous quantity
d_CN - d_AD for every participant, so the distribution is visible rather than
only its dichotomised summary. Zero is the nearest-centroid decision boundary,
so the count above the line is the AD-nearest proportion reported in the text.

Colours match the UMAP panel from `plot_psd_umap_transfer.py`.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from train_helpers import ROOT_DIR


INPUT_CSV = ROOT_DIR / "outputs/psd_umap_45hz/psd_umap_subject_coordinates_and_distances.csv"
GROUPS = ["N", "A+P-", "A+P+"]
GROUP_COLORS = {"N": "#A8A8A8", "A+P-": "#EDC46F", "A+P+": "#B694D6"}
CONTRASTS = [("A+P+", "N"), ("A+P+", "A+P-"), ("A+P-", "N")]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
JITTER_OFFSET = -0.14
SUMMARY_OFFSET = 0.20
# Geometry and grid copied from the UMAP panel, so 11 pt text lands on the same
# number of pixels in both panels and the two frames read as one figure.
PANEL_FIGSIZE = (6.4115, 5.567)
PANEL_DPI = 180
PANEL_RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 15.5,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
}
GRID_STYLE = {"linestyle": (0, (1.5, 3.0)), "linewidth": 0.7, "color": "#d2d2d2", "alpha": 0.9}
# Significance brackets, stacked shortest span first so the widest sits on top.
BRACKET_LEVELS = {("A+P-", "N"): 0, ("A+P+", "A+P-"): 1, ("A+P+", "N"): 2}
BRACKET_BASE = 1.45   # clearance above the tallest point, for the count labels
BRACKET_STEP = 1.05
BRACKET_TICK = 0.30


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--figure-dir", type=Path, default=ROOT_DIR / "figures")
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)
    data["group"] = data["group"].replace({"A+P−": "A+P-"})
    data = data[data["group"].isin(GROUPS)].copy()
    data["closeness"] = data["distance_to_cn_centroid"] - data["distance_to_ad_centroid"]

    plt.rcParams.update(PANEL_RCPARAMS)
    figure, axis = plt.subplots(figsize=PANEL_FIGSIZE)
    rng = np.random.default_rng(7)
    axis.set_axisbelow(True)
    # Horizontal rules only: the x axis is categorical, so a vertical rule would
    # fall down the centre of each violin.
    axis.yaxis.grid(True, **GRID_STYLE)

    axis.axhline(0, color=INK_SECONDARY, linewidth=1.1, linestyle=(0, (5, 4)), zorder=2)

    group_values = {}
    for position, group in enumerate(GROUPS):
        values = data.loc[data["group"].eq(group), "closeness"].to_numpy()
        group_values[group] = values

        body = axis.violinplot(values, positions=[position], widths=0.74,
                               showextrema=False, showmedians=False)
        for part in body["bodies"]:
            part.set_facecolor(GROUP_COLORS[group])
            part.set_alpha(0.32)
            part.set_edgecolor("none")

        axis.scatter(position + JITTER_OFFSET + rng.uniform(-0.085, 0.085, values.size), values,
                     s=22, color=GROUP_COLORS[group], alpha=0.95, linewidths=0.5,
                     edgecolors="white", zorder=3)

        # Slim box beside the cloud, so the summary never crosses the points.
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        x = position + SUMMARY_OFFSET
        axis.add_patch(plt.Rectangle((x - 0.045, q25), 0.09, q75 - q25, facecolor="white",
                                     edgecolor=INK_PRIMARY, linewidth=1.0, zorder=4))
        axis.hlines(median, x - 0.045, x + 0.045, color=INK_PRIMARY, linewidth=1.8, zorder=5)

        above = int((values > 0).sum())
        axis.text(position, values.max() + 0.45, f"{above}/{values.size}", ha="center",
                  va="bottom", fontsize=13, fontweight="bold", color=INK_PRIMARY)

    axis.annotate("$\\uparrow$ nearer AD", xy=(-0.56, 0), xytext=(0, 4),
                  textcoords="offset points", fontsize=11, color=INK_SECONDARY,
                  ha="left", va="bottom")
    axis.annotate("$\\downarrow$ nearer CN", xy=(-0.56, 0), xytext=(0, -5),
                  textcoords="offset points", fontsize=11, color=INK_SECONDARY,
                  ha="left", va="top")

    # Mann-Whitney on the continuous score, Holm-corrected across the three
    # contrasts, matching the correction family used elsewhere.
    raw = [
        mannwhitneyu(
            data.loc[data["group"].eq(a), "closeness"],
            data.loc[data["group"].eq(b), "closeness"],
            alternative="two-sided",
        ).pvalue
        for a, b in CONTRASTS
    ]
    adjusted = holm_adjust(np.array(raw))

    base = max(values.max() for values in group_values.values()) + BRACKET_BASE
    for (a, b), p_value, q_value in zip(CONTRASTS, raw, adjusted):
        left, right = sorted((GROUPS.index(a), GROUPS.index(b)))
        y = base + BRACKET_LEVELS[(a, b)] * BRACKET_STEP
        axis.plot([left, left, right, right],
                  [y, y + BRACKET_TICK, y + BRACKET_TICK, y],
                  color=INK_SECONDARY, linewidth=1.0, solid_joinstyle="miter", zorder=6)
        axis.text((left + right) / 2, y + BRACKET_TICK + 0.10,
                  f"{p_value:.3f} ({q_value:.3f})", ha="center", va="bottom",
                  fontsize=11.5, color=INK_SECONDARY)

    top = base + max(BRACKET_LEVELS.values()) * BRACKET_STEP + BRACKET_TICK + 0.95
    axis.set_ylim(axis.get_ylim()[0], top)

    axis.set_xticks(range(len(GROUPS)))
    axis.set_xticklabels(GROUPS, fontweight="bold")
    axis.set_xlim(-0.6, len(GROUPS) - 0.4)
    axis.set_ylabel(r"AD-closeness ($d_{\mathrm{CN}} - d_{\mathrm{AD}}$)")
    # Full box in the same weight and colour as the UMAP panel beside it.
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    figure.tight_layout()
    output_path = args.figure_dir / "psd_centroid_panel_c_closeness_45hz.png"
    figure.savefig(output_path, dpi=PANEL_DPI, facecolor="white")
    plt.close(figure)
    print(f"Saved {output_path}")
    for group in GROUPS:
        values = data.loc[data["group"].eq(group), "closeness"]
        print(f"  {group:5s} above boundary {int((values > 0).sum())}/{len(values)}   "
              f"median {values.median():+.2f}")


if __name__ == "__main__":
    main()
