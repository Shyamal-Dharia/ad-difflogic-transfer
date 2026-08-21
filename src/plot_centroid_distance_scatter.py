"""Subject distances to the clinical CN and AD centroids.

Each participant is plotted by distance to the clinical AD centroid (x) against
distance to the clinical CN centroid (y), both in the standardized
75-dimensional PSD space. The identity line is the nearest-centroid decision
boundary: points above it are closer to AD, points below are closer to CN.

The two axes are independent measurements, unlike closeness scores of the form
d_CN - d_disease, which share a common term and are correlated by construction.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, theilslopes
from sklearn.linear_model import HuberRegressor

from train_helpers import ROOT_DIR


INPUT_CSV = ROOT_DIR / "outputs/psd_umap_45hz/psd_umap_subject_coordinates_and_distances.csv"
OUTPUT_DIR = ROOT_DIR / "outputs/psd_umap_45hz"

# Validated all-pairs on the light surface (worst CVD dE 9.2, normal-vision 24.0).
GROUP_COLORS = {"N": "#2a78d6", "A+P-": "#eb6834", "A+P+": "#1baf7a"}
GENETIC_GROUPS = ["N", "A+P-", "A+P+"]
CLINICAL_MARKERS = {"CN": "o", "AD": "s"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e4dd"
REFERENCE = "#a9a69c"
ROBUST_METHOD = "huber"  # "huber" or "theilsen"


def load_distances(input_csv):
    data = pd.read_csv(input_csv)
    data["group"] = data["group"].replace({"A+P−": "A+P-"})
    return data


def plot_scatter(data, output_path):
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.edgecolor": "#b8b6ae",
            "axes.labelcolor": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
        }
    )
    genetic = data[data["group"].isin(GENETIC_GROUPS)]
    figure, axis = plt.subplots(figsize=(6.6, 6.2))

    low = min(data["distance_to_ad_centroid"].min(), data["distance_to_cn_centroid"].min()) - 0.6
    high = max(data["distance_to_ad_centroid"].max(), data["distance_to_cn_centroid"].max()) + 0.6

    axis.set_axisbelow(True)
    axis.grid(color=GRID, linewidth=0.6)

    # Above the identity line the participant is nearer the AD centroid.
    axis.fill_between([low, high], [low, high], [high, high], color="#c44e52", alpha=0.045, zorder=0)
    axis.plot([low, high], [low, high], color=REFERENCE, linewidth=1.1, linestyle=(0, (5, 4)), zorder=1)

    # Region labels in the corners, which the data does not reach.
    axis.text(0.03, 0.97, "nearer AD centroid", transform=axis.transAxes, fontsize=9.5,
              color=INK_SECONDARY, ha="left", va="top")
    axis.text(0.97, 0.03, "nearer CN centroid", transform=axis.transAxes, fontsize=9.5,
              color=INK_SECONDARY, ha="right", va="bottom")

    for group, marker in CLINICAL_MARKERS.items():
        subset = data[data["group"].eq(group)]
        axis.scatter(
            subset["distance_to_ad_centroid"], subset["distance_to_cn_centroid"],
            s=30, marker=marker, facecolors="none", edgecolors=REFERENCE,
            linewidths=1.0, zorder=2, label=f"Clinical {group}",
        )

    for group in GENETIC_GROUPS:
        subset = data[data["group"].eq(group)]
        axis.scatter(
            subset["distance_to_ad_centroid"], subset["distance_to_cn_centroid"],
            s=42, color=GROUP_COLORS[group], alpha=0.55, linewidths=0.4,
            edgecolors="white", zorder=3, label=group,
        )

    # Group means, so the shift is legible without averaging 77 points by eye.
    for group in GENETIC_GROUPS:
        subset = data[data["group"].eq(group)]
        mean_ad = subset["distance_to_ad_centroid"].mean()
        mean_cn = subset["distance_to_cn_centroid"].mean()
        # N and A+P- means nearly coincide, so colour carries identity, not text.
        axis.scatter(mean_ad, mean_cn, s=300, marker="X", color=GROUP_COLORS[group],
                     edgecolors=INK_PRIMARY, linewidths=1.5, zorder=6,
                     label="group mean" if group == GENETIC_GROUPS[0] else None)

    # One least-squares fit per genetic-risk group. Each line passes through its
    # own group mean by construction, so the X markers sit on their line.
    fits = {}
    for group in GENETIC_GROUPS:
        subset = data[data["group"].eq(group)]
        x = subset["distance_to_ad_centroid"].to_numpy()
        y = subset["distance_to_cn_centroid"].to_numpy()
        # Robust fit, so a few atypical participants cannot drive the line;
        # ordinary least squares inflates the A+P- slope here. Theil-Sen is
        # retained alongside because it supplies the slope confidence interval.
        if ROBUST_METHOD == "huber":
            model = HuberRegressor(epsilon=1.35).fit(x.reshape(-1, 1), y)
            slope, intercept = float(model.coef_[0]), float(model.intercept_)
        else:
            slope, intercept, _, _ = theilslopes(y, x, 0.95)
        _, _, low_slope, high_slope = theilslopes(y, x, 0.95)
        r, p_value = pearsonr(x, y)
        fits[group] = (slope, intercept, r, p_value, low_slope, high_slope)
        line_x = np.linspace(x.min(), x.max(), 50)
        axis.plot(line_x, slope * line_x + intercept, color=GROUP_COLORS[group],
                  linewidth=2.0, zorder=5, solid_capstyle="round")

    axis.set_xlabel("Distance to clinical AD centroid")
    axis.set_ylabel("Distance to clinical CN centroid")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal")
    axis.tick_params(length=0)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    legend = axis.legend(frameon=False, fontsize=9.5, loc="upper right",
                         handletextpad=0.5, borderaxespad=1.6, labelspacing=0.45)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_distances(args.input_csv)
    output_path = args.output_dir / "psd_centroid_distance_scatter_45hz.png"
    fits = plot_scatter(data, output_path)

    assignments = (
        data[data["group"].isin(GENETIC_GROUPS)]
        .assign(ad_nearest=lambda frame: frame["nearest_clinical_centroid"].eq("AD"))
        .groupby("group")["ad_nearest"]
        .agg(["sum", "count"])
    )
    print(f"Saved {output_path}")
    for group, (slope, _, r, p_value, low, high) in fits.items():
        print(f"  {group:5s} {ROBUST_METHOD} slope {slope:5.2f}  (Theil-Sen CI [{low:5.2f}, {high:5.2f}])"
              f"   Pearson r = {r:.3f} (p = {p_value:.3f})")
    print(assignments)


if __name__ == "__main__":
    main()
