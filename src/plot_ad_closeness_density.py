"""AD-closeness densities per genetic-risk group.

Companion view to `plot_ad_closeness_violin.py`: the same quantity
d_CN - d_AD, shown as a smoothed density per group rather than as individual
points, so the shift of the A+P+ distribution toward the boundary is visible as
a whole-distribution effect. Zero is the nearest-centroid decision boundary.

Densities use `gaussian_kde` with Scott's rule, the same default bandwidth as
MATLAB's `ksdensity`. Dashed verticals mark each group's median.

Panel geometry, frame and grid are copied from the UMAP panel produced by
`plot_psd_umap_transfer.py`, so the figures read as one set.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from train_helpers import ROOT_DIR


INPUT_CSV = ROOT_DIR / "outputs/psd_umap_45hz/psd_umap_subject_coordinates_and_distances.csv"
GROUPS = ["N", "A+P-", "A+P+"]
GROUP_COLORS = {"N": "#A8A8A8", "A+P-": "#EDC46F", "A+P+": "#B694D6"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
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
    axis.set_axisbelow(True)
    axis.grid(True, **GRID_STYLE)

    low = data["closeness"].min() - 1.5
    high = data["closeness"].max() + 1.5
    grid = np.linspace(low, high, 400)

    # Everything right of zero is nearer the clinical AD centroid.
    axis.axvspan(0, high, color="#c44e52", alpha=0.045, zorder=0)
    axis.axvline(0, color=INK_SECONDARY, linewidth=1.1, linestyle=(0, (5, 4)), zorder=2)

    peak = 0.0
    for group in GROUPS:
        values = data.loc[data["group"].eq(group), "closeness"].to_numpy()
        density = gaussian_kde(values)(grid)
        peak = max(peak, density.max())
        axis.fill_between(grid, density, color=GROUP_COLORS[group], alpha=0.18, zorder=3)
        axis.plot(grid, density, color=GROUP_COLORS[group], linewidth=2.4, zorder=4,
                  solid_capstyle="round", label=f"{group} (n={values.size})")

        # Median marked to its own curve, not to the top of the axes, so three
        # verticals do not dominate the panel.
        median = float(np.median(values))
        axis.vlines(median, 0, gaussian_kde(values)(median)[0], color=GROUP_COLORS[group],
                    linewidth=1.8, linestyle=(0, (4, 3)), zorder=5)

    # Low on the panel, where no curve or the legend can reach them. Arrows point
    # away from the boundary, matching the vertical pair on the violin panel.
    axis.text(-0.25, peak * 0.045, "$\\leftarrow$ nearer CN", fontsize=11.5,
              color=INK_SECONDARY, ha="right", va="bottom")
    axis.text(0.25, peak * 0.045, "nearer AD $\\rightarrow$", fontsize=11.5,
              color=INK_SECONDARY, ha="left", va="bottom")

    axis.set_xlabel(r"AD-closeness ($d_{\mathrm{CN}} - d_{\mathrm{AD}}$)")
    axis.set_ylabel("Density")
    axis.set_xlim(low, high)
    axis.set_ylim(0, peak * 1.22)

    legend = axis.legend(loc="upper right", frameon=True, fontsize=12, borderpad=0.35,
                         labelspacing=0.35, handletextpad=0.6, borderaxespad=0.45)
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_linewidth(0.8)

    # Full box in the same weight and colour as the UMAP panel.
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    figure.tight_layout()
    output_path = args.figure_dir / "psd_closeness_density_45hz.png"
    figure.savefig(output_path, dpi=PANEL_DPI, facecolor="white")
    plt.close(figure)
    print(f"Saved {output_path}")
    for group in GROUPS:
        values = data.loc[data["group"].eq(group), "closeness"]
        print(f"  {group:5s} n={len(values):3d}  median {values.median():+.2f}   "
              f"above boundary {int((values > 0).sum())}/{len(values)}")


if __name__ == "__main__":
    main()
