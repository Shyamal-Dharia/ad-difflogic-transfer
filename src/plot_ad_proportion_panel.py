"""AD-nearest proportion per genetic-risk group, with corrected significance.

Panel B of the PSD feature-space figure. Colours match the UMAP panel produced
by `plot_psd_umap_transfer.py` so the two read as one figure.

Significance is Fisher's exact test on the 2x2 assignment tables, Holm-corrected
across the three pairwise contrasts, matching the correction family used for the
transfer-score contrasts elsewhere in the manuscript.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from train_helpers import ROOT_DIR


INPUT_CSV = ROOT_DIR / "outputs/psd_umap_45hz/psd_umap_subject_coordinates_and_distances.csv"
GROUPS = ["N", "A+P-", "A+P+"]
GROUP_COLORS = {"N": "#A8A8A8", "A+P-": "#EDC46F", "A+P+": "#B694D6"}
CONTRASTS = [("A+P-", "N"), ("A+P+", "N"), ("A+P+", "A+P-")]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def assignment_table(data):
    return {
        group: [
            int(data[data["group"].eq(group)]["nearest_clinical_centroid"].eq("AD").sum()),
            int(data[data["group"].eq(group)]["nearest_clinical_centroid"].eq("CN").sum()),
        ]
        for group in GROUPS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--figure-dir", type=Path, default=ROOT_DIR / "figures")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs/psd_umap_45hz")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input_csv)
    data["group"] = data["group"].replace({"A+P−": "A+P-"})
    data = data[data["group"].isin(GROUPS)]
    table = assignment_table(data)

    raw = [fisher_exact([table[a], table[b]]).pvalue for a, b in CONTRASTS]
    adjusted = holm_adjust(np.array(raw))
    tests = pd.DataFrame(
        {
            "contrast": [f"{a} vs {b}" for a, b in CONTRASTS],
            "p_uncorrected": raw,
            "p_holm": adjusted,
        }
    )
    tests.to_csv(args.output_dir / "psd_ad_proportion_tests.csv", index=False)

    plt.rcParams.update({"font.size": 11, "text.color": INK_PRIMARY,
                         "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY})
    figure, axis = plt.subplots(figsize=(5.77, 5.01))
    proportions = [table[g][0] / sum(table[g]) for g in GROUPS]
    axis.bar(GROUPS, proportions, color=[GROUP_COLORS[g] for g in GROUPS],
             edgecolor=INK_PRIMARY, linewidth=0.8, width=0.62, zorder=3)
    for position, group in enumerate(GROUPS):
        ad, cn = table[group]
        axis.text(position, proportions[position] + 0.015, f"{ad}/{ad + cn}",
                  ha="center", va="bottom", fontsize=11, color=INK_PRIMARY)

    # Tests are reported in the text, not annotated on the panel.
    axis.set_ylabel("Proportion nearest the clinical AD centroid")
    axis.set_ylim(0, max(proportions) * 1.22)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="#e6e4dd", linewidth=0.6)
    axis.tick_params(length=0)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    figure.tight_layout()
    output_path = args.figure_dir / "psd_centroid_panel_b_ad_proportion_45hz.png"
    figure.savefig(output_path, dpi=200, facecolor="white")
    plt.close(figure)
    print(f"Saved {output_path}")
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
