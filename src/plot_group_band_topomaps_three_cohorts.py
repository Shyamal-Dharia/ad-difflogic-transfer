"""Exploratory topomaps for all three cohorts, including PSEN1 (DS007427).

Same construction as `plot_group_band_topomaps.py` (deviation from cohort mean,
symmetric scale shared across groups within a band) extended with the PSEN1
carriers and non-carriers. Visualization aid; not a manuscript figure.

Note on the PSEN1 block: with only two groups, each group's deviation from the
cohort mean is exactly half the G1-G2 difference, so those two maps are mirror
images by construction. They carry one contrast, not two.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from plot_group_band_topomaps import BAND_LIMITS, CHANNELS, INK_PRIMARY, INK_SECONDARY, build_info
from train_helpers import BANDS, ROOT_DIR, load_target_dataset


CLINICAL_CSV = ROOT_DIR / "datasets/statistics_psd_45hz/subject_log_relative_bandpower.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "revision_2026" / "psen1_ds007427"
BAND_NAMES = list(BANDS)
GROUP_NAMES = {"A": "AD", "C": "CN", "F": "FTD", "A+P-": "A+P−"}
ROWS = [
    ("Clinical", "ALZ_FTD", "CN"),
    ("Clinical", "ALZ_FTD", "FTD"),
    ("Clinical", "ALZ_FTD", "AD"),
    ("Genetic-Risk", "PEARL", "N"),
    ("Genetic-Risk", "PEARL", "A+P−"),
    ("Genetic-Risk", "PEARL", "A+P+"),
    ("PSEN1", "DS007427", "G2"),
    ("PSEN1", "DS007427", "G1"),
]
COHORT_BLOCKS = (("Clinical", (0, 2)), ("Genetic-Risk", (3, 5)), ("PSEN1", (6, 7)))


def load_channel_bandpower():
    """Per-subject, per-channel, per-band log-relative power for all three cohorts."""
    clinical = pd.read_csv(CLINICAL_CSV)
    clinical["group"] = clinical["group"].replace(GROUP_NAMES)
    clinical = clinical[["dataset", "participant_id", "group", "channel", "band", "log_relative_bandpower"]]

    subjects, _ = load_target_dataset("DS007427", feature_kind="psd")
    rows = []
    for subject in subjects:
        if subject["group"] == "CTR":
            continue
        per_channel_band = subject["x"].mean(axis=0)  # (channels, bands)
        for channel_index, channel in enumerate(subject["channel_names"]):
            for band_index, band in enumerate(BAND_NAMES):
                rows.append(
                    {
                        "dataset": "DS007427",
                        "participant_id": subject["participant_id"],
                        "group": subject["group"],
                        "channel": str(channel),
                        "band": band,
                        "log_relative_bandpower": float(per_channel_band[channel_index, band_index]),
                    }
                )
    return pd.concat([clinical, pd.DataFrame(rows)], ignore_index=True)


def group_topography(data, dataset, group, band):
    subset = data[
        data["dataset"].eq(dataset) & data["group"].eq(group) & data["band"].eq(band)
    ]
    per_channel = subset.groupby("channel")["log_relative_bandpower"].mean()
    return per_channel.reindex(CHANNELS).to_numpy(), subset["participant_id"].nunique()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_channel_bandpower()
    info = build_info()

    deviations = {}
    band_limits = {}
    for band in BAND_NAMES:
        cohort_means = {}
        for cohort in {row[0] for row in ROWS}:
            cohort_rows = [row for row in ROWS if row[0] == cohort]
            cohort_means[cohort] = np.mean(
                [group_topography(data, ds, g, band)[0] for _, ds, g in cohort_rows], axis=0
            )
        for cohort, dataset, group in ROWS:
            values, n_subjects = group_topography(data, dataset, group, band)
            deviations[(group, band)] = (values - cohort_means[cohort], n_subjects)
        band_limits[band] = max(
            float(np.abs(deviations[(row[2], band)][0]).max()) for row in ROWS
        )

    figure, axes = plt.subplots(
        len(ROWS), len(BAND_NAMES), figsize=(11.0, 16.6),
        gridspec_kw={"wspace": 0.06, "hspace": 0.06, "bottom": 0.058},
    )

    for row, (cohort, dataset, group) in enumerate(ROWS):
        for column, band in enumerate(BAND_NAMES):
            axis = axes[row, column]
            values, n_subjects = deviations[(group, band)]
            limit = band_limits[band]
            mne.viz.plot_topomap(
                values, info, axes=axis, cmap="RdBu_r", vlim=(-limit, limit),
                sensors=True, contours=0, outlines="head",
                image_interp="nearest", sphere=None, show=False,
            )
            if row == 0:
                axis.set_title(
                    f"{band.capitalize()}\n{BAND_LIMITS[band]}",
                    fontsize=12.5, color=INK_PRIMARY, pad=10,
                )
            if column == 0:
                axis.text(
                    -0.30, 0.5, f"{group}\n(n={n_subjects})", transform=axis.transAxes,
                    ha="center", va="center", fontsize=12.5, color=INK_PRIMARY,
                )

    for column, band in enumerate(BAND_NAMES):
        limit = band_limits[band]
        position = axes[-1, column].get_position()
        colorbar_axis = figure.add_axes(
            [position.x0 + 0.012, position.y0 - 0.018, position.width - 0.024, 0.006]
        )
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=-limit, vmax=limit), cmap="RdBu_r"
            ),
            cax=colorbar_axis, orientation="horizontal",
        )
        colorbar.outline.set_visible(False)
        colorbar.set_ticks([-limit, 0.0, limit])
        colorbar.ax.set_xticklabels([f"{-limit:+.2f}", "0", f"{limit:+.2f}"])
        colorbar.ax.tick_params(labelsize=9.5, length=0, colors=INK_SECONDARY, pad=2)

    for cohort, (first, last) in COHORT_BLOCKS:
        top = axes[first, 0].get_position().y1
        bottom = axes[last, 0].get_position().y0
        figure.lines.append(
            plt.Line2D([0.055, 0.055], [bottom, top], transform=figure.transFigure,
                       color=INK_SECONDARY, linewidth=1.0)
        )
        figure.text(
            0.040, (top + bottom) / 2, cohort, rotation=90,
            ha="center", va="center", fontsize=13.5, color=INK_PRIMARY,
        )

    output_path = args.output_dir / "psd_group_band_topomaps_three_cohorts.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
