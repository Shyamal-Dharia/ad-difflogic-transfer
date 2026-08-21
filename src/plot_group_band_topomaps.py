"""Scalp topographies of log-relative band power for every cohort group.

Addresses reviewer 3's request for group-level power distributions, complementing
the violin summary in `plot_harmonization_distributions.py` with spatial detail.

The values plotted are the same channel-band features used for every reported
result, computed with `scipy.signal.welch` in `feature_extraction_psd.py`. MNE is
used for the topographic projection only, not to recompute spectra: calling
`compute_psd` here would produce a second estimate that disagrees with the
features the models were trained on.

Maps show each group's deviation from its own cohort mean for that band, not
absolute power. Absolute log-relative power is strictly negative and has no
meaningful midpoint, so a diverging scale centred on zero would be arbitrary;
centring on the cohort mean creates a real zero (the average of that cohort's
groups) and makes the between-group structure legible. Centring within cohort
rather than across both also prevents the cross-cohort domain shift from
dominating every map. The scale is symmetric and shared across all six groups
within a band, so groups are directly comparable down a column, and is not
shared across bands because band magnitudes differ by roughly an order of
magnitude.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT_DIR / "datasets/statistics_psd_45hz/subject_log_relative_bandpower.csv"
OUTPUT_DIR = ROOT_DIR / "outputs/harmonization_45hz"

CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2",
]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_LIMITS = {
    "delta": "1-4 Hz",
    "theta": "4-8 Hz",
    "alpha": "8-13 Hz",
    "beta": "13-30 Hz",
    "gamma": "30-45 Hz",
}
GROUP_NAMES = {"A": "AD", "C": "CN", "F": "FTD", "A+P-": "A+P−"}
# Both blocks run control first, then the intermediate group, then the group of
# interest: CN, FTD, AD mirrors N, A+P-, A+P+.
ROWS = [
    ("Clinical", "ALZ_FTD", "CN"),
    ("Clinical", "ALZ_FTD", "FTD"),
    ("Clinical", "ALZ_FTD", "AD"),
    ("Genetic-Risk", "PEARL", "N"),
    ("Genetic-Risk", "PEARL", "A+P−"),
    ("Genetic-Risk", "PEARL", "A+P+"),
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"


def build_info():
    info = mne.create_info(ch_names=CHANNELS, sfreq=250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    return info


def load_channel_bandpower(input_csv):
    data = pd.read_csv(input_csv)
    data["group"] = data["group"].replace(GROUP_NAMES)
    missing = set(CHANNELS) - set(data["channel"])
    if missing:
        raise ValueError(f"Missing channels: {sorted(missing)}")
    return data


def group_topography(data, dataset, group, band):
    subset = data[
        data["dataset"].eq(dataset) & data["group"].eq(group) & data["band"].eq(band)
    ]
    if subset.empty:
        raise ValueError(f"No data for {dataset}/{group}/{band}")
    per_channel = subset.groupby("channel")["log_relative_bandpower"].mean()
    return per_channel.reindex(CHANNELS).to_numpy(), subset["participant_id"].nunique()


def plot_topomaps(data, output_path, mode="deviation"):
    info = build_info()
    figure, axes = plt.subplots(
        len(ROWS),
        len(BANDS),
        figsize=(11.0, 12.8),
        gridspec_kw={"wspace": 0.06, "hspace": 0.06, "bottom": 0.075},
    )

    # "deviation": each group minus its cohort mean, symmetric diverging scale, so
    # zero is a real reference and between-group structure is emphasised.
    # "raw": absolute log-relative power on a sequential scale, which is what the
    # values actually are and needs no reference concept to read.
    colormap = "RdBu_r" if mode == "deviation" else "Blues"
    deviations = {}
    band_limits = {}
    for band in BANDS:
        cohort_means = {}
        for cohort in {row[0] for row in ROWS}:
            cohort_rows = [row for row in ROWS if row[0] == cohort]
            cohort_means[cohort] = np.mean(
                [group_topography(data, dataset, group, band)[0] for _, dataset, group in cohort_rows],
                axis=0,
            )
        for cohort, dataset, group in ROWS:
            values, n_subjects = group_topography(data, dataset, group, band)
            offset = cohort_means[cohort] if mode == "deviation" else 0.0
            deviations[(group, band)] = (values - offset, n_subjects)
        stacked = np.concatenate([deviations[(row[2], band)][0] for row in ROWS])
        if mode == "deviation":
            max_abs = float(np.abs(stacked).max())
            band_limits[band] = (-max_abs, max_abs)
        else:
            band_limits[band] = (float(stacked.min()), float(stacked.max()))

    for row, (cohort, dataset, group) in enumerate(ROWS):
        for column, band in enumerate(BANDS):
            axis = axes[row, column]
            values, n_subjects = deviations[(group, band)]
            vmin, vmax = band_limits[band]
            mne.viz.plot_topomap(
                values,
                info,
                axes=axis,
                cmap=colormap,
                vlim=(vmin, vmax),
                sensors=True,
                contours=0,
                outlines="head",
                image_interp="nearest",
                sphere=None,
                show=False,
            )
            if row == 0:
                axis.set_title(
                    f"{band.capitalize()}\n{BAND_LIMITS[band]}",
                    fontsize=12.5,
                    color=INK_PRIMARY,
                    pad=10,
                )
            if column == 0:
                axis.text(
                    -0.30,
                    0.5,
                    f"{group}\n(n={n_subjects})",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12.5,
                    color=INK_PRIMARY,
                )
    # One colour bar per band, placed under its own column, because the scale is
    # shared down a column and not across the row.
    for column, band in enumerate(BANDS):
        vmin, vmax = band_limits[band]
        position = axes[-1, column].get_position()
        colorbar_axis = figure.add_axes(
            [position.x0 + 0.012, position.y0 - 0.022, position.width - 0.024, 0.008]
        )
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=colormap
            ),
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.outline.set_visible(False)
        if mode == "deviation":
            colorbar.set_ticks([vmin, 0.0, vmax])
            colorbar.ax.set_xticklabels([f"{vmin:+.2f}", "0", f"{vmax:+.2f}"])
        else:
            colorbar.set_ticks([vmin, vmax])
            colorbar.ax.set_xticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
        colorbar.ax.tick_params(labelsize=9.5, length=0, colors=INK_SECONDARY, pad=2)

    # Cohort brackets, so the two datasets read as blocks rather than six rows.
    for cohort, rows in (("Clinical", (0, 2)), ("Genetic-Risk", (3, 5))):
        top = axes[rows[0], 0].get_position().y1
        bottom = axes[rows[1], 0].get_position().y0
        figure.lines.append(
            plt.Line2D(
                [0.055, 0.055],
                [bottom, top],
                transform=figure.transFigure,
                color=INK_SECONDARY,
                linewidth=1.0,
            )
        )
        figure.text(
            0.040,
            (top + bottom) / 2,
            cohort,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13.5,
            color=INK_PRIMARY,
        )

    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_summary(data, output_path):
    summary = (
        data.groupby(["dataset", "group", "band", "channel"], as_index=False)
        .agg(
            n_subjects=("participant_id", "nunique"),
            mean_log_relative_bandpower=("log_relative_bandpower", "mean"),
            sd_log_relative_bandpower=("log_relative_bandpower", "std"),
        )
    )
    summary.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mode", choices=["deviation", "raw"], default="deviation")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_channel_bandpower(args.input_csv)
    suffix = "" if args.mode == "deviation" else "_raw"
    figure_path = args.output_dir / f"psd_group_band_topomaps_45hz{suffix}.png"
    summary_path = args.output_dir / "group_channel_bandpower_summary.csv"
    plot_topomaps(data, figure_path, mode=args.mode)
    write_summary(data, summary_path)
    print(f"Saved {figure_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
