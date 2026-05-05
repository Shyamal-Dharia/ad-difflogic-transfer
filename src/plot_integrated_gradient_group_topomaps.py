from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from train_helpers import ROOT_DIR


CHANNEL_ORDER = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "C3",
    "Cz",
    "C4",
    "P3",
    "Pz",
    "P4",
    "O1",
    "O2",
]
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
GROUP_ORDER = ["N", "A+P-", "A+P+"]
P_THRESHOLD = 0.10
VALUE_COLUMN = "mean_signed_relevance"

INTERPRETATION_DIR = (
    ROOT_DIR
    / "outputs/difflogic/medium_interpretable/interpretation/soft_integrated_gradients"
)
TESTS_PATH = ROOT_DIR / "outputs/model_relevance_statistics/integrated_gradient_relevance_tests.csv"
OUTPUT_DIR = ROOT_DIR / "outputs/model_relevance_statistics"


def make_info():
    info = mne.create_info(CHANNEL_ORDER, sfreq=250, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="raise")
    return info


def load_group_values(value_column):
    group_summary = pd.read_csv(INTERPRETATION_DIR / "pearl_gradient_relevance_group_summary.csv")
    required_columns = {"group", "channel", "band", value_column}
    missing_columns = required_columns.difference(group_summary.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in group summary: {sorted(missing_columns)}")

    return group_summary


def load_significant_channels(p_threshold):
    tests = pd.read_csv(TESTS_PATH)
    significant = tests[tests["p_uncorrected"].lt(p_threshold)].copy()
    significant["groups"] = significant["comparison"].str.split("_vs_")

    rows = []
    for _, row in significant.iterrows():
        for group in row["groups"]:
            rows.append(
                {
                    "channel": row["channel"],
                    "band": row["band"],
                    "group": group,
                    "comparison": row["comparison"],
                    "p_uncorrected": row["p_uncorrected"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["channel", "band", "group", "comparison", "p_uncorrected"])

    significant_channels = pd.DataFrame(rows)
    return (
        significant_channels
        .sort_values("p_uncorrected")
        .drop_duplicates(["channel", "band", "group"], keep="first")
        .sort_values(["band", "group", "channel"])
    )


def channel_values(group_summary, band, group, value_column):
    band_group = (
        group_summary[group_summary["band"].eq(band) & group_summary["group"].eq(group)]
        .set_index("channel")
        .reindex(CHANNEL_ORDER)
    )
    if band_group[value_column].isna().any():
        missing = band_group[band_group[value_column].isna()].index.tolist()
        raise ValueError(f"Missing relevance values for {band} {group}: {missing}")

    return band_group[value_column].to_numpy()


def significance_mask(significant_channels, band, group):
    channels = set(
        significant_channels[
            significant_channels["band"].eq(band) & significant_channels["group"].eq(group)
        ]["channel"]
    )
    return np.array([channel in channels for channel in CHANNEL_ORDER], dtype=bool)


def topomap_positions(info):
    from mne.channels.layout import _find_topomap_coords

    return _find_topomap_coords(info, picks=np.arange(len(CHANNEL_ORDER)))


def draw_significance_labels(axis, positions, mask):
    for channel, (x_pos, y_pos), is_significant in zip(CHANNEL_ORDER, positions, mask):
        if not is_significant:
            continue

        axis.scatter(
            x_pos,
            y_pos,
            s=240,
            marker="o",
            facecolor="#FFD84D",
            edgecolor="black",
            linewidth=1.2,
            zorder=10,
        )
        axis.text(
            x_pos,
            y_pos,
            channel,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="black",
            zorder=11,
        )


def plot_one_topomap(axis, values, info, positions, mask, max_abs, image_interpolation):
    image, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=axis,
        show=False,
        cmap="RdBu_r",
        vlim=(-max_abs, max_abs),
        contours=0,
        sensors=True,
        image_interp=image_interpolation,
    )
    draw_significance_labels(axis, positions, mask)
    return image


def save_figure(fig, output_prefix):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{output_prefix}.png"
    pdf_path = OUTPUT_DIR / f"{output_prefix}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def plot_combined_group_topomaps(
    group_summary,
    significant_channels,
    info,
    positions,
    value_column,
    output_prefix,
    image_interpolation,
):
    max_abs = group_summary[value_column].abs().max()
    fig, axes = plt.subplots(
        len(BAND_ORDER),
        len(GROUP_ORDER),
        figsize=(6.8, 9.2),
        constrained_layout=True,
    )

    image = None
    for row_index, band in enumerate(BAND_ORDER):
        for column_index, group in enumerate(GROUP_ORDER):
            axis = axes[row_index, column_index]
            values = channel_values(group_summary, band, group, value_column)
            mask = significance_mask(significant_channels, band, group)
            image = plot_one_topomap(
                axis,
                values,
                info,
                positions,
                mask,
                max_abs,
                image_interpolation,
            )

            if row_index == 0:
                axis.set_title(group, fontsize=11, pad=8)
            if column_index == 0:
                axis.text(
                    -0.18,
                    0.5,
                    band,
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=11,
                    fontweight="bold",
                )

    cbar = fig.colorbar(image, ax=axes, shrink=0.62, pad=0.02)
    cbar.set_label("")
    save_figure(fig, output_prefix)


def plot_band_topomaps(
    group_summary,
    significant_channels,
    info,
    positions,
    band,
    value_column,
    output_prefix,
    image_interpolation,
):
    max_abs = group_summary[group_summary["band"].eq(band)][value_column].abs().max()
    fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(6.8, 2.45), constrained_layout=True)

    image = None
    for axis, group in zip(axes, GROUP_ORDER):
        values = channel_values(group_summary, band, group, value_column)
        mask = significance_mask(significant_channels, band, group)
        image = plot_one_topomap(
            axis,
            values,
            info,
            positions,
            mask,
            max_abs,
            image_interpolation,
        )
        axis.set_title(group, fontsize=11, pad=8)

    cbar = fig.colorbar(image, ax=axes, shrink=0.80, pad=0.02)
    cbar.set_label("")
    save_figure(fig, f"{output_prefix}_{band}")


def plot_group_topomaps(value_column, output_prefix, p_threshold, image_interpolation):
    group_summary = load_group_values(value_column)
    significant_channels = load_significant_channels(p_threshold)
    info = make_info()
    positions = topomap_positions(info)

    plot_combined_group_topomaps(
        group_summary,
        significant_channels,
        info,
        positions,
        value_column,
        output_prefix,
        image_interpolation,
    )

    for band in BAND_ORDER:
        plot_band_topomaps(
            group_summary,
            significant_channels,
            info,
            positions,
            band,
            value_column,
            output_prefix,
            image_interpolation,
        )

    significant_channels.to_csv(OUTPUT_DIR / f"{output_prefix}_significant_channels.csv", index=False)
    print(f"Saved {OUTPUT_DIR / f'{output_prefix}_significant_channels.csv'}")


def main():
    plot_group_topomaps(
        value_column=VALUE_COLUMN,
        output_prefix="integrated_gradient_group_topomaps_mean_nearest_p010",
        p_threshold=P_THRESHOLD,
        image_interpolation="nearest",
    )
    plot_group_topomaps(
        value_column=VALUE_COLUMN,
        output_prefix="integrated_gradient_group_topomaps_mean_cubic_p010",
        p_threshold=P_THRESHOLD,
        image_interpolation="cubic",
    )


if __name__ == "__main__":
    main()
