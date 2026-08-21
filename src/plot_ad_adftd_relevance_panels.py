from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import plot_transfer_relevance_contrast_topomaps as topomaps


ROOT = Path(__file__).resolve().parents[1]
AD_STATS = ROOT / "outputs/model_relevance_statistics_45hz_benchmark_250k"
ADFTD_INTERPRETATION = (
    ROOT
    / "outputs/difflogic_45hz/cn_vs_ad_ftd_benchmark_250k_psd"
    / "interpretation/soft_integrated_gradients"
)
ADFTD_TESTS = (
    ROOT
    / "outputs/model_relevance_statistics_45hz_cn_vs_ad_ftd_250k"
    / "integrated_gradient_relevance_tests.csv"
)
FIGURES = ROOT / "figures/45hz_benchmark_250k"

PANELS = [
    (
        "(A) AD model",
        [
            ("clinical_A_minus_C", "Clinical\nAD - CN"),
            ("pearl_A+P+_minus_N", "PEARL\nA+P+ - N"),
            ("pearl_A+P+_minus_A+P-", "PEARL\nA+P+ - A+P-"),
        ],
    ),
    (
        "(B) AD+FTD model",
        [
            ("clinical_AD+FTD_minus_C", "Clinical\nAD+FTD - CN"),
            ("adftd_pearl_A+P+_minus_N", "PEARL\nA+P+ - N"),
            ("adftd_pearl_A+P+_minus_A+P-", "PEARL\nA+P+ - A+P-"),
        ],
    ),
]


def make_adftd_pearl_contrasts():
    summary = pd.read_csv(ADFTD_INTERPRETATION / "pearl_gradient_relevance_group_summary.csv")
    pivot = summary.pivot(
        index=["channel", "band"],
        columns="group",
        values=topomaps.VALUE_COLUMN,
    ).reset_index()
    contrasts = {
        "adftd_pearl_A+P+_minus_N": pivot["A+P+"] - pivot["N"],
        "adftd_pearl_A+P+_minus_A+P-": pivot["A+P+"] - pivot["A+P-"],
    }
    return pd.concat(
        [
            pivot[["channel", "band"]].assign(contrast=name, value=values)
            for name, values in contrasts.items()
        ],
        ignore_index=True,
    )


def make_significant_channels():
    significant = pd.read_csv(
        AD_STATS / "source_target_integrated_gradient_contrast_significant_channels.csv"
    )
    tests = pd.read_csv(ADFTD_TESTS)
    mapping = {
        "A+P+_vs_N": "adftd_pearl_A+P+_minus_N",
        "A+P+_vs_A+P-": "adftd_pearl_A+P+_minus_A+P-",
    }
    tests = tests[tests["comparison"].isin(mapping)].copy()
    tests["contrast"] = tests["comparison"].map(mapping)
    tests = tests[tests["p_uncorrected"].lt(topomaps.P_THRESHOLD)]
    return pd.concat(
        [significant, tests[["channel", "band", "contrast", "p_uncorrected"]]],
        ignore_index=True,
    )


def plot_panels(contrast_table, significant_channels):
    info = topomaps.make_info()
    positions = topomaps.topomap_positions(info)
    fig, axes = plt.subplots(10, 3, figsize=(7.2, 16.2), constrained_layout=True)
    image = None

    for panel_index, (panel_title, contrasts) in enumerate(PANELS):
        row_offset = panel_index * len(topomaps.BAND_ORDER)
        for band_index, band in enumerate(topomaps.BAND_ORDER):
            row = row_offset + band_index
            for column, (contrast, title) in enumerate(contrasts):
                axis = axes[row, column]
                values = topomaps.contrast_values(contrast_table, band, contrast)
                mask = topomaps.significance_mask(significant_channels, band, contrast)
                image = topomaps.plot_one_topomap(
                    axis,
                    values,
                    info,
                    positions,
                    mask,
                    max_abs=1.0,
                    image_interpolation="nearest",
                )
                if band_index == 0:
                    axis.set_title(title, fontsize=topomaps.COLUMN_TITLE_FONTSIZE, pad=9)
                if column == 0:
                    axis.text(
                        -0.18,
                        0.5,
                        topomaps.BAND_LABELS[band],
                        transform=axis.transAxes,
                        ha="right",
                        va="center",
                        rotation=90,
                        fontsize=topomaps.ROW_LABEL_FONTSIZE,
                        fontweight="bold",
                    )

        axes[row_offset, 1].text(
            0.5,
            1.38,
            panel_title,
            transform=axes[row_offset, 1].transAxes,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )

    colorbar = fig.colorbar(image, ax=axes, shrink=0.45, pad=0.02)
    colorbar.ax.tick_params(labelsize=topomaps.COLORBAR_TICK_FONTSIZE)
    topomaps.save_figure(
        fig,
        "source_target_integrated_gradient_contrasts_ad_adftd_panels_nearest_p010",
    )


def main():
    topomaps.OUTPUT_DIR = AD_STATS
    topomaps.FIGURES_DIR = FIGURES
    primary = pd.read_csv(AD_STATS / "source_target_integrated_gradient_contrasts.csv")
    contrasts = pd.concat([primary, make_adftd_pearl_contrasts()], ignore_index=True)
    assert contrasts.groupby("contrast").size().eq(75).all()
    normalized = topomaps.normalize_by_contrast(contrasts)
    normalized.to_csv(
        AD_STATS / "source_target_integrated_gradient_contrasts_ad_adftd_panels.csv",
        index=False,
    )
    significant_channels = make_significant_channels()
    plot_panels(normalized, significant_channels)


if __name__ == "__main__":
    main()
