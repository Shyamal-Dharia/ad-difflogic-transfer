from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns

from train_helpers import ROOT_DIR


OUTPUT_DIR = ROOT_DIR / "outputs/publication_figures_v2"
DIFFLOGIC_DIR = ROOT_DIR / "outputs/difflogic"

GROUP_ORDER = ["N", "A+P-", "A+P+"]
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
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


def save_figure(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def make_info():
    info = mne.create_info(CHANNEL_ORDER, sfreq=250, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    return info


def read_seed_average(run_name):
    return pd.read_csv(
        DIFFLOGIC_DIR / run_name / "summary/pearl_subject_predictions_seed_average.csv"
    )


def read_metrics(run_name):
    return pd.read_csv(DIFFLOGIC_DIR / run_name / "summary/alz_test_metrics_all_seeds.csv")


def plot_model_relevance_topomaps():
    contrast = pd.read_csv(
        DIFFLOGIC_DIR
        / "medium_interpretable/interpretation/soft_integrated_gradients"
        / "pearl_signed_relevance_group_contrasts.csv"
    )
    tests = pd.read_csv(
        ROOT_DIR / "outputs/model_relevance_statistics/integrated_gradient_relevance_tests.csv"
    )
    tests = tests[tests["comparison"].eq("A+P+_vs_N")]

    info = make_info()
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.2))
    max_abs = contrast["A+P+_minus_N"].abs().max()

    image = None
    for axis, band in zip(axes, BAND_ORDER):
        band_data = (
            contrast[contrast["band"].eq(band)]
            .set_index("channel")
            .loc[CHANNEL_ORDER, "A+P+_minus_N"]
            .to_numpy()
        )
        band_tests = (
            tests[tests["band"].eq(band)]
            .set_index("channel")
            .reindex(CHANNEL_ORDER)
        )
        mask = band_tests["p_uncorrected"].lt(0.05).fillna(False).to_numpy()
        image, _ = mne.viz.plot_topomap(
            band_data,
            info,
            axes=axis,
            show=False,
            cmap="RdBu_r",
            vlim=(-max_abs, max_abs),
            contours=0,
            sensors=True,
            mask=mask,
            mask_params={
                "marker": "*",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 10,
                "linewidth": 0,
            },
        )
        axis.set_title(band)

    cbar = fig.colorbar(image, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("A+P+ minus N relevance")
    fig.suptitle("Integrated-gradient model relevance by frequency band", y=1.03)
    fig.text(
        0.5,
        -0.03,
        "Stars mark uncorrected channel-level p < 0.05; no channel survived FDR correction.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, "figure_01_model_relevance_topomaps")


def plot_theta_psd_pairwise_topomaps():
    theta_tests = pd.read_csv(ROOT_DIR / "outputs/theta_psd_group_statistics/theta_psd_pairwise_tests.csv")
    info = make_info()
    comparisons = [
        ("A+P+_vs_N", "A+P+ minus N"),
        ("A+P+_vs_A+P-", "A+P+ minus A+P-"),
        ("A+P-_vs_N", "A+P- minus N"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2))
    max_abs = theta_tests["median_difference"].abs().max()

    image = None
    for axis, (comparison, title) in zip(axes, comparisons):
        data = (
            theta_tests[theta_tests["comparison"].eq(comparison)]
            .set_index("channel")
            .loc[CHANNEL_ORDER]
        )
        values = data["median_difference"].to_numpy()
        mask = data["p_uncorrected"].lt(0.05).fillna(False).to_numpy()
        image, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=axis,
            show=False,
            cmap="RdBu_r",
            vlim=(-max_abs, max_abs),
            contours=0,
            sensors=True,
            mask=mask,
            mask_params={
                "marker": "*",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 10,
                "linewidth": 0,
            },
        )
        axis.set_title(title)

    cbar = fig.colorbar(image, ax=axes, shrink=0.75, pad=0.03)
    cbar.set_label("Median theta log-relative PSD difference")
    fig.suptitle("Theta-band PSD group comparisons", y=1.03)
    fig.text(
        0.5,
        -0.03,
        "Stars mark uncorrected p < 0.05; no theta-channel comparison was significant.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, "figure_02_theta_psd_pairwise_topomaps")


def make_performance_table():
    runs = [
        ("medium_interpretable", "PSD"),
        ("medium_no_fp1_fp2", "PSD without Fp1/Fp2"),
        ("medium_shuffled_alz_labels", "PSD shuffled labels"),
        ("hfd_kmax16_medium", "HFD kmax=16"),
        ("hfd_kmaxnone_medium", "HFD kmax=None"),
    ]
    rows = []

    for run_name, label in runs:
        metrics = read_metrics(run_name)
        group_summary = pd.read_csv(DIFFLOGIC_DIR / run_name / "summary/pearl_group_summary_seed_average.csv")
        medians = group_summary.set_index("group")["median_p_ad"]
        rows.append(
            {
                "analysis": label,
                "balanced_accuracy": f"{metrics['balanced_accuracy'].mean():.3f} +/- {metrics['balanced_accuracy'].std():.3f}",
                "auroc": f"{metrics['auroc'].mean():.3f} +/- {metrics['auroc'].std():.3f}",
                "median_N": f"{medians['N']:.3f}",
                "median_A+P-": f"{medians['A+P-']:.3f}",
                "median_A+P+": f"{medians['A+P+']:.3f}",
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "performance_summary_table.csv", index=False)

    fig, axis = plt.subplots(figsize=(9.4, 2.4))
    axis.axis("off")
    display_table = axis.table(
        cellText=table.values,
        colLabels=[
            "Analysis",
            "Bal. acc.",
            "AUROC",
            "N",
            "A+P-",
            "A+P+",
        ],
        loc="center",
        cellLoc="center",
    )
    display_table.auto_set_font_size(False)
    display_table.set_fontsize(8.5)
    display_table.scale(1, 1.35)
    axis.set_title("Subject-level performance and PEARL median AD-like scores", pad=10)
    save_figure(fig, "figure_03_performance_summary_table")


def plot_family_control_panels():
    runs = [
        ("medium_interpretable", "PSD"),
        ("medium_shuffled_alz_labels", "Shuffled labels"),
        ("hfd_kmax16_medium", "HFD kmax=16"),
        ("hfd_kmaxnone_medium", "HFD kmax=None"),
    ]
    rows = []

    for run_name, label in runs:
        data = read_seed_average(run_name)
        data["analysis"] = label
        rows.append(data)

    data = pd.concat(rows, ignore_index=True)
    grid = sns.catplot(
        data=data,
        x="group",
        y="mean_p_ad",
        col="analysis",
        order=GROUP_ORDER,
        kind="box",
        color="white",
        showfliers=False,
        col_wrap=4,
        height=3.1,
        aspect=0.85,
        sharey=True,
    )

    for axis, (_, label) in zip(grid.axes.flat, runs):
        subset = data[data["analysis"].eq(label)]
        sns.stripplot(
            data=subset,
            x="group",
            y="mean_p_ad",
            order=GROUP_ORDER,
            color="black",
            alpha=0.55,
            jitter=0.18,
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel("AD-like EEG score")
        axis.set_ylim(0, 1)
        axis.set_title(label)

    grid.fig.suptitle("PEARL transfer scores across controls and feature families", y=1.05)
    save_figure(grid.fig, "figure_04_family_control_panels")


def plot_theta_psd_group_heatmap():
    theta_tests = pd.read_csv(ROOT_DIR / "outputs/theta_psd_group_statistics/theta_psd_pairwise_tests.csv")
    comparison_order = ["A+P+_vs_N", "A+P+_vs_A+P-", "A+P-_vs_N"]
    heatmap_data = (
        theta_tests
        .pivot(index="channel", columns="comparison", values="median_difference")
        .loc[CHANNEL_ORDER, comparison_order]
    )
    p_data = (
        theta_tests
        .pivot(index="channel", columns="comparison", values="p_uncorrected")
        .loc[CHANNEL_ORDER, comparison_order]
    )
    annotations = p_data.applymap(lambda value: f"p={value:.2f}")

    fig, axis = plt.subplots(figsize=(5.7, 6.2))
    sns.heatmap(
        heatmap_data,
        cmap="RdBu_r",
        center=0.0,
        linewidths=0.4,
        linecolor="white",
        annot=annotations,
        fmt="",
        cbar_kws={"label": "Median theta PSD difference"},
        ax=axis,
    )
    axis.set_xlabel("Group comparison")
    axis.set_ylabel("Channel")
    axis.set_title("Theta PSD pairwise effects and p-values")
    save_figure(fig, "figure_05_theta_psd_group_heatmap")


def main():
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_relevance_topomaps()
    plot_theta_psd_pairwise_topomaps()
    make_performance_table()
    plot_family_control_panels()
    plot_theta_psd_group_heatmap()
    print(f"Saved revised figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
