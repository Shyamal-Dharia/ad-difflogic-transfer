from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from train_helpers import ROOT_DIR


OUTPUT_DIR = ROOT_DIR / "outputs/publication_figures"
DIFFLOGIC_DIR = ROOT_DIR / "outputs/difflogic"
FOCUSED_PSD_DIR = ROOT_DIR / "outputs/focused_psd_statistics"

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


def read_main_predictions():
    return pd.read_csv(
        DIFFLOGIC_DIR
        / "medium_interpretable/summary/pearl_subject_predictions_seed_average.csv"
    )


def read_run_group_summary(run_name):
    return pd.read_csv(DIFFLOGIC_DIR / run_name / "summary/pearl_group_summary_seed_average.csv")


def plot_alz_performance():
    metrics = pd.read_csv(
        DIFFLOGIC_DIR / "medium_interpretable/summary/alz_test_metrics_all_seeds.csv"
    )
    plot_data = metrics.melt(
        value_vars=["balanced_accuracy", "auroc"],
        var_name="metric",
        value_name="value",
    )
    labels = {
        "balanced_accuracy": "Balanced accuracy",
        "auroc": "AUROC",
    }
    plot_data["metric"] = plot_data["metric"].map(labels)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.boxplot(data=plot_data, x="metric", y="value", color="white", showfliers=False, ax=ax)
    sns.stripplot(data=plot_data, x="metric", y="value", color="black", alpha=0.55, jitter=0.18, ax=ax)
    ax.axhline(0.5, color="0.5", linewidth=1, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Subject-level held-out performance")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("ALZ_FTD AD-vs-control performance")
    save_figure(fig, "figure_01_alz_performance")


def plot_pearl_scores():
    data = read_main_predictions()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    sns.boxplot(
        data=data,
        x="group",
        y="mean_p_ad",
        order=GROUP_ORDER,
        color="white",
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="group",
        y="mean_p_ad",
        order=GROUP_ORDER,
        color="black",
        alpha=0.65,
        jitter=0.18,
        ax=ax,
    )
    ax.set_xlabel("PEARL genetic-risk group")
    ax.set_ylabel("Transferred AD-like EEG score")
    ax.set_title("PEARL transfer scores")
    ax.set_ylim(0.0, 1.0)
    save_figure(fig, "figure_02_pearl_transfer_scores")


def plot_feature_family_controls():
    rows = []
    run_labels = {
        "medium_interpretable": "PSD",
        "medium_shuffled_alz_labels": "PSD shuffled labels",
        "hfd_kmax16_medium": "HFD kmax=16",
        "hfd_kmaxnone_medium": "HFD kmax=None",
    }

    for run_name, run_label in run_labels.items():
        summary = read_run_group_summary(run_name)
        summary["run"] = run_label
        rows.append(summary[["run", "group", "median_p_ad", "mean_p_ad"]])

    data = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.pointplot(
        data=data,
        x="run",
        y="median_p_ad",
        hue="group",
        hue_order=GROUP_ORDER,
        order=list(run_labels.values()),
        dodge=0.35,
        markers="o",
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Median PEARL AD-like EEG score")
    ax.set_title("Transfer pattern across controls and feature families")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Group", frameon=False)
    save_figure(fig, "figure_03_feature_family_controls")


def plot_interpretation_heatmap():
    path = (
        DIFFLOGIC_DIR
        / "medium_interpretable/interpretation/soft_integrated_gradients"
        / "pearl_signed_relevance_group_contrasts.csv"
    )
    data = pd.read_csv(path)
    heatmap_data = data.pivot(index="channel", columns="band", values="A+P+_minus_N")
    heatmap_data = heatmap_data.loc[CHANNEL_ORDER, BAND_ORDER]

    fig, ax = plt.subplots(figsize=(5.4, 5.8))
    sns.heatmap(
        heatmap_data,
        cmap="vlag",
        center=0.0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Integrated-gradient relevance difference"},
        ax=ax,
    )
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("EEG channel")
    ax.set_title("Model relevance: A+P+ minus N")
    save_figure(fig, "figure_04_integrated_gradients_heatmap")


def plot_focused_theta_score():
    path = FOCUSED_PSD_DIR / "focused_theta_subject_score.csv"
    if not path.exists():
        return

    data = pd.read_csv(path)
    y_column = "mean_model_relevant_theta_log_relative_bandpower"

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    sns.boxplot(data=data, x="group", y=y_column, order=GROUP_ORDER, color="white", showfliers=False, ax=ax)
    sns.stripplot(data=data, x="group", y=y_column, order=GROUP_ORDER, color="black", alpha=0.65, jitter=0.18, ax=ax)
    ax.set_xlabel("PEARL genetic-risk group")
    ax.set_ylabel("Mean theta log-relative bandpower")
    ax.set_title("Focused PSD statistic: model-relevant theta")
    save_figure(fig, "figure_05_focused_theta_psd")


def plot_top_focused_psd_features():
    subject_values_path = FOCUSED_PSD_DIR / "focused_psd_subject_values.csv"
    relevance_path = (
        DIFFLOGIC_DIR
        / "medium_interpretable/interpretation/soft_integrated_gradients"
        / "pearl_signed_relevance_group_contrasts.csv"
    )
    if not subject_values_path.exists():
        return

    subject_values = pd.read_csv(subject_values_path)
    relevance = pd.read_csv(relevance_path)
    top_features = (
        relevance
        .sort_values("A+P+_minus_N", ascending=False)
        .head(6)
        [["channel", "band"]]
    )
    data = subject_values.merge(top_features, on=["channel", "band"], how="inner")
    data["feature"] = data["channel"] + " " + data["band"]

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.6), sharey=False)
    axes = axes.ravel()

    for axis, feature in zip(axes, top_features.assign(feature=top_features["channel"] + " " + top_features["band"])["feature"]):
        feature_data = data[data["feature"].eq(feature)]
        sns.boxplot(
            data=feature_data,
            x="group",
            y="log_relative_bandpower",
            order=GROUP_ORDER,
            color="white",
            showfliers=False,
            ax=axis,
        )
        sns.stripplot(
            data=feature_data,
            x="group",
            y="log_relative_bandpower",
            order=GROUP_ORDER,
            color="black",
            alpha=0.55,
            jitter=0.18,
            ax=axis,
        )
        axis.set_title(feature)
        axis.set_xlabel("")
        axis.set_ylabel("Log-relative bandpower")

    fig.suptitle("Focused PSD values for top model-relevant features", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure_06_top_focused_psd_features")


def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plot_alz_performance()
    plot_pearl_scores()
    plot_feature_family_controls()
    plot_interpretation_heatmap()
    plot_focused_theta_score()
    plot_top_focused_psd_features()
    print(f"Saved publication figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
