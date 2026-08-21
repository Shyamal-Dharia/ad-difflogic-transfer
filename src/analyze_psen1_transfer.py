"""Exploratory zero-shot transfer to the DS007427 PSEN1-E280A cohort.

Requested by reviewer 3. The trained clinical AD/CN models are applied to
asymptomatic PSEN1-E280A carriers (G1) and matched non-carriers (G2) using the
same zero-shot procedure as the primary genetic-risk analysis, with no
retraining. The single CTR recording is reported but excluded from statistics.

This is a secondary analysis of a distinct genetic aetiology and age range, not
a validation of the APOE/PICALM result.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from train_helpers import BANDS, ROOT_DIR, load_target_dataset


DIFFLOGIC_DIR = ROOT_DIR / "outputs" / "difflogic_45hz"
RUN = "benchmark_250k_psd"
CLINICAL_CSV = ROOT_DIR / "datasets/statistics_psd_45hz/subject_log_relative_bandpower.csv"
BAND_NAMES = list(BANDS)

# Cohort palette, validated all-pairs on the light surface: worst CVD dE 9.2,
# worst normal-vision dE 24.0. Group identity is carried by axis labels.
COHORT_COLORS = {"Clinical": "#2a78d6", "PEARL": "#eb6834", "PSEN1": "#1baf7a"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GROUP_NAMES = {"A": "AD", "C": "CN", "F": "FTD", "A+P-": "A+P−"}
COHORTS = [
    ("Clinical", "ALZ_FTD", ["CN", "FTD", "AD"]),
    ("PEARL", "PEARL", ["N", "A+P−", "A+P+"]),
    ("PSEN1", "DS007427", ["G2", "G1"]),
]
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260816


def auc_from_groups(target, reference):
    differences = target[:, None] - reference[None, :]
    return (
        np.count_nonzero(differences > 0) + 0.5 * np.count_nonzero(differences == 0)
    ) / differences.size


def load_psen1_bandpower():
    """Subject-level log-relative band power, channel-averaged.

    Uses the repository's own loader so the band aggregation is identical to the
    one the models consume: the cached `x` array holds per-frequency-bin
    log-relative PSD, not band powers, and must be aggregated with `band_mask`.
    """
    subjects, _ = load_target_dataset("DS007427", feature_kind="psd")
    rows = []
    for subject in subjects:
        # subject["x"] is (epochs, channels, bands) after band aggregation.
        per_band = subject["x"].mean(axis=0).mean(axis=0)
        for band_index, band in enumerate(BAND_NAMES):
            rows.append(
                {
                    "dataset": "DS007427",
                    "participant_id": subject["participant_id"],
                    "group": subject["group"],
                    "band": band,
                    "log_relative_bandpower": float(per_band[band_index]),
                }
            )
    return pd.DataFrame(rows)


def transfer_contrast(scores):
    carriers = scores.loc[scores["group"].eq("G1"), "mean_p_ad"].to_numpy()
    controls = scores.loc[scores["group"].eq("G2"), "mean_p_ad"].to_numpy()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(N_BOOTSTRAP)
    aurocs = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sampled_carriers = rng.choice(carriers, carriers.size, replace=True)
        sampled_controls = rng.choice(controls, controls.size, replace=True)
        differences[index] = sampled_carriers.mean() - sampled_controls.mean()
        aurocs[index] = auc_from_groups(sampled_carriers, sampled_controls)
    difference_ci = np.quantile(differences, [0.025, 0.975])
    auroc_ci = np.quantile(aurocs, [0.025, 0.975])

    return pd.DataFrame(
        [
            {
                "contrast": "G1 (PSEN1 carriers) vs G2 (non-carriers)",
                "n_carriers": carriers.size,
                "n_controls": controls.size,
                "mean_carriers": carriers.mean(),
                "mean_controls": controls.mean(),
                "mean_difference": carriers.mean() - controls.mean(),
                "ci_low": difference_ci[0],
                "ci_high": difference_ci[1],
                "ranking_auroc": auc_from_groups(carriers, controls),
                "auroc_ci_low": auroc_ci[0],
                "auroc_ci_high": auroc_ci[1],
                "p_mannwhitney": mannwhitneyu(
                    carriers, controls, alternative="two-sided"
                ).pvalue,
            }
        ]
    )


def plot_three_cohorts(bandpower, output_path):
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
    figure, axes = plt.subplots(3, 5, figsize=(13.5, 8.4), sharey=True)
    y_min = bandpower["log_relative_bandpower"].min() - 0.10
    y_max = bandpower["log_relative_bandpower"].max() + 0.10
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for row, (cohort_name, dataset, groups) in enumerate(COHORTS):
        cohort_data = bandpower[bandpower["dataset"].eq(dataset)]
        color = COHORT_COLORS[cohort_name]
        counts = cohort_data.groupby("group")["participant_id"].nunique()

        for column, band in enumerate(BAND_NAMES):
            axis = axes[row, column]
            axis.set_axisbelow(True)
            axis.yaxis.grid(True, color="#e3e1da", linewidth=0.6)
            axis.xaxis.grid(False)

            for position, group in enumerate(groups):
                values = cohort_data.loc[
                    cohort_data["group"].eq(group) & cohort_data["band"].eq(band),
                    "log_relative_bandpower",
                ].to_numpy()
                body = axis.violinplot(
                    values, positions=[position], widths=0.78,
                    showextrema=False, showmedians=False,
                )
                for part in body["bodies"]:
                    part.set_facecolor(color)
                    part.set_alpha(0.16)
                    part.set_edgecolor("none")

                jitter = rng.uniform(-0.11, 0.11, size=values.size)
                axis.scatter(
                    position + jitter, values, s=5.5, color=color, alpha=0.55,
                    linewidths=0.25, edgecolors="white", zorder=3,
                )
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                axis.vlines(position, q25, q75, color=INK_PRIMARY, linewidth=1.4, zorder=4)
                axis.hlines(
                    median, position - 0.20, position + 0.20,
                    color=INK_PRIMARY, linewidth=2.0, zorder=5,
                )

            axis.set_xlim(-0.62, len(groups) - 0.38)
            axis.set_ylim(y_min, y_max)
            axis.set_xticks(range(len(groups)))
            axis.set_xticklabels(
                [f"{group}\n({counts[group]})" for group in groups], fontsize=10
            )
            axis.set_xlabel("")
            axis.tick_params(length=0)
            if row == 0:
                axis.set_title(band.capitalize(), fontsize=12.5, color=INK_PRIMARY, pad=7)
            if column == 0:
                axis.set_ylabel(
                    f"{cohort_name}\nlog$_{{10}}$ relative bandpower", fontsize=11.5
                )
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "revision_2026" / "psen1_ds007427",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(
        DIFFLOGIC_DIR / RUN / "summary" / "ds007427_subject_predictions_seed_average.csv"
    )
    contrast = transfer_contrast(scores)
    contrast.to_csv(args.output_dir / "psen1_transfer_contrast.csv", index=False)

    psen1 = load_psen1_bandpower()
    clinical = pd.read_csv(CLINICAL_CSV)
    clinical["group"] = clinical["group"].replace(GROUP_NAMES)
    clinical = (
        clinical.groupby(["dataset", "participant_id", "group", "band"], as_index=False)
        .agg(log_relative_bandpower=("log_relative_bandpower", "mean"))
    )
    bandpower = pd.concat([clinical, psen1[psen1["group"].ne("CTR")]], ignore_index=True)
    bandpower.to_csv(args.output_dir / "subject_bandpower_three_cohorts.csv", index=False)

    figure_path = args.output_dir / "psd_group_distributions_with_psen1.png"
    plot_three_cohorts(bandpower, figure_path)

    band_summary = (
        psen1[psen1["group"].ne("CTR")]
        .groupby(["group", "band"], as_index=False)
        .agg(
            n=("participant_id", "nunique"),
            mean=("log_relative_bandpower", "mean"),
            sd=("log_relative_bandpower", "std"),
        )
    )
    band_summary.to_csv(args.output_dir / "psen1_bandpower_summary.csv", index=False)

    print(contrast.round(4).to_string(index=False))
    print()
    print(band_summary.round(3).to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
