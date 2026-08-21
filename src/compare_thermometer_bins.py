"""Thermometer-bin sensitivity for the Diff-Logic transfer analysis.

Addresses reviewer 1 minor comment 1 and reviewer 2 comment 4.

Only the number of thermometer bins changes. The logic-layer width is
target_parameters / (depth * 16) and is independent of the input dimension, so
every setting keeps exactly 249,984 trainable Boolean-function logits. T = 15 is
re-run under the same conditions and used as the internal reference so the sweep
is self-consistent.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from train_helpers import ROOT_DIR


RUNS = [
    (5, "benchmark_250k_psd_tbins_05"),
    (10, "benchmark_250k_psd_tbins_10"),
    (15, "benchmark_250k_psd_tbins_15"),
    (20, "benchmark_250k_psd_tbins_20"),
]
REFERENCE_BINS = 15
PUBLISHED_RUN = "benchmark_250k_psd"
CONTRASTS = [("A+P-", "N"), ("A+P+", "N"), ("A+P+", "A+P-")]
GROUP_ORDER = ["N", "A+P-", "A+P+"]


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def bootstrap_interval(values):
    return np.quantile(values, [0.025, 0.975])


def auc_from_groups(target, reference):
    pairwise_differences = target[:, None] - reference[None, :]
    return (
        np.count_nonzero(pairwise_differences > 0)
        + 0.5 * np.count_nonzero(pairwise_differences == 0)
    ) / pairwise_differences.size


def load_clinical_by_seed(run_dir):
    rows = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        predictions = pd.read_csv(seed_dir / "alz_test_subject_predictions_all_folds.csv")
        metrics = pd.read_csv(seed_dir / "alz_test_metrics_all_folds.csv")
        rows.append(
            {
                "seed": int(seed_dir.name.split("_")[1]),
                "balanced_accuracy": balanced_accuracy_score(
                    predictions["true_label"], predictions["pred_label"]
                ),
                "auroc": roc_auc_score(predictions["true_label"], predictions["mean_p_ad"]),
                "input_dim": int(metrics["input_dim"].iloc[0]),
                "parameter_count": int(metrics["parameter_count"].iloc[0]),
                "mean_best_epoch": float(metrics["best_epoch"].mean()),
            }
        )
    return pd.DataFrame(rows)


def load_pearl_subject_scores(run_dir):
    predictions = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        seed_predictions = pd.read_csv(seed_dir / "pearl_subject_predictions_ensemble.csv")
        predictions.append(seed_predictions)
    return (
        pd.concat(predictions, ignore_index=True)
        .groupby(["participant_id", "group"], as_index=False)
        .agg(mean_positive_score=("mean_p_ad", "mean"))
    )


def summarize_run(bins, run_dir, n_bootstrap, seed):
    clinical = load_clinical_by_seed(run_dir)
    pearl = load_pearl_subject_scores(run_dir)
    group_summary = pearl.groupby("group")["mean_positive_score"].agg(["mean", "std"])

    summary = {
        "thermometer_bins": bins,
        "input_dim": int(clinical["input_dim"].iloc[0]),
        "parameter_count": int(clinical["parameter_count"].iloc[0]),
        "balanced_accuracy_mean": clinical["balanced_accuracy"].mean(),
        "balanced_accuracy_sd": clinical["balanced_accuracy"].std(),
        "clinical_auroc_mean": clinical["auroc"].mean(),
        "clinical_auroc_sd": clinical["auroc"].std(),
        "mean_best_epoch": clinical["mean_best_epoch"].mean(),
    }
    for group in GROUP_ORDER:
        summary[f"{group}_mean"] = group_summary.loc[group, "mean"]
        summary[f"{group}_sd"] = group_summary.loc[group, "std"]

    rng = np.random.default_rng(seed)
    contrast_rows = []
    for target_group, reference_group in CONTRASTS:
        target = pearl.loc[pearl["group"].eq(target_group), "mean_positive_score"].to_numpy()
        reference = pearl.loc[pearl["group"].eq(reference_group), "mean_positive_score"].to_numpy()
        bootstrap_differences = np.empty(n_bootstrap)
        bootstrap_aurocs = np.empty(n_bootstrap)
        for index in range(n_bootstrap):
            sampled_target = rng.choice(target, len(target), replace=True)
            sampled_reference = rng.choice(reference, len(reference), replace=True)
            bootstrap_differences[index] = sampled_target.mean() - sampled_reference.mean()
            bootstrap_aurocs[index] = auc_from_groups(sampled_target, sampled_reference)
        difference_ci = bootstrap_interval(bootstrap_differences)
        auroc_ci = bootstrap_interval(bootstrap_aurocs)
        contrast_rows.append(
            {
                "thermometer_bins": bins,
                "contrast": f"{target_group} vs {reference_group}",
                "mean_difference": target.mean() - reference.mean(),
                "mean_difference_ci_low": difference_ci[0],
                "mean_difference_ci_high": difference_ci[1],
                "ranking_auroc": auc_from_groups(target, reference),
                "ranking_auroc_ci_low": auroc_ci[0],
                "ranking_auroc_ci_high": auroc_ci[1],
                "p_uncorrected": mannwhitneyu(target, reference, alternative="two-sided").pvalue,
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_holm"] = holm_adjust(contrasts["p_uncorrected"].to_numpy())
    return summary, contrasts, pearl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=ROOT_DIR / "outputs/difflogic_45hz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs/revision_2026/thermometer_bin_sensitivity",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    contrasts = []
    scores = {}
    for run_index, (bins, run_name) in enumerate(RUNS):
        run_dir = args.runs_dir / run_name
        if not run_dir.exists():
            print(f"Skipping missing run: {run_name}")
            continue
        summary, run_contrasts, pearl = summarize_run(
            bins, run_dir, args.bootstrap, args.seed + run_index
        )
        summaries.append(summary)
        contrasts.append(run_contrasts)
        scores[bins] = pearl

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "model_summary.csv", index=False)
    contrast_frame = pd.concat(contrasts, ignore_index=True)
    contrast_frame.to_csv(args.output_dir / "group_contrasts.csv", index=False)

    agreement_rows = []
    if REFERENCE_BINS in scores:
        reference_scores = scores[REFERENCE_BINS]
        for bins, pearl in scores.items():
            if bins == REFERENCE_BINS:
                continue
            merged = reference_scores.merge(
                pearl, on=["participant_id", "group"], suffixes=("_reference", "_variant")
            )
            pearson_r, pearson_p = pearsonr(
                merged["mean_positive_score_reference"], merged["mean_positive_score_variant"]
            )
            spearman_rho, spearman_p = spearmanr(
                merged["mean_positive_score_reference"], merged["mean_positive_score_variant"]
            )
            agreement_rows.append(
                {
                    "thermometer_bins": bins,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "mean_absolute_difference": float(
                        np.mean(
                            np.abs(
                                merged["mean_positive_score_variant"]
                                - merged["mean_positive_score_reference"]
                            )
                        )
                    ),
                }
            )
    pd.DataFrame(agreement_rows).to_csv(
        args.output_dir / "agreement_with_t15.csv", index=False
    )

    published_dir = args.runs_dir / PUBLISHED_RUN
    if published_dir.exists() and REFERENCE_BINS in scores:
        published = load_pearl_subject_scores(published_dir)
        merged = published.merge(
            scores[REFERENCE_BINS], on=["participant_id", "group"], suffixes=("_published", "_rerun")
        )
        published_clinical = load_clinical_by_seed(published_dir)
        pearson_r, _ = pearsonr(
            merged["mean_positive_score_published"], merged["mean_positive_score_rerun"]
        )
        pd.DataFrame(
            [
                {
                    "published_balanced_accuracy": published_clinical["balanced_accuracy"].mean(),
                    "rerun_balanced_accuracy": summary_frame.loc[
                        summary_frame["thermometer_bins"].eq(REFERENCE_BINS),
                        "balanced_accuracy_mean",
                    ].iloc[0],
                    "published_auroc": published_clinical["auroc"].mean(),
                    "rerun_auroc": summary_frame.loc[
                        summary_frame["thermometer_bins"].eq(REFERENCE_BINS),
                        "clinical_auroc_mean",
                    ].iloc[0],
                    "score_pearson_r": pearson_r,
                    "score_mean_absolute_difference": float(
                        np.mean(
                            np.abs(
                                merged["mean_positive_score_rerun"]
                                - merged["mean_positive_score_published"]
                            )
                        )
                    ),
                }
            ]
        ).to_csv(args.output_dir / "rerun_vs_published_t15.csv", index=False)

    print(
        summary_frame[
            [
                "thermometer_bins",
                "input_dim",
                "parameter_count",
                "balanced_accuracy_mean",
                "balanced_accuracy_sd",
                "clinical_auroc_mean",
                "N_mean",
                "A+P-_mean",
                "A+P+_mean",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print()
    print(
        contrast_frame[
            [
                "thermometer_bins",
                "contrast",
                "mean_difference",
                "mean_difference_ci_low",
                "mean_difference_ci_high",
                "ranking_auroc",
                "p_uncorrected",
                "p_holm",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print()
    print(pd.DataFrame(agreement_rows).round(4).to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
