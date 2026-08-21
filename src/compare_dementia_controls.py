import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from train_helpers import ROOT_DIR


RUNS = [
    ("CN vs AD", "250k", "benchmark_250k_psd"),
    ("CN vs FTD", "250k", "cn_vs_ftd_benchmark_250k_psd"),
    ("CN vs AD+FTD", "250k", "cn_vs_ad_ftd_benchmark_250k_psd"),
    ("CN vs AD", "500k", "benchmark_500k_psd"),
    ("CN vs FTD", "500k", "cn_vs_ftd_benchmark_500k_psd"),
    ("CN vs AD+FTD", "500k", "cn_vs_ad_ftd_benchmark_500k_psd"),
]
CONTRASTS = [("A+P+", "N"), ("A+P+", "A+P-"), ("A+P-", "N")]


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
        rows.append(
            {
                "seed": int(seed_dir.name.split("_")[1]),
                "n_source_subjects": len(predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    predictions["true_label"], predictions["pred_label"]
                ),
                "auroc": roc_auc_score(predictions["true_label"], predictions["mean_p_ad"]),
            }
        )
    return pd.DataFrame(rows)


def load_pearl_subject_scores(run_dir):
    predictions = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        seed_predictions = pd.read_csv(seed_dir / "pearl_subject_predictions_ensemble.csv")
        seed_predictions["seed"] = int(seed_dir.name.split("_")[1])
        predictions.append(seed_predictions)
    return (
        pd.concat(predictions, ignore_index=True)
        .groupby(["participant_id", "group"], as_index=False)
        .agg(mean_positive_score=("mean_p_ad", "mean"))
    )


def summarize_run(task, size, run_name, run_dir, n_bootstrap, seed):
    clinical = load_clinical_by_seed(run_dir)
    pearl = load_pearl_subject_scores(run_dir)
    group_summary = pearl.groupby("group")["mean_positive_score"].agg(["mean", "std"])
    summary = {
        "task": task,
        "size": size,
        "run_name": run_name,
        "n_source_subjects": int(clinical["n_source_subjects"].iloc[0]),
        "balanced_accuracy_mean": clinical["balanced_accuracy"].mean(),
        "balanced_accuracy_sd": clinical["balanced_accuracy"].std(),
        "clinical_auroc_mean": clinical["auroc"].mean(),
        "clinical_auroc_sd": clinical["auroc"].std(),
    }
    for group in ["N", "A+P-", "A+P+"]:
        summary[f"{group}_mean"] = group_summary.loc[group, "mean"]
        summary[f"{group}_sd"] = group_summary.loc[group, "std"]

    rng = np.random.default_rng(seed)
    contrast_rows = []
    for target_group, reference_group in CONTRASTS:
        target = pearl.loc[pearl["group"].eq(target_group), "mean_positive_score"].to_numpy()
        reference = pearl.loc[
            pearl["group"].eq(reference_group), "mean_positive_score"
        ].to_numpy()
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
                "task": task,
                "size": size,
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


def paired_contrast_difference(ad, comparator, target_group, reference_group, n_bootstrap, rng):
    paired = ad.merge(
        comparator,
        on=["participant_id", "group"],
        suffixes=("_ad", "_comparator"),
        validate="one_to_one",
    )
    target = paired[paired["group"].eq(target_group)]
    reference = paired[paired["group"].eq(reference_group)]

    def difference(target_rows, reference_rows):
        ad_difference = (
            target_rows["mean_positive_score_ad"].mean()
            - reference_rows["mean_positive_score_ad"].mean()
        )
        comparator_difference = (
            target_rows["mean_positive_score_comparator"].mean()
            - reference_rows["mean_positive_score_comparator"].mean()
        )
        return ad_difference - comparator_difference

    bootstrap = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sampled_target = target.iloc[rng.integers(len(target), size=len(target))]
        sampled_reference = reference.iloc[rng.integers(len(reference), size=len(reference))]
        bootstrap[index] = difference(sampled_target, sampled_reference)
    ci_low, ci_high = bootstrap_interval(bootstrap)
    return difference(target, reference), ci_low, ci_high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir", type=Path, default=ROOT_DIR / "outputs/difflogic_45hz"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs/revision_2026/dementia_control_comparison",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    summaries = []
    contrasts = []
    pearl_scores = {}
    for run_index, (task, size, run_name) in enumerate(RUNS):
        summary, run_contrasts, pearl = summarize_run(
            task,
            size,
            run_name,
            args.runs_dir / run_name,
            args.bootstrap,
            args.seed + run_index,
        )
        summaries.append(summary)
        contrasts.append(run_contrasts)
        pearl_scores[(task, size)] = pearl

    paired_rows = []
    rng = np.random.default_rng(args.seed + 100)
    for size in ["250k", "500k"]:
        ad = pearl_scores[("CN vs AD", size)]
        for comparator_task in ["CN vs FTD", "CN vs AD+FTD"]:
            for target_group, reference_group in CONTRASTS[:2]:
                difference, ci_low, ci_high = paired_contrast_difference(
                    ad,
                    pearl_scores[(comparator_task, size)],
                    target_group,
                    reference_group,
                    args.bootstrap,
                    rng,
                )
                paired_rows.append(
                    {
                        "size": size,
                        "comparison": f"CN vs AD minus {comparator_task}",
                        "contrast": f"{target_group} vs {reference_group}",
                        "difference_of_mean_differences": difference,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.output_dir / "model_summary.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(
        args.output_dir / "pearl_group_contrasts.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "paired_contrast_differences.csv", index=False
    )
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nSaved comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
