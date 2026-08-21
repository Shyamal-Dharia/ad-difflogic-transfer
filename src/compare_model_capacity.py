"""Model-capacity scaling for Diff-Logic and the neural baselines.

Addresses reviewer 3's request for results from models much larger than 250k
parameters.

Each capacity tier uses identical folds, seeds, features, and the same zero-shot
transfer protocol, so only the trainable parameter budget differs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from train_helpers import ROOT_DIR


RUNS = [
    ("Diff-Logic", "250k", "benchmark_250k_psd"),
    ("Diff-Logic", "500k", "benchmark_500k_psd"),
    ("Diff-Logic", "1M", "benchmark_1M_psd"),
    ("Diff-Logic", "4M", "benchmark_4M_psd"),
    ("MLP", "250k", "mlp_250k_psd"),
    ("MLP", "1M", "mlp_1M_psd"),
    ("1D-Conv", "250k", "conv1d_250k_psd"),
    ("1D-Conv", "1M", "conv1d_1M_psd"),
]
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
                "balanced_accuracy": balanced_accuracy_score(
                    predictions["true_label"], predictions["pred_label"]
                ),
                "auroc": roc_auc_score(predictions["true_label"], predictions["mean_p_ad"]),
                "parameter_count": int(metrics["parameter_count"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def load_pearl_subject_scores(run_dir):
    predictions = [
        pd.read_csv(seed_dir / "pearl_subject_predictions_ensemble.csv")
        for seed_dir in sorted(run_dir.glob("seed_*"))
    ]
    return (
        pd.concat(predictions, ignore_index=True)
        .groupby(["participant_id", "group"], as_index=False)
        .agg(mean_positive_score=("mean_p_ad", "mean"))
    )


def summarize_run(model, size, run_dir):
    clinical = load_clinical_by_seed(run_dir)
    pearl = load_pearl_subject_scores(run_dir)
    group_summary = pearl.groupby("group")["mean_positive_score"].agg(["mean", "std"])

    summary = {
        "model": model,
        "size": size,
        "n_seeds": len(clinical),
        "parameter_count": int(clinical["parameter_count"].iloc[0]),
        "balanced_accuracy_mean": clinical["balanced_accuracy"].mean(),
        "balanced_accuracy_sd": clinical["balanced_accuracy"].std(),
        "clinical_auroc_mean": clinical["auroc"].mean(),
        "clinical_auroc_sd": clinical["auroc"].std(),
    }
    for group in GROUP_ORDER:
        summary[f"{group}_mean"] = group_summary.loc[group, "mean"]

    contrast_rows = []
    for target_group, reference_group in CONTRASTS:
        target = pearl.loc[pearl["group"].eq(target_group), "mean_positive_score"].to_numpy()
        reference = pearl.loc[pearl["group"].eq(reference_group), "mean_positive_score"].to_numpy()
        contrast_rows.append(
            {
                "model": model,
                "size": size,
                "contrast": f"{target_group} vs {reference_group}",
                "mean_difference": target.mean() - reference.mean(),
                "ranking_auroc": auc_from_groups(target, reference),
                "p_uncorrected": mannwhitneyu(target, reference, alternative="two-sided").pvalue,
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_holm"] = holm_adjust(contrasts["p_uncorrected"].to_numpy())
    return summary, contrasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=ROOT_DIR / "outputs/difflogic_45hz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs/revision_2026/model_capacity_scaling",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    contrasts = []
    for model, size, run_name in RUNS:
        run_dir = args.runs_dir / run_name
        if not run_dir.exists() or not list(run_dir.glob("seed_*")):
            print(f"Skipping missing run: {run_name}")
            continue
        summary, run_contrasts = summarize_run(model, size, run_dir)
        summaries.append(summary)
        contrasts.append(run_contrasts)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "capacity_summary.csv", index=False)
    contrast_frame = pd.concat(contrasts, ignore_index=True)
    contrast_frame.to_csv(args.output_dir / "capacity_group_contrasts.csv", index=False)

    print(summary_frame.round(4).to_string(index=False))
    print()
    print(
        contrast_frame[contrast_frame["contrast"].ne("A+P- vs N")]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
