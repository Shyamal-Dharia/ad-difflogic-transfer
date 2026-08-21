"""Single canonical source for every genetic-risk transfer contrast.

Tables 4, 5, and 6 of the submitted manuscript each reported the same contrasts
from a separately seeded bootstrap, producing inconsistent confidence intervals
for identical quantities (for example A+P+ vs A+P- as [0.019, 0.198] in one table
and [0.017, 0.197] in another). This script computes each contrast exactly once,
with one fixed bootstrap seed, so every table can read the same numbers.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from train_helpers import ROOT_DIR


RUNS = [
    ("Diff-Logic", "250k", "cn_vs_ad", "benchmark_250k_psd"),
    ("MLP", "250k", "cn_vs_ad", "mlp_250k_psd"),
    ("1D-Conv", "250k", "cn_vs_ad", "conv1d_250k_psd"),
    ("Transformer", "250k", "cn_vs_ad", "transformer_250k_psd"),
    ("Diff-Logic", "250k", "cn_vs_ftd", "cn_vs_ftd_benchmark_250k_psd"),
    ("Diff-Logic", "250k", "cn_vs_ad_ftd", "cn_vs_ad_ftd_benchmark_250k_psd"),
]
CONTRASTS = [("A+P-", "N"), ("A+P+", "N"), ("A+P+", "A+P-")]
GROUP_ORDER = ["N", "A+P-", "A+P+"]
BOOTSTRAP_SEED = 20260816
N_BOOTSTRAP = 10_000


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
        rows.append(
            {
                "balanced_accuracy": balanced_accuracy_score(
                    predictions["true_label"], predictions["pred_label"]
                ),
                "auroc": roc_auc_score(predictions["true_label"], predictions["mean_p_ad"]),
            }
        )
    return pd.DataFrame(rows)


def load_subject_scores(run_dir):
    frames = [
        pd.read_csv(seed_dir / "pearl_subject_predictions_ensemble.csv")
        for seed_dir in sorted(run_dir.glob("seed_*"))
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(["participant_id", "group"], as_index=False)
        .agg(score=("mean_p_ad", "mean"))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=ROOT_DIR / "outputs/difflogic_45hz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs/revision_2026/canonical_contrasts",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    group_rows = []
    contrast_rows = []
    omnibus_rows = []

    for model, size, task, run_name in RUNS:
        run_dir = args.runs_dir / run_name
        if not run_dir.exists():
            print(f"Skipping missing run: {run_name}")
            continue

        clinical = load_clinical_by_seed(run_dir)
        scores = load_subject_scores(run_dir)
        label = {"model": model, "size": size, "clinical_task": task}

        for group in GROUP_ORDER:
            values = scores.loc[scores["group"].eq(group), "score"].to_numpy()
            group_rows.append(
                {
                    **label,
                    "group": group,
                    "n_subjects": values.size,
                    "mean": values.mean(),
                    "sd_across_subjects": values.std(ddof=1),
                    "median": float(np.median(values)),
                    "clinical_balanced_accuracy_mean": clinical["balanced_accuracy"].mean(),
                    "clinical_balanced_accuracy_sd": clinical["balanced_accuracy"].std(),
                    "clinical_auroc_mean": clinical["auroc"].mean(),
                    "clinical_auroc_sd": clinical["auroc"].std(),
                }
            )

        omnibus_rows.append(
            {
                **label,
                "kruskal_p": kruskal(
                    *[scores.loc[scores["group"].eq(g), "score"].to_numpy() for g in GROUP_ORDER]
                ).pvalue,
            }
        )

        # One generator, one seed, for every contrast of this run.
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        rows = []
        for target_group, reference_group in CONTRASTS:
            target = scores.loc[scores["group"].eq(target_group), "score"].to_numpy()
            reference = scores.loc[scores["group"].eq(reference_group), "score"].to_numpy()
            differences = np.empty(N_BOOTSTRAP)
            aurocs = np.empty(N_BOOTSTRAP)
            for index in range(N_BOOTSTRAP):
                sampled_target = rng.choice(target, target.size, replace=True)
                sampled_reference = rng.choice(reference, reference.size, replace=True)
                differences[index] = sampled_target.mean() - sampled_reference.mean()
                aurocs[index] = auc_from_groups(sampled_target, sampled_reference)
            difference_ci = np.quantile(differences, [0.025, 0.975])
            auroc_ci = np.quantile(aurocs, [0.025, 0.975])
            rows.append(
                {
                    **label,
                    "contrast": f"{target_group} vs {reference_group}",
                    "mean_difference": target.mean() - reference.mean(),
                    "mean_difference_ci_low": difference_ci[0],
                    "mean_difference_ci_high": difference_ci[1],
                    "ranking_auroc": auc_from_groups(target, reference),
                    "ranking_auroc_ci_low": auroc_ci[0],
                    "ranking_auroc_ci_high": auroc_ci[1],
                    "p_uncorrected": mannwhitneyu(
                        target, reference, alternative="two-sided"
                    ).pvalue,
                }
            )
        frame = pd.DataFrame(rows)
        frame["p_holm"] = holm_adjust(frame["p_uncorrected"].to_numpy())
        contrast_rows.append(frame)

    group_frame = pd.DataFrame(group_rows)
    contrast_frame = pd.concat(contrast_rows, ignore_index=True)
    group_frame.to_csv(args.output_dir / "canonical_group_summary.csv", index=False)
    contrast_frame.to_csv(args.output_dir / "canonical_contrasts.csv", index=False)
    pd.DataFrame(omnibus_rows).to_csv(args.output_dir / "canonical_omnibus.csv", index=False)

    print("Group summary (subject-level)")
    print(
        group_frame[["model", "clinical_task", "group", "n_subjects", "mean", "sd_across_subjects"]]
        .round(4)
        .to_string(index=False)
    )
    print("\nCanonical contrasts (single bootstrap seed)")
    print(
        contrast_frame[
            [
                "model",
                "clinical_task",
                "contrast",
                "mean_difference",
                "mean_difference_ci_low",
                "mean_difference_ci_high",
                "ranking_auroc",
                "ranking_auroc_ci_low",
                "ranking_auroc_ci_high",
                "p_uncorrected",
                "p_holm",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
