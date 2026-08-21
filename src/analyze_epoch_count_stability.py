import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from compare_dementia_controls import auc_from_groups
from train_helpers import ROOT_DIR


SUBSET_EPOCH_COUNTS = [15, 30, 45, 60]
CONTRASTS = [("A+P+", "N"), ("A+P+", "A+P-")]


def load_epoch_ensemble(run_dir):
    predictions = []
    for prediction_path in sorted(run_dir.glob("seed_*/fold_*/pearl_epoch_predictions.csv")):
        table = pd.read_csv(prediction_path)
        table["epoch_index"] = table.groupby("participant_id").cumcount()
        predictions.append(
            table[["participant_id", "group", "epoch_index", "p_ad"]]
        )

    if not predictions:
        raise FileNotFoundError(f"No PEARL epoch predictions found in {run_dir}")

    return (
        pd.concat(predictions, ignore_index=True)
        .groupby(["participant_id", "group", "epoch_index"], as_index=False)
        .agg(mean_p_ad=("p_ad", "mean"))
    )


def subsample_scores(subject_epochs, epoch_count, repeats, rng):
    scores = np.empty((repeats, len(subject_epochs)))
    for repeat in range(repeats):
        for subject_index, epoch_scores in enumerate(subject_epochs):
            sampled_indices = rng.choice(len(epoch_scores), epoch_count, replace=False)
            scores[repeat, subject_index] = epoch_scores[sampled_indices].mean()
    return scores


def summarize_duration(scores, full_scores, groups, epoch_count):
    metric_rows = []
    for subset_scores in scores:
        row = {
            "pearson_r_with_full": pearsonr(subset_scores, full_scores)[0],
            "spearman_rho_with_full": spearmanr(subset_scores, full_scores)[0],
            "mean_absolute_error": np.abs(subset_scores - full_scores).mean(),
        }
        for target_group, reference_group in CONTRASTS:
            target = subset_scores[groups == target_group]
            reference = subset_scores[groups == reference_group]
            contrast_name = f"{target_group}_vs_{reference_group}".replace("+", "plus").replace("-", "minus")
            row[f"{contrast_name}_mean_difference"] = target.mean() - reference.mean()
            row[f"{contrast_name}_ranking_auroc"] = auc_from_groups(target, reference)
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    summary = {
        "epochs": epoch_count,
        "minutes": epoch_count * 4 / 60,
        "repeats": len(scores),
        "median_within_subject_sd": np.median(scores.std(axis=0, ddof=1)),
    }
    for column in metrics:
        summary[f"{column}_mean"] = metrics[column].mean()
        summary[f"{column}_ci_low"] = metrics[column].quantile(0.025)
        summary[f"{column}_ci_high"] = metrics[column].quantile(0.975)
    return summary, metrics


def split_half_reliability(subject_epochs, repeats, rng):
    rows = []
    for _ in range(repeats):
        first_scores = []
        second_scores = []
        for epoch_scores in subject_epochs:
            selected = rng.choice(len(epoch_scores), 60, replace=False)
            first_scores.append(epoch_scores[selected[:30]].mean())
            second_scores.append(epoch_scores[selected[30:]].mean())
        pearson_r = pearsonr(first_scores, second_scores)[0]
        rows.append(
            {
                "pearson_r": pearson_r,
                "spearman_rho": spearmanr(first_scores, second_scores)[0],
                "spearman_brown_reliability": 2 * pearson_r / (1 + pearson_r),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT_DIR / "outputs/difflogic_45hz/benchmark_250k_psd",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs/revision_2026/epoch_count_stability",
    )
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    epoch_predictions = load_epoch_ensemble(args.run_dir)
    subject_table = (
        epoch_predictions.groupby(["participant_id", "group"], as_index=False)
        .agg(n_epochs=("epoch_index", "count"), full_score=("mean_p_ad", "mean"))
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    subject_epochs = [
        epoch_predictions.loc[
            epoch_predictions["participant_id"].eq(participant_id), "mean_p_ad"
        ].to_numpy()
        for participant_id in subject_table["participant_id"]
    ]
    groups = subject_table["group"].to_numpy()
    full_scores = subject_table["full_score"].to_numpy()
    rng = np.random.default_rng(args.seed)

    duration_summaries = []
    replicate_tables = []
    for epoch_count in SUBSET_EPOCH_COUNTS:
        scores = subsample_scores(subject_epochs, epoch_count, args.repeats, rng)
        summary, metrics = summarize_duration(
            scores, full_scores, groups, epoch_count
        )
        duration_summaries.append(summary)
        metrics.insert(0, "repeat", np.arange(1, args.repeats + 1))
        metrics.insert(0, "epochs", epoch_count)
        replicate_tables.append(metrics)

    count_correlations = pd.DataFrame(
        [
            {
                "scope": "All PEARL",
                "n_subjects": len(subject_table),
                "pearson_r": pearsonr(subject_table["n_epochs"], full_scores)[0],
                "pearson_p": pearsonr(subject_table["n_epochs"], full_scores)[1],
                "spearman_rho": spearmanr(subject_table["n_epochs"], full_scores)[0],
                "spearman_p": spearmanr(subject_table["n_epochs"], full_scores)[1],
            }
        ]
    )
    split_half = split_half_reliability(subject_epochs, args.repeats, rng)
    split_half_summary = split_half.agg(["mean", "std", "min", "max"])
    split_half_summary.loc["ci_low"] = split_half.quantile(0.025)
    split_half_summary.loc["ci_high"] = split_half.quantile(0.975)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_table.to_csv(args.output_dir / "subject_epoch_counts_and_scores.csv", index=False)
    pd.DataFrame(duration_summaries).to_csv(
        args.output_dir / "duration_stability_summary.csv", index=False
    )
    pd.concat(replicate_tables, ignore_index=True).to_csv(
        args.output_dir / "duration_stability_replicates.csv", index=False
    )
    count_correlations.to_csv(args.output_dir / "epoch_count_score_correlation.csv", index=False)
    split_half.to_csv(args.output_dir / "split_half_replicates.csv", index=False)
    split_half_summary.to_csv(args.output_dir / "split_half_summary.csv")

    print(subject_table["n_epochs"].describe().to_string())
    print("\nDuration stability")
    print(pd.DataFrame(duration_summaries).to_string(index=False))
    print("\nEpoch-count correlation")
    print(count_correlations.to_string(index=False))
    print("\nIndependent two-minute split halves")
    print(split_half_summary.to_string())
    print(f"\nSaved epoch-count sensitivity results to {args.output_dir}")


if __name__ == "__main__":
    main()
