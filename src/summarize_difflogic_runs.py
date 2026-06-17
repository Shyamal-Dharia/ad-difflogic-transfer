import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from train_helpers import ROOT_DIR


OUTPUT_DIR = ROOT_DIR / "outputs/difflogic"


def read_seed_outputs(run_name, output_dir):
    run_dirs = sorted((output_dir / run_name).glob("seed_*"))
    metrics = []
    pearl = []

    for run_dir in run_dirs:
        metrics_path = run_dir / "alz_test_metrics_all_folds.csv"
        pearl_path = run_dir / "pearl_subject_predictions_ensemble.csv"

        if metrics_path.exists():
            metrics.append(pd.read_csv(metrics_path))
        if pearl_path.exists():
            seed_pearl = pd.read_csv(pearl_path)
            seed_pearl["seed"] = int(run_dir.name.split("_")[1])
            pearl.append(seed_pearl)

    if not metrics:
        raise FileNotFoundError(f"No metric files found for {output_dir / run_name}")
    if not pearl:
        raise FileNotFoundError(f"No PEARL prediction files found for {output_dir / run_name}")

    return pd.concat(metrics, ignore_index=True), pd.concat(pearl, ignore_index=True)


def summarize_model(run_name, output_dir):
    summary_dir = output_dir / run_name / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    metrics, pearl = read_seed_outputs(run_name, output_dir)
    pearl_mean = (
        pearl
        .groupby(["participant_id", "group", "true_label"])
        .agg(
            n_seed_runs=("seed", "count"),
            mean_p_ad=("mean_p_ad", "mean"),
            std_p_ad=("mean_p_ad", "std"),
            mean_p_control=("mean_p_control", "mean"),
            mean_fraction_epochs_predicted_ad=("mean_fraction_epochs_predicted_ad", "mean"),
        )
        .reset_index()
    )
    pearl_mean["pred_label"] = (pearl_mean["mean_p_ad"] >= 0.5).astype(int)

    group_summary = pearl_mean.groupby("group").agg(
        n_subjects=("participant_id", "count"),
        mean_p_ad=("mean_p_ad", "mean"),
        std_p_ad=("mean_p_ad", "std"),
        median_p_ad=("mean_p_ad", "median"),
        pred_ad_rate=("pred_label", "mean"),
    )

    groups = [group["mean_p_ad"].to_numpy() for _, group in pearl_mean.groupby("group")]
    test_rows = [
        {
            "comparison": "N_vs_A+P-_vs_A+P+",
            "test": "kruskal",
            "p_value": kruskal(*groups).pvalue,
        }
    ]

    for group_a, group_b in [("N", "A+P-"), ("N", "A+P+"), ("A+P-", "A+P+")]:
        x_a = pearl_mean.loc[pearl_mean["group"].eq(group_a), "mean_p_ad"]
        x_b = pearl_mean.loc[pearl_mean["group"].eq(group_b), "mean_p_ad"]
        test_rows.append(
            {
                "comparison": "{}_vs_{}".format(group_a, group_b),
                "test": "mann_whitney_u",
                "p_value": mannwhitneyu(x_a, x_b, alternative="two-sided").pvalue,
                "median_a": x_a.median(),
                "median_b": x_b.median(),
            }
        )

    tests = pd.DataFrame(test_rows)

    metrics.to_csv(summary_dir / "alz_test_metrics_all_seeds.csv", index=False)
    pearl.to_csv(summary_dir / "pearl_subject_predictions_all_seeds.csv", index=False)
    pearl_mean.to_csv(summary_dir / "pearl_subject_predictions_seed_average.csv", index=False)
    group_summary.to_csv(summary_dir / "pearl_group_summary_seed_average.csv")
    tests.to_csv(summary_dir / "pearl_group_tests_seed_average.csv", index=False)

    print(metrics[["balanced_accuracy", "auroc"]].agg(["mean", "std", "min", "max"]))
    print()
    print(group_summary)
    print()
    print(tests)
    print("\nSaved summary to {}".format(summary_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="medium")
    parser.add_argument("--model-size", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.model_size is not None:
        args.run_name = args.model_size

    return args


if __name__ == "__main__":
    args = parse_args()
    summarize_model(args.run_name, args.output_dir)
