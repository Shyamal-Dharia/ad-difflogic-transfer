import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

from train_helpers import ROOT_DIR


PSD_STATS_DIR = ROOT_DIR / "datasets/statistics_psd"
OUTPUT_DIR = ROOT_DIR / "outputs/focused_psd_statistics"
INTERPRETATION_DIR = (
    ROOT_DIR
    / "outputs/difflogic/medium_interpretable/interpretation/soft_integrated_gradients"
)


def median_difference(values_a, values_b):
    return np.median(values_a) - np.median(values_b)


def permutation_median_test(values_a, values_b, n_permutations, seed):
    rng = np.random.default_rng(seed)
    observed = median_difference(values_a, values_b)
    values = np.concatenate([values_a, values_b])
    n_a = len(values_a)
    null = np.zeros(n_permutations)

    for index in range(n_permutations):
        shuffled = rng.permutation(values)
        null[index] = median_difference(shuffled[:n_a], shuffled[n_a:])

    p_two_sided = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1)
    return observed, p_two_sided


def load_model_relevant_features(top_n, interpretation_dir):
    signed = pd.read_csv(interpretation_dir / "pearl_signed_relevance_group_contrasts.csv")
    top_apn = signed.sort_values("A+P+_minus_N", ascending=False).head(top_n)
    top_apm = signed.sort_values("A+P+_minus_A+P-", ascending=False).head(top_n)
    features = (
        pd.concat([top_apn, top_apm], ignore_index=True)
        [["channel", "band"]]
        .drop_duplicates()
        .sort_values(["channel", "band"])
        .reset_index(drop=True)
    )
    return features


def load_focused_psd(features, psd_stats_dir):
    subject_psd = pd.read_csv(psd_stats_dir / "subject_log_relative_bandpower.csv")
    pearl_psd = subject_psd[subject_psd["dataset"].eq("PEARL")].copy()
    focused = pearl_psd.merge(features, on=["channel", "band"], how="inner")
    return focused


def summarize_groups(focused):
    return (
        focused
        .groupby(["channel", "band", "group"])["log_relative_bandpower"]
        .agg(
            n_subjects="count",
            mean="mean",
            std="std",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )


def run_feature_tests(focused, n_permutations, seed):
    rows = []

    for (channel, band), feature_data in focused.groupby(["channel", "band"]):
        group_values = {
            group: values["log_relative_bandpower"].to_numpy()
            for group, values in feature_data.groupby("group")
        }
        h_statistic, h_p = kruskal(*group_values.values())
        rows.append(
            {
                "channel": channel,
                "band": band,
                "comparison": "N_vs_A+P-_vs_A+P+",
                "test": "kruskal",
                "statistic": h_statistic,
                "median_difference": np.nan,
                "p_uncorrected": h_p,
            }
        )

        for group_a, group_b in [("A+P+", "N"), ("A+P+", "A+P-"), ("A+P-", "N")]:
            values_a = group_values[group_a]
            values_b = group_values[group_b]
            statistic, p_mannwhitney = mannwhitneyu(values_a, values_b, alternative="two-sided")
            observed, p_permutation = permutation_median_test(
                values_a,
                values_b,
                n_permutations,
                seed,
            )
            rows.append(
                {
                    "channel": channel,
                    "band": band,
                    "comparison": f"{group_a}_vs_{group_b}",
                    "test": "mann_whitney_u",
                    "statistic": statistic,
                    "median_difference": observed,
                    "p_uncorrected": p_mannwhitney,
                }
            )
            rows.append(
                {
                    "channel": channel,
                    "band": band,
                    "comparison": f"{group_a}_vs_{group_b}",
                    "test": "permutation_median",
                    "statistic": np.nan,
                    "median_difference": observed,
                    "p_uncorrected": p_permutation,
                }
            )

    tests = pd.DataFrame(rows)
    tests["p_fdr"] = np.nan
    tests["reject_fdr_0_05"] = False

    for test_name, test_data in tests.groupby("test"):
        rejected, p_fdr = fdrcorrection(test_data["p_uncorrected"].to_numpy())
        tests.loc[test_data.index, "p_fdr"] = p_fdr
        tests.loc[test_data.index, "reject_fdr_0_05"] = rejected

    return tests.sort_values(["test", "p_fdr", "p_uncorrected"])


def make_composite_theta_score(focused):
    theta = focused[focused["band"].eq("theta")]
    theta_score = (
        theta
        .groupby(["participant_id", "group"])["log_relative_bandpower"]
        .mean()
        .reset_index(name="mean_model_relevant_theta_log_relative_bandpower")
    )
    return theta_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--psd-stats-dir", type=Path, default=PSD_STATS_DIR)
    parser.add_argument("--interpretation-dir", type=Path, default=INTERPRETATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = load_model_relevant_features(args.top_n, args.interpretation_dir)
    focused = load_focused_psd(features, args.psd_stats_dir)
    summary = summarize_groups(focused)
    tests = run_feature_tests(focused, args.n_permutations, args.seed)
    theta_score = make_composite_theta_score(focused)

    features.to_csv(args.output_dir / "model_relevant_channel_band_features.csv", index=False)
    focused.to_csv(args.output_dir / "focused_psd_subject_values.csv", index=False)
    summary.to_csv(args.output_dir / "focused_psd_group_summary.csv", index=False)
    tests.to_csv(args.output_dir / "focused_psd_tests.csv", index=False)
    theta_score.to_csv(args.output_dir / "focused_theta_subject_score.csv", index=False)

    print(f"Saved focused PSD statistics to {args.output_dir}")
    print("\nModel-relevant features")
    print(features.to_string(index=False))
    print("\nTop focused PSD tests")
    print(tests.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
