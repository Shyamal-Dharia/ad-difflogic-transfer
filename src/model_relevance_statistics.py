import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

from train_helpers import ROOT_DIR


INTERPRETATION_DIR = (
    ROOT_DIR
    / "outputs/difflogic/medium_interpretable/interpretation/soft_integrated_gradients"
)
OUTPUT_DIR = ROOT_DIR / "outputs/model_relevance_statistics"


def run_relevance_tests(interpretation_dir):
    relevance = pd.read_csv(interpretation_dir / "pearl_gradient_relevance_seed_average.csv")
    rows = []

    for (channel, band), feature_data in relevance.groupby(["channel", "band"]):
        group_values = {
            group: values["signed_relevance"].to_numpy()
            for group, values in feature_data.groupby("group")
        }

        for group_a, group_b in [("A+P+", "N"), ("A+P+", "A+P-"), ("A+P-", "N")]:
            values_a = group_values[group_a]
            values_b = group_values[group_b]
            statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
            rows.append(
                {
                    "channel": channel,
                    "band": band,
                    "comparison": f"{group_a}_vs_{group_b}",
                    "median_a": np.median(values_a),
                    "median_b": np.median(values_b),
                    "median_difference": np.median(values_a) - np.median(values_b),
                    "statistic": statistic,
                    "p_uncorrected": p_value,
                }
            )

    tests = pd.DataFrame(rows)
    tests["p_fdr"] = np.nan
    tests["reject_fdr_0_05"] = False

    for comparison, comparison_data in tests.groupby("comparison"):
        rejected, p_fdr = fdrcorrection(comparison_data["p_uncorrected"].to_numpy())
        tests.loc[comparison_data.index, "p_fdr"] = p_fdr
        tests.loc[comparison_data.index, "reject_fdr_0_05"] = rejected

    return tests.sort_values(["comparison", "p_uncorrected"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpretation-dir", type=Path, default=INTERPRETATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tests = run_relevance_tests(args.interpretation_dir)
    tests.to_csv(args.output_dir / "integrated_gradient_relevance_tests.csv", index=False)
    print(f"Saved {args.output_dir / 'integrated_gradient_relevance_tests.csv'}")
    print(tests.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
