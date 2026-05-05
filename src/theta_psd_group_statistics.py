from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

from train_helpers import ROOT_DIR


PSD_STATS_DIR = ROOT_DIR / "datasets/statistics_psd"
OUTPUT_DIR = ROOT_DIR / "outputs/theta_psd_group_statistics"


def run_theta_tests():
    psd = pd.read_csv(PSD_STATS_DIR / "subject_log_relative_bandpower.csv")
    theta = psd[psd["dataset"].eq("PEARL") & psd["band"].eq("theta")].copy()
    rows = []

    for channel, channel_data in theta.groupby("channel"):
        group_values = {
            group: values["log_relative_bandpower"].to_numpy()
            for group, values in channel_data.groupby("group")
        }

        for group_a, group_b in [("A+P+", "N"), ("A+P+", "A+P-"), ("A+P-", "N")]:
            values_a = group_values[group_a]
            values_b = group_values[group_b]
            statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
            rows.append(
                {
                    "channel": channel,
                    "band": "theta",
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tests = run_theta_tests()
    tests.to_csv(OUTPUT_DIR / "theta_psd_pairwise_tests.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'theta_psd_pairwise_tests.csv'}")
    print(tests.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
