from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection


ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT_DIR / "datasets/features_psd"
OUTPUT_DIR = ROOT_DIR / "datasets/statistics_psd"

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def band_mask(freqs, band_name):
    low_freq, high_freq = BANDS[band_name]

    if band_name == "gamma":
        return (freqs >= low_freq) & (freqs <= high_freq)

    return (freqs >= low_freq) & (freqs < high_freq)


def subject_bandpower_rows(dataset_name, feature_path):
    data = np.load(feature_path, allow_pickle=True)
    freqs = data["freqs"]
    channel_names = data["channel_names"].astype(str)
    relative_psd = data["relative_psd"]

    rows = []

    for band_name in BANDS:
        mask = band_mask(freqs, band_name)
        bandpower = relative_psd[:, :, mask].sum(axis=-1)
        log_relative_bandpower = np.log10(np.maximum(bandpower, 1e-20))
        subject_values = log_relative_bandpower.mean(axis=0)

        for channel_index, channel_name in enumerate(channel_names):
            rows.append(
                {
                    "dataset": dataset_name,
                    "participant_id": str(scalar(data["participant_id"])),
                    "group": str(scalar(data["group"])),
                    "label": int(scalar(data["label"])),
                    "age": int(scalar(data["age"])),
                    "channel": channel_name,
                    "band": band_name,
                    "log_relative_bandpower": subject_values[channel_index],
                    "n_epochs": relative_psd.shape[0],
                }
            )

    return rows


def make_subject_bandpower_table(dataset_name):
    rows = []

    for feature_path in sorted((FEATURE_DIR / dataset_name).glob("*.npz")):
        rows.extend(subject_bandpower_rows(dataset_name, feature_path))

    return pd.DataFrame(rows)


def kruskal_effect_size(statistic, n_subjects, n_groups):
    effect_size = (statistic - n_groups + 1) / (n_subjects - n_groups)
    return max(0.0, effect_size)


def run_group_tests(subject_bandpower):
    rows = []

    for (dataset_name, channel, band), group_df in subject_bandpower.groupby(["dataset", "channel", "band"]):
        groups = []
        group_names = []

        for group_name, values in group_df.groupby("group")["log_relative_bandpower"]:
            groups.append(values.to_numpy())
            group_names.append(group_name)

        statistic, p_value = kruskal(*groups)
        rows.append(
            {
                "dataset": dataset_name,
                "channel": channel,
                "band": band,
                "test": "kruskal_wallis",
                "groups": ",".join(group_names),
                "statistic": statistic,
                "p_uncorrected": p_value,
                "effect_size_epsilon_squared": kruskal_effect_size(
                    statistic,
                    len(group_df["participant_id"].unique()),
                    len(groups),
                ),
            }
        )

    results = pd.DataFrame(rows)
    rejected, p_fdr = fdrcorrection(results["p_uncorrected"])
    results["p_fdr"] = p_fdr
    results["reject_fdr_0_05"] = rejected
    return results.sort_values(["p_fdr", "p_uncorrected"])


def summarize_groups(subject_bandpower):
    summary = (
        subject_bandpower
        .groupby(["dataset", "group", "channel", "band"])["log_relative_bandpower"]
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
    return summary


def run_pairwise_tests(subject_bandpower):
    rows = []

    for (dataset_name, channel, band), group_df in subject_bandpower.groupby(["dataset", "channel", "band"]):
        group_values = {
            group_name: values["log_relative_bandpower"].to_numpy()
            for group_name, values in group_df.groupby("group")
        }

        for group_a, group_b in combinations(sorted(group_values), 2):
            statistic, p_value = mannwhitneyu(
                group_values[group_a],
                group_values[group_b],
                alternative="two-sided",
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "channel": channel,
                    "band": band,
                    "test": "mann_whitney_u",
                    "group_a": group_a,
                    "group_b": group_b,
                    "median_a": np.median(group_values[group_a]),
                    "median_b": np.median(group_values[group_b]),
                    "median_difference_a_minus_b": np.median(group_values[group_a]) - np.median(group_values[group_b]),
                    "statistic": statistic,
                    "p_uncorrected": p_value,
                }
            )

    results = pd.DataFrame(rows)
    rejected, p_fdr = fdrcorrection(results["p_uncorrected"])
    results["p_fdr"] = p_fdr
    results["reject_fdr_0_05"] = rejected
    return results.sort_values(["p_fdr", "p_uncorrected"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alz_ftd = make_subject_bandpower_table("ALZ_FTD")
    pearl = make_subject_bandpower_table("PEARL")
    subject_bandpower = pd.concat([alz_ftd, pearl], ignore_index=True)

    group_summary = summarize_groups(subject_bandpower)
    group_tests = run_group_tests(subject_bandpower)
    pairwise_tests = run_pairwise_tests(subject_bandpower)

    subject_bandpower.to_csv(OUTPUT_DIR / "subject_log_relative_bandpower.csv", index=False)
    group_summary.to_csv(OUTPUT_DIR / "group_log_relative_bandpower_summary.csv", index=False)
    group_tests.to_csv(OUTPUT_DIR / "group_kruskal_tests.csv", index=False)
    pairwise_tests.to_csv(OUTPUT_DIR / "pairwise_mannwhitney_tests.csv", index=False)

    print(f"Saved {OUTPUT_DIR / 'subject_log_relative_bandpower.csv'}")
    print(f"Saved {OUTPUT_DIR / 'group_log_relative_bandpower_summary.csv'}")
    print(f"Saved {OUTPUT_DIR / 'group_kruskal_tests.csv'}")
    print(f"Saved {OUTPUT_DIR / 'pairwise_mannwhitney_tests.csv'}")


if __name__ == "__main__":
    main()
