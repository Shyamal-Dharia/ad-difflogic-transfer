import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal


CONTRASTS = [
    ("A+P+", "N"),
    ("A+P+", "A+P-"),
    ("A+P-", "N"),
]


def empirical_upper_tail(observed, null_values):
    return (1 + np.sum(null_values >= observed)) / (1 + len(null_values))


def empirical_two_sided(observed, null_values):
    return (1 + np.sum(np.abs(null_values) >= abs(observed))) / (1 + len(null_values))


def pairwise_permutation_tests(predictions, n_permutations, seed):
    rng = np.random.default_rng(seed)
    rows = []

    for target_group, reference_group in CONTRASTS:
        pair = predictions[predictions["group"].isin([target_group, reference_group])].copy()
        scores = pair["mean_p_ad"].to_numpy()
        labels = pair["group"].to_numpy()

        target_scores = pair.loc[pair["group"].eq(target_group), "mean_p_ad"].to_numpy()
        reference_scores = pair.loc[pair["group"].eq(reference_group), "mean_p_ad"].to_numpy()
        observed_mean = target_scores.mean() - reference_scores.mean()
        observed_median = np.median(target_scores) - np.median(reference_scores)

        null_mean = np.zeros(n_permutations)
        null_median = np.zeros(n_permutations)
        for index in range(n_permutations):
            shuffled_labels = rng.permutation(labels)
            shuffled_target = scores[shuffled_labels == target_group]
            shuffled_reference = scores[shuffled_labels == reference_group]
            null_mean[index] = shuffled_target.mean() - shuffled_reference.mean()
            null_median[index] = np.median(shuffled_target) - np.median(shuffled_reference)

        rows.append(
            {
                "comparison": f"{target_group} - {reference_group}",
                "observed_mean_difference": observed_mean,
                "observed_median_difference": observed_median,
                "p_mean_one_sided_positive": empirical_upper_tail(observed_mean, null_mean),
                "p_mean_two_sided": empirical_two_sided(observed_mean, null_mean),
                "p_median_one_sided_positive": empirical_upper_tail(observed_median, null_median),
                "p_median_two_sided": empirical_two_sided(observed_median, null_median),
                "n_permutations": n_permutations,
                "seed": seed,
            }
        )

    return pd.DataFrame(rows)


def overall_permutation_test(predictions, n_permutations, seed):
    rng = np.random.default_rng(seed)
    scores = predictions["mean_p_ad"].to_numpy()
    labels = predictions["group"].to_numpy()
    groups = sorted(predictions["group"].unique())

    observed_groups = [
        predictions.loc[predictions["group"].eq(group), "mean_p_ad"].to_numpy()
        for group in groups
    ]
    observed = kruskal(*observed_groups).statistic

    null_values = np.zeros(n_permutations)
    for index in range(n_permutations):
        shuffled_labels = rng.permutation(labels)
        shuffled_groups = [scores[shuffled_labels == group] for group in groups]
        null_values[index] = kruskal(*shuffled_groups).statistic

    return pd.DataFrame(
        [
            {
                "comparison": "N vs A+P- vs A+P+",
                "statistic": "kruskal_h",
                "observed": observed,
                "p_value": empirical_upper_tail(observed, null_values),
                "n_permutations": n_permutations,
                "seed": seed,
            }
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Seed-averaged PEARL subject prediction CSV.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    pairwise = pairwise_permutation_tests(predictions, args.n_permutations, args.seed)
    overall = overall_permutation_test(predictions, args.n_permutations, args.seed)

    pairwise.to_csv(args.output_dir / "pearl_permutation_tests.csv", index=False)
    overall.to_csv(args.output_dir / "pearl_overall_permutation_test.csv", index=False)

    print(pairwise)
    print()
    print(overall)
    print(f"\nSaved permutation tests to {args.output_dir}")


if __name__ == "__main__":
    main()
