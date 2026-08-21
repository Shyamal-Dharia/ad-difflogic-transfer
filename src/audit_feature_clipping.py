"""Out-of-range feature clipping audit for the genetic-risk cohort.

Addresses reviewer 1 minor comment 2.

Min-max scaling parameters are fitted on the clinical training subjects of each
fold and then applied to the genetic-risk cohort, where the result is clipped to
[0, 1]. This script counts how often genetic-risk features fall outside the
clinical training range before clipping, summarizes the rate by group, channel,
and band, and tests whether the clipping rate explains the reported group
contrast.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from train_helpers import (
    BANDS,
    ROOT_DIR,
    fit_minmax_scaler,
    load_source_dataset,
    load_target_dataset,
    make_subject_stratified_folds,
    stack_subjects,
)


DIFFLOGIC_DIR = ROOT_DIR / "outputs" / "difflogic_45hz"
TRUE_RUN = "benchmark_250k_psd"
SEEDS = [1, 2, 3, 4, 5]
BAND_NAMES = list(BANDS)
CONTRASTS = [("A+P-", "N"), ("A+P+", "N"), ("A+P+", "A+P-")]


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "revision_2026" / "clipping_audit",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_subjects, source_table = load_source_dataset(
        dataset_name="ALZ_FTD",
        clinical_task="cn_vs_ad",
        feature_kind="psd",
    )
    target_subjects, target_table = load_target_dataset("PEARL", feature_kind="psd")

    x_target, _, target_subject_ids = stack_subjects(
        target_subjects,
        target_table["participant_id"].to_numpy(),
    )
    channel_names = list(target_subjects[0]["channel_names"])
    group_by_subject = dict(zip(target_table["participant_id"], target_table["group"]))

    below_counts = np.zeros(x_target.shape[1:], dtype=np.int64)
    above_counts = np.zeros(x_target.shape[1:], dtype=np.int64)
    subject_rows = []
    fold_rows = []
    n_fold_models = 0

    for seed in SEEDS:
        folds = make_subject_stratified_folds(source_table, random_state=seed)
        for fold in folds:
            x_train, _, _ = stack_subjects(source_subjects, fold["train_subject_ids"])
            scaler = fit_minmax_scaler(x_train)
            below = x_target < scaler["feature_min"]
            above = x_target > scaler["feature_max"]

            below_counts += below.sum(axis=0)
            above_counts += above.sum(axis=0)
            n_fold_models += 1

            outside = below | above
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold["fold"],
                    "clipped_fraction": float(outside.mean()),
                    "below_fraction": float(below.mean()),
                    "above_fraction": float(above.mean()),
                }
            )

            per_epoch = outside.reshape(outside.shape[0], -1).mean(axis=1)
            subject_rows.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "fold": fold["fold"],
                        "participant_id": target_subject_ids,
                        "clipped_fraction": per_epoch,
                    }
                )
                .groupby(["seed", "fold", "participant_id"], as_index=False)["clipped_fraction"]
                .mean()
            )

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(args.output_dir / "clipping_by_fold.csv", index=False)

    total_values = n_fold_models * x_target.shape[0]
    cell_frame = pd.DataFrame(
        [
            {
                "channel": channel_names[channel_index],
                "band": BAND_NAMES[band_index],
                "below_fraction": below_counts[channel_index, band_index] / total_values,
                "above_fraction": above_counts[channel_index, band_index] / total_values,
                "clipped_fraction": (
                    below_counts[channel_index, band_index] + above_counts[channel_index, band_index]
                )
                / total_values,
            }
            for channel_index in range(len(channel_names))
            for band_index in range(len(BAND_NAMES))
        ]
    )
    cell_frame.to_csv(args.output_dir / "clipping_by_channel_band.csv", index=False)

    subject_frame = (
        pd.concat(subject_rows, ignore_index=True)
        .groupby("participant_id", as_index=False)["clipped_fraction"]
        .mean()
    )
    subject_frame["group"] = subject_frame["participant_id"].map(group_by_subject)

    scores = pd.read_csv(
        DIFFLOGIC_DIR / TRUE_RUN / "summary" / "pearl_subject_predictions_seed_average.csv"
    )[["participant_id", "mean_p_ad"]]
    subject_frame = subject_frame.merge(scores, on="participant_id", how="inner")
    subject_frame.to_csv(args.output_dir / "clipping_by_subject.csv", index=False)

    group_frame = (
        subject_frame.groupby("group")["clipped_fraction"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reindex(["N", "A+P-", "A+P+"])
    )
    group_frame.to_csv(args.output_dir / "clipping_by_group.csv")

    pearson_r, pearson_p = pearsonr(subject_frame["clipped_fraction"], subject_frame["mean_p_ad"])
    spearman_rho, spearman_p = spearmanr(subject_frame["clipped_fraction"], subject_frame["mean_p_ad"])

    group_p_values = []
    group_rows = []
    for target_group, reference_group in CONTRASTS:
        target = subject_frame.loc[subject_frame["group"] == target_group, "clipped_fraction"]
        reference = subject_frame.loc[subject_frame["group"] == reference_group, "clipped_fraction"]
        p_value = mannwhitneyu(target, reference, alternative="two-sided").pvalue
        group_p_values.append(p_value)
        group_rows.append(
            {
                "contrast": f"{target_group} vs {reference_group}",
                "mean_difference": target.mean() - reference.mean(),
                "p_uncorrected": p_value,
            }
        )
    group_contrast_frame = pd.DataFrame(group_rows)
    group_contrast_frame["p_holm"] = holm_adjust(np.array(group_p_values))
    group_contrast_frame.to_csv(args.output_dir / "clipping_group_contrasts.csv", index=False)

    model_frame = subject_frame.copy()
    model_frame["group"] = pd.Categorical(model_frame["group"], categories=["N", "A+P-", "A+P+"])
    unadjusted = smf.ols("mean_p_ad ~ C(group)", data=model_frame).fit()
    adjusted = smf.ols("mean_p_ad ~ C(group) + clipped_fraction", data=model_frame).fit()

    adjustment_rows = []
    for term in ["C(group)[T.A+P-]", "C(group)[T.A+P+]"]:
        adjustment_rows.append(
            {
                "term": term,
                "beta_unadjusted": unadjusted.params[term],
                "p_unadjusted": unadjusted.pvalues[term],
                "beta_clipping_adjusted": adjusted.params[term],
                "p_clipping_adjusted": adjusted.pvalues[term],
            }
        )
    adjustment_rows.append(
        {
            "term": "clipped_fraction",
            "beta_unadjusted": np.nan,
            "p_unadjusted": np.nan,
            "beta_clipping_adjusted": adjusted.params["clipped_fraction"],
            "p_clipping_adjusted": adjusted.pvalues["clipped_fraction"],
        }
    )
    pd.DataFrame(adjustment_rows).to_csv(
        args.output_dir / "score_model_adjusted_for_clipping.csv", index=False
    )

    correlation_frame = pd.DataFrame(
        [
            {
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "n_subjects": len(subject_frame),
                "n_fold_models": n_fold_models,
                "overall_clipped_fraction": fold_frame["clipped_fraction"].mean(),
                "overall_below_fraction": fold_frame["below_fraction"].mean(),
                "overall_above_fraction": fold_frame["above_fraction"].mean(),
            }
        ]
    )
    correlation_frame.to_csv(args.output_dir / "clipping_score_association.csv", index=False)

    print("Overall clipped fraction: {:.4f}".format(fold_frame["clipped_fraction"].mean()))
    print("  below training minimum: {:.4f}".format(fold_frame["below_fraction"].mean()))
    print("  above training maximum: {:.4f}".format(fold_frame["above_fraction"].mean()))
    print("\nBy group:")
    print(group_frame.round(4))
    print("\nGroup contrasts in clipping rate:")
    print(group_contrast_frame.round(4).to_string(index=False))
    print(
        "\nClipping vs score: Pearson r = {:.3f} (p = {:.3f}); Spearman rho = {:.3f} (p = {:.3f})".format(
            pearson_r, pearson_p, spearman_rho, spearman_p
        )
    )
    print("\nScore model before and after adjusting for clipping fraction:")
    print(pd.DataFrame(adjustment_rows).round(4).to_string(index=False))
    print("\nMost frequently clipped channel-band cells:")
    print(cell_frame.sort_values("clipped_fraction", ascending=False).head(10).round(4).to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
