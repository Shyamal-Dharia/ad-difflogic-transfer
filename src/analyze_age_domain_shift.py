"""Age distribution, domain separability, and age-sensitivity of the transfer score.

Addresses reviewer 1 major comment 5 and reviewer 2 comments 1 and 7.

Three questions are separated:
1. How far apart are the source and target cohorts in age, and how much do they
   overlap?
2. How separable are the two cohorts from the PSD features themselves, that is,
   how large is the domain shift the zero-shot transfer has to survive?
3. Does the reported genetic-risk group contrast survive age adjustment within
   the narrow genetic-risk age range?
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import gaussian_kde, kruskal, mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_helpers import ROOT_DIR, load_source_dataset, load_target_dataset


DIFFLOGIC_DIR = ROOT_DIR / "outputs" / "difflogic_45hz"
TRUE_RUN = "benchmark_250k_psd"
CONTRASTS = [("A+P-", "N"), ("A+P+", "N"), ("A+P+", "A+P-")]
GROUP_ORDER = ["N", "A+P-", "A+P+"]
SOURCE_GROUP_LABELS = {"A": "AD", "C": "CN", "F": "FTD"}


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def standardized_mean_difference(values_a, values_b):
    pooled_sd = np.sqrt(
        ((values_a.size - 1) * values_a.var(ddof=1) + (values_b.size - 1) * values_b.var(ddof=1))
        / (values_a.size + values_b.size - 2)
    )
    return (values_a.mean() - values_b.mean()) / pooled_sd


def overlapping_coefficient(values_a, values_b):
    grid = np.linspace(
        min(values_a.min(), values_b.min()) - 5.0,
        max(values_a.max(), values_b.max()) + 5.0,
        1000,
    )
    density_a = gaussian_kde(values_a)(grid)
    density_b = gaussian_kde(values_b)(grid)
    return float(np.trapezoid(np.minimum(density_a, density_b), grid))


def subject_frame(subjects, cohort):
    return pd.DataFrame(
        [
            {
                "cohort": cohort,
                "participant_id": subject["participant_id"],
                "group": SOURCE_GROUP_LABELS.get(subject["group"], subject["group"]),
                "age": subject["age"],
                "features": subject["x"].mean(axis=0).ravel(),
            }
            for subject in subjects
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "revision_2026" / "age_domain_shift",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_subjects, _ = load_source_dataset(
        dataset_name="ALZ_FTD", clinical_task="cn_vs_ad", feature_kind="psd"
    )
    target_subjects, _ = load_target_dataset("PEARL", feature_kind="psd")

    subjects = pd.concat(
        [subject_frame(source_subjects, "Clinical"), subject_frame(target_subjects, "Genetic-Risk")],
        ignore_index=True,
    )

    age_summary = (
        subjects.groupby(["cohort", "group"])["age"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    age_summary.to_csv(args.output_dir / "age_summary_by_group.csv", index=False)

    clinical_age = subjects.loc[subjects["cohort"].eq("Clinical"), "age"].to_numpy()
    genetic_age = subjects.loc[subjects["cohort"].eq("Genetic-Risk"), "age"].to_numpy()
    ad_age = subjects.loc[subjects["group"].eq("AD"), "age"].to_numpy()
    cn_age = subjects.loc[subjects["group"].eq("CN"), "age"].to_numpy()

    genetic_low, genetic_high = genetic_age.min(), genetic_age.max()
    clinical_in_range = ((clinical_age >= genetic_low) & (clinical_age <= genetic_high)).sum()
    ad_in_range = ((ad_age >= genetic_low) & (ad_age <= genetic_high)).sum()
    cn_in_range = ((cn_age >= genetic_low) & (cn_age <= genetic_high)).sum()

    shift_frame = pd.DataFrame(
        [
            {
                "comparison": "Clinical CN/AD vs Genetic-Risk",
                "mean_a": clinical_age.mean(),
                "sd_a": clinical_age.std(ddof=1),
                "mean_b": genetic_age.mean(),
                "sd_b": genetic_age.std(ddof=1),
                "mean_difference_years": clinical_age.mean() - genetic_age.mean(),
                "standardized_mean_difference": standardized_mean_difference(
                    clinical_age, genetic_age
                ),
                "overlapping_coefficient": overlapping_coefficient(clinical_age, genetic_age),
                "mann_whitney_p": mannwhitneyu(
                    clinical_age, genetic_age, alternative="two-sided"
                ).pvalue,
            },
            {
                "comparison": "Clinical AD vs Clinical CN",
                "mean_a": ad_age.mean(),
                "sd_a": ad_age.std(ddof=1),
                "mean_b": cn_age.mean(),
                "sd_b": cn_age.std(ddof=1),
                "mean_difference_years": ad_age.mean() - cn_age.mean(),
                "standardized_mean_difference": standardized_mean_difference(ad_age, cn_age),
                "overlapping_coefficient": overlapping_coefficient(ad_age, cn_age),
                "mann_whitney_p": mannwhitneyu(ad_age, cn_age, alternative="two-sided").pvalue,
            },
        ]
    )
    shift_frame.to_csv(args.output_dir / "age_shift_summary.csv", index=False)

    genetic_groups = [
        subjects.loc[subjects["group"].eq(group), "age"].to_numpy() for group in GROUP_ORDER
    ]
    overlap_frame = pd.DataFrame(
        [
            {
                "genetic_risk_age_min": genetic_low,
                "genetic_risk_age_max": genetic_high,
                "clinical_subjects_total": clinical_age.size,
                "clinical_subjects_in_genetic_range": int(clinical_in_range),
                "clinical_ad_in_genetic_range": int(ad_in_range),
                "clinical_cn_in_genetic_range": int(cn_in_range),
                "genetic_risk_kruskal_p": kruskal(*genetic_groups).pvalue,
            }
        ]
    )
    overlap_frame.to_csv(args.output_dir / "age_overlap.csv", index=False)

    features = np.vstack(subjects["features"].to_numpy())
    domain_labels = subjects["cohort"].eq("Genetic-Risk").astype(int).to_numpy()
    domain_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    domain_scores = cross_val_predict(
        domain_model,
        features,
        domain_labels,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed),
        method="predict_proba",
    )[:, 1]
    domain_auroc = roc_auc_score(domain_labels, domain_scores)

    age_model_scores = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        subjects[["age"]].to_numpy(),
        domain_labels,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed),
        method="predict_proba",
    )[:, 1]

    pd.DataFrame(
        [
            {
                "classifier": "Log-relative PSD (75 features)",
                "cohort_auroc": domain_auroc,
                "n_subjects": len(subjects),
            },
            {
                "classifier": "Age only",
                "cohort_auroc": roc_auc_score(domain_labels, age_model_scores),
                "n_subjects": len(subjects),
            },
        ]
    ).to_csv(args.output_dir / "domain_separability.csv", index=False)

    scores = pd.read_csv(
        DIFFLOGIC_DIR / TRUE_RUN / "summary" / "pearl_subject_predictions_seed_average.csv"
    )[["participant_id", "group", "mean_p_ad"]]
    genetic = subjects.loc[subjects["cohort"].eq("Genetic-Risk"), ["participant_id", "age"]]
    scores = scores.merge(genetic, on="participant_id", validate="one_to_one")

    pearson_r, pearson_p = pearsonr(scores["age"], scores["mean_p_ad"])
    spearman_rho, spearman_p = spearmanr(scores["age"], scores["mean_p_ad"])
    correlation_rows = [
        {
            "subset": "All genetic-risk",
            "n": len(scores),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
        }
    ]
    for group in GROUP_ORDER:
        subset = scores[scores["group"].eq(group)]
        group_pearson_r, group_pearson_p = pearsonr(subset["age"], subset["mean_p_ad"])
        group_spearman_rho, group_spearman_p = spearmanr(subset["age"], subset["mean_p_ad"])
        correlation_rows.append(
            {
                "subset": group,
                "n": len(subset),
                "pearson_r": group_pearson_r,
                "pearson_p": group_pearson_p,
                "spearman_rho": group_spearman_rho,
                "spearman_p": group_spearman_p,
            }
        )
    pd.DataFrame(correlation_rows).to_csv(
        args.output_dir / "age_score_correlation.csv", index=False
    )

    residual_model = smf.ols("mean_p_ad ~ age", data=scores).fit()
    scores = scores.assign(age_residual_score=residual_model.resid)

    contrast_rows = []
    raw_p_values = []
    residual_p_values = []
    for target_group, reference_group in CONTRASTS:
        target = scores[scores["group"].eq(target_group)]
        reference = scores[scores["group"].eq(reference_group)]
        raw_p = mannwhitneyu(
            target["mean_p_ad"], reference["mean_p_ad"], alternative="two-sided"
        ).pvalue
        residual_p = mannwhitneyu(
            target["age_residual_score"],
            reference["age_residual_score"],
            alternative="two-sided",
        ).pvalue
        raw_p_values.append(raw_p)
        residual_p_values.append(residual_p)
        contrast_rows.append(
            {
                "contrast": f"{target_group} vs {reference_group}",
                "raw_mean_difference": target["mean_p_ad"].mean() - reference["mean_p_ad"].mean(),
                "raw_p": raw_p,
                "age_residual_mean_difference": target["age_residual_score"].mean()
                - reference["age_residual_score"].mean(),
                "age_residual_p": residual_p,
            }
        )
    contrast_frame = pd.DataFrame(contrast_rows)
    contrast_frame["raw_p_holm"] = holm_adjust(np.array(raw_p_values))
    contrast_frame["age_residual_p_holm"] = holm_adjust(np.array(residual_p_values))
    contrast_frame.to_csv(args.output_dir / "age_residualized_contrasts.csv", index=False)

    print("Age summary by group")
    print(age_summary.round(2).to_string(index=False))
    print("\nCohort age shift")
    print(shift_frame.round(3).to_string(index=False))
    print("\nAge overlap")
    print(overlap_frame.round(3).to_string(index=False))
    print("\nDomain separability from PSD features")
    print(pd.read_csv(args.output_dir / "domain_separability.csv").round(4).to_string(index=False))
    print("\nAge versus score in the genetic-risk cohort")
    print(pd.DataFrame(correlation_rows).round(3).to_string(index=False))
    print("\nGroup contrasts before and after age residualization")
    print(contrast_frame.round(4).to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
