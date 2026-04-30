import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, kruskal
import statsmodels.formula.api as smf

from train_helpers import ROOT_DIR


DEFAULT_SUMMARY_DIR = ROOT_DIR / "outputs/difflogic/medium/summary"
PEARL_PARTICIPANTS = ROOT_DIR / "datasets/PEARL/participants.tsv"


def load_pearl_predictions_with_covariates(summary_dir):
    predictions = pd.read_csv(summary_dir / "pearl_subject_predictions_seed_average.csv")
    participants = pd.read_csv(PEARL_PARTICIPANTS, sep="\t")
    covariates = participants[["participant_id", "age", "sex", "risk_group"]]
    data = predictions.merge(covariates, on="participant_id", how="left")
    data["group"] = pd.Categorical(data["group"], categories=["N", "A+P-", "A+P+"], ordered=True)
    data["age_centered"] = data["age"] - data["age"].mean()
    data["sex"] = data["sex"].astype(int)
    return data


def summarize_age_sex_balance(data):
    age_summary = data.groupby("group", observed=True)["age"].agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
    )
    sex_table = pd.crosstab(data["group"], data["sex"])
    age_groups = [group["age"].to_numpy() for _, group in data.groupby("group", observed=True)]
    age_test = kruskal(*age_groups)
    sex_test = chi2_contingency(sex_table)
    return age_summary, sex_table, age_test, sex_test


def fit_adjusted_models(data):
    model_n_reference = smf.ols(
        'mean_p_ad ~ C(group, Treatment(reference="N")) + age_centered + sex',
        data=data,
    ).fit(cov_type="HC3")

    data = data.copy()
    data["group_apm_reference"] = pd.Categorical(
        data["group"],
        categories=["A+P-", "N", "A+P+"],
        ordered=True,
    )
    model_apm_reference = smf.ols(
        'mean_p_ad ~ C(group_apm_reference, Treatment(reference="A+P-")) + age_centered + sex',
        data=data,
    ).fit(cov_type="HC3")

    return model_n_reference, model_apm_reference


def model_to_table(name, model):
    conf = model.conf_int()
    rows = []

    for term in model.params.index:
        rows.append(
            {
                "model": name,
                "term": term,
                "coef": model.params[term],
                "ci_low": conf.loc[term, 0],
                "ci_high": conf.loc[term, 1],
                "p_value": model.pvalues[term],
            }
        )

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    summary_dir = args.summary_dir
    data = load_pearl_predictions_with_covariates(summary_dir)
    age_summary, sex_table, age_test, sex_test = summarize_age_sex_balance(data)
    model_n_reference, model_apm_reference = fit_adjusted_models(data)

    model_rows = []
    model_rows.extend(model_to_table("reference_N", model_n_reference))
    model_rows.extend(model_to_table("reference_A+P-", model_apm_reference))
    model_table = pd.DataFrame(model_rows)

    data.to_csv(summary_dir / "pearl_subject_predictions_seed_average_with_covariates.csv", index=False)
    age_summary.to_csv(summary_dir / "pearl_age_group_summary.csv")
    sex_table.to_csv(summary_dir / "pearl_sex_group_counts.csv")
    model_table.to_csv(summary_dir / "pearl_age_sex_adjusted_ols.csv", index=False)

    print("Age by group")
    print(age_summary)
    print("\nAge Kruskal p =", age_test.pvalue)
    print("\nSex counts")
    print(sex_table)
    print("Sex chi-square p =", sex_test.pvalue)
    print("\nAge/sex adjusted OLS")
    print(model_table)


if __name__ == "__main__":
    main()
