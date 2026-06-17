import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from make_model_comparison_tables import bootstrap_mean_difference_ci
from train_helpers import ROOT_DIR


RUNS = [
    ("DiffLogic", "benchmark_250k_psd"),
    ("MLP 250k", "mlp_250k_psd"),
    ("1D-Conv 250k", "conv1d_250k_psd"),
    ("Transformer 250k", "transformer_250k_psd"),
]

LATEX_LABELS = {
    "DiffLogic": r"$\textbf{Diff-Logic}_{\textbf{250k}}$",
    "MLP 250k": r"MLP$_{250\mathrm{k}}$",
    "1D-Conv 250k": r"1D-Conv$_{250\mathrm{k}}$",
    "Transformer 250k": r"Transformer$_{250\mathrm{k}}$",
}

INFERENCE_PROFILE = {
    "DiffLogic": {"size_kb": 132.0, "latency_ms": 0.22, "throughput_per_second": 35_980},
    "MLP 250k": {"size_kb": 977.4, "latency_ms": 0.34, "throughput_per_second": 23_682},
    "1D-Conv 250k": {"size_kb": 978.4, "latency_ms": 0.38, "throughput_per_second": 21_263},
    "Transformer 250k": {"size_kb": 978.3, "latency_ms": 3.54, "throughput_per_second": 2_263},
}

GROUPS = ["N", "A+P-", "A+P+"]
CONTRASTS = [
    ("A+P-", "N", "A+P- - N"),
    ("A+P+", "N", "A+P+ - N"),
    ("A+P+", "A+P-", "A+P+ - A+P-"),
]


def mean_sd(mean, sd):
    return f"${mean:.3f} \\pm {sd:.3f}$"


def p_with_marker(p_value):
    marker = ""
    if p_value < 0.05:
        marker = "*"
    elif p_value < 0.10:
        marker = r"^{\dagger}"
    return f"${p_value:.3f}{marker}$"


def maybe_bold(value, bold):
    return rf"\textbf{{{value}}}" if bold else value


def seed_from_path(path):
    return int(path.parent.name.split("_")[1])


def load_clinical_subject_predictions(run_dir):
    tables = []
    for path in sorted(run_dir.glob("seed_*/alz_test_subject_predictions_all_folds.csv")):
        table = pd.read_csv(path)
        table["seed"] = seed_from_path(path)
        tables.append(table)

    if not tables:
        raise FileNotFoundError(f"No clinical predictions found in {run_dir}")

    return pd.concat(tables, ignore_index=True)


def summarize_clinical_by_seed(run_dir):
    predictions = load_clinical_subject_predictions(run_dir)
    rows = []

    for seed, seed_predictions in predictions.groupby("seed", sort=True):
        rows.append(
            {
                "seed": seed,
                "n_subjects": len(seed_predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    seed_predictions["true_label"],
                    seed_predictions["pred_label"],
                ),
                "auroc": roc_auc_score(
                    seed_predictions["true_label"],
                    seed_predictions["mean_p_ad"],
                ),
            }
        )

    return pd.DataFrame(rows)


def load_pearl_by_seed(run_dir):
    tables = []
    for path in sorted(run_dir.glob("seed_*/pearl_subject_predictions_ensemble.csv")):
        table = pd.read_csv(path)
        table["seed"] = seed_from_path(path)
        tables.append(table)

    if not tables:
        raise FileNotFoundError(f"No PEARL predictions found in {run_dir}")

    return pd.concat(tables, ignore_index=True)


def summarize_pearl_groups_by_seed(run_dir):
    pearl = load_pearl_by_seed(run_dir)
    return (
        pearl.groupby(["seed", "group"], as_index=False)
        .agg(
            n_subjects=("participant_id", "nunique"),
            group_mean_p_ad=("mean_p_ad", "mean"),
        )
    )


def pearl_seed_average_subject_scores(run_dir):
    pearl = load_pearl_by_seed(run_dir)
    return (
        pearl.groupby(["participant_id", "group", "true_label"], as_index=False)
        .agg(mean_p_ad=("mean_p_ad", "mean"))
    )


def make_performance_table(difflogic_dir, inference_profiles):
    rows = []

    for model_name, run_name in RUNS:
        run_dir = difflogic_dir / run_name
        clinical = summarize_clinical_by_seed(run_dir)
        pearl_groups = summarize_pearl_groups_by_seed(run_dir)
        group_summary = pearl_groups.groupby("group")["group_mean_p_ad"].agg(["mean", "std"])
        inference = inference_profiles[model_name]

        row = {
            "model": model_name,
            "run_name": run_name,
            "balanced_accuracy_mean": clinical["balanced_accuracy"].mean(),
            "balanced_accuracy_sd": clinical["balanced_accuracy"].std(),
            "auroc_mean": clinical["auroc"].mean(),
            "auroc_sd": clinical["auroc"].std(),
            "size_kb": inference["size_kb"],
            "latency_ms": inference["latency_ms"],
            "throughput_per_second": inference["throughput_per_second"],
        }
        for group in GROUPS:
            row[f"{group}_mean"] = group_summary.loc[group, "mean"]
            row[f"{group}_sd"] = group_summary.loc[group, "std"]
        rows.append(row)

    return pd.DataFrame(rows)


def make_transfer_contrasts(difflogic_dir):
    rows = []

    for model_index, (model_name, run_name) in enumerate(RUNS):
        scores = pearl_seed_average_subject_scores(difflogic_dir / run_name)
        group_scores = {
            group: group_data["mean_p_ad"].to_numpy()
            for group, group_data in scores.groupby("group")
        }

        for contrast_index, (target_group, reference_group, contrast_label) in enumerate(CONTRASTS):
            target = group_scores[target_group]
            reference = group_scores[reference_group]
            ci_low, ci_high = bootstrap_mean_difference_ci(
                target,
                reference,
                seed=45_000 + model_index * 100 + contrast_index,
            )
            statistic, p_value = mannwhitneyu(target, reference, alternative="two-sided")
            rows.append(
                {
                    "model": model_name,
                    "run_name": run_name,
                    "contrast": contrast_label,
                    "target_group": target_group,
                    "reference_group": reference_group,
                    "mean_difference": target.mean() - reference.mean(),
                    "mean_difference_ci_low": ci_low,
                    "mean_difference_ci_high": ci_high,
                    "mannwhitney_u": statistic,
                    "p_uncorrected": p_value,
                }
            )

    return pd.DataFrame(rows)


def format_inference(row, column):
    if column == "throughput_per_second":
        return f"{row[column]:,.0f}"
    if column == "latency_ms":
        return f"{row[column]:.2f}"
    return f"{row[column]:.1f}"


def write_performance_latex(performance, output_path):
    lines = [
        r"\begin{table*}[t]",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\centering",
        r"\caption{Subject-level clinical classification performance, zero-shot genetic-risk transfer scores, and Nvidia-Jetson Orin Nano 7W-inference profiling. Clinical metrics are reported as mean $\pm$ SD across random seeds, with each seed computed from pooled held-out cross-validation subject predictions. Genetic-risk AD-Signature EEG Score values are seed-level group means reported as mean $\pm$ SD across random seeds. Inference metrics (model size (KB), latency (ms), and throughput) reflect optimized single-core CPU execution.}",
        r"\label{tab:model_performance_transfer}",
        r"\begin{tabular}{@{}l|cc|ccc|ccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c|}{\textbf{Clinical (Source)}} & \multicolumn{3}{c|}{\textbf{Genetic-risk (Target)}} & \multicolumn{3}{c}{\textbf{Inference (CPU)}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(l){7-9}",
        r" & \textbf{Bal. Accuracy} & \textbf{AUROC} & \textbf{N} & \textbf{A+P-} & \textbf{A+P+} & \textbf{Size} & \textbf{Latency} & \textbf{Throughput} \\",
        r"\midrule",
    ]

    for _, row in performance.iterrows():
        bold = row["model"] == "DiffLogic"
        inference_values = [
            maybe_bold(format_inference(row, "size_kb"), bold),
            maybe_bold(format_inference(row, "latency_ms"), bold),
            maybe_bold(format_inference(row, "throughput_per_second"), bold),
        ]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                LATEX_LABELS[row["model"]],
                mean_sd(row["balanced_accuracy_mean"], row["balanced_accuracy_sd"]),
                mean_sd(row["auroc_mean"], row["auroc_sd"]),
                mean_sd(row["N_mean"], row["N_sd"]),
                mean_sd(row["A+P-_mean"], row["A+P-_sd"]),
                mean_sd(row["A+P+_mean"], row["A+P+_sd"]),
                inference_values[0],
                inference_values[1],
                inference_values[2],
            )
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    output_path.write_text("\n".join(lines))


def write_contrasts_latex(contrasts, output_path):
    lines = [
        r"\begin{table*}[t]",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\centering",
        r"\scriptsize",
        r"\caption{Genetic-risk transfer contrasts from seed/fold-averaged subject-level AD-Signature EEG Scores. Mean differences and bootstrap 95\% confidence intervals are reported. Uncorrected $p$ values are derived from Mann-Whitney U tests. Significance levels: * indicates $p < 0.05$, $\dagger$ indicates marginal significance $p < 0.10$.}",
        r"\label{tab:model_transfer_contrasts}",
        r"\begin{tabular}{@{}l|ccc|ccc|ccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{3}{c|}{\textbf{A+P- - N}} & \multicolumn{3}{c|}{\textbf{A+P+ - N}} & \multicolumn{3}{c}{\textbf{A+P+ - A+P-}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(l){8-10}",
        r" & \textbf{Mean Diff.} & \textbf{95\% CI} & \textbf{$p$} & \textbf{Mean Diff.} & \textbf{95\% CI} & \textbf{$p$} & \textbf{Mean Diff.} & \textbf{95\% CI} & \textbf{$p$} \\",
        r"\midrule",
    ]

    for model_name, model_data in contrasts.groupby("model", sort=False):
        pieces = []
        for _, contrast in model_data.iterrows():
            pieces.extend(
                [
                    f"${contrast['mean_difference']:.3f}$",
                    (
                        f"$[{contrast['mean_difference_ci_low']:.3f}, "
                        f"{contrast['mean_difference_ci_high']:.3f}]$"
                    ),
                    p_with_marker(contrast["p_uncorrected"]),
                ]
            )

        lines.append("{} & {} \\\\".format(LATEX_LABELS[model_name], " & ".join(pieces)))

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    output_path.write_text("\n".join(lines))


def ols_row(table, model, term, label):
    row = table[table["model"].eq(model) & table["term"].eq(term)].iloc[0]
    return {
        "contrast": label,
        "coef": row["coef"],
        "ci_low": row["ci_low"],
        "ci_high": row["ci_high"],
        "p_value": row["p_value"],
    }


def write_ols_latex(ols_csv, output_path):
    table = pd.read_csv(ols_csv)
    rows = [
        ols_row(
            table,
            "reference_N",
            'C(group, Treatment(reference="N"))[T.A+P+]',
            "A+P+ vs N",
        ),
        ols_row(
            table,
            "reference_N",
            'C(group, Treatment(reference="N"))[T.A+P-]',
            "A+P- vs N",
        ),
        ols_row(
            table,
            "reference_A+P-",
            'C(group_apm_reference, Treatment(reference="A+P-"))[T.A+P+]',
            "A+P+ vs A+P-",
        ),
        ols_row(table, "reference_N", "age_centered", "Age (centered)"),
        ols_row(table, "reference_N", "sex", "Sex"),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\centering",
        r"\caption{Multivariable linear regression coefficients for the AD-Signature EEG Score in the genetic-risk cohort, adjusting for age and sex. Significance levels: * indicates $p < 0.05$, $\dagger$ indicates marginal significance $p < 0.10$.}",
        r"\label{tab:ols_covariates}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\hline",
        r"\textbf{Contrast} & \textbf{Coefficient ($\beta$)} & \textbf{95\% CI} & \textbf{$p$-value} \\",
        r"\hline",
    ]

    for row in rows:
        significant = row["p_value"] < 0.05
        coef = f"${row['coef']:.3f}$"
        ci = f"$[{row['ci_low']:.3f}, {row['ci_high']:.3f}]$"
        p_value = p_with_marker(row["p_value"])
        if significant:
            coef = rf"\textbf{{{coef}}}"
            p_value = rf"\textbf{{{p_value}}}"
        lines.append(f"{row['contrast']} & {coef} & {ci} & {p_value} \\\\")

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    output_path.write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difflogic-dir", type=Path, default=ROOT_DIR / "outputs/difflogic_45hz")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs/model_comparison_tables_45hz_benchmark_250k")
    parser.add_argument(
        "--ols-csv",
        type=Path,
        default=ROOT_DIR / "outputs/difflogic_45hz/benchmark_250k_psd/summary/pearl_age_sex_adjusted_ols.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    performance = make_performance_table(args.difflogic_dir, INFERENCE_PROFILE)
    contrasts = make_transfer_contrasts(args.difflogic_dir)

    performance.to_csv(args.output_dir / "model_performance_transfer_table.csv", index=False)
    contrasts.to_csv(args.output_dir / "model_transfer_contrasts.csv", index=False)
    write_performance_latex(performance, args.output_dir / "model_performance_transfer_table.tex")
    write_contrasts_latex(contrasts, args.output_dir / "model_transfer_contrasts_table.tex")

    if args.ols_csv.exists():
        write_ols_latex(args.ols_csv, args.output_dir / "ols_covariates_table.tex")

    print(f"Saved benchmark 45 Hz tables to {args.output_dir}")
    print(performance.to_string(index=False))
    print()
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
