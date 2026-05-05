from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from statsmodels.stats.multitest import fdrcorrection

from train_helpers import ROOT_DIR


DIFFLOGIC_DIR = ROOT_DIR / "outputs/difflogic"
OUTPUT_DIR = ROOT_DIR / "outputs/model_comparison_tables"

MODEL_RUNS = [
    ("DiffLogic", "medium_interpretable"),
    ("MLP 250k", "mlp_250k_psd"),
    ("1-Conv 250k", "conv1d_250k_psd"),
    ("Transformer 250k", "transformer_250k_psd"),
]

MODEL_LATEX_LABELS = {
    "DiffLogic": r"$\textbf{Diff-Logic}_{\textbf{250k}}$",
    "MLP 250k": r"MLP$_{250\mathrm{k}}$",
    "1-Conv 250k": r"1D-Conv$_{250\mathrm{k}}$",
    "Transformer 250k": r"Transformer$_{250\mathrm{k}}$",
}

INFERENCE_PROFILE = {
    "DiffLogic": {"size_kb": 132.0, "latency_ms": 0.22, "throughput_per_second": 35_980},
    "MLP 250k": {"size_kb": 977.4, "latency_ms": 0.34, "throughput_per_second": 23_682},
    "1-Conv 250k": {"size_kb": 978.4, "latency_ms": 0.38, "throughput_per_second": 21_263},
    "Transformer 250k": {"size_kb": 978.3, "latency_ms": 3.54, "throughput_per_second": 2_263},
}

PEARL_GROUPS = ["N", "A+P-", "A+P+"]
TRANSFER_CONTRASTS = [
    ("A+P-", "N", 2),
    ("A+P+", "N", 0),
    ("A+P+", "A+P-", 1),
]


def format_mean_sd(mean, sd):
    return f"${mean:.3f} \\pm {sd:.3f}$"


def format_inference_value(value, decimals=1, bold=False):
    if decimals == 0:
        text = f"{value:,.0f}"
    else:
        text = f"{value:.{decimals}f}"

    if bold:
        return rf"\textbf{{{text}}}"
    return text


def load_clinical_metrics(run_name):
    metric_tables = []

    for metrics_path in sorted((DIFFLOGIC_DIR / run_name).glob("seed_*/alz_test_metrics_all_folds.csv")):
        metric_tables.append(pd.read_csv(metrics_path))

    if not metric_tables:
        raise FileNotFoundError(f"No clinical metric files found for {run_name}")

    return pd.concat(metric_tables, ignore_index=True)


def load_clinical_subject_predictions(run_name):
    prediction_tables = []

    for predictions_path in sorted((DIFFLOGIC_DIR / run_name).glob("seed_*/alz_test_subject_predictions_all_folds.csv")):
        predictions = pd.read_csv(predictions_path)
        predictions["seed"] = int(predictions_path.parent.name.split("_")[1])
        prediction_tables.append(predictions)

    if not prediction_tables:
        raise FileNotFoundError(f"No clinical subject prediction files found for {run_name}")

    return pd.concat(prediction_tables, ignore_index=True)


def summarize_clinical_metrics_by_seed(run_name):
    subject_predictions = load_clinical_subject_predictions(run_name)
    rows = []

    for seed, seed_predictions in subject_predictions.groupby("seed", sort=True):
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


def load_pearl_subject_scores(run_name):
    pearl_tables = []

    for pearl_path in sorted((DIFFLOGIC_DIR / run_name).glob("seed_*/pearl_subject_predictions_ensemble.csv")):
        pearl = pd.read_csv(pearl_path)
        pearl["seed"] = pearl_path.parent.name
        pearl_tables.append(pearl)

    if not pearl_tables:
        raise FileNotFoundError(f"No PEARL prediction files found for {run_name}")

    pearl = pd.concat(pearl_tables, ignore_index=True)
    return (
        pearl
        .groupby(["participant_id", "group", "true_label"], as_index=False)
        .agg(mean_p_ad=("mean_p_ad", "mean"))
    )


def load_pearl_subject_scores_by_seed(run_name):
    pearl_tables = []

    for pearl_path in sorted((DIFFLOGIC_DIR / run_name).glob("seed_*/pearl_subject_predictions_ensemble.csv")):
        pearl = pd.read_csv(pearl_path)
        pearl["seed"] = int(pearl_path.parent.name.split("_")[1])
        pearl_tables.append(pearl)

    if not pearl_tables:
        raise FileNotFoundError(f"No PEARL prediction files found for {run_name}")

    return pd.concat(pearl_tables, ignore_index=True)


def summarize_pearl_group_scores_by_seed(run_name):
    pearl_subject_scores = load_pearl_subject_scores_by_seed(run_name)
    return (
        pearl_subject_scores
        .groupby(["seed", "group"], as_index=False)
        .agg(
            n_subjects=("participant_id", "nunique"),
            group_mean_p_ad=("mean_p_ad", "mean"),
        )
    )


def cohens_d(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    pooled_variance = (
        ((len(values_a) - 1) * values_a.var(ddof=1))
        + ((len(values_b) - 1) * values_b.var(ddof=1))
    ) / (len(values_a) + len(values_b) - 2)
    return (values_a.mean() - values_b.mean()) / np.sqrt(pooled_variance)


def cliffs_delta(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    n_greater = sum((value > values_b).sum() for value in values_a)
    n_lower = sum((value < values_b).sum() for value in values_a)
    return (n_greater - n_lower) / (len(values_a) * len(values_b))


def bootstrap_mean_difference_ci(values_a, values_b, n_bootstrap=10_000, seed=42):
    rng = np.random.default_rng(seed)
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    differences = np.zeros(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        sample_a = rng.choice(values_a, size=len(values_a), replace=True)
        sample_b = rng.choice(values_b, size=len(values_b), replace=True)
        differences[index] = sample_a.mean() - sample_b.mean()

    return np.quantile(differences, [0.025, 0.975])


def make_performance_table():
    rows = []

    for model_name, run_name in MODEL_RUNS:
        seed_metrics = summarize_clinical_metrics_by_seed(run_name)
        pearl_seed_group_scores = summarize_pearl_group_scores_by_seed(run_name)
        pearl_seed_summary = (
            pearl_seed_group_scores
            .groupby("group")["group_mean_p_ad"]
            .agg(["mean", "std", "count"])
        )
        pearl_subject_counts = (
            pearl_seed_group_scores
            .groupby("group")["n_subjects"]
            .first()
        )
        inference = INFERENCE_PROFILE[model_name]

        row = {
            "model": model_name,
            "run_name": run_name,
            "n_clinical_seeds": len(seed_metrics),
            "n_clinical_subjects_per_seed": int(seed_metrics["n_subjects"].iloc[0]),
            "balanced_accuracy_mean": seed_metrics["balanced_accuracy"].mean(),
            "balanced_accuracy_sd": seed_metrics["balanced_accuracy"].std(),
            "auroc_mean": seed_metrics["auroc"].mean(),
            "auroc_sd": seed_metrics["auroc"].std(),
            "size_kb": inference["size_kb"],
            "latency_ms": inference["latency_ms"],
            "throughput_per_second": inference["throughput_per_second"],
        }

        for group in PEARL_GROUPS:
            row[f"{group}_mean"] = pearl_seed_summary.loc[group, "mean"]
            row[f"{group}_sd"] = pearl_seed_summary.loc[group, "std"]
            row[f"{group}_n_seeds"] = int(pearl_seed_summary.loc[group, "count"])
            row[f"{group}_n_subjects"] = int(pearl_subject_counts.loc[group])

        rows.append(row)

    return pd.DataFrame(rows)


def make_transfer_contrast_table():
    rows = []

    for model_index, (model_name, run_name) in enumerate(MODEL_RUNS):
        pearl_subject_scores = load_pearl_subject_scores(run_name)
        group_scores = {
            group: group_data["mean_p_ad"].to_numpy()
            for group, group_data in pearl_subject_scores.groupby("group")
        }

        for target_group, reference_group, bootstrap_seed_index in TRANSFER_CONTRASTS:
            target_values = group_scores[target_group]
            reference_values = group_scores[reference_group]
            mean_difference = target_values.mean() - reference_values.mean()
            ci_low, ci_high = bootstrap_mean_difference_ci(
                target_values,
                reference_values,
                seed=10_000 + model_index * 100 + bootstrap_seed_index,
            )
            statistic, p_uncorrected = mannwhitneyu(
                target_values,
                reference_values,
                alternative="two-sided",
            )
            rows.append(
                {
                    "model": model_name,
                    "run_name": run_name,
                    "contrast": f"{target_group}_minus_{reference_group}",
                    "target_group": target_group,
                    "reference_group": reference_group,
                    "target_n": len(target_values),
                    "reference_n": len(reference_values),
                    "target_mean": target_values.mean(),
                    "reference_mean": reference_values.mean(),
                    "mean_difference": mean_difference,
                    "mean_difference_ci_low": ci_low,
                    "mean_difference_ci_high": ci_high,
                    "cohens_d": cohens_d(target_values, reference_values),
                    "cliffs_delta": cliffs_delta(target_values, reference_values),
                    "mannwhitney_u": statistic,
                    "p_uncorrected": p_uncorrected,
                }
            )

    contrasts = pd.DataFrame(rows)
    rejected, p_fdr = fdrcorrection(contrasts["p_uncorrected"].to_numpy())
    contrasts["p_fdr"] = p_fdr
    contrasts["reject_fdr_0_05"] = rejected
    return contrasts


def write_performance_latex(performance, output_path):
    lines = [
        r"\begin{table*}[t]",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\centering",
        r"\caption{Subject-level clinical classification performance, zero-shot PEARL-Neuro transfer scores, and Nvidia-Jetson Orin Nano 7W-inference profiling. Clinical metrics are reported as mean $\pm$ SD across random seeds, with each seed computed from pooled held-out cross-validation subject predictions. PEARL-Neuro AD-Signature EEG Score values are seed-level group means reported as mean $\pm$ SD across random seeds. Inference metrics (model size (KB), latency (ms), and throughput) reflect optimized single-core CPU execution.}",
        r"\label{tab:model_performance_transfer}",
        r"\begin{tabular}{@{}l|cc|ccc|ccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c|}{\textbf{Clinical (Source)}} & \multicolumn{3}{c|}{\textbf{PEARL-Neuro (Target)}} & \multicolumn{3}{c}{\textbf{Inference (CPU)}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(l){7-9}",
        r" & \textbf{Bal. Accuracy} & \textbf{AUROC} & \textbf{N} & \textbf{A+P-} & \textbf{A+P+} & \textbf{Size} & \textbf{Latency} & \textbf{Throughput} \\",
        r"\midrule",
    ]

    for _, row in performance.iterrows():
        model = MODEL_LATEX_LABELS[row["model"]]
        bold_inference = row["model"] == "DiffLogic"

        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                model,
                format_mean_sd(row["balanced_accuracy_mean"], row["balanced_accuracy_sd"]),
                format_mean_sd(row["auroc_mean"], row["auroc_sd"]),
                format_mean_sd(row["N_mean"], row["N_sd"]),
                format_mean_sd(row["A+P-_mean"], row["A+P-_sd"]),
                format_mean_sd(row["A+P+_mean"], row["A+P+_sd"]),
                format_inference_value(row["size_kb"], decimals=1, bold=bold_inference),
                format_inference_value(row["latency_ms"], decimals=2, bold=bold_inference),
                format_inference_value(row["throughput_per_second"], decimals=0, bold=bold_inference),
            )
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    output_path.write_text("\n".join(lines))


def write_contrast_latex(contrasts, output_path):
    pivot_rows = []

    for model_name, model_data in contrasts.groupby("model", sort=False):
        row = {"model": model_name}
        for _, contrast_data in model_data.iterrows():
            contrast = contrast_data["contrast"]
            row[f"{contrast}_mean"] = f"${contrast_data['mean_difference']:.3f}$"
            row[f"{contrast}_ci"] = (
                f"$[{contrast_data['mean_difference_ci_low']:.3f}, "
                f"{contrast_data['mean_difference_ci_high']:.3f}]$"
            )
            row[f"{contrast}_p"] = f"${contrast_data['p_uncorrected']:.3f}$"
        pivot_rows.append(row)

    lines = [
        r"\begin{table*}[t]",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\centering",
        r"\scriptsize",
        r"\caption{PEARL-Neuro transfer contrasts from seed/fold-averaged subject-level AD-Signature EEG Scores. Mean differences and bootstrap 95\% confidence intervals are reported. Uncorrected $p$ values are derived from Mann-Whitney U tests.}",
        r"\label{tab:model_transfer_contrasts}",
        r"\begin{tabular}{@{}l|ccc|ccc|ccc@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{3}{c|}{\textbf{A+P- - N}} & \multicolumn{3}{c|}{\textbf{A+P+ - N}} & \multicolumn{3}{c}{\textbf{A+P+ - A+P-}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(l){8-10}",
        r" & \textbf{Mean Diff.} & \textbf{95\% CI} & \textbf{$p$} & \textbf{Mean Diff.} & \textbf{95\% CI} & \textbf{$p$} & \textbf{Mean Diff.} & \textbf{95\% CI} & \textbf{$p$} \\",
        r"\midrule",
    ]

    for row in pivot_rows:
        model = row["model"]
        if model == "DiffLogic":
            model = r"\textbf{DiffLogic}"

        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                model,
                row["A+P-_minus_N_mean"],
                row["A+P-_minus_N_ci"],
                row["A+P-_minus_N_p"],
                row["A+P+_minus_N_mean"],
                row["A+P+_minus_N_ci"],
                row["A+P+_minus_N_p"],
                row["A+P+_minus_A+P-_mean"],
                row["A+P+_minus_A+P-_ci"],
                row["A+P+_minus_A+P-_p"],
            )
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    output_path.write_text("\n".join(lines))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    performance = make_performance_table()
    contrasts = make_transfer_contrast_table()

    performance.to_csv(OUTPUT_DIR / "model_performance_transfer_table.csv", index=False)
    contrasts.to_csv(OUTPUT_DIR / "model_transfer_contrasts_fdr.csv", index=False)
    write_performance_latex(performance, OUTPUT_DIR / "model_performance_transfer_table.tex")
    write_contrast_latex(contrasts, OUTPUT_DIR / "model_transfer_contrasts_table.tex")

    print(f"Saved model comparison tables to {OUTPUT_DIR}")
    print()
    print(performance.to_string(index=False))
    print()
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
