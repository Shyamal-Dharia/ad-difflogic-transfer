import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from train_helpers import ROOT_DIR, make_subject_stratified_folds


FEATURE_CSV = ROOT_DIR / "datasets/statistics_psd_45hz/subject_log_relative_bandpower.csv"
DIFFLOGIC_RUN = ROOT_DIR / "outputs/difflogic_45hz/benchmark_250k_psd"
OUTPUT_DIR = ROOT_DIR / "outputs/revision_2026/centroid_comparison"
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def load_feature_tables(path):
    data = pd.read_csv(path)
    data["feature"] = data["channel"] + "_" + data["band"]
    columns = [f"{channel}_{band}" for channel in CHANNELS for band in BANDS]
    features = data.pivot(index="participant_id", columns="feature", values="log_relative_bandpower")
    features = features.reindex(columns=columns)
    if features.isna().any().any():
        raise ValueError("Missing values in the 75-dimensional subject PSD vectors")

    metadata = data[["participant_id", "dataset", "group"]].drop_duplicates()
    clinical = metadata[metadata["dataset"].eq("ALZ_FTD") & metadata["group"].isin(["A", "C"])].copy()
    clinical["label"] = clinical["group"].map({"C": 0, "A": 1})
    pearl = metadata[metadata["dataset"].eq("PEARL")].copy()
    return features, clinical, pearl


def centroid_scores(train_features, train_labels, test_features):
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_features)
    cn_centroid = train_scaled[train_labels == 0].mean(axis=0)
    ad_centroid = train_scaled[train_labels == 1].mean(axis=0)
    distance_cn = np.linalg.norm(test_scaled - cn_centroid, axis=1)
    distance_ad = np.linalg.norm(test_scaled - ad_centroid, axis=1)
    return distance_cn - distance_ad


def binary_metrics(labels, scores, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "auroc": roc_auc_score(labels, scores),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }


def run_centroid(features, clinical, pearl, seeds):
    clinical_rows = []
    pearl_rows = []
    clinical_index = clinical.set_index("participant_id")
    pearl_ids = pearl["participant_id"].sort_values().to_numpy()

    for seed in seeds:
        folds = make_subject_stratified_folds(clinical, random_state=seed)
        for fold in folds:
            train_ids = fold["train_subject_ids"]
            test_ids = fold["test_subject_ids"]
            train_labels = clinical_index.loc[train_ids, "label"].to_numpy()
            test_scores = centroid_scores(
                features.loc[train_ids].to_numpy(),
                train_labels,
                features.loc[test_ids].to_numpy(),
            )
            for participant_id, score in zip(test_ids, test_scores):
                clinical_rows.append(
                    {
                        "seed": seed,
                        "fold": fold["fold"],
                        "participant_id": participant_id,
                        "true_label": clinical_index.loc[participant_id, "label"],
                        "centroid_score": score,
                        "pred_label": int(score > 0),
                    }
                )

            pearl_scores = centroid_scores(
                features.loc[train_ids].to_numpy(),
                train_labels,
                features.loc[pearl_ids].to_numpy(),
            )
            for participant_id, score in zip(pearl_ids, pearl_scores):
                pearl_rows.append(
                    {
                        "seed": seed,
                        "fold": fold["fold"],
                        "participant_id": participant_id,
                        "group": pearl.set_index("participant_id").loc[participant_id, "group"],
                        "centroid_score": score,
                    }
                )

    return pd.DataFrame(clinical_rows), pd.DataFrame(pearl_rows)


def load_difflogic_predictions(run_dir, seeds):
    tables = []
    for seed in seeds:
        path = run_dir / f"seed_{seed:03d}/alz_test_subject_predictions_all_folds.csv"
        table = pd.read_csv(path)[["participant_id", "true_label", "mean_p_ad", "pred_label"]]
        table["seed"] = seed
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def metrics_by_seed(centroid, difflogic, seeds):
    rows = []
    for seed in seeds:
        centroid_seed = centroid[centroid["seed"].eq(seed)]
        difflogic_seed = difflogic[difflogic["seed"].eq(seed)]
        for model, table, score_column in [
            ("Nearest centroid", centroid_seed, "centroid_score"),
            ("Diff-Logic", difflogic_seed, "mean_p_ad"),
        ]:
            row = binary_metrics(table["true_label"], table[score_column], table["pred_label"])
            row.update({"model": model, "seed": seed, "n_subjects": len(table)})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(metrics):
    return (
        metrics.groupby("model", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            sensitivity_sd=("sensitivity", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_sd=("specificity", "std"),
        )
    )


def paired_bootstrap(centroid, difflogic, n_bootstrap, seed):
    centroid_mean = centroid.groupby(["participant_id", "true_label"], as_index=False).agg(
        score=("centroid_score", "mean"),
    )
    logic_mean = difflogic.groupby(["participant_id", "true_label"], as_index=False).agg(
        score=("mean_p_ad", "mean"),
    )
    centroid_mean["prediction"] = (centroid_mean["score"] > 0).astype(int)
    logic_mean["prediction"] = (logic_mean["score"] >= 0.5).astype(int)
    paired = logic_mean.merge(
        centroid_mean,
        on=["participant_id", "true_label"],
        suffixes=("_difflogic", "_centroid"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    negative = paired[paired["true_label"].eq(0)].reset_index(drop=True)
    positive = paired[paired["true_label"].eq(1)].reset_index(drop=True)
    negative_indices = rng.integers(0, len(negative), size=(n_bootstrap, len(negative)))
    positive_indices = rng.integers(0, len(positive), size=(n_bootstrap, len(positive)))

    differences = {}
    for model in ["difflogic", "centroid"]:
        negative_scores = negative[f"score_{model}"].to_numpy()[negative_indices]
        positive_scores = positive[f"score_{model}"].to_numpy()[positive_indices]
        negative_predictions = negative[f"prediction_{model}"].to_numpy()[negative_indices]
        positive_predictions = positive[f"prediction_{model}"].to_numpy()[positive_indices]
        sensitivity = positive_predictions.mean(axis=1)
        specificity = 1 - negative_predictions.mean(axis=1)
        auc = (
            (positive_scores[:, :, None] > negative_scores[:, None, :]).mean(axis=(1, 2))
            + 0.5
            * (positive_scores[:, :, None] == negative_scores[:, None, :]).mean(axis=(1, 2))
        )
        model_values = {
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "auroc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }
        if model == "difflogic":
            differences = model_values
        else:
            differences = {
                metric: differences[metric] - model_values[metric]
                for metric in differences
            }

    rows = []
    logic_observed = binary_metrics(
        paired["true_label"], paired["score_difflogic"], paired["prediction_difflogic"]
    )
    centroid_observed = binary_metrics(
        paired["true_label"], paired["score_centroid"], paired["prediction_centroid"]
    )
    for metric, values in differences.items():
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "difference": logic_observed[metric] - centroid_observed[metric],
                "ci_low": low,
                "ci_high": high,
                "direction": "Diff-Logic minus nearest centroid",
            }
        )
    return pd.DataFrame(rows), paired


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", type=Path, default=FEATURE_CSV)
    parser.add_argument("--difflogic-run", type=Path, default=DIFFLOGIC_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, clinical, pearl = load_feature_tables(args.feature_csv)
    centroid, pearl_centroid = run_centroid(features, clinical, pearl, args.seeds)
    difflogic = load_difflogic_predictions(args.difflogic_run, args.seeds)
    if centroid.groupby("seed")["participant_id"].nunique().ne(65).any():
        raise ValueError("Each seed must contain exactly 65 held-out clinical predictions")
    if not centroid[["seed", "participant_id", "true_label"]].merge(
        difflogic[["seed", "participant_id", "true_label"]],
        on=["seed", "participant_id", "true_label"],
        validate="one_to_one",
    ).shape[0] == len(centroid):
        raise ValueError("Centroid and Diff-Logic subject predictions do not align")

    metrics = metrics_by_seed(centroid, difflogic, args.seeds)
    summary = summarize_metrics(metrics)
    differences, paired = paired_bootstrap(
        centroid, difflogic, args.bootstrap, args.bootstrap_seed
    )
    pearl_subject = pearl_centroid.groupby(["participant_id", "group"], as_index=False).agg(
        centroid_score=("centroid_score", "mean")
    )

    centroid.to_csv(args.output_dir / "clinical_centroid_predictions.csv", index=False)
    pearl_subject.to_csv(args.output_dir / "pearl_centroid_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "clinical_metrics_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "clinical_metrics_summary.csv", index=False)
    differences.to_csv(args.output_dir / "clinical_metric_differences_bootstrap.csv", index=False)
    paired.to_csv(args.output_dir / "clinical_paired_ensemble_predictions.csv", index=False)
    print(summary.to_string(index=False))
    print("\nPaired subject-bootstrap differences:\n" + differences.to_string(index=False))


if __name__ == "__main__":
    main()
