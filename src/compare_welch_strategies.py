import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr

from train_helpers import BANDS, ROOT_DIR, relative_psd_to_bandpower, scalar


FREQ_MIN = 1.0
FREQ_MAX = 45.0
EPS = 1e-20
PRIMARY_WINDOW_SECONDS = 2.0
PRIMARY_OVERLAP_SECONDS = 1.0


def epochwise_subject_features(x, sfreq, window_seconds, overlap_seconds):
    nperseg = min(int(window_seconds * sfreq), x.shape[-1])
    noverlap = min(int(overlap_seconds * sfreq), nperseg - 1)
    freqs, psd = welch(
        x,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    frequency_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    psd = psd[:, :, frequency_mask]
    relative_psd = psd / np.maximum(psd.sum(axis=-1, keepdims=True), EPS)
    epoch_features = relative_psd_to_bandpower(relative_psd, freqs[frequency_mask])
    return epoch_features.mean(axis=0).reshape(-1)


def whole_recording_subject_features(x, sfreq):
    continuous = x.transpose(1, 0, 2).reshape(1, x.shape[1], -1)
    return epochwise_subject_features(
        continuous,
        sfreq,
        PRIMARY_WINDOW_SECONDS,
        PRIMARY_OVERLAP_SECONDS,
    )


def load_subject_features(processed_dir):
    rows = []
    feature_names = None
    methods = {
        "epochwise_2s_50pct": lambda x, sfreq: epochwise_subject_features(
            x, sfreq, 2.0, 1.0
        ),
        "whole_recording_2s_50pct": whole_recording_subject_features,
        "epochwise_1s_50pct": lambda x, sfreq: epochwise_subject_features(
            x, sfreq, 1.0, 0.5
        ),
        "epochwise_2s_0pct": lambda x, sfreq: epochwise_subject_features(
            x, sfreq, 2.0, 0.0
        ),
        "epochwise_4s_0pct": lambda x, sfreq: epochwise_subject_features(
            x, sfreq, 4.0, 0.0
        ),
    }

    for dataset_name in ["ALZ_FTD", "PEARL"]:
        for input_path in sorted((processed_dir / dataset_name).glob("*.npz")):
            data = np.load(input_path, allow_pickle=True)
            group = str(scalar(data["group"]))
            if dataset_name == "ALZ_FTD" and group == "F":
                continue

            channel_names = data["channel_names"].astype(str).tolist()
            if feature_names is None:
                feature_names = [
                    f"{channel}_{band}" for channel in channel_names for band in BANDS
                ]
            x = data["x"]
            sfreq = float(scalar(data["sfreq"]))
            row = {
                "dataset": "Clinical CN/AD" if dataset_name == "ALZ_FTD" else "PEARL",
                "participant_id": str(scalar(data["participant_id"])),
                "group": group,
            }
            for method_name, method in methods.items():
                row[method_name] = method(x, sfreq)
            rows.append(row)

    return rows, feature_names


def correlation_tables(rows, feature_names):
    primary_method = "epochwise_2s_50pct"
    comparison_methods = [
        "whole_recording_2s_50pct",
        "epochwise_1s_50pct",
        "epochwise_2s_0pct",
        "epochwise_4s_0pct",
    ]
    summary_rows = []
    feature_rows = []

    for dataset_name in ["Clinical CN/AD", "PEARL", "Combined"]:
        dataset_rows = rows if dataset_name == "Combined" else [
            row for row in rows if row["dataset"] == dataset_name
        ]
        primary = np.stack([row[primary_method] for row in dataset_rows])

        for comparison_method in comparison_methods:
            comparison = np.stack(
                [row[comparison_method] for row in dataset_rows]
            )
            flattened_pearson = pearsonr(primary.ravel(), comparison.ravel())[0]
            flattened_spearman = spearmanr(primary.ravel(), comparison.ravel())[0]
            per_feature_pearson = np.array(
                [
                    pearsonr(primary[:, index], comparison[:, index])[0]
                    for index in range(primary.shape[1])
                ]
            )
            per_subject_pearson = np.array(
                [
                    pearsonr(primary[index], comparison[index])[0]
                    for index in range(primary.shape[0])
                ]
            )
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "comparison": comparison_method,
                    "n_subjects": len(dataset_rows),
                    "flattened_pearson_r": flattened_pearson,
                    "flattened_spearman_rho": flattened_spearman,
                    "featurewise_pearson_median": np.median(per_feature_pearson),
                    "featurewise_pearson_q1": np.quantile(per_feature_pearson, 0.25),
                    "featurewise_pearson_q3": np.quantile(per_feature_pearson, 0.75),
                    "featurewise_pearson_min": per_feature_pearson.min(),
                    "subjectwise_pearson_median": np.median(per_subject_pearson),
                    "mean_absolute_difference": np.abs(primary - comparison).mean(),
                }
            )
            for feature_name, correlation in zip(
                feature_names, per_feature_pearson
            ):
                feature_rows.append(
                    {
                        "dataset": dataset_name,
                        "comparison": comparison_method,
                        "feature": feature_name,
                        "pearson_r": correlation,
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(feature_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT_DIR / "datasets/processed"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs/revision_2026/welch_sensitivity",
    )
    args = parser.parse_args()

    rows, feature_names = load_subject_features(args.processed_dir)
    summary, feature_correlations = correlation_tables(rows, feature_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "correlation_summary.csv", index=False)
    feature_correlations.to_csv(
        args.output_dir / "featurewise_correlations.csv", index=False
    )
    print(summary.to_string(index=False))
    print(f"\nSaved Welch sensitivity results to {args.output_dir}")


if __name__ == "__main__":
    main()
