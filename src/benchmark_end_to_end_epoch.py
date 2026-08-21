import argparse
import gc
import time
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from feature_extraction_psd import compute_welch_psd, make_psd_features
from preprocessing import (
    PEARL_DIR,
    crop_pearl_eyes_closed,
    make_clean_epochs,
    preprocess_raw,
)
from train_helpers import ROOT_DIR, relative_psd_to_bandpower


def extract_bandpower(clean_epochs, sfreq):
    psd, freqs = compute_welch_psd(clean_epochs, sfreq)
    _, relative_psd, _ = make_psd_features(psd)
    return relative_psd_to_bandpower(relative_psd, freqs)


def benchmark_recording(raw, dataset_name, repeats):
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for repeat in range(repeats + 1):
            gc.collect()
            start = time.perf_counter_ns()
            processed = preprocess_raw(raw)
            preprocessed = time.perf_counter_ns()
            clean_epochs, _, _ = make_clean_epochs(processed)
            epoched = time.perf_counter_ns()
            features = extract_bandpower(clean_epochs, processed.info["sfreq"])
            finished = time.perf_counter_ns()

            if repeat == 0:
                continue
            n_epochs = len(clean_epochs)
            rows.append(
                {
                    "dataset": dataset_name,
                    "repeat": repeat,
                    "input_sfreq": raw.info["sfreq"],
                    "input_channels": len(raw.ch_names),
                    "recording_seconds": raw.n_times / raw.info["sfreq"],
                    "retained_epochs": n_epochs,
                    "preprocessing_total_ms": (preprocessed - start) / 1e6,
                    "epoching_artifact_total_ms": (epoched - preprocessed) / 1e6,
                    "feature_total_ms": (finished - epoched) / 1e6,
                    "total_compute_ms": (finished - start) / 1e6,
                    "preprocessing_per_epoch_ms": (preprocessed - start) / 1e6 / n_epochs,
                    "epoching_artifact_per_epoch_ms": (epoched - preprocessed) / 1e6 / n_epochs,
                    "feature_per_epoch_ms": (finished - epoched) / 1e6 / n_epochs,
                    "total_compute_per_epoch_ms": (finished - start) / 1e6 / n_epochs,
                    "feature_shape": str(features.shape),
                }
            )
    return rows, processed, clean_epochs


def benchmark_single_epoch_feature(clean_epoch, sfreq, repeats):
    timings = np.empty(repeats)
    for repeat in range(repeats + 1):
        start = time.perf_counter_ns()
        extract_bandpower(clean_epoch[None, :, :], sfreq)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if repeat > 0:
            timings[repeat - 1] = elapsed_ms
    return timings


def benchmark_single_raw_block(raw, dataset_name, repeats):
    start_sample = int(10 * raw.info["sfreq"])
    stop_sample = start_sample + int(4 * raw.info["sfreq"]) - 1
    block = raw.copy().crop(
        tmin=start_sample / raw.info["sfreq"],
        tmax=stop_sample / raw.info["sfreq"],
    )
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for repeat in range(repeats + 1):
            start = time.perf_counter_ns()
            processed = preprocess_raw(block)
            data = processed.get_data()[None, :, :]
            np.ptp(data, axis=2).max(axis=1)
            preprocessed = time.perf_counter_ns()
            extract_bandpower(data, processed.info["sfreq"])
            finished = time.perf_counter_ns()
            if repeat > 0:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "repeat": repeat,
                        "preprocessing_ms": (preprocessed - start) / 1e6,
                        "feature_ms": (finished - preprocessed) / 1e6,
                        "total_compute_ms": (finished - start) / 1e6,
                    }
                )
    return rows


def load_example_recordings():
    clinical_path = sorted(
        (ROOT_DIR / "datasets/ALZ_FTD").glob(
            "sub-*/eeg/*_task-eyesclosed_eeg.set"
        )
    )[0]
    clinical = mne.io.read_raw_eeglab(
        clinical_path, preload=True, verbose="ERROR"
    )

    pearl_path = sorted(
        PEARL_DIR.glob("sub-*/eeg/*_task-rest_eeg.vhdr")
    )[0]
    events_path = pearl_path.with_name(
        pearl_path.name.replace("_eeg.vhdr", "_events.tsv")
    )
    pearl = mne.io.read_raw_brainvision(
        pearl_path, preload=True, verbose="ERROR"
    )
    crop_summary = pd.read_csv(
        ROOT_DIR / "datasets/processed/PEARL/pearl_crop_summary.csv"
    )
    typical_seconds = crop_summary["eyes_closed_seconds"].median()
    pearl, _, _ = crop_pearl_eyes_closed(
        pearl, events_path, typical_seconds
    )
    return [("Clinical", clinical), ("PEARL", pearl)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--feature-repeats", type=int, default=1000)
    parser.add_argument("--block-repeats", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs/revision_2026/end_to_end_latency",
    )
    args = parser.parse_args()

    gc.disable()
    recording_rows = []
    feature_rows = []
    block_rows = []
    for dataset_name, raw in load_example_recordings():
        rows, processed, clean_epochs = benchmark_recording(
            raw, dataset_name, args.repeats
        )
        recording_rows.extend(rows)
        timings = benchmark_single_epoch_feature(
            clean_epochs[0], processed.info["sfreq"], args.feature_repeats
        )
        feature_rows.append(
            {
                "dataset": dataset_name,
                "repeats": args.feature_repeats,
                "mean_ms": timings.mean(),
                "median_ms": np.median(timings),
                "p95_ms": np.quantile(timings, 0.95),
                "min_ms": timings.min(),
                "max_ms": timings.max(),
            }
        )
        block_rows.extend(
            benchmark_single_raw_block(raw, dataset_name, args.block_repeats)
        )
    gc.enable()

    recordings = pd.DataFrame(recording_rows)
    features = pd.DataFrame(feature_rows)
    blocks = pd.DataFrame(block_rows)
    summary = recordings.groupby("dataset").agg(
        repeats=("repeat", "count"),
        input_sfreq=("input_sfreq", "first"),
        input_channels=("input_channels", "first"),
        recording_seconds=("recording_seconds", "first"),
        retained_epochs=("retained_epochs", "first"),
        preprocessing_median_ms=("preprocessing_total_ms", "median"),
        epoching_artifact_median_ms=("epoching_artifact_total_ms", "median"),
        feature_median_ms=("feature_total_ms", "median"),
        total_compute_median_ms=("total_compute_ms", "median"),
        preprocessing_per_epoch_median_ms=("preprocessing_per_epoch_ms", "median"),
        epoching_artifact_per_epoch_median_ms=("epoching_artifact_per_epoch_ms", "median"),
        feature_per_epoch_median_ms=("feature_per_epoch_ms", "median"),
        total_compute_per_epoch_median_ms=("total_compute_per_epoch_ms", "median"),
        total_compute_per_epoch_p95_ms=("total_compute_per_epoch_ms", lambda x: np.quantile(x, 0.95)),
    ).reset_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recordings.to_csv(args.output_dir / "recording_benchmark_repeats.csv", index=False)
    summary.to_csv(args.output_dir / "recording_benchmark_summary.csv", index=False)
    features.to_csv(args.output_dir / "single_epoch_feature_benchmark.csv", index=False)
    blocks.to_csv(args.output_dir / "single_raw_block_benchmark_repeats.csv", index=False)
    block_summary = blocks.groupby("dataset").agg(
        repeats=("repeat", "count"),
        preprocessing_median_ms=("preprocessing_ms", "median"),
        preprocessing_p95_ms=("preprocessing_ms", lambda x: np.quantile(x, 0.95)),
        feature_median_ms=("feature_ms", "median"),
        feature_p95_ms=("feature_ms", lambda x: np.quantile(x, 0.95)),
        total_compute_median_ms=("total_compute_ms", "median"),
        total_compute_p95_ms=("total_compute_ms", lambda x: np.quantile(x, 0.95)),
    ).reset_index()
    block_summary.to_csv(
        args.output_dir / "single_raw_block_benchmark_summary.csv", index=False
    )
    print(summary.to_string(index=False))
    print("\nSingle-epoch feature extraction")
    print(features.to_string(index=False))
    print("\nIsolated four-second raw block")
    print(block_summary.to_string(index=False))
    print(f"\nSaved latency benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
