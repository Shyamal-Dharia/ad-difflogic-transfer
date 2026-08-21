import argparse
import gc
import re
import time
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from feature_extraction_psd import compute_welch_psd, make_psd_features
from train_helpers import ROOT_DIR, relative_psd_to_bandpower


COMMON_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2",
]
TARGET_SFREQ = 250
EPOCH_SECONDS = 4
EPOCH_REJECT_VOLTS = 150e-6
NOTCH_B, NOTCH_A = iirnotch(50.0, 30.0, fs=TARGET_SFREQ)
BANDPASS_SOS = butter(
    4, [1.0, 45.0], btype="bandpass", fs=TARGET_SFREQ, output="sos"
)


def load_eeglab_set(path):
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    channel_names = [channel.labels for channel in data["chanlocs"].flat]
    return data["data"].astype(np.float32) * 1e-6, channel_names, float(data["srate"])


def parse_brainvision_header(path):
    text = path.read_text(encoding="utf-8-sig")
    values = {}
    channels = []
    resolutions = []
    section = None
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if "=" not in line or line.startswith(";"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
        if section == "Channel Infos" and re.fullmatch(r"Ch\d+", key.strip()):
            fields = value.split(",")
            channels.append(fields[0])
            resolutions.append(float(fields[2]))
    return values, channels, np.asarray(resolutions)


def brainvision_crop_samples(marker_path, sfreq):
    start_sample = None
    stop_sample = None
    for line in marker_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("Mk") or "=" not in line:
            continue
        fields = line.split("=", 1)[1].split(",")
        if len(fields) < 3:
            continue
        code_match = re.search(r"(\d+)", fields[1])
        if code_match is None:
            continue
        code = int(code_match.group(1))
        position = int(fields[2]) - 1
        if code == 4 and start_sample is None:
            start_sample = position
        elif code == 11 and start_sample is not None:
            stop_sample = position
            break
    if start_sample is None or stop_sample is None:
        raise ValueError(f"Could not find S4--S11 crop markers in {marker_path}")
    return start_sample, stop_sample


def load_brainvision(path):
    values, channel_names, resolutions = parse_brainvision_header(path)
    sfreq = 1e6 / float(values["SamplingInterval"])
    n_channels = int(values["NumberOfChannels"])
    eeg_path = path.with_name(values["DataFile"])
    raw = np.memmap(eeg_path, dtype="<f4", mode="r").reshape(-1, n_channels).T
    raw = np.asarray(raw, dtype=np.float32) * resolutions[:, None] * 1e-6
    marker_path = path.with_name(values["MarkerFile"])
    start_sample, stop_sample = brainvision_crop_samples(marker_path, sfreq)
    return raw[:, start_sample:stop_sample], channel_names, sfreq


def scipy_preprocess(data, channel_names, sfreq):
    channel_indices = [channel_names.index(channel) for channel in COMMON_CHANNELS]
    selected = np.ascontiguousarray(data[channel_indices])
    sampling_gcd = gcd(int(round(sfreq)), TARGET_SFREQ)
    resampled = resample_poly(
        selected,
        TARGET_SFREQ // sampling_gcd,
        int(round(sfreq)) // sampling_gcd,
        axis=-1,
    )
    notched = filtfilt(NOTCH_B, NOTCH_A, resampled, axis=-1)
    filtered = sosfiltfilt(BANDPASS_SOS, notched, axis=-1)
    return (filtered - filtered.mean(axis=0, keepdims=True)).astype(np.float32)


def make_clean_epoch_array(data):
    samples_per_epoch = TARGET_SFREQ * EPOCH_SECONDS
    n_epochs = data.shape[-1] // samples_per_epoch
    epochs = data[:, : n_epochs * samples_per_epoch]
    epochs = epochs.reshape(len(COMMON_CHANNELS), n_epochs, samples_per_epoch)
    epochs = epochs.transpose(1, 0, 2)
    keep = np.ptp(epochs, axis=2).max(axis=1) <= EPOCH_REJECT_VOLTS
    return epochs[keep]


def extract_bandpower(epochs):
    psd, freqs = compute_welch_psd(epochs, TARGET_SFREQ)
    _, relative_psd, _ = make_psd_features(psd)
    return relative_psd_to_bandpower(relative_psd, freqs)


def benchmark_recording(data, channel_names, sfreq, dataset_name, repeats):
    rows = []
    for repeat in range(repeats + 1):
        gc.collect()
        start = time.perf_counter_ns()
        processed = scipy_preprocess(data, channel_names, sfreq)
        preprocessed = time.perf_counter_ns()
        epochs = make_clean_epoch_array(processed)
        epoched = time.perf_counter_ns()
        features = extract_bandpower(epochs)
        finished = time.perf_counter_ns()
        if repeat == 0:
            continue
        n_epochs = len(epochs)
        rows.append(
            {
                "dataset": dataset_name,
                "repeat": repeat,
                "input_sfreq": sfreq,
                "input_channels": len(channel_names),
                "recording_seconds": data.shape[-1] / sfreq,
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
    return rows, epochs


def benchmark_block(data, channel_names, sfreq, dataset_name, repeats):
    start_sample = int(10 * sfreq)
    stop_sample = start_sample + int(EPOCH_SECONDS * sfreq)
    block = data[:, start_sample:stop_sample]
    rows = []
    for repeat in range(repeats + 1):
        start = time.perf_counter_ns()
        processed = scipy_preprocess(block, channel_names, sfreq)
        np.ptp(processed, axis=1).max()
        preprocessed = time.perf_counter_ns()
        extract_bandpower(processed[None])
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


def example_recordings():
    clinical_path = sorted(
        (ROOT_DIR / "datasets/ALZ_FTD").glob("sub-*/eeg/*_task-eyesclosed_eeg.set")
    )[0]
    pearl_path = sorted(
        (ROOT_DIR / "datasets/PEARL").glob("sub-*/eeg/*_task-rest_eeg.vhdr")
    )[0]
    return [
        ("Clinical",) + load_eeglab_set(clinical_path),
        ("PEARL",) + load_brainvision(pearl_path),
    ]


def summarize(rows):
    table = pd.DataFrame(rows)
    return table.groupby("dataset").agg(
        repeats=("repeat", "count"),
        preprocessing_median_ms=("preprocessing_ms", "median"),
        preprocessing_p95_ms=("preprocessing_ms", lambda x: np.quantile(x, 0.95)),
        feature_median_ms=("feature_ms", "median"),
        feature_p95_ms=("feature_ms", lambda x: np.quantile(x, 0.95)),
        total_compute_median_ms=("total_compute_ms", "median"),
        total_compute_p95_ms=("total_compute_ms", lambda x: np.quantile(x, 0.95)),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-repeats", type=int, default=10)
    parser.add_argument("--block-repeats", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs/revision_2026/end_to_end_latency_scipy",
    )
    args = parser.parse_args()

    gc.disable()
    recording_rows = []
    block_rows = []
    for dataset_name, data, channel_names, sfreq in example_recordings():
        rows, _ = benchmark_recording(
            data, channel_names, sfreq, dataset_name, args.recording_repeats
        )
        recording_rows.extend(rows)
        block_rows.extend(
            benchmark_block(
                data, channel_names, sfreq, dataset_name, args.block_repeats
            )
        )
    gc.enable()

    recordings = pd.DataFrame(recording_rows)
    blocks = pd.DataFrame(block_rows)
    recording_summary = recordings.groupby("dataset").agg(
        repeats=("repeat", "count"),
        input_sfreq=("input_sfreq", "first"),
        input_channels=("input_channels", "first"),
        recording_seconds=("recording_seconds", "first"),
        retained_epochs=("retained_epochs", "first"),
        preprocessing_per_epoch_median_ms=("preprocessing_per_epoch_ms", "median"),
        epoching_artifact_per_epoch_median_ms=("epoching_artifact_per_epoch_ms", "median"),
        feature_per_epoch_median_ms=("feature_per_epoch_ms", "median"),
        total_compute_per_epoch_median_ms=("total_compute_per_epoch_ms", "median"),
        total_compute_per_epoch_p95_ms=("total_compute_per_epoch_ms", lambda x: np.quantile(x, 0.95)),
    ).reset_index()
    block_summary = summarize(block_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recordings.to_csv(args.output_dir / "recording_repeats.csv", index=False)
    recording_summary.to_csv(args.output_dir / "recording_summary.csv", index=False)
    blocks.to_csv(args.output_dir / "block_repeats.csv", index=False)
    block_summary.to_csv(args.output_dir / "block_summary.csv", index=False)
    print("Whole-recording amortized")
    print(recording_summary.to_string(index=False))
    print("\nIsolated four-second raw block")
    print(block_summary.to_string(index=False))
    print(f"\nSaved SciPy benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
