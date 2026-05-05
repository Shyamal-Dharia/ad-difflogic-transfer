import argparse
from pathlib import Path

import numpy as np
from scipy.signal import welch


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "datasets/processed"
OUTPUT_DIR = ROOT_DIR / "datasets/features_psd"

FREQ_MIN = 1.0
FREQ_MAX = 45.0
WELCH_SECONDS = 2.0
WELCH_OVERLAP_SECONDS = 1.0
EPS = 1e-20


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def compute_welch_psd(x, sfreq):
    n_times = x.shape[-1]
    nperseg = min(int(WELCH_SECONDS * sfreq), n_times)
    noverlap = min(int(WELCH_OVERLAP_SECONDS * sfreq), max(0, nperseg // 2))

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

    freq_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    return psd[:, :, freq_mask].astype(np.float32), freqs[freq_mask].astype(np.float32)


def make_psd_features(psd):
    total_power = psd.sum(axis=-1, keepdims=True)
    relative_psd = psd / np.maximum(total_power, EPS)
    log_psd = np.log10(np.maximum(psd, EPS))
    log_relative_psd = np.log10(np.maximum(relative_psd, EPS))
    return log_psd, relative_psd, log_relative_psd


def save_psd_features(input_path, output_dir):
    data = np.load(input_path, allow_pickle=True)
    sfreq = float(scalar(data["sfreq"]))
    psd, freqs = compute_welch_psd(data["x"], sfreq)
    log_psd, relative_psd, log_relative_psd = make_psd_features(psd)

    output_path = output_dir / input_path.name.replace("_preprocessed.npz", "_psd.npz")
    output = {
        "x": log_relative_psd.astype(np.float32),
        "psd": psd,
        "log_psd": log_psd.astype(np.float32),
        "relative_psd": relative_psd.astype(np.float32),
        "log_relative_psd": log_relative_psd.astype(np.float32),
        "freqs": freqs,
        "participant_id": data["participant_id"],
        "group": data["group"],
        "label": data["label"],
        "age": data["age"],
        "channel_names": data["channel_names"],
        "sfreq": data["sfreq"],
        "feature_type": "welch_log_relative_psd",
        "x_description": "log10(relative Welch PSD); relative PSD is normalized by total 1-45 Hz power per epoch and channel",
        "freq_min": np.float64(FREQ_MIN),
        "freq_max": np.float64(FREQ_MAX),
        "welch_seconds": np.float64(WELCH_SECONDS),
        "welch_overlap_seconds": np.float64(WELCH_OVERLAP_SECONDS),
        "eps": np.float64(EPS),
        "source_file": str(input_path),
    }

    for key in ["gender", "sex", "mmse", "eyes_closed_start", "eyes_closed_end"]:
        if key in data.files:
            output[key] = data[key]

    np.savez_compressed(output_path, **output)


def extract_dataset_psd(dataset_name):
    input_dir = PROCESSED_DIR / dataset_name
    output_dir = OUTPUT_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in sorted(input_dir.glob("*.npz")):
        print(f"Extracting PSD {dataset_name} {input_path.name}")
        save_psd_features(input_path, output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ALZ_FTD", "PEARL"])
    return parser.parse_args()


def main():
    args = parse_args()

    for dataset_name in args.datasets:
        extract_dataset_psd(dataset_name)


if __name__ == "__main__":
    main()
