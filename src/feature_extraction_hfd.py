import argparse
from pathlib import Path

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.signal import butter, sosfiltfilt

from hfd import interval_t, lin_fit_hfd
from train_helpers import BANDS


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "datasets/processed"
OUTPUT_ROOT = ROOT_DIR / "datasets/features_hfd"

DEFAULT_K_MAX = 16
DEFAULT_NUM_K = 16
FILTER_ORDER = 4


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def init_hfd_library():
    lib_path = Path(__file__).resolve().with_name("libhfd.so")
    lib = ctypes.CDLL(str(lib_path))

    rwptr = ndpointer(float, flags=("C", "A", "W"))
    rwptr_sizet = ndpointer(ctypes.c_size_t, flags=("C", "A", "W"))

    lib.curve_length.restype = ctypes.c_int
    lib.curve_length.argtypes = [
        rwptr_sizet,
        ctypes.c_size_t,
        rwptr,
        ctypes.c_size_t,
        rwptr,
    ]
    return lib


def higuchi_fd_1d(x, k_values, lib):
    x = np.require(x, dtype=float, requirements=("C", "A"))
    k_values = np.require(k_values, dtype=ctypes.c_size_t, requirements=("C", "A"))
    curve_lengths = np.zeros(k_values.size)
    curve_lengths = np.require(curve_lengths, dtype=float, requirements=("C", "A"))
    lib.curve_length(k_values, k_values.size, x, x.size, curve_lengths)
    return lin_fit_hfd(k_values, curve_lengths)


def compute_epoch_hfd(x, k_max, num_k):
    lib = init_hfd_library()
    k_values = interval_t(x.shape[-1], num_val=num_k, kmax=k_max)
    hfd_features = np.zeros(x.shape[:2], dtype=np.float32)

    for epoch_index in range(x.shape[0]):
        for channel_index in range(x.shape[1]):
            hfd_features[epoch_index, channel_index] = higuchi_fd_1d(
                x[epoch_index, channel_index],
                k_values,
                lib,
            )

    return hfd_features, k_values.astype(np.int64)


def bandpass_epochs(x, sfreq, low_freq, high_freq):
    sos = butter(
        FILTER_ORDER,
        [low_freq, high_freq],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def compute_band_hfd(x, sfreq, k_max, num_k):
    band_features = []
    k_values = None

    for band_name, (low_freq, high_freq) in BANDS.items():
        print(f"  bandpass HFD {band_name} {low_freq}-{high_freq} Hz")
        band_data = bandpass_epochs(x, sfreq, low_freq, high_freq)
        hfd_features, k_values = compute_epoch_hfd(band_data, k_max, num_k)
        band_features.append(hfd_features)

    return np.stack(band_features, axis=-1).astype(np.float32), k_values


def save_hfd_features(input_path, output_dir, k_max, num_k):
    data = np.load(input_path, allow_pickle=True)
    sfreq = float(scalar(data["sfreq"]))
    hfd_features, k_values = compute_band_hfd(data["x"], sfreq, k_max, num_k)

    output_path = output_dir / input_path.name.replace("_preprocessed.npz", "_hfd.npz")
    output = {
        "x": hfd_features,
        "hfd": hfd_features,
        "bands": np.array(list(BANDS)),
        "participant_id": data["participant_id"],
        "group": data["group"],
        "label": data["label"],
        "age": data["age"],
        "channel_names": data["channel_names"],
        "sfreq": data["sfreq"],
        "feature_type": "bandpass_higuchi_fractal_dimension",
        "k_max": np.array("none" if k_max is None else str(k_max)),
        "k_max_effective": np.int64(k_values.max()),
        "num_k": np.int64(num_k),
        "k_values": k_values,
        "filter_order": np.int64(FILTER_ORDER),
        "source_file": str(input_path),
    }

    for key in ["gender", "sex", "mmse", "eyes_closed_start", "eyes_closed_end"]:
        if key in data.files:
            output[key] = data[key]

    np.savez_compressed(output_path, **output)


def extract_dataset_hfd(dataset_name, k_max, num_k, output_name):
    input_dir = PROCESSED_DIR / dataset_name
    output_dir = OUTPUT_ROOT / output_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in sorted(input_dir.glob("*.npz")):
        print(f"Extracting HFD {dataset_name} {input_path.name}")
        save_hfd_features(input_path, output_dir, k_max, num_k)


def parse_k_max(value):
    if value.lower() in ["none", "auto"]:
        return None

    return int(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-max", type=parse_k_max, default=DEFAULT_K_MAX)
    parser.add_argument("--num-k", type=int, default=DEFAULT_NUM_K)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    k_max_name = "none" if args.k_max is None else args.k_max
    output_name = args.output_name or f"kmax_{k_max_name}"
    extract_dataset_hfd("ALZ_FTD", args.k_max, args.num_k, output_name)
    extract_dataset_hfd("PEARL", args.k_max, args.num_k, output_name)


if __name__ == "__main__":
    main()
