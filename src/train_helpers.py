import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


ROOT_DIR = Path(__file__).resolve().parents[1]


def path_from_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default

    path = Path(value)
    if path.is_absolute():
        return path

    return ROOT_DIR / path


PSD_FEATURE_DIR = path_from_env("PSD_FEATURE_DIR", ROOT_DIR / "datasets/features_psd")
HFD_FEATURE_ROOT = ROOT_DIR / "datasets/features_hfd"
FOLD_DIR = ROOT_DIR / "datasets/folds"

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

EPS = 1e-20
THERMOMETER_BINS = 15


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def band_mask(freqs, band_name):
    low_freq, high_freq = BANDS[band_name]

    if band_name == "gamma":
        return (freqs >= low_freq) & (freqs <= high_freq)

    return (freqs >= low_freq) & (freqs < high_freq)


def relative_psd_to_bandpower(relative_psd, freqs):
    bandpowers = []

    for band_name in BANDS:
        mask = band_mask(freqs, band_name)
        bandpower = relative_psd[:, :, mask].sum(axis=-1)
        bandpowers.append(bandpower)

    relative_bandpower = np.stack(bandpowers, axis=-1)
    log_relative_bandpower = np.log10(np.maximum(relative_bandpower, EPS))
    return log_relative_bandpower.astype(np.float32)


def alz_c_vs_a_label(group):
    if group == "C":
        return 0, "C"
    if group == "A":
        return 1, "A"

    return None, None


def alz_c_vs_f_label(group):
    if group == "C":
        return 0, "C"
    if group == "F":
        return 1, "F"

    return None, None


def pearl_group_label(group):
    if group == "N":
        return 0, "N"
    if group == "A+P-":
        return 1, "A+P-"
    if group == "A+P+":
        return 2, "A+P+"

    return None, None


def ad_hc_group_label(group):
    if group == "HC":
        return 0, "HC"
    if group == "AD":
        return 1, "AD"

    return None, None


def normalize_excluded_channels(exclude_channels=None):
    if exclude_channels is None:
        return set()

    return {channel.strip() for channel in exclude_channels if channel.strip()}


def make_subject(
    dataset_name,
    data,
    x,
    channel_names,
    label,
    selected_group,
    participant_id,
):
    return {
        "participant_id": participant_id,
        "dataset": dataset_name,
        "x": x,
        "label": label,
        "group": selected_group,
        "age": int(scalar(data["age"])),
        "channel_names": channel_names,
        "n_epochs": x.shape[0],
        "n_channels": x.shape[1],
        "n_bands": x.shape[2],
    }


def load_subject_feature(dataset_name, feature_path, label_function, exclude_channels=None):
    data = np.load(feature_path, allow_pickle=True)
    group = str(scalar(data["group"]))
    label, selected_group = label_function(group)

    if label is None:
        return None

    channel_names = data["channel_names"].astype(str)
    excluded_channels = normalize_excluded_channels(exclude_channels)
    keep_channels = np.array([channel not in excluded_channels for channel in channel_names])
    relative_psd = data["relative_psd"][:, keep_channels, :]
    x = relative_psd_to_bandpower(relative_psd, data["freqs"])
    channel_names = channel_names[keep_channels]
    participant_id = str(scalar(data["participant_id"]))

    return make_subject(
        dataset_name,
        data,
        x,
        channel_names,
        label,
        selected_group,
        participant_id,
    )


def load_bandpower_dataset(dataset_name, label_function, exclude_channels=None):
    subjects = []

    for feature_path in sorted((PSD_FEATURE_DIR / dataset_name).glob("*.npz")):
        subject = load_subject_feature(
            dataset_name,
            feature_path,
            label_function,
            exclude_channels=exclude_channels,
        )

        if subject is not None:
            subjects.append(subject)

    subject_table = pd.DataFrame(
        [
            {
                "participant_id": subject["participant_id"],
                "dataset": subject["dataset"],
                "label": subject["label"],
                "group": subject["group"],
                "age": subject["age"],
                "n_epochs": subject["n_epochs"],
                "n_channels": subject["n_channels"],
                "n_bands": subject["n_bands"],
            }
            for subject in subjects
        ]
    )

    return subjects, subject_table


def load_alz_c_vs_a_bandpower_dataset(exclude_channels=None):
    return load_bandpower_dataset(
        "ALZ_FTD",
        alz_c_vs_a_label,
        exclude_channels=exclude_channels,
    )


def load_alz_c_vs_f_bandpower_dataset(exclude_channels=None):
    return load_bandpower_dataset(
        "ALZ_FTD",
        alz_c_vs_f_label,
        exclude_channels=exclude_channels,
    )


def load_ad_hc_bandpower_dataset(dataset_name, exclude_channels=None):
    return load_bandpower_dataset(
        dataset_name,
        ad_hc_group_label,
        exclude_channels=exclude_channels,
    )


def load_pearl_bandpower_dataset(dataset_name="PEARL", exclude_channels=None):
    return load_bandpower_dataset(
        dataset_name,
        pearl_group_label,
        exclude_channels=exclude_channels,
    )


def load_subject_hfd_feature(dataset_name, feature_path, label_function, exclude_channels=None):
    data = np.load(feature_path, allow_pickle=True)
    group = str(scalar(data["group"]))
    label, selected_group = label_function(group)

    if label is None:
        return None

    channel_names = data["channel_names"].astype(str)
    excluded_channels = normalize_excluded_channels(exclude_channels)
    keep_channels = np.array([channel not in excluded_channels for channel in channel_names])
    x = data["x"][:, keep_channels, :].astype(np.float32)
    channel_names = channel_names[keep_channels]
    participant_id = str(scalar(data["participant_id"]))

    return make_subject(
        dataset_name,
        data,
        x,
        channel_names,
        label,
        selected_group,
        participant_id,
    )


def load_hfd_dataset(dataset_name, label_function, hfd_name="kmax_16", exclude_channels=None):
    subjects = []
    feature_dir = HFD_FEATURE_ROOT / hfd_name / dataset_name

    for feature_path in sorted(feature_dir.glob("*.npz")):
        subject = load_subject_hfd_feature(
            dataset_name,
            feature_path,
            label_function,
            exclude_channels=exclude_channels,
        )

        if subject is not None:
            subjects.append(subject)

    subject_table = pd.DataFrame(
        [
            {
                "participant_id": subject["participant_id"],
                "dataset": subject["dataset"],
                "label": subject["label"],
                "group": subject["group"],
                "age": subject["age"],
                "n_epochs": subject["n_epochs"],
                "n_channels": subject["n_channels"],
                "n_bands": subject["n_bands"],
            }
            for subject in subjects
        ]
    )

    return subjects, subject_table


def load_alz_c_vs_a_hfd_dataset(hfd_name="kmax_16", exclude_channels=None):
    return load_hfd_dataset(
        "ALZ_FTD",
        alz_c_vs_a_label,
        hfd_name=hfd_name,
        exclude_channels=exclude_channels,
    )


def load_alz_c_vs_f_hfd_dataset(hfd_name="kmax_16", exclude_channels=None):
    return load_hfd_dataset(
        "ALZ_FTD",
        alz_c_vs_f_label,
        hfd_name=hfd_name,
        exclude_channels=exclude_channels,
    )


def load_pearl_hfd_dataset(dataset_name="PEARL", hfd_name="kmax_16", exclude_channels=None):
    return load_hfd_dataset(
        dataset_name,
        pearl_group_label,
        hfd_name=hfd_name,
        exclude_channels=exclude_channels,
    )


def load_alz_c_vs_a_dataset(feature_kind="psd", hfd_name="kmax_16", exclude_channels=None):
    if feature_kind == "hfd":
        return load_alz_c_vs_a_hfd_dataset(
            hfd_name=hfd_name,
            exclude_channels=exclude_channels,
        )

    return load_alz_c_vs_a_bandpower_dataset(exclude_channels=exclude_channels)


def load_alz_c_vs_f_dataset(feature_kind="psd", hfd_name="kmax_16", exclude_channels=None):
    if feature_kind == "hfd":
        return load_alz_c_vs_f_hfd_dataset(
            hfd_name=hfd_name,
            exclude_channels=exclude_channels,
        )

    return load_alz_c_vs_f_bandpower_dataset(exclude_channels=exclude_channels)


def load_alz_dataset(
    clinical_task="cn_vs_ad",
    feature_kind="psd",
    hfd_name="kmax_16",
    exclude_channels=None,
):
    if clinical_task == "cn_vs_ftd":
        return load_alz_c_vs_f_dataset(
            feature_kind=feature_kind,
            hfd_name=hfd_name,
            exclude_channels=exclude_channels,
        )

    return load_alz_c_vs_a_dataset(
        feature_kind=feature_kind,
        hfd_name=hfd_name,
        exclude_channels=exclude_channels,
    )


def load_source_dataset(
    dataset_name="ALZ_FTD",
    clinical_task="cn_vs_ad",
    feature_kind="psd",
    hfd_name="kmax_16",
    exclude_channels=None,
):
    if dataset_name == "ALZ_FTD":
        return load_alz_dataset(
            clinical_task=clinical_task,
            feature_kind=feature_kind,
            hfd_name=hfd_name,
            exclude_channels=exclude_channels,
        )

    if feature_kind != "psd":
        raise ValueError("Only PSD features are supported for generic AD/HC source datasets.")

    return load_ad_hc_bandpower_dataset(dataset_name, exclude_channels=exclude_channels)


def load_pearl_dataset(
    dataset_name="PEARL",
    feature_kind="psd",
    hfd_name="kmax_16",
    exclude_channels=None,
):
    if feature_kind == "hfd":
        return load_pearl_hfd_dataset(
            dataset_name=dataset_name,
            hfd_name=hfd_name,
            exclude_channels=exclude_channels,
        )

    return load_pearl_bandpower_dataset(dataset_name=dataset_name, exclude_channels=exclude_channels)


def stack_subjects(subjects, subject_ids):
    selected_subjects = [
        subject
        for subject in subjects
        if subject["participant_id"] in set(subject_ids)
    ]

    x = np.concatenate([subject["x"] for subject in selected_subjects], axis=0)
    y = np.concatenate(
        [
            np.full(subject["n_epochs"], subject["label"], dtype=np.int64)
            for subject in selected_subjects
        ]
    )
    epoch_subject_ids = np.concatenate(
        [
            np.full(subject["n_epochs"], subject["participant_id"])
            for subject in selected_subjects
        ]
    )

    return x, y, epoch_subject_ids


def fit_minmax_scaler(x_train):
    feature_min = x_train.min(axis=0, keepdims=True)
    feature_max = x_train.max(axis=0, keepdims=True)

    return {
        "feature_min": feature_min.astype(np.float32),
        "feature_max": feature_max.astype(np.float32),
    }


def apply_minmax_scaler(x, scaler):
    feature_min = scaler["feature_min"]
    feature_max = scaler["feature_max"]
    denominator = np.maximum(feature_max - feature_min, EPS)
    x_scaled = (x - feature_min) / denominator
    return np.clip(x_scaled, 0.0, 1.0).astype(np.float32)


def fit_thermometer_encoder(x_train, n_bins=THERMOMETER_BINS):
    scaler = fit_minmax_scaler(x_train)
    thresholds = np.linspace(1.0 / n_bins, 1.0, n_bins, dtype=np.float32)

    return {
        "feature_min": scaler["feature_min"],
        "feature_max": scaler["feature_max"],
        "thresholds": thresholds,
        "n_bins": n_bins,
    }


def apply_thermometer_encoder(x, encoder, flatten=True):
    x_scaled = apply_minmax_scaler(x, encoder)
    encoded = x_scaled[..., None] >= encoder["thresholds"]
    encoded = encoded.astype(np.float32)

    if flatten:
        encoded = encoded.reshape(encoded.shape[0], -1)

    return encoded


def encode_fold_for_difflogic(x_train, x_val, x_test, x_pearl=None, n_bins=THERMOMETER_BINS):
    encoder = fit_thermometer_encoder(x_train, n_bins=n_bins)
    x_train_encoded = apply_thermometer_encoder(x_train, encoder)
    x_val_encoded = apply_thermometer_encoder(x_val, encoder)
    x_test_encoded = apply_thermometer_encoder(x_test, encoder)
    x_pearl_encoded = None

    if x_pearl is not None:
        x_pearl_encoded = apply_thermometer_encoder(x_pearl, encoder)

    return encoder, x_train_encoded, x_val_encoded, x_test_encoded, x_pearl_encoded


def scale_fold_features(x_train, x_val, x_test, x_pearl=None):
    scaler = fit_minmax_scaler(x_train)
    x_train_scaled = apply_minmax_scaler(x_train, scaler)
    x_val_scaled = apply_minmax_scaler(x_val, scaler)
    x_test_scaled = apply_minmax_scaler(x_test, scaler)
    x_pearl_scaled = None

    if x_pearl is not None:
        x_pearl_scaled = apply_minmax_scaler(x_pearl, scaler)

    return scaler, x_train_scaled, x_val_scaled, x_test_scaled, x_pearl_scaled


def make_subject_stratified_folds(subject_table, n_splits=10, val_size=0.20, random_state=42):
    subject_table = subject_table.sort_values("participant_id").reset_index(drop=True)
    subject_ids = subject_table["participant_id"].to_numpy()
    labels = subject_table["label"].to_numpy()

    folds = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_index, train_val_index, test_index in split_with_index(splitter, subject_ids, labels):
        train_val_labels = labels[train_val_index]
        validation_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=random_state + fold_index,
        )
        train_relative_index, val_relative_index = next(
            validation_splitter.split(subject_ids[train_val_index], train_val_labels)
        )

        train_index = train_val_index[train_relative_index]
        val_index = train_val_index[val_relative_index]

        folds.append(
            {
                "fold": fold_index,
                "train_subject_ids": subject_ids[train_index],
                "val_subject_ids": subject_ids[val_index],
                "test_subject_ids": subject_ids[test_index],
            }
        )

    return folds


def split_with_index(splitter, x, y):
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y), start=1):
        yield fold_index, train_index, test_index


def count_split(subject_table, subject_ids):
    split_table = subject_table[subject_table["participant_id"].isin(subject_ids)]
    counts = split_table.groupby("group").agg(
        n_subjects=("participant_id", "count"),
        n_epochs=("n_epochs", "sum"),
    )
    return counts


def summarize_folds(subject_table, folds):
    rows = []

    for fold in folds:
        for split_name in ["train", "val", "test"]:
            subject_ids = fold[f"{split_name}_subject_ids"]
            counts = count_split(subject_table, subject_ids)

            for group in sorted(subject_table["group"].unique()):
                n_subjects = 0
                n_epochs = 0

                if group in counts.index:
                    n_subjects = int(counts.loc[group, "n_subjects"])
                    n_epochs = int(counts.loc[group, "n_epochs"])

                rows.append(
                    {
                        "fold": fold["fold"],
                        "split": split_name,
                        "group": group,
                        "n_subjects": n_subjects,
                        "n_epochs": n_epochs,
                    }
                )

    return pd.DataFrame(rows)


def make_fold_assignment_table(subject_table, folds):
    rows = []

    for fold in folds:
        for split_name in ["train", "val", "test"]:
            subject_ids = fold[f"{split_name}_subject_ids"]
            split_table = subject_table[subject_table["participant_id"].isin(subject_ids)]

            for _, row in split_table.iterrows():
                rows.append(
                    {
                        "fold": fold["fold"],
                        "split": split_name,
                        "participant_id": row["participant_id"],
                        "group": row["group"],
                        "label": row["label"],
                        "n_epochs": row["n_epochs"],
                    }
                )

    return pd.DataFrame(rows)


def print_dataset_summary(name, subject_table):
    print(f"\n{name}")
    print(subject_table.groupby("group").agg(
        n_subjects=("participant_id", "count"),
        n_epochs=("n_epochs", "sum"),
    ))


def main():
    alz_subjects, alz_table = load_alz_c_vs_a_bandpower_dataset()
    pearl_subjects, pearl_table = load_pearl_bandpower_dataset()

    print_dataset_summary("ALZ_FTD binary training dataset: C vs A", alz_table)
    print_dataset_summary("PEARL transfer dataset: N vs A+P- vs A+P+", pearl_table)

    folds = make_subject_stratified_folds(alz_table)
    fold_summary = summarize_folds(alz_table, folds)
    fold_assignments = make_fold_assignment_table(alz_table, folds)

    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    fold_summary.to_csv(FOLD_DIR / "alz_c_vs_a_10fold_summary.csv", index=False)
    fold_assignments.to_csv(FOLD_DIR / "alz_c_vs_a_10fold_assignments.csv", index=False)

    print("\nALZ_FTD 10-fold subject-stratified split summary")
    print(fold_summary.to_string(index=False))
    print(f"\nSaved {FOLD_DIR / 'alz_c_vs_a_10fold_summary.csv'}")
    print(f"Saved {FOLD_DIR / 'alz_c_vs_a_10fold_assignments.csv'}")

    x_train, y_train, train_subject_ids = stack_subjects(alz_subjects, folds[0]["train_subject_ids"])
    x_val, y_val, val_subject_ids = stack_subjects(alz_subjects, folds[0]["val_subject_ids"])
    x_test, y_test, test_subject_ids = stack_subjects(alz_subjects, folds[0]["test_subject_ids"])
    x_pearl, y_pearl, pearl_subject_ids = stack_subjects(
        pearl_subjects,
        pearl_table["participant_id"].to_numpy(),
    )
    encoder, x_train_encoded, x_val_encoded, x_test_encoded, x_pearl_encoded = encode_fold_for_difflogic(
        x_train,
        x_val,
        x_test,
        x_pearl,
    )

    print("\nExample fold 1 train arrays")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("train_subject_ids:", train_subject_ids.shape)
    print("\nExample fold 1 thermometer encoded arrays")
    print("n_bins:", encoder["n_bins"])
    print("x_train_encoded:", x_train_encoded.shape)
    print("x_val_encoded:", x_val_encoded.shape)
    print("x_test_encoded:", x_test_encoded.shape)
    print("x_pearl_encoded:", x_pearl_encoded.shape)


if __name__ == "__main__":
    main()
