from pathlib import Path

import mne
import numpy as np
import pandas as pd


ALZ_FTD_DIR = Path("datasets/ALZ_FTD")
PEARL_DIR = Path("datasets/PEARL")
DS007427_DIR = Path("datasets/ds007427")
OUTPUT_DIR = Path("datasets/processed")

COMMON_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "C3",
    "Cz",
    "C4",
    "P3",
    "Pz",
    "P4",
    "O1",
    "O2",
]

GROUP_LABELS = {
    "A": 0,
    "F": 1,
    "C": 2,
}

PEARL_GROUP_LABELS = {
    "N": 0,
    "A+P-": 1,
    "A+P+": 2,
}

DS007427_GROUP_LABELS = {
    "CTR": 0,
    "G1": 1,
    "G2": 0,
}

SFREQ = 250
LOW_FREQ = 1.0
HIGH_FREQ = 45.0
LINE_FREQ = 50.0
EPOCH_SECONDS = 4.0
EPOCH_REJECT_VOLTS = 150e-6


def preprocess_raw(raw):
    raw = raw.copy()
    common_channel_names = {name.lower(): name for name in COMMON_CHANNELS}
    raw.rename_channels(
        {
            name: common_channel_names[name.lower()]
            for name in raw.ch_names
            if name.lower() in common_channel_names
        }
    )
    raw.pick(COMMON_CHANNELS)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="raise")
    raw.resample(SFREQ, verbose="ERROR")
    raw.notch_filter(LINE_FREQ, verbose="ERROR")
    raw.filter(LOW_FREQ, HIGH_FREQ, verbose="ERROR")
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    return raw


def make_clean_epochs(raw):
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=EPOCH_SECONDS,
        overlap=0.0,
        preload=True,
        verbose="ERROR",
    )
    data = epochs.get_data()
    epoch_peak_to_peak = np.ptp(data, axis=2).max(axis=1)
    epoch_keep_mask = epoch_peak_to_peak <= EPOCH_REJECT_VOLTS
    clean_data = data[epoch_keep_mask].astype(np.float32)
    return clean_data, epoch_keep_mask, epoch_peak_to_peak


def save_alz_ftd_subject(row, output_dir):
    participant_id = row["participant_id"]
    eeg_path = ALZ_FTD_DIR / participant_id / "eeg" / f"{participant_id}_task-eyesclosed_eeg.set"

    print(f"Preprocessing ALZ_FTD {participant_id}")
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose="ERROR")
    raw = preprocess_raw(raw)
    data, epoch_keep_mask, epoch_peak_to_peak = make_clean_epochs(raw)

    output_path = output_dir / f"{participant_id}_task-eyesclosed_preprocessed.npz"
    np.savez_compressed(
        output_path,
        x=data,
        participant_id=participant_id,
        gender=row["Gender"],
        age=np.int64(row["Age"]),
        group=row["Group"],
        label=np.int64(GROUP_LABELS[row["Group"]]),
        mmse=np.int64(row["MMSE"]),
        channel_names=np.array(COMMON_CHANNELS),
        sfreq=np.float64(SFREQ),
        epoch_seconds=np.float64(EPOCH_SECONDS),
        low_freq=np.float64(LOW_FREQ),
        high_freq=np.float64(HIGH_FREQ),
        line_freq=np.float64(LINE_FREQ),
        average_reference_channels=np.array(COMMON_CHANNELS),
        epoch_keep_mask=epoch_keep_mask,
        epoch_peak_to_peak=epoch_peak_to_peak,
    )


def read_pearl_event_onsets(events_path):
    events = pd.read_csv(events_path, sep="\t")
    event_codes = events["event_type"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    return events, event_codes


def make_pearl_crop_summary():
    rows = []

    for events_path in sorted(PEARL_DIR.glob("sub-*/eeg/*_task-rest_events.tsv")):
        participant_id = events_path.parent.parent.name
        eeg_path = events_path.with_name(events_path.name.replace("_events.tsv", "_eeg.vhdr"))
        raw = mne.io.read_raw_brainvision(eeg_path, preload=False, verbose="ERROR")
        events, event_codes = read_pearl_event_onsets(events_path)
        s4_start = events.loc[event_codes == 4, "onset"]
        s11_end = events.loc[event_codes == 11, "onset"]
        eyes_closed_start = np.nan
        eyes_closed_end = np.nan

        if len(s4_start) > 0:
            eyes_closed_start = s4_start.iloc[0]
        if len(s11_end) > 0:
            eyes_closed_end = s11_end.iloc[0]

        rows.append(
            {
                "participant_id": participant_id,
                "recording_seconds": raw.n_times / raw.info["sfreq"],
                "s4_start": eyes_closed_start,
                "s11_end": eyes_closed_end,
                "eyes_closed_seconds": eyes_closed_end - eyes_closed_start,
                "missing_s4": pd.isna(eyes_closed_start),
                "missing_s11": pd.isna(eyes_closed_end),
            }
        )

    return pd.DataFrame(rows)


def crop_pearl_eyes_closed(raw, events_path, typical_eyes_closed_seconds):
    events, event_codes = read_pearl_event_onsets(events_path)
    s4_start = events.loc[event_codes == 4, "onset"]
    s11_end = events.loc[event_codes == 11, "onset"]
    recording_end = raw.times[-1]

    eyes_closed_start = max(0.0, recording_end - typical_eyes_closed_seconds)
    eyes_closed_end = eyes_closed_start + typical_eyes_closed_seconds

    if len(s4_start) > 0:
        eyes_closed_start = s4_start.iloc[0]
        eyes_closed_end = eyes_closed_start + typical_eyes_closed_seconds
    if len(s11_end) > 0:
        eyes_closed_end = s11_end.iloc[0]
        if len(s4_start) == 0:
            eyes_closed_start = max(0.0, eyes_closed_end - typical_eyes_closed_seconds)

    eyes_closed_end = min(eyes_closed_end, recording_end)
    raw = raw.crop(tmin=eyes_closed_start, tmax=eyes_closed_end)
    return raw, np.float64(eyes_closed_start), np.float64(eyes_closed_end)


def save_pearl_subject(row, output_dir, typical_eyes_closed_seconds):
    participant_id = row["participant_id"]
    eeg_dir = PEARL_DIR / participant_id / "eeg"
    eeg_path = eeg_dir / f"{participant_id}_task-rest_eeg.vhdr"
    events_path = eeg_dir / f"{participant_id}_task-rest_events.tsv"

    print(f"Preprocessing PEARL {participant_id}")
    raw = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose="ERROR")
    raw, eyes_closed_start, eyes_closed_end = crop_pearl_eyes_closed(
        raw,
        events_path,
        typical_eyes_closed_seconds,
    )
    raw = preprocess_raw(raw)
    data, epoch_keep_mask, epoch_peak_to_peak = make_clean_epochs(raw)

    output_path = output_dir / f"{participant_id}_task-rest_eyesclosed_preprocessed.npz"
    np.savez_compressed(
        output_path,
        x=data,
        participant_id=participant_id,
        age=np.int64(row["age"]),
        sex=np.int64(row["sex"]),
        group=row["risk_group"],
        label=np.int64(PEARL_GROUP_LABELS[row["risk_group"]]),
        eyes_closed_start=eyes_closed_start,
        eyes_closed_end=eyes_closed_end,
        channel_names=np.array(COMMON_CHANNELS),
        sfreq=np.float64(SFREQ),
        epoch_seconds=np.float64(EPOCH_SECONDS),
        low_freq=np.float64(LOW_FREQ),
        high_freq=np.float64(HIGH_FREQ),
        line_freq=np.float64(LINE_FREQ),
        average_reference_channels=np.array(COMMON_CHANNELS),
        epoch_keep_mask=epoch_keep_mask,
        epoch_peak_to_peak=epoch_peak_to_peak,
    )


def preprocess_alz_ftd():
    participants = pd.read_csv(ALZ_FTD_DIR / "participants.tsv", sep="\t")
    subject_ids = sorted(
        path.parent.parent.name
        for path in ALZ_FTD_DIR.glob("sub-*/eeg/*_task-eyesclosed_eeg.set")
    )
    participants = participants.set_index("participant_id").loc[subject_ids].reset_index()
    output_dir = OUTPUT_DIR / "ALZ_FTD"
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in participants.iterrows():
        save_alz_ftd_subject(row, output_dir)


def preprocess_pearl():
    participants = pd.read_csv(PEARL_DIR / "participants.tsv", sep="\t")
    participants = participants[participants["risk_group"].isin(PEARL_GROUP_LABELS)]
    participants = participants.set_index("participant_id")
    subject_ids = sorted(
        path.parent.parent.name
        for path in PEARL_DIR.glob("sub-*/eeg/*_task-rest_eeg.vhdr")
    )
    subject_ids = sorted(set(subject_ids) & set(participants.index))
    participants = participants.loc[subject_ids].reset_index()
    output_dir = OUTPUT_DIR / "PEARL"
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_summary = make_pearl_crop_summary()
    crop_summary.to_csv(output_dir / "pearl_crop_summary.csv", index=False)
    typical_eyes_closed_seconds = crop_summary["eyes_closed_seconds"].median()

    for _, row in participants.iterrows():
        save_pearl_subject(row, output_dir, typical_eyes_closed_seconds)


def ds007427_group(participant_id):
    for group in DS007427_GROUP_LABELS:
        if participant_id.startswith(f"sub-{group}"):
            return group
    raise ValueError(f"Unknown DS007427 participant group: {participant_id}")


def save_ds007427_subject(row, eeg_path, output_dir):
    participant_id = row["participant_id"]
    group = ds007427_group(participant_id)

    print(f"Preprocessing DS007427 {participant_id}")
    raw = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose="ERROR")
    raw = preprocess_raw(raw)
    data, epoch_keep_mask, epoch_peak_to_peak = make_clean_epochs(raw)

    output_path = output_dir / f"{participant_id}_task-CE_preprocessed.npz"
    np.savez_compressed(
        output_path,
        x=data,
        participant_id=participant_id,
        age=np.float64(row["age"]),
        sex=np.str_(row["sex"]),
        group=group,
        label=np.int64(DS007427_GROUP_LABELS[group]),
        channel_names=np.array(COMMON_CHANNELS),
        sfreq=np.float64(SFREQ),
        epoch_seconds=np.float64(EPOCH_SECONDS),
        low_freq=np.float64(LOW_FREQ),
        high_freq=np.float64(HIGH_FREQ),
        line_freq=np.float64(LINE_FREQ),
        average_reference_channels=np.array(COMMON_CHANNELS),
        epoch_keep_mask=epoch_keep_mask,
        epoch_peak_to_peak=epoch_peak_to_peak,
    )


def preprocess_ds007427():
    participants = pd.read_csv(
        DS007427_DIR / "participants.tsv",
        sep="\t",
        keep_default_na=False,
    ).set_index("participant_id")
    participants["age"] = pd.to_numeric(participants["age"], errors="coerce")
    eeg_paths = sorted(
        DS007427_DIR.glob("sub-*/ses-V0/eeg/*_task-CE_eeg.vhdr")
    )
    output_dir = OUTPUT_DIR / "DS007427"
    output_dir.mkdir(parents=True, exist_ok=True)

    for eeg_path in eeg_paths:
        participant_id = eeg_path.parents[2].name
        row = participants.loc[participant_id].copy()
        row["participant_id"] = participant_id
        save_ds007427_subject(row, eeg_path, output_dir)


def main():
    preprocess_alz_ftd()
    preprocess_pearl()
    preprocess_ds007427()


if __name__ == "__main__":
    main()
