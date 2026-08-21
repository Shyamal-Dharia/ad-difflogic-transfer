"""Recording duration and artifact-rejection summary by dataset and group.

Addresses reviewer 1 minor comment 3 and reviewer 3's question on how
artifact-free epochs were identified.

Counts are read from the stored preprocessing masks, so no recomputation of the
pipeline is required.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_helpers import ROOT_DIR


PROCESSED_DIR = ROOT_DIR / "datasets" / "processed"
GROUP_LABELS = {
    "ALZ_FTD": {"A": "AD", "C": "CN", "F": "FTD"},
}
GROUP_ORDER = {
    "ALZ_FTD": ["AD", "CN", "FTD"],
    "PEARL": ["N", "A+P-", "A+P+"],
}


def subject_rows(dataset_name):
    rows = []
    for path in sorted((PROCESSED_DIR / dataset_name).glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        keep_mask = data["epoch_keep_mask"]
        epoch_seconds = float(data["epoch_seconds"])
        group = str(data["group"].item() if data["group"].ndim == 0 else data["group"])
        group = GROUP_LABELS.get(dataset_name, {}).get(group, group)
        rows.append(
            {
                "dataset": dataset_name,
                "participant_id": str(
                    data["participant_id"].item()
                    if data["participant_id"].ndim == 0
                    else data["participant_id"]
                ),
                "group": group,
                "candidate_epochs": int(keep_mask.size),
                "retained_epochs": int(keep_mask.sum()),
                "rejected_epochs": int(keep_mask.size - keep_mask.sum()),
                "recording_minutes": keep_mask.size * epoch_seconds / 60.0,
                "retained_minutes": int(keep_mask.sum()) * epoch_seconds / 60.0,
                "median_peak_to_peak_uv": float(np.median(data["epoch_peak_to_peak"])) * 1e6,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ALZ_FTD", "PEARL"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "revision_2026" / "epoch_rejection",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subjects = pd.DataFrame(
        [row for dataset_name in args.datasets for row in subject_rows(dataset_name)]
    )
    subjects["rejected_percent"] = 100.0 * subjects["rejected_epochs"] / subjects["candidate_epochs"]
    subjects.to_csv(args.output_dir / "epoch_rejection_by_subject.csv", index=False)

    group_summary = (
        subjects.groupby(["dataset", "group"])
        .agg(
            participants=("participant_id", "count"),
            candidate_epochs=("candidate_epochs", "sum"),
            retained_epochs=("retained_epochs", "sum"),
            rejected_epochs=("rejected_epochs", "sum"),
            mean_recording_minutes=("recording_minutes", "mean"),
            mean_retained_epochs=("retained_epochs", "mean"),
            min_retained_epochs=("retained_epochs", "min"),
            median_subject_rejected_percent=("rejected_percent", "median"),
            max_subject_rejected_percent=("rejected_percent", "max"),
        )
        .reset_index()
    )
    group_summary["rejected_percent"] = (
        100.0 * group_summary["rejected_epochs"] / group_summary["candidate_epochs"]
    )
    group_summary["order"] = group_summary.apply(
        lambda row: GROUP_ORDER.get(row["dataset"], []).index(row["group"])
        if row["group"] in GROUP_ORDER.get(row["dataset"], [])
        else 99,
        axis=1,
    )
    group_summary = group_summary.sort_values(["dataset", "order"]).drop(columns="order")
    group_summary.to_csv(args.output_dir / "epoch_rejection_by_group.csv", index=False)

    print(
        group_summary[
            [
                "dataset",
                "group",
                "participants",
                "candidate_epochs",
                "retained_epochs",
                "rejected_percent",
                "mean_recording_minutes",
                "mean_retained_epochs",
                "min_retained_epochs",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
