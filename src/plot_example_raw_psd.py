from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import pandas as pd

from preprocessing import COMMON_CHANNELS, HIGH_FREQ, LOW_FREQ, preprocess_raw


ROOT_DIR = Path(__file__).resolve().parents[1]
ALZ_FTD_DIR = ROOT_DIR / "datasets" / "ALZ_FTD"
OUTPUT_DIR = ROOT_DIR / "outputs" / "psd_examples"
WELCH_SECONDS = 2.0
WELCH_OVERLAP_SECONDS = 1.0
FREQ_BANDS = [
    ("Delta", 1.0, 4.0, "#e8dcc7"),
    ("Theta", 4.0, 8.0, "#dbe5cc"),
    ("Alpha", 8.0, 13.0, "#d9e3de"),
    ("Beta", 13.0, 30.0, "#e7d6c5"),
    ("Gamma", 30.0, 45.0, "#e5dbe8"),
]


def read_raw_eeg(raw_path):
    suffix = raw_path.suffix.lower()

    if suffix == ".set":
        return mne.io.read_raw_eeglab(raw_path, preload=True, verbose="ERROR")
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(raw_path, preload=True, verbose="ERROR")
    if suffix in {".edf", ".bdf"}:
        return mne.io.read_raw_edf(raw_path, preload=True, verbose="ERROR")
    if suffix == ".fif":
        return mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")

    raise ValueError(f"Unsupported raw EEG file type: {raw_path.suffix}")


def find_default_raw_path():
    participants = pd.read_csv(ALZ_FTD_DIR / "participants.tsv", sep="\t")
    ad_participants = participants[participants["Group"].eq("A")]

    for participant_id in sorted(ad_participants["participant_id"]):
        raw_path = ALZ_FTD_DIR / participant_id / "eeg" / f"{participant_id}_task-eyesclosed_eeg.set"
        if raw_path.exists():
            return raw_path

    raw_paths = sorted((ROOT_DIR / "datasets" / "PEARL").glob("sub-*/eeg/*_eeg.vhdr"))
    if raw_paths:
        return raw_paths[0]

    raise FileNotFoundError("No ALZ_FTD .set or PEARL .vhdr raw EEG file was found.")


def alz_ftd_group_label(participant_id):
    participants_path = ALZ_FTD_DIR / "participants.tsv"
    if not participants_path.exists():
        return None

    participants = pd.read_csv(participants_path, sep="\t")
    matching_rows = participants[participants["participant_id"].eq(participant_id)]
    if matching_rows.empty:
        return None

    group = matching_rows["Group"].iloc[0]
    if group == "A":
        return "AD"
    if group == "C":
        return "CN"
    if group == "F":
        return "FTD"
    return group


def make_psd_plot(raw_path, output_dir):
    raw = read_raw_eeg(raw_path)
    raw = preprocess_raw(raw)
    raw.reorder_channels(COMMON_CHANNELS)

    sfreq = raw.info["sfreq"]
    n_per_seg = int(WELCH_SECONDS * sfreq)
    n_overlap = int(WELCH_OVERLAP_SECONDS * sfreq)

    psd = raw.compute_psd(
        method="welch",
        fmin=LOW_FREQ,
        fmax=HIGH_FREQ,
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        picks="eeg",
        verbose="ERROR",
    )

    fig = psd.plot(
        average=False,
        spatial_colors=True,
        dB=True,
        show=False,
    )
    fig.set_size_inches(10.0, 4.5)
    axis = fig.axes[0]
    participant_id = raw_path.parent.parent.name
    group_label = alz_ftd_group_label(participant_id)
    title_label = f"{participant_id} {group_label}" if group_label else participant_id
    # axis.set_title(
    #     f"Example raw EEG PSD, {title_label}, 15 common channels",
    #     pad=10,
    # )
    axis.set_xlabel("Frequency (Hz)")
    axis.set_xlim(LOW_FREQ, HIGH_FREQ)
    ymin, ymax = axis.get_ylim()
    label_y = ymax - 0.06 * (ymax - ymin)
    for band_name, fmin, fmax, color in FREQ_BANDS:
        axis.axvspan(fmin, fmax, color=color, alpha=0.45, zorder=0)
        axis.text(
            (fmin + fmax) / 2.0,
            label_y,
            band_name,
            ha="center",
            va="top",
            fontsize=12,
            color="0.25",
            zorder=3,
        )
    axis.set_ylim(ymin, ymax)
    axis.grid(True, color="0.86", linewidth=0.8, zorder=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename_group = f"_{group_label}" if group_label else ""
    png_path = output_dir / f"{participant_id}{filename_group}_raw_psd_spatial_colors.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return png_path


def main():
    parser = ArgumentParser(description="Plot an example raw EEG PSD with MNE spatial colors.")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Path to a raw EEG file. Defaults to one available ALZ_FTD recording.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the PSD figure should be saved.",
    )
    args = parser.parse_args()

    raw_path = args.raw_path if args.raw_path is not None else find_default_raw_path()
    png_path = make_psd_plot(raw_path.resolve(), args.output_dir.resolve())
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
