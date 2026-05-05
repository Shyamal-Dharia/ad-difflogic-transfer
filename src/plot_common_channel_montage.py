from pathlib import Path

import matplotlib.pyplot as plt
import mne


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "publication_figures_v2"

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

CHANNEL_COLORS = {
    "Fp1": "#9fc4e6",
    "Fp2": "#9fc4e6",
    "F7": "#9fc4e6",
    "F3": "#9fc4e6",
    "Fz": "#9fc4e6",
    "F4": "#9fc4e6",
    "F8": "#9fc4e6",
    "C3": "#a9dfad",
    "Cz": "#a9dfad",
    "C4": "#a9dfad",
    "P3": "#eceb8d",
    "Pz": "#eceb8d",
    "P4": "#eceb8d",
    "O1": "#b8b3e6",
    "O2": "#b8b3e6",
}


def make_common_channel_info():
    info = mne.create_info(COMMON_CHANNELS, sfreq=250, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="raise")
    return info


def plot_common_channels(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    info = make_common_channel_info()

    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    mne.viz.plot_sensors(
        info,
        kind="topomap",
        show_names=True,
        axes=ax,
        show=False,
        pointsize=0,
        linewidth=1,
    )

    sensor_collection = ax.collections[0]
    positions = sensor_collection.get_offsets()
    sensor_collection.remove()

    for text in list(ax.texts):
        text.remove()

    for channel, (x, y) in zip(COMMON_CHANNELS, positions):
        ax.scatter(
            x,
            y,
            s=560,
            c=CHANNEL_COLORS[channel],
            edgecolors="black",
            linewidths=1.3,
            zorder=3,
        )
        ax.text(
            x,
            y,
            channel,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
            zorder=4,
        )

    png_path = output_dir / "common_15_channel_montage.png"
    pdf_path = output_dir / "common_15_channel_montage.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return png_path, pdf_path


def main():
    png_path, pdf_path = plot_common_channels(OUTPUT_DIR)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
