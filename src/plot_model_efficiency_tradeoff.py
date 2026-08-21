from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train_helpers import ROOT_DIR


MODELS = [
    {
        "name": "Diff-Logic",
        "balanced_accuracy": 80.9,
        "latency_ms": 0.22,
        "size_kb": 132.0,
        "color": "#5BAFC8",
        "label_x": 0.25,
        "label_y": 80.86,
        "ha": "left",
    },
    {
        "name": "MLP",
        "balanced_accuracy": 79.6,
        "latency_ms": 0.34,
        "size_kb": 977.4,
        "color": "#7067B8",
        "label_x": 0.43,
        "label_y": 79.58,
        "ha": "left",
    },
    {
        "name": "1D-Conv",
        "balanced_accuracy": 80.1,
        "latency_ms": 0.38,
        "size_kb": 978.4,
        "color": "#9B63C8",
        "label_x": 0.465,
        "label_y": 80.14,
        "ha": "left",
    },
    {
        "name": "Transformer",
        "balanced_accuracy": 78.4,
        "latency_ms": 3.54,
        "size_kb": 978.3,
        "color": "#7298D2",
        "label_x": 2.92,
        "label_y": 78.40,
        "ha": "right",
    },
]


def bubble_area(size_kb):
    return 290.0 * (size_kb / 132.0)


def plot_efficiency(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.dpi": 180,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for model in MODELS:
        ax.scatter(
            model["latency_ms"],
            model["balanced_accuracy"],
            s=bubble_area(model["size_kb"]),
            color=model["color"],
            alpha=0.9,
            edgecolor="black",
            linewidth=1.3,
            zorder=3,
        )
        ax.text(
            model["label_x"],
            model["label_y"],
            model["name"],
            ha=model.get("ha", "left"),
            va="center",
            fontsize=17,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.18, 4.2)
    ax.set_ylim(78.1, 81.3)
    ax.set_xticks([0.2, 0.3, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2", "4"])
    ax.set_yticks([78.5, 79.0, 79.5, 80.0, 80.5, 81.0])

    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Clinical balanced accuracy (%)")
    ax.grid(True, which="major", linestyle=(0, (1.5, 3.0)), linewidth=1.2, color="#b7b7b7")
    ax.grid(False, which="minor")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(width=1.2, length=6)

    fig.tight_layout()
    png_path = output_dir / "model_efficiency_tradeoff_45hz_benchmark_250k.png"
    pdf_path = output_dir / "model_efficiency_tradeoff_45hz_benchmark_250k.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    plot_efficiency(ROOT_DIR / "figures")
