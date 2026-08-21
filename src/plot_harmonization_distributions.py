import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT_DIR / "datasets/statistics_psd_45hz/subject_log_relative_bandpower.csv"
OUTPUT_DIR = ROOT_DIR / "outputs/harmonization_45hz"

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
# Control first, then intermediate, then group of interest, matching the row
# order of the topomap figure so the two read together.
COHORTS = {
    "Clinical": {"dataset": "ALZ_FTD", "groups": ["CN", "FTD", "AD"]},
    "PEARL": {"dataset": "PEARL", "groups": ["N", "A+P−", "A+P+"]},
}
GROUP_NAMES = {"A": "AD", "C": "CN", "F": "FTD", "A+P-": "A+P−"}

# Colour encodes cohort, the categorical split this figure exists to show.
# Group identity is carried by x position and axis label, so identity is never
# colour-alone. Validated with the data-viz palette validator (light surface,
# all pairs): worst-pair CVD dE 24.7 protan, normal-vision dE 33.6, both >= 3:1
# contrast on the chart surface.
COHORT_COLORS = {"Clinical": "#2a78d6", "PEARL": "#eb6834"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"


def load_subject_bandpower(input_csv):
    data = pd.read_csv(input_csv)
    data["group"] = data["group"].replace(GROUP_NAMES)
    channel_counts = data.groupby(["dataset", "participant_id", "band"])["channel"].nunique()
    if not channel_counts.eq(15).all():
        raise ValueError("Every subject-band must contain the same 15 common channels")

    subject_bandpower = (
        data.groupby(["dataset", "participant_id", "group", "band"], as_index=False)
        .agg(log_relative_bandpower=("log_relative_bandpower", "mean"))
    )
    observed_bands = set(subject_bandpower["band"])
    if observed_bands != set(BANDS):
        raise ValueError(f"Expected bands {BANDS}, found {sorted(observed_bands)}")
    return subject_bandpower


def summarize(subject_bandpower):
    return (
        subject_bandpower.groupby(["dataset", "group", "band"], as_index=False)
        .agg(
            n_subjects=("participant_id", "nunique"),
            mean=("log_relative_bandpower", "mean"),
            std=("log_relative_bandpower", "std"),
            median=("log_relative_bandpower", "median"),
            q25=("log_relative_bandpower", lambda values: values.quantile(0.25)),
            q75=("log_relative_bandpower", lambda values: values.quantile(0.75)),
        )
    )


def plot_distributions(subject_bandpower, output_path):
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.edgecolor": "#b8b6ae",
            "axes.labelcolor": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
        }
    )
    # No shared x: the two cohorts have different group labels on the same columns.
    figure, axes = plt.subplots(2, 5, figsize=(13.5, 5.8), sharey=True)
    y_min = subject_bandpower["log_relative_bandpower"].min() - 0.10
    y_max = subject_bandpower["log_relative_bandpower"].max() + 0.10
    rng = np.random.default_rng(20260816)

    for row, (cohort_name, cohort) in enumerate(COHORTS.items()):
        cohort_data = subject_bandpower[subject_bandpower["dataset"].eq(cohort["dataset"])]
        missing_groups = set(cohort["groups"]) - set(cohort_data["group"])
        if missing_groups:
            raise ValueError(f"Missing {cohort_name} groups: {sorted(missing_groups)}")

        color = COHORT_COLORS[cohort_name]
        counts = cohort_data.groupby("group")["participant_id"].nunique()

        for column, band in enumerate(BANDS):
            axis = axes[row, column]
            axis.set_axisbelow(True)
            axis.yaxis.grid(True, color="#e3e1da", linewidth=0.6)
            axis.xaxis.grid(False)

            for position, group in enumerate(cohort["groups"]):
                values = cohort_data.loc[
                    cohort_data["group"].eq(group) & cohort_data["band"].eq(band),
                    "log_relative_bandpower",
                ].to_numpy()

                # Recessive silhouette: distribution shape without competing edges.
                body = axis.violinplot(
                    values,
                    positions=[position],
                    widths=0.78,
                    showextrema=False,
                    showmedians=False,
                )
                for part in body["bodies"]:
                    part.set_facecolor(color)
                    part.set_alpha(0.16)
                    part.set_edgecolor("none")

                # Every participant, since n is 20-36 and the cloud is the evidence.
                jitter = rng.uniform(-0.11, 0.11, size=values.size)
                axis.scatter(
                    position + jitter,
                    values,
                    s=5.5,
                    color=color,
                    alpha=0.55,
                    linewidths=0.25,
                    edgecolors="white",
                    zorder=3,
                )

                # Median and interquartile range in ink, so the summary reads on top.
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                axis.vlines(position, q25, q75, color=INK_PRIMARY, linewidth=1.4, zorder=4)
                axis.hlines(
                    median,
                    position - 0.20,
                    position + 0.20,
                    color=INK_PRIMARY,
                    linewidth=2.0,
                    zorder=5,
                )

            axis.set_xlim(-0.62, len(cohort["groups"]) - 0.38)
            axis.set_ylim(y_min, y_max)
            axis.set_xticks(range(len(cohort["groups"])))
            axis.set_xticklabels(
                [f"{group}\n({counts[group]})" for group in cohort["groups"]],
                fontsize=10,
            )
            axis.set_xlabel("")
            axis.tick_params(length=0)

            if row == 0:
                axis.set_title(band.capitalize(), fontsize=12.5, color=INK_PRIMARY, pad=7)
            if column == 0:
                axis.set_ylabel(
                    f"{cohort_name}\nlog$_{{10}}$ relative bandpower",
                    fontsize=11.5,
                )
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_bandpower = load_subject_bandpower(args.input_csv)
    summary = summarize(subject_bandpower)
    subject_path = args.output_dir / "subject_mean_log_relative_bandpower.csv"
    summary_path = args.output_dir / "group_bandpower_summary.csv"
    figure_path = args.output_dir / "psd_group_distributions_45hz.png"
    subject_bandpower.to_csv(subject_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_distributions(subject_bandpower, figure_path)
    print(f"Saved {figure_path}")
    print(f"Saved {subject_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
