import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from interpret_difflogic_gradients import (
    integrated_gradient_relevance,
    load_encoder,
    load_model,
    resolve_device,
)
from train_helpers import (
    BANDS,
    ROOT_DIR,
    apply_thermometer_encoder,
    load_alz_c_vs_a_dataset,
    make_subject_stratified_folds,
    stack_subjects,
)


RUN_NAME = "medium_interpretable"
MODEL_KIND = "difflogic"
MODEL_SIZE = "medium"
TARGET_PARAMETERS = 250_000
IG_STEPS = 16
BATCH_SIZE = 256
P_THRESHOLD = 0.10
VALUE_COLUMN = "mean_signed_relevance"

CHANNEL_ORDER = [
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
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_LABELS = {
    "delta": "Delta",
    "theta": "Theta",
    "alpha": "Alpha",
    "beta": "Beta",
    "gamma": "Gamma",
}
CONTRASTS = [
    ("clinical_A_minus_C", "Clinical\nAD - CN"),
    ("pearl_A+P+_minus_N", "Genetic-Risk\nA+P+ - N"),
    ("pearl_A+P+_minus_A+P-", "Genetic-Risk\nA+P+ - A+P-"),
]

COLUMN_TITLE_FONTSIZE = 13
ROW_LABEL_FONTSIZE = 14
CHANNEL_LABEL_FONTSIZE = 7.2
COLORBAR_TICK_FONTSIZE = 11

RUN_DIR = ROOT_DIR / "outputs/difflogic" / RUN_NAME
PEARL_INTERPRETATION_DIR = RUN_DIR / "interpretation/soft_integrated_gradients"
OUTPUT_DIR = ROOT_DIR / "outputs/model_relevance_statistics"
FIGURES_DIR = ROOT_DIR / "figures"
CLINICAL_RELEVANCE_DIR = OUTPUT_DIR / "clinical_integrated_gradients"
PEARL_TESTS_PATH = OUTPUT_DIR / "integrated_gradient_relevance_tests.csv"


def make_info():
    info = mne.create_info(CHANNEL_ORDER, sfreq=250, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="raise")
    return info


def topomap_positions(info):
    from mne.channels.layout import _find_topomap_coords

    return _find_topomap_coords(info, picks=np.arange(len(CHANNEL_ORDER)))


def compute_epoch_relevance(model, x_encoded, n_channels, n_bands, n_bins, device):
    signed_relevance = []
    absolute_relevance = []

    for start in range(0, len(x_encoded), BATCH_SIZE):
        end = start + BATCH_SIZE
        x_batch = torch.tensor(x_encoded[start:end], dtype=torch.float32, device=device)
        relevance = integrated_gradient_relevance(model, x_batch, IG_STEPS)
        relevance = relevance.detach().cpu().numpy()
        relevance = relevance.reshape(len(x_batch), n_channels, n_bands, n_bins)
        signed_relevance.append(relevance.sum(axis=-1))
        absolute_relevance.append(np.abs(relevance).sum(axis=-1))

    return np.concatenate(signed_relevance), np.concatenate(absolute_relevance)


def make_subject_relevance_table(
    signed_relevance,
    absolute_relevance,
    subject_ids,
    subject_table,
    channel_names,
    seed,
    fold,
):
    metadata = subject_table.set_index("participant_id")
    rows = []

    for participant_id in np.unique(subject_ids):
        subject_mask = subject_ids == participant_id
        signed_subject = signed_relevance[subject_mask].mean(axis=0)
        absolute_subject = absolute_relevance[subject_mask].mean(axis=0)

        for channel_index, channel in enumerate(channel_names):
            for band_index, band in enumerate(BANDS):
                rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "participant_id": participant_id,
                        "group": metadata.loc[participant_id, "group"],
                        "channel": channel,
                        "band": band,
                        "signed_relevance": signed_subject[channel_index, band_index],
                        "absolute_relevance": absolute_subject[channel_index, band_index],
                    }
                )

    return pd.DataFrame(rows)


def summarize_relevance(subject_relevance):
    seed_average = (
        subject_relevance
        .groupby(["participant_id", "group", "channel", "band"], as_index=False)
        .agg(
            signed_relevance=("signed_relevance", "mean"),
            absolute_relevance=("absolute_relevance", "mean"),
        )
    )

    group_summary = (
        seed_average
        .groupby(["group", "channel", "band"], as_index=False)
        .agg(
            mean_signed_relevance=("signed_relevance", "mean"),
            median_signed_relevance=("signed_relevance", "median"),
            mean_absolute_relevance=("absolute_relevance", "mean"),
            median_absolute_relevance=("absolute_relevance", "median"),
        )
    )

    return seed_average, group_summary


def benjamini_hochberg(p_values, alpha=0.05):
    p_values = np.asarray(p_values, dtype=float)
    n_tests = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(n_tests, dtype=float)
    running_min = 1.0

    for index in range(n_tests - 1, -1, -1):
        rank = index + 1
        running_min = min(running_min, sorted_p[index] * n_tests / rank)
        adjusted_sorted[index] = running_min

    adjusted = np.empty(n_tests, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    rejected = adjusted <= alpha
    return rejected, adjusted


def run_pairwise_tests(seed_average, group_a, group_b, comparison):
    rows = []

    for (channel, band), feature_data in seed_average.groupby(["channel", "band"]):
        values_a = feature_data[feature_data["group"].eq(group_a)]["signed_relevance"].to_numpy()
        values_b = feature_data[feature_data["group"].eq(group_b)]["signed_relevance"].to_numpy()
        statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
        rows.append(
            {
                "channel": channel,
                "band": band,
                "comparison": comparison,
                "group_a": group_a,
                "group_b": group_b,
                "mean_a": values_a.mean(),
                "mean_b": values_b.mean(),
                "mean_difference": values_a.mean() - values_b.mean(),
                "median_a": np.median(values_a),
                "median_b": np.median(values_b),
                "median_difference": np.median(values_a) - np.median(values_b),
                "statistic": statistic,
                "p_uncorrected": p_value,
            }
        )

    tests = pd.DataFrame(rows)
    rejected, p_fdr = benjamini_hochberg(tests["p_uncorrected"].to_numpy())
    tests["p_fdr"] = p_fdr
    tests["reject_fdr_0_05"] = rejected
    return tests.sort_values("p_uncorrected")


def compute_clinical_relevance():
    clinical_all_path = CLINICAL_RELEVANCE_DIR / "clinical_gradient_relevance_all_models.csv"
    clinical_seed_average_path = CLINICAL_RELEVANCE_DIR / "clinical_gradient_relevance_seed_average.csv"
    clinical_group_summary_path = CLINICAL_RELEVANCE_DIR / "clinical_gradient_relevance_group_summary.csv"
    clinical_tests_path = CLINICAL_RELEVANCE_DIR / "clinical_integrated_gradient_relevance_tests.csv"

    if clinical_group_summary_path.exists() and clinical_tests_path.exists():
        return (
            pd.read_csv(clinical_group_summary_path),
            pd.read_csv(clinical_tests_path),
        )

    CLINICAL_RELEVANCE_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cuda")
    alz_subjects, alz_table = load_alz_c_vs_a_dataset(feature_kind="psd", exclude_channels=[])
    channel_names = alz_subjects[0]["channel_names"]
    n_channels = alz_subjects[0]["n_channels"]
    n_bands = alz_subjects[0]["n_bands"]
    tables = []

    for seed_dir in sorted(RUN_DIR.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        folds = make_subject_stratified_folds(alz_table, random_state=seed)

        for fold in folds:
            fold_index = fold["fold"]
            fold_dir = seed_dir / f"fold_{fold_index:02d}"
            encoder = load_encoder(fold_dir / "thermometer_encoder.npz")
            x_test, _, test_subject_ids = stack_subjects(alz_subjects, fold["test_subject_ids"])
            x_test_encoded = apply_thermometer_encoder(x_test, encoder)
            model = load_model(
                fold_dir / "best_model.pt",
                MODEL_SIZE,
                input_dim=x_test_encoded.shape[1],
                device=device,
                logic_mode="soft",
                model_kind=MODEL_KIND,
                target_parameters=TARGET_PARAMETERS,
            )
            signed_relevance, absolute_relevance = compute_epoch_relevance(
                model,
                x_test_encoded,
                n_channels,
                n_bands,
                encoder["n_bins"],
                device,
            )
            tables.append(
                make_subject_relevance_table(
                    signed_relevance,
                    absolute_relevance,
                    test_subject_ids,
                    alz_table,
                    channel_names,
                    seed,
                    fold_index,
                )
            )
            print(f"computed clinical relevance seed {seed} fold {fold_index}")

    subject_relevance = pd.concat(tables, ignore_index=True)
    seed_average, group_summary = summarize_relevance(subject_relevance)
    clinical_tests = run_pairwise_tests(
        seed_average,
        group_a="A",
        group_b="C",
        comparison="clinical_A_vs_C",
    )

    subject_relevance.to_csv(clinical_all_path, index=False)
    seed_average.to_csv(clinical_seed_average_path, index=False)
    group_summary.to_csv(clinical_group_summary_path, index=False)
    clinical_tests.to_csv(clinical_tests_path, index=False)

    return group_summary, clinical_tests


def make_clinical_contrast(clinical_group_summary):
    pivot = clinical_group_summary.pivot(
        index=["channel", "band"],
        columns="group",
        values=VALUE_COLUMN,
    ).reset_index()
    pivot["contrast"] = "clinical_A_minus_C"
    pivot["value"] = pivot["A"] - pivot["C"]
    return pivot[["channel", "band", "contrast", "value"]]


def make_pearl_contrasts():
    group_summary = pd.read_csv(PEARL_INTERPRETATION_DIR / "pearl_gradient_relevance_group_summary.csv")
    pivot = group_summary.pivot(
        index=["channel", "band"],
        columns="group",
        values=VALUE_COLUMN,
    ).reset_index()

    rows = []
    for _, row in pivot.iterrows():
        rows.extend(
            [
                {
                    "channel": row["channel"],
                    "band": row["band"],
                    "contrast": "pearl_A+P+_minus_N",
                    "value": row["A+P+"] - row["N"],
                },
                {
                    "channel": row["channel"],
                    "band": row["band"],
                    "contrast": "pearl_A+P+_minus_A+P-",
                    "value": row["A+P+"] - row["A+P-"],
                },
            ]
        )

    return pd.DataFrame(rows)


def make_significant_channels(clinical_tests):
    pearl_tests = pd.read_csv(PEARL_TESTS_PATH)
    mapping = {
        "A+P+_vs_N": "pearl_A+P+_minus_N",
        "A+P+_vs_A+P-": "pearl_A+P+_minus_A+P-",
    }
    pearl_tests = pearl_tests[pearl_tests["comparison"].isin(mapping)].copy()
    pearl_tests["contrast"] = pearl_tests["comparison"].map(mapping)

    clinical_tests = clinical_tests.copy()
    clinical_tests["contrast"] = "clinical_A_minus_C"

    tests = pd.concat(
        [
            clinical_tests[["channel", "band", "contrast", "p_uncorrected"]],
            pearl_tests[["channel", "band", "contrast", "p_uncorrected"]],
        ],
        ignore_index=True,
    )
    tests = tests[tests["p_uncorrected"].lt(P_THRESHOLD)].copy()
    return tests.sort_values(["contrast", "band", "p_uncorrected"])


def contrast_values(contrast_table, band, contrast):
    data = (
        contrast_table[contrast_table["band"].eq(band) & contrast_table["contrast"].eq(contrast)]
        .set_index("channel")
        .reindex(CHANNEL_ORDER)
    )
    if data["value"].isna().any():
        missing = data[data["value"].isna()].index.tolist()
        raise ValueError(f"Missing values for {band} {contrast}: {missing}")

    return data["value"].to_numpy()


def normalize_by_contrast(contrast_table):
    normalized = contrast_table.copy()
    normalized["raw_value"] = normalized["value"]
    max_abs_by_contrast = normalized.groupby("contrast")["raw_value"].transform(
        lambda values: values.abs().max()
    )
    normalized["contrast_max_abs"] = max_abs_by_contrast
    normalized["value"] = normalized["raw_value"] / max_abs_by_contrast
    return normalized


def significance_mask(significant_channels, band, contrast):
    channels = set(
        significant_channels[
            significant_channels["band"].eq(band)
            & significant_channels["contrast"].eq(contrast)
        ]["channel"]
    )
    return np.array([channel in channels for channel in CHANNEL_ORDER], dtype=bool)


def draw_significance_labels(axis, positions, mask):
    for channel, (x_pos, y_pos), is_significant in zip(CHANNEL_ORDER, positions, mask):
        if not is_significant:
            continue

        axis.scatter(
            x_pos,
            y_pos,
            s=240,
            marker="o",
            facecolor="#FFD84D",
            edgecolor="black",
            linewidth=1.2,
            zorder=10,
        )
        axis.text(
            x_pos,
            y_pos,
            channel,
            ha="center",
            va="center",
            fontsize=CHANNEL_LABEL_FONTSIZE,
            fontweight="bold",
            color="black",
            zorder=11,
        )


def plot_one_topomap(axis, values, info, positions, mask, max_abs, image_interpolation):
    image, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=axis,
        show=False,
        cmap="RdBu_r",
        vlim=(-max_abs, max_abs),
        contours=0,
        sensors=True,
        image_interp=image_interpolation,
    )
    draw_significance_labels(axis, positions, mask)
    return image


def save_figure(fig, output_prefix):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{output_prefix}.png"
    pdf_path = OUTPUT_DIR / f"{output_prefix}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{output_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def make_overlap_table(contrast_table, significant_channels):
    clinical_hits = significant_channels[
        significant_channels["contrast"].eq("clinical_A_minus_C")
    ][["channel", "band", "p_uncorrected"]].rename(
        columns={"p_uncorrected": "clinical_p_uncorrected"}
    )
    pearl_hits = significant_channels[
        significant_channels["contrast"].ne("clinical_A_minus_C")
    ][["channel", "band", "contrast", "p_uncorrected"]].rename(
        columns={"p_uncorrected": "pearl_p_uncorrected"}
    )
    overlap = clinical_hits.merge(pearl_hits, on=["channel", "band"], how="inner")

    values = contrast_table.rename(columns={"value": "contrast_value"})
    overlap = overlap.merge(
        values[values["contrast"].eq("clinical_A_minus_C")][
            ["channel", "band", "contrast_value"]
        ].rename(columns={"contrast_value": "clinical_AD_minus_CN"}),
        on=["channel", "band"],
        how="left",
    )
    overlap = overlap.merge(
        values[values["contrast"].ne("clinical_A_minus_C")][
            ["channel", "band", "contrast", "contrast_value"]
        ].rename(columns={"contrast_value": "pearl_contrast_value"}),
        on=["channel", "band", "contrast"],
        how="left",
    )
    overlap["same_direction"] = (
        np.sign(overlap["clinical_AD_minus_CN"])
        == np.sign(overlap["pearl_contrast_value"])
    )
    return overlap.sort_values(["contrast", "band", "pearl_p_uncorrected"])


def plot_combined_contrasts(
    contrast_table,
    significant_channels,
    image_interpolation,
    output_prefix,
    fixed_max_abs=None,
):
    info = make_info()
    positions = topomap_positions(info)
    max_abs = fixed_max_abs if fixed_max_abs is not None else contrast_table["value"].abs().max()
    fig, axes = plt.subplots(
        len(BAND_ORDER),
        len(CONTRASTS),
        figsize=(6.9, 9.2),
        constrained_layout=True,
    )

    image = None
    for row_index, band in enumerate(BAND_ORDER):
        for column_index, (contrast, title) in enumerate(CONTRASTS):
            axis = axes[row_index, column_index]
            values = contrast_values(contrast_table, band, contrast)
            mask = significance_mask(significant_channels, band, contrast)
            image = plot_one_topomap(
                axis,
                values,
                info,
                positions,
                mask,
                max_abs,
                image_interpolation,
            )

            if row_index == 0:
                axis.set_title(title, fontsize=COLUMN_TITLE_FONTSIZE, pad=10)
            if column_index == 0:
                axis.text(
                    -0.18,
                    0.5,
                    BAND_LABELS[band],
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=ROW_LABEL_FONTSIZE,
                    fontweight="bold",
                )

    cbar = fig.colorbar(image, ax=axes, shrink=0.62, pad=0.02)
    cbar.set_label("")
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    save_figure(fig, output_prefix)


def plot_band_contrasts(
    contrast_table,
    significant_channels,
    band,
    image_interpolation,
    output_prefix,
    fixed_max_abs=None,
):
    info = make_info()
    positions = topomap_positions(info)
    if fixed_max_abs is None:
        max_abs = contrast_table[contrast_table["band"].eq(band)]["value"].abs().max()
    else:
        max_abs = fixed_max_abs
    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(6.9, 2.45), constrained_layout=True)

    image = None
    for axis, (contrast, title) in zip(axes, CONTRASTS):
        values = contrast_values(contrast_table, band, contrast)
        mask = significance_mask(significant_channels, band, contrast)
        image = plot_one_topomap(
            axis,
            values,
            info,
            positions,
            mask,
            max_abs,
            image_interpolation,
        )
        axis.set_title(title, fontsize=COLUMN_TITLE_FONTSIZE, pad=10)

    cbar = fig.colorbar(image, ax=axes, shrink=0.80, pad=0.02)
    cbar.set_label("")
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    save_figure(fig, f"{output_prefix}_{band}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--model-kind", default=MODEL_KIND, choices=["difflogic", "difflogic_medium"])
    parser.add_argument("--model-size", default=MODEL_SIZE, choices=["small", "medium", "large"])
    parser.add_argument("--target-parameters", type=int, default=TARGET_PARAMETERS)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--pearl-interpretation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--pearl-tests-path", type=Path)
    return parser.parse_args()


def configure_paths(args):
    global RUN_NAME, MODEL_KIND, MODEL_SIZE, TARGET_PARAMETERS
    global RUN_DIR, PEARL_INTERPRETATION_DIR, OUTPUT_DIR, FIGURES_DIR
    global CLINICAL_RELEVANCE_DIR, PEARL_TESTS_PATH

    RUN_NAME = args.run_name
    MODEL_KIND = args.model_kind
    MODEL_SIZE = args.model_size
    TARGET_PARAMETERS = args.target_parameters
    RUN_DIR = args.run_dir if args.run_dir is not None else ROOT_DIR / "outputs/difflogic" / RUN_NAME
    PEARL_INTERPRETATION_DIR = (
        args.pearl_interpretation_dir
        if args.pearl_interpretation_dir is not None
        else RUN_DIR / "interpretation/soft_integrated_gradients"
    )
    OUTPUT_DIR = args.output_dir
    FIGURES_DIR = args.figures_dir
    CLINICAL_RELEVANCE_DIR = OUTPUT_DIR / "clinical_integrated_gradients"
    PEARL_TESTS_PATH = (
        args.pearl_tests_path
        if args.pearl_tests_path is not None
        else OUTPUT_DIR / "integrated_gradient_relevance_tests.csv"
    )


def main():
    configure_paths(parse_args())
    clinical_group_summary, clinical_tests = compute_clinical_relevance()
    contrast_table = pd.concat(
        [
            make_clinical_contrast(clinical_group_summary),
            make_pearl_contrasts(),
        ],
        ignore_index=True,
    )
    significant_channels = make_significant_channels(clinical_tests)

    contrast_table.to_csv(OUTPUT_DIR / "source_target_integrated_gradient_contrasts.csv", index=False)
    significant_channels.to_csv(
        OUTPUT_DIR / "source_target_integrated_gradient_contrast_significant_channels.csv",
        index=False,
    )
    normalized_contrast_table = normalize_by_contrast(contrast_table)
    normalized_contrast_table.to_csv(
        OUTPUT_DIR / "source_target_integrated_gradient_contrasts_relative_by_contrast.csv",
        index=False,
    )
    make_overlap_table(contrast_table, significant_channels).to_csv(
        OUTPUT_DIR / "source_target_integrated_gradient_contrast_overlap.csv",
        index=False,
    )

    for image_interpolation in ["nearest", "cubic"]:
        output_prefix = f"source_target_integrated_gradient_contrasts_mean_{image_interpolation}_p010"
        plot_combined_contrasts(contrast_table, significant_channels, image_interpolation, output_prefix)

        for band in BAND_ORDER:
            plot_band_contrasts(
                contrast_table,
                significant_channels,
                band,
                image_interpolation,
                output_prefix,
            )

        relative_output_prefix = (
            "source_target_integrated_gradient_contrasts_relative_by_contrast_"
            f"{image_interpolation}_p010"
        )
        plot_combined_contrasts(
            normalized_contrast_table,
            significant_channels,
            image_interpolation,
            relative_output_prefix,
            fixed_max_abs=1.0,
        )

        for band in BAND_ORDER:
            plot_band_contrasts(
                normalized_contrast_table,
                significant_channels,
                band,
                image_interpolation,
                relative_output_prefix,
                fixed_max_abs=1.0,
            )


if __name__ == "__main__":
    main()
