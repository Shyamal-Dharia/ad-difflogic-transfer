import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from PIL import Image
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from train_helpers import ROOT_DIR, relative_psd_to_bandpower


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

CLINICAL_GROUPS = {"C": "CN", "A": "AD"}
FTD_CLINICAL_GROUPS = {"C": "CN", "F": "FTD"}
ALL_CLINICAL_GROUPS = {"C": "CN", "A": "AD", "F": "FTD"}
PEARL_GROUPS = {"N": "N", "A+P-": "A+P-", "A+P+": "A+P+"}
GROUP_ORDER = ["CN", "AD", "N", "A+P-", "A+P+"]
PLOT_STYLES = {
    "CN": {"color": "#7EAED3", "marker": "o", "size": 58, "alpha": 0.86},
    "AD": {"color": "#D88484", "marker": "o", "size": 58, "alpha": 0.86},
    "FTD": {"color": "#D88484", "marker": "o", "size": 58, "alpha": 0.86},
    "N": {"color": "#A8A8A8", "marker": "^", "size": 68, "alpha": 0.78},
    "A+P-": {"color": "#EDC46F", "marker": "^", "size": 68, "alpha": 0.82},
    "A+P+": {"color": "#B694D6", "marker": "^", "size": 68, "alpha": 0.86},
}
PANEL_FIGSIZE = (6.2, 6.2)
PANEL_RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 180,
}


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def pad_png_to_match_height(image_path, reference_path):
    image = Image.open(image_path).convert("RGBA")
    reference = Image.open(reference_path)

    if image.height >= reference.height:
        return

    top_padding = (reference.height - image.height) // 2
    output = Image.new("RGBA", (image.width, reference.height), (255, 255, 255, 255))
    output.paste(image, (0, top_padding), image)
    output.convert("RGB").save(image_path)


def load_subject_features(feature_dir, dataset_name, group_map):
    rows = []
    dataset_dir = feature_dir / dataset_name

    for feature_path in sorted(dataset_dir.glob("*.npz")):
        data = np.load(feature_path, allow_pickle=True)
        raw_group = str(scalar(data["group"]))
        group = group_map.get(raw_group)

        if group is None:
            continue

        channel_names = data["channel_names"].astype(str).tolist()
        if channel_names != COMMON_CHANNELS:
            raise ValueError(
                f"{feature_path} has channel order {channel_names}, expected {COMMON_CHANNELS}"
            )

        epoch_bandpower = relative_psd_to_bandpower(data["relative_psd"], data["freqs"])
        subject_feature = epoch_bandpower.mean(axis=0).reshape(-1)
        rows.append(
            {
                "participant_id": str(scalar(data["participant_id"])),
                "dataset": dataset_name,
                "group": group,
                "n_epochs": int(epoch_bandpower.shape[0]),
                "feature": subject_feature,
            }
        )

    return rows


def make_subject_table(feature_dir, clinical_groups=CLINICAL_GROUPS):
    clinical_rows = load_subject_features(feature_dir, "ALZ_FTD", clinical_groups)
    pearl_rows = load_subject_features(feature_dir, "PEARL", PEARL_GROUPS)
    rows = clinical_rows + pearl_rows
    table = pd.DataFrame(rows)
    features = np.stack(table["feature"].to_numpy())
    table = table.drop(columns=["feature"]).reset_index(drop=True)
    return table, features


def add_centroid_distances(table, features_scaled, disease_group="AD"):
    clinical_mask = table["dataset"].eq("ALZ_FTD")
    cn_mask = clinical_mask & table["group"].eq("CN")
    disease_mask = clinical_mask & table["group"].eq(disease_group)
    cn_centroid = features_scaled[cn_mask.to_numpy()].mean(axis=0)
    disease_centroid = features_scaled[disease_mask.to_numpy()].mean(axis=0)

    distance_to_cn = np.linalg.norm(features_scaled - cn_centroid, axis=1)
    distance_to_disease = np.linalg.norm(features_scaled - disease_centroid, axis=1)
    disease_name = disease_group.lower()
    table = table.copy()
    table["distance_to_cn_centroid"] = distance_to_cn
    table[f"distance_to_{disease_name}_centroid"] = distance_to_disease
    table[f"{disease_name}_closeness"] = distance_to_cn - distance_to_disease
    table["nearest_clinical_centroid"] = np.where(
        table[f"{disease_name}_closeness"] > 0.0,
        disease_group,
        "CN",
    )
    return table


def summarize_closeness(table):
    summary = (
        table.groupby(["dataset", "group"], sort=False)
        .agg(
            n=("participant_id", "size"),
            mean_ad_closeness=("ad_closeness", "mean"),
            sd_ad_closeness=("ad_closeness", "std"),
            mean_distance_to_cn=("distance_to_cn_centroid", "mean"),
            mean_distance_to_ad=("distance_to_ad_centroid", "mean"),
        )
        .reset_index()
    )
    return summary


def summarize_nearest_centroid(table, disease_group="AD"):
    counts = (
        table.groupby(["dataset", "group", "nearest_clinical_centroid"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for centroid in ["CN", disease_group]:
        if centroid not in counts.columns:
            counts[centroid] = 0

    fraction_column = f"{disease_group.lower()}_nearest_fraction"
    counts["n"] = counts["CN"] + counts[disease_group]
    counts[fraction_column] = counts[disease_group] / counts["n"]
    return counts[["dataset", "group", "n", "CN", disease_group, fraction_column]]


def pearl_pairwise_tests(table):
    pearl = table[table["dataset"].eq("PEARL")]
    comparisons = [("A+P+", "N"), ("A+P+", "A+P-"), ("A+P-", "N")]
    rows = []

    for group_a, group_b in comparisons:
        values_a = pearl.loc[pearl["group"].eq(group_a), "ad_closeness"].to_numpy()
        values_b = pearl.loc[pearl["group"].eq(group_b), "ad_closeness"].to_numpy()
        statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
        rows.append(
            {
                "contrast": f"{group_a} - {group_b}",
                "mean_difference": values_a.mean() - values_b.mean(),
                "mannwhitney_u": statistic,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows)


def fit_umap(table, features_scaled, random_state):
    return fit_umap_components(table, features_scaled, random_state, n_components=2)


def fit_umap_components(table, features_scaled, random_state, n_components):
    clinical_mask = table["dataset"].eq("ALZ_FTD").to_numpy()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.25,
        metric="euclidean",
        random_state=random_state,
        transform_seed=random_state,
    )
    clinical_embedding = reducer.fit_transform(features_scaled[clinical_mask])
    pearl_embedding = reducer.transform(features_scaled[~clinical_mask])

    embedding = np.zeros((len(table), n_components), dtype=np.float32)
    embedding[clinical_mask] = clinical_embedding
    embedding[~clinical_mask] = pearl_embedding
    return embedding


def fit_pca(table, features_scaled):
    clinical_mask = table["dataset"].eq("ALZ_FTD").to_numpy()
    reducer = PCA(n_components=2)
    reducer.fit(features_scaled[clinical_mask])
    return reducer.transform(features_scaled)


def fit_tsne(features_scaled, random_state):
    reducer = TSNE(
        n_components=2,
        perplexity=20,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        metric="euclidean",
    )
    return reducer.fit_transform(features_scaled)


def plot_embedding(table, embedding, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.dpi": 180,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    for group in GROUP_ORDER:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        style = PLOT_STYLES[group]
        label = group if group in {"CN", "AD"} else f"Genetic-risk {group}"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            edgecolor="black",
            linewidth=0.65,
            label=label,
            zorder=3,
        )

    for group in GROUP_ORDER:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        centroid = embedding[mask].mean(axis=0)
        style = PLOT_STYLES[group]
        ax.scatter(
            centroid[0],
            centroid[1],
            s=180,
            c=style["color"],
            marker="X",
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("45 Hz PSD Feature Space")
    ax.grid(True, linestyle=(0, (1.5, 3.0)), linewidth=0.8, color="#c6c6c6", alpha=0.9)
    ax.legend(loc="best", frameon=True, fontsize=9)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    png_path = output_dir / "psd_umap_clinical_fit_genetic_risk_45hz.png"
    pdf_path = output_dir / "psd_umap_clinical_fit_genetic_risk_45hz.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_embedding_triptych(table, pca_embedding, tsne_embedding, umap_embedding, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 180,
        }
    )

    embeddings = [
        ("PCA", pca_embedding, "PC 1", "PC 2"),
        ("t-SNE", tsne_embedding, "t-SNE 1", "t-SNE 2"),
        ("UMAP", umap_embedding, "UMAP 1", "UMAP 2"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    legend_handles = []
    legend_labels = []

    for ax, (title, embedding, xlabel, ylabel) in zip(axes, embeddings):
        for group in GROUP_ORDER:
            mask = table["group"].eq(group).to_numpy()
            if not np.any(mask):
                continue
            style = PLOT_STYLES[group]
            label = group if group in {"CN", "AD"} else f"Genetic-risk {group}"
            scatter = ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=36 if group in {"CN", "AD"} else 44,
                c=style["color"],
                marker=style["marker"],
                alpha=0.78,
                edgecolor="black",
                linewidth=0.45,
                label=label,
                zorder=3,
            )
            if ax is axes[0]:
                legend_handles.append(scatter)
                legend_labels.append(label)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=(0, (1.5, 3.0)), linewidth=0.65, color="#d0d0d0")
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.3,
    )
    fig.suptitle("Subject-Level 45 Hz PSD Feature Embeddings", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])

    png_path = output_dir / "psd_pca_tsne_umap_comparison_45hz.png"
    pdf_path = output_dir / "psd_pca_tsne_umap_comparison_45hz.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_embedding_with_centroid_circles(table, embedding, output_dir, disease_group="AD"):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PANEL_RCPARAMS)

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    disease_groups = [disease_group] if isinstance(disease_group, str) else list(disease_group)
    clinical_groups = {"CN", *disease_groups}
    group_order = ["CN", *disease_groups, "N", "A+P-", "A+P+"]
    plot_styles = PLOT_STYLES.copy()
    if len(disease_groups) > 1:
        plot_styles["FTD"] = {"color": "#72B7A5", "marker": "o", "size": 58, "alpha": 0.86}
    for group in group_order:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        style = plot_styles[group]
        label = group if group in clinical_groups else f"Genetic-risk {group}"
        point_size = 48 if group in clinical_groups else 60
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            c=style["color"],
            marker=style["marker"],
            alpha=0.80 if group in clinical_groups else 0.72,
            edgecolor="black",
            linewidth=0.55,
            label=label,
            zorder=3,
        )

    centroid_points = {}
    for group in ["CN", *disease_groups]:
        mask = table["group"].eq(group).to_numpy()
        centroid = embedding[mask].mean(axis=0)
        distances = np.linalg.norm(embedding[mask] - centroid, axis=1)
        radius = np.percentile(distances, 65)
        style = plot_styles[group]
        centroid_points[group] = centroid
        circle = plt.Circle(
            centroid,
            radius,
            fill=True,
            color=style["color"],
            linewidth=1.8,
            linestyle=(0, (5.0, 3.0)),
            alpha=0.08,
            zorder=2,
        )
        ax.add_patch(circle)
        outline = plt.Circle(
            centroid,
            radius,
            fill=False,
            color=style["color"],
            linewidth=1.8,
            linestyle=(0, (5.0, 3.0)),
            alpha=0.95,
            zorder=4,
        )
        ax.add_patch(outline)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=170,
            c=style["color"],
            marker="X",
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
            label=f"{group} centroid",
        )
        ax.annotate(
            group,
            xy=centroid,
            xytext=(9, 9 if group == "CN" else -16),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.68},
            zorder=7,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, linestyle=(0, (1.5, 3.0)), linewidth=0.7, color="#d2d2d2", alpha=0.9)
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=9 if len(disease_groups) > 1 else 10,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.45,
        borderaxespad=0.45,
        scatterpoints=1,
        markerscale=0.75,
    )
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_linewidth(0.8)
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    if disease_groups == ["AD"]:
        ax.set_xlim(-2.0, 7.0)
    else:
        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        ax.set_xlim(x_min - 0.55, x_max + 0.55)
    ax.set_ylim(y_min - 0.55, y_max + 0.55)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    if disease_groups == ["AD"]:
        disease_suffix = ""
    else:
        disease_suffix = "_" + "_".join(group.lower() for group in disease_groups)
    png_path = output_dir / f"psd_umap_clinical_centroid_circles{disease_suffix}_45hz.png"
    pdf_path = output_dir / f"psd_umap_clinical_centroid_circles{disease_suffix}_45hz.pdf"
    panel_png_path = output_dir / f"psd_centroid_panel_a_umap{disease_suffix}_45hz.png"
    panel_pdf_path = output_dir / f"psd_centroid_panel_a_umap{disease_suffix}_45hz.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(panel_png_path)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(panel_pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_tsne_with_centroid_circles(table, embedding, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.dpi": 180,
        }
    )

    fig, ax = plt.subplots(figsize=(4.6, 5.78))

    for group in GROUP_ORDER:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        style = PLOT_STYLES[group]
        label = group if group in {"CN", "AD"} else f"Genetic-risk {group}"
        point_size = 48 if group in {"CN", "AD"} else 60
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            c=style["color"],
            marker=style["marker"],
            alpha=0.80 if group in {"CN", "AD"} else 0.72,
            edgecolor="black",
            linewidth=0.55,
            label=label,
            zorder=3,
        )

    centroid_points = {}
    for group in ["CN", "AD"]:
        mask = table["group"].eq(group).to_numpy()
        centroid = embedding[mask].mean(axis=0)
        distances = np.linalg.norm(embedding[mask] - centroid, axis=1)
        radius = np.percentile(distances, 65)
        style = PLOT_STYLES[group]
        centroid_points[group] = centroid
        circle = plt.Circle(
            centroid,
            radius,
            fill=True,
            color=style["color"],
            linewidth=1.8,
            linestyle=(0, (5.0, 3.0)),
            alpha=0.08,
            zorder=2,
        )
        ax.add_patch(circle)
        outline = plt.Circle(
            centroid,
            radius,
            fill=False,
            color=style["color"],
            linewidth=1.8,
            linestyle=(0, (5.0, 3.0)),
            alpha=0.95,
            zorder=4,
        )
        ax.add_patch(outline)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=170,
            c=style["color"],
            marker="X",
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
            label=f"{group} centroid",
        )
        ax.annotate(
            group,
            xy=centroid,
            xytext=(9, 9 if group == "CN" else -16),
            textcoords="offset points",
            fontsize=12.0,
            weight="bold",
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.68},
            zorder=7,
        )

    if {"CN", "AD"}.issubset(centroid_points):
        cn_centroid = centroid_points["CN"]
        ad_centroid = centroid_points["AD"]
        ax.plot(
            [cn_centroid[0], ad_centroid[0]],
            [cn_centroid[1], ad_centroid[1]],
            color="#333333",
            linewidth=1.0,
            linestyle=(0, (2.0, 3.0)),
            alpha=0.75,
            zorder=1,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linestyle=(0, (1.5, 3.0)), linewidth=0.7, color="#d2d2d2", alpha=0.9)
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=11.3,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.45,
        borderaxespad=0.45,
        scatterpoints=1,
        markerscale=0.75,
    )
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_linewidth(0.8)

    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    ax.set_xlim(x_min - 1.5, x_max + 1.5)
    ax.set_ylim(y_min - 1.5, y_max + 1.5)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    png_path = output_dir / "psd_tsne_clinical_centroid_circles_45hz.png"
    pdf_path = output_dir / "psd_tsne_clinical_centroid_circles_45hz.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def centroid_table_rows(nearest_summary):
    dataset_labels = {"ALZ_FTD": "Clinical", "PEARL": "Genetic-Risk"}
    rows = []

    for dataset, group in [
        ("ALZ_FTD", "CN"),
        ("ALZ_FTD", "AD"),
        ("PEARL", "N"),
        ("PEARL", "A+P-"),
        ("PEARL", "A+P+"),
    ]:
        row = nearest_summary[
            nearest_summary["dataset"].eq(dataset) & nearest_summary["group"].eq(group)
        ].iloc[0]
        rows.append(
            [
                dataset_labels[dataset],
                group.replace("A+P-", "A+P-").replace("A+P+", "A+P+"),
                f"{int(row['n'])}",
                f"{int(row['CN'])}",
                f"{int(row['AD'])}",
                f"{row['ad_nearest_fraction']:.3f}",
            ]
        )

    return rows


def draw_nearest_centroid_table(ax, nearest_summary):
    ax.axis("off")
    columns = ["Dataset", "Group", "N", "CN-nearest", "AD-nearest", "AD-fraction"]
    rows = centroid_table_rows(nearest_summary)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.22, 0.15, 0.09, 0.18, 0.18, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    table.scale(1.0, 1.55)

    header_color = "#34495E"
    clinical_color = "#EAF3FB"
    pearl_color = "#F7F1E4"
    highlight_color = "#EBDDF7"
    edge_color = "#C9CED6"

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.65)

        if row_index == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.set_linewidth(0.0)
            continue

        if row_index in [1, 2]:
            cell.set_facecolor(clinical_color)
        else:
            cell.set_facecolor(pearl_color)

        if row_index == 5:
            cell.set_facecolor(highlight_color)
            cell.get_text().set_weight("bold")

        if col_index in [0, 1]:
            cell.get_text().set_ha("left")
            cell.PAD = 0.06

    ax.text(
        0.0,
        0.98,
        "Nearest-Centroid Assignment",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        weight="bold",
    )
    ax.text(
        0.0,
        0.90,
        "Distances computed in standardized 75D PSD feature space",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        color="#444444",
    )


def plot_nearest_centroid_table(nearest_summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.dpi": 180,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    draw_nearest_centroid_table(ax, nearest_summary)
    fig.tight_layout()

    png_path = output_dir / "psd_nearest_centroid_table_45hz.png"
    pdf_path = output_dir / "psd_nearest_centroid_table_45hz.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_umap_with_nearest_centroid_table(table, embedding, nearest_summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 180,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax, table_ax = axes

    for group in GROUP_ORDER:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        style = PLOT_STYLES[group]
        label = group if group in {"CN", "AD"} else f"Genetic-risk {group}"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=40 if group in {"CN", "AD"} else 50,
            c=style["color"],
            marker=style["marker"],
            alpha=0.78,
            edgecolor="black",
            linewidth=0.48,
            label=label,
            zorder=3,
        )

    for group in ["CN", "AD"]:
        mask = table["group"].eq(group).to_numpy()
        centroid = embedding[mask].mean(axis=0)
        distances = np.linalg.norm(embedding[mask] - centroid, axis=1)
        radius = np.percentile(distances, 65)
        style = PLOT_STYLES[group]
        ax.add_patch(
            plt.Circle(
                centroid,
                radius,
                fill=True,
                color=style["color"],
                linewidth=1.5,
                alpha=0.08,
                zorder=2,
            )
        )
        ax.add_patch(
            plt.Circle(
                centroid,
                radius,
                fill=False,
                color=style["color"],
                linewidth=1.5,
                linestyle=(0, (5.0, 3.0)),
                alpha=0.95,
                zorder=4,
            )
        )
        ax.scatter(
            centroid[0],
            centroid[1],
            s=155,
            c=style["color"],
            marker="X",
            edgecolor="black",
            linewidth=1.0,
            zorder=6,
            label=f"{group} centroid",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xlim(-2.0, 7.0)
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    ax.set_ylim(y_min - 0.55, y_max + 0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=(0, (1.5, 3.0)), linewidth=0.65, color="#d2d2d2", alpha=0.9)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5, markerscale=0.72)

    draw_nearest_centroid_table(table_ax, nearest_summary)

    fig.tight_layout(w_pad=2.0)
    png_path = output_dir / "psd_umap_nearest_centroid_panel_45hz.png"
    pdf_path = output_dir / "psd_umap_nearest_centroid_panel_45hz.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def draw_ad_fraction_bars(ax, nearest_summary, disease_group="AD"):
    pearl = nearest_summary[nearest_summary["dataset"].eq("PEARL")].copy()
    group_order = ["N", "A+P-", "A+P+"]
    pearl["group"] = pd.Categorical(pearl["group"], categories=group_order, ordered=True)
    pearl = pearl.sort_values("group")

    colors = ["#CFCFCF", "#F1D387", "#C8AEDF"]
    edge_colors = ["#6A6A6A", "#A77A22", "#73519B"]
    x = np.arange(len(pearl))
    disease_name = disease_group.lower()
    fractions = pearl[f"{disease_name}_nearest_fraction"].to_numpy()
    disease_counts = pearl[disease_group].astype(int).to_numpy()
    totals = pearl["n"].astype(int).to_numpy()

    bars = ax.bar(
        x,
        fractions,
        color=colors,
        edgecolor=edge_colors,
        linewidth=1.2,
        width=0.62,
        alpha=1.0,
        zorder=3,
    )

    for bar, fraction, disease_count, total in zip(bars, fractions, disease_counts, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            fraction + 0.035,
            f"{disease_count}/{total}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["N", "A+P-", "A+P+"], fontsize=10)
    ax.set_ylabel(f"{disease_group}-proportion")
    ax.set_ylim(0.0, 0.70)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.grid(True, axis="y", linestyle=(0, (1.5, 3.0)), linewidth=0.8, color="#d0d0d0")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_ad_fraction_bars(nearest_summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 180,
        }
    )

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    draw_ad_fraction_bars(ax, nearest_summary)
    fig.tight_layout()

    png_path = output_dir / "psd_genetic_risk_ad_nearest_fraction_45hz.png"
    pdf_path = output_dir / "psd_genetic_risk_ad_nearest_fraction_45hz.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_ad_fraction_bars_matched_panel(nearest_summary, output_dir, disease_group="AD"):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PANEL_RCPARAMS)

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    draw_ad_fraction_bars(ax, nearest_summary, disease_group)
    fig.tight_layout()

    disease_name = disease_group.lower()
    png_path = output_dir / f"psd_centroid_panel_b_{disease_name}_fraction_45hz.png"
    pdf_path = output_dir / f"psd_centroid_panel_b_{disease_name}_fraction_45hz.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_umap_with_ad_fraction_bars(table, embedding, nearest_summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 180,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.1), gridspec_kw={"width_ratios": [1.35, 0.9]})
    ax, bar_ax = axes

    for group in GROUP_ORDER:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        style = PLOT_STYLES[group]
        label = group if group in {"CN", "AD"} else f"Genetic-risk {group}"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=44 if group in {"CN", "AD"} else 55,
            c=style["color"],
            marker=style["marker"],
            alpha=0.80 if group in {"CN", "AD"} else 0.74,
            edgecolor="black",
            linewidth=0.5,
            label=label,
            zorder=3,
        )

    for group in ["CN", "AD"]:
        mask = table["group"].eq(group).to_numpy()
        centroid = embedding[mask].mean(axis=0)
        distances = np.linalg.norm(embedding[mask] - centroid, axis=1)
        radius = np.percentile(distances, 65)
        style = PLOT_STYLES[group]
        ax.add_patch(
            plt.Circle(centroid, radius, fill=True, color=style["color"], alpha=0.08, zorder=2)
        )
        ax.add_patch(
            plt.Circle(
                centroid,
                radius,
                fill=False,
                color=style["color"],
                linewidth=1.6,
                linestyle=(0, (5.0, 3.0)),
                alpha=0.95,
                zorder=4,
            )
        )
        ax.scatter(
            centroid[0],
            centroid[1],
            s=165,
            c=style["color"],
            marker="X",
            edgecolor="black",
            linewidth=1.0,
            zorder=6,
            label=f"{group} centroid",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xlim(-2.0, 7.0)
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    ax.set_ylim(y_min - 0.55, y_max + 0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=(0, (1.5, 3.0)), linewidth=0.7, color="#d2d2d2", alpha=0.9)
    ax.legend(loc="lower right", frameon=True, fontsize=8.2, markerscale=0.72)

    draw_ad_fraction_bars(bar_ax, nearest_summary)

    fig.tight_layout(w_pad=2.0)
    png_path = output_dir / "psd_umap_ad_nearest_fraction_panel_45hz.png"
    pdf_path = output_dir / "psd_umap_ad_nearest_fraction_panel_45hz.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_embedding_and_closeness(table, embedding, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 180,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax_umap, ax_distance = axes

    for group in GROUP_ORDER:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        style = PLOT_STYLES[group]
        label = group if group in {"CN", "AD"} else f"Genetic-risk {group}"
        ax_umap.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            edgecolor="black",
            linewidth=0.65,
            label=label,
            zorder=3,
        )

    for group in GROUP_ORDER:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        centroid = embedding[mask].mean(axis=0)
        style = PLOT_STYLES[group]
        ax_umap.scatter(
            centroid[0],
            centroid[1],
            s=180,
            c=style["color"],
            marker="X",
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )

    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")
    ax_umap.set_title("Clinical UMAP Fit")
    ax_umap.grid(True, linestyle=(0, (1.5, 3.0)), linewidth=0.8, color="#c6c6c6", alpha=0.9)
    ax_umap.legend(loc="best", frameon=True, fontsize=8.5)

    positions = np.arange(len(GROUP_ORDER), dtype=float)
    values_by_group = [
        table.loc[table["group"].eq(group), "ad_closeness"].to_numpy() for group in GROUP_ORDER
    ]
    box = ax_distance.boxplot(
        values_by_group,
        positions=positions,
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
    )

    for patch, group in zip(box["boxes"], GROUP_ORDER):
        patch.set_facecolor(PLOT_STYLES[group]["color"])
        patch.set_alpha(0.42)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.9)

    rng = np.random.default_rng(17)
    for position, group, values in zip(positions, GROUP_ORDER, values_by_group):
        style = PLOT_STYLES[group]
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax_distance.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=26,
            c=style["color"],
            alpha=0.72,
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )

    ax_distance.axhline(0.0, color="black", linestyle=(0, (3.0, 3.0)), linewidth=1.0)
    ax_distance.set_xticks(positions)
    ax_distance.set_xticklabels(GROUP_ORDER, rotation=30, ha="right")
    ax_distance.set_ylabel("AD-closeness score")
    ax_distance.set_title(r"$d_{\mathrm{CN}} - d_{\mathrm{AD}}$")
    ax_distance.grid(True, axis="y", linestyle=(0, (1.5, 3.0)), linewidth=0.8, color="#c6c6c6")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    fig.tight_layout()
    png_path = output_dir / "psd_umap_and_ad_closeness_45hz.png"
    pdf_path = output_dir / "psd_umap_and_ad_closeness_45hz.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_embedding_3d(table, embedding_3d, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 180,
        }
    )

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    for group in GROUP_ORDER:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        style = PLOT_STYLES[group]
        label = group if group in {"CN", "AD"} else f"Genetic-risk {group}"
        ax.scatter(
            embedding_3d[mask, 0],
            embedding_3d[mask, 1],
            embedding_3d[mask, 2],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            alpha=style["alpha"],
            edgecolor="black",
            linewidth=0.45,
            label=label,
            depthshade=False,
        )

    for group in GROUP_ORDER:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        centroid = embedding_3d[mask].mean(axis=0)
        style = PLOT_STYLES[group]
        ax.scatter(
            centroid[0],
            centroid[1],
            centroid[2],
            s=190,
            c=style["color"],
            marker="X",
            edgecolor="black",
            linewidth=0.9,
            depthshade=False,
        )

    ax.set_xlabel("UMAP 1", labelpad=8)
    ax.set_ylabel("UMAP 2", labelpad=8)
    ax.set_zlabel("UMAP 3", labelpad=8)
    ax.set_title("3D Clinical UMAP Fit")
    ax.view_init(elev=22, azim=-62)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=8, frameon=True)
    ax.grid(True)

    fig.tight_layout()
    png_path = output_dir / "psd_umap_3d_clinical_fit_genetic_risk_45hz.png"
    pdf_path = output_dir / "psd_umap_3d_clinical_fit_genetic_risk_45hz.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_embedding_3d_interactive(table, embedding_3d, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()

    for group in GROUP_ORDER:
        mask = table["group"].eq(group).to_numpy()
        if not np.any(mask):
            continue
        style = PLOT_STYLES[group]
        label = group if group in {"CN", "AD"} else f"Genetic-risk {group}"
        marker_symbol = "circle" if group in {"CN", "AD"} else "diamond"
        hover_text = [
            f"{row.participant_id}<br>{row.dataset}<br>{row.group}<br>"
            f"AD-closeness: {row.ad_closeness:.3f}<br>"
            f"Nearest centroid: {row.nearest_clinical_centroid}"
            for row in table.loc[mask].itertuples()
        ]
        fig.add_trace(
            go.Scatter3d(
                x=embedding_3d[mask, 0],
                y=embedding_3d[mask, 1],
                z=embedding_3d[mask, 2],
                mode="markers",
                name=label,
                text=hover_text,
                hoverinfo="text",
                marker={
                    "size": 6,
                    "color": style["color"],
                    "opacity": style["alpha"],
                    "symbol": marker_symbol,
                    "line": {"color": "black", "width": 1.0},
                },
            )
        )

    fig.update_layout(
        title="3D UMAP Projection of 45 Hz PSD Features",
        scene={
            "xaxis_title": "UMAP 1",
            "yaxis_title": "UMAP 2",
            "zaxis_title": "UMAP 3",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        width=900,
        height=720,
    )

    html_path = output_dir / "psd_umap_3d_clinical_fit_genetic_risk_45hz.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path


def save_latex_summary(summary, output_path):
    summary_for_tex = summary.copy()
    summary_for_tex["mean_ad_closeness"] = summary_for_tex["mean_ad_closeness"].map(
        lambda value: f"{value:.3f}"
    )
    summary_for_tex["sd_ad_closeness"] = summary_for_tex["sd_ad_closeness"].map(
        lambda value: f"{value:.3f}"
    )
    summary_for_tex.to_latex(
        output_path,
        index=False,
        escape=False,
        columns=["dataset", "group", "n", "mean_ad_closeness", "sd_ad_closeness"],
        header=["Dataset", "Group", "N", "Mean AD-closeness", "SD"],
    )


def make_ftd_centroid_panels(args):
    table, features = make_subject_table(args.feature_dir, FTD_CLINICAL_GROUPS)
    clinical_mask = table["dataset"].eq("ALZ_FTD").to_numpy()
    scaler = StandardScaler().fit(features[clinical_mask])
    features_scaled = scaler.transform(features)
    embedding = fit_umap(table, features_scaled, args.random_state)
    table = add_centroid_distances(table, features_scaled, "FTD")
    nearest_summary = summarize_nearest_centroid(table, "FTD")

    summary_path = args.output_dir / "psd_nearest_centroid_ftd_summary.csv"
    nearest_summary.to_csv(summary_path, index=False)
    plot_embedding_with_centroid_circles(
        table,
        embedding,
        args.figure_dir,
        disease_group="FTD",
    )
    bar_path, _ = plot_ad_fraction_bars_matched_panel(
        nearest_summary,
        args.figure_dir,
        disease_group="FTD",
    )
    panel_path = args.figure_dir / "psd_centroid_panel_a_umap_ftd_45hz.png"
    pad_png_to_match_height(bar_path, panel_path)
    print(f"Saved {panel_path}")
    print(f"Saved {bar_path}")
    print(f"Saved {summary_path}")


def make_all_centroid_panel(args):
    table, features = make_subject_table(args.feature_dir, ALL_CLINICAL_GROUPS)
    clinical_mask = table["dataset"].eq("ALZ_FTD").to_numpy()
    scaler = StandardScaler().fit(features[clinical_mask])
    embedding = fit_umap(table, scaler.transform(features), args.random_state)
    plot_embedding_with_centroid_circles(
        table,
        embedding,
        args.figure_dir,
        disease_group=("AD", "FTD"),
    )
    panel_path = args.figure_dir / "psd_centroid_panel_a_umap_ad_ftd_45hz.png"
    print(f"Saved {panel_path}")


def make_ad_ftd_correlation_panel(args):
    score_tables = []
    for clinical_groups, disease_group in [
        (CLINICAL_GROUPS, "AD"),
        (FTD_CLINICAL_GROUPS, "FTD"),
    ]:
        table, features = make_subject_table(args.feature_dir, clinical_groups)
        clinical_mask = table["dataset"].eq("ALZ_FTD").to_numpy()
        scaler = StandardScaler().fit(features[clinical_mask])
        table = add_centroid_distances(
            table,
            scaler.transform(features),
            disease_group,
        )
        score_tables.append(
            table[table["dataset"].eq("PEARL")][
                ["participant_id", "group", f"{disease_group.lower()}_closeness"]
            ]
        )

    scores = score_tables[0].merge(
        score_tables[1],
        on=["participant_id", "group"],
        validate="one_to_one",
    )
    all_pearson = pearsonr(scores["ad_closeness"], scores["ftd_closeness"])
    all_spearman = spearmanr(scores["ad_closeness"], scores["ftd_closeness"])
    dual_risk = scores[scores["group"].eq("A+P+")]
    dual_pearson = pearsonr(dual_risk["ad_closeness"], dual_risk["ftd_closeness"])
    dual_spearman = spearmanr(dual_risk["ad_closeness"], dual_risk["ftd_closeness"])

    plt.rcParams.update(PANEL_RCPARAMS)
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for group in ["N", "A+P-", "A+P+"]:
        group_scores = scores[scores["group"].eq(group)]
        style = PLOT_STYLES[group]
        ax.scatter(
            group_scores["ad_closeness"],
            group_scores["ftd_closeness"],
            s=68 if group == "A+P+" else 58,
            c=style["color"],
            marker="^",
            edgecolor="black",
            linewidth=0.65,
            alpha=0.86,
            label=group,
            zorder=3,
        )

    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=1.0)
    fit = np.polyfit(scores["ad_closeness"], scores["ftd_closeness"], 1)
    x_line = np.linspace(scores["ad_closeness"].min(), scores["ad_closeness"].max(), 100)
    ax.plot(x_line, np.polyval(fit, x_line), color="#333333", linewidth=1.2, alpha=0.75)
    ax.set_xlabel(r"AD-closeness ($d_{CN}-d_{AD}$)")
    ax.set_ylabel(r"FTD-closeness ($d_{CN}-d_{FTD}$)")
    ax.grid(True, linestyle=(0, (1.5, 3.0)), linewidth=0.7, color="#d2d2d2")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.text(
        0.03,
        0.97,
        f"All PEARL: $r={all_pearson.statistic:.3f}$\n"
        f"A+P+: $r={dual_pearson.statistic:.3f}$",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#888888", "alpha": 0.9},
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.tight_layout()

    png_path = args.figure_dir / "psd_ad_ftd_closeness_correlation_45hz.png"
    pdf_path = args.figure_dir / "psd_ad_ftd_closeness_correlation_45hz.pdf"
    csv_path = args.output_dir / "psd_ad_ftd_subject_closeness.csv"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    scores.to_csv(csv_path, index=False)
    print(f"Saved {png_path}")
    print(f"All PEARL: Pearson r={all_pearson.statistic:.3f}, Spearman rho={all_spearman.statistic:.3f}")
    print(f"A+P+: Pearson r={dual_pearson.statistic:.3f}, Spearman rho={dual_spearman.statistic:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=ROOT_DIR / "datasets/features_psd_45hz")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs/psd_umap_45hz")
    parser.add_argument("--figure-dir", type=Path, default=ROOT_DIR / "figures")
    parser.add_argument("--random-state", type=int, default=11)
    parser.add_argument(
        "--clinical-centroid",
        choices=["ad", "ftd", "all", "correlation"],
        default="ad",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clinical_centroid == "ftd":
        make_ftd_centroid_panels(args)
        return
    if args.clinical_centroid == "all":
        make_all_centroid_panel(args)
        return
    if args.clinical_centroid == "correlation":
        make_ad_ftd_correlation_panel(args)
        return

    table, features = make_subject_table(args.feature_dir)
    clinical_mask = table["dataset"].eq("ALZ_FTD").to_numpy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features[clinical_mask])
    all_features_scaled = scaler.transform(features)

    embedding = fit_umap(table, all_features_scaled, args.random_state)
    embedding_3d = fit_umap_components(
        table,
        all_features_scaled,
        args.random_state,
        n_components=3,
    )
    pca_embedding = fit_pca(table, all_features_scaled)
    tsne_embedding = fit_tsne(all_features_scaled, args.random_state)
    table["umap_1"] = embedding[:, 0]
    table["umap_2"] = embedding[:, 1]
    table = add_centroid_distances(table, all_features_scaled)
    table["umap_3d_1"] = embedding_3d[:, 0]
    table["umap_3d_2"] = embedding_3d[:, 1]
    table["umap_3d_3"] = embedding_3d[:, 2]
    table["pca_1"] = pca_embedding[:, 0]
    table["pca_2"] = pca_embedding[:, 1]
    table["tsne_1"] = tsne_embedding[:, 0]
    table["tsne_2"] = tsne_embedding[:, 1]

    summary = summarize_closeness(table)
    nearest_summary = summarize_nearest_centroid(table)
    tests = pearl_pairwise_tests(table)

    subject_path = args.output_dir / "psd_umap_subject_coordinates_and_distances.csv"
    summary_path = args.output_dir / "psd_umap_ad_closeness_summary.csv"
    nearest_summary_path = args.output_dir / "psd_nearest_centroid_summary.csv"
    tests_path = args.output_dir / "psd_umap_pearl_ad_closeness_tests.csv"
    tex_path = args.output_dir / "psd_umap_ad_closeness_summary.tex"
    table.to_csv(subject_path, index=False)
    summary.to_csv(summary_path, index=False)
    nearest_summary.to_csv(nearest_summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    save_latex_summary(summary, tex_path)

    png_path, pdf_path = plot_embedding(table, embedding, args.figure_dir)
    triptych_png_path, triptych_pdf_path = plot_embedding_triptych(
        table,
        pca_embedding,
        tsne_embedding,
        embedding,
        args.figure_dir,
    )
    centroid_png_path, centroid_pdf_path = plot_embedding_with_centroid_circles(
        table, embedding, args.figure_dir
    )
    tsne_centroid_png_path, tsne_centroid_pdf_path = plot_tsne_with_centroid_circles(
        table, tsne_embedding, args.figure_dir
    )
    centroid_table_png_path, centroid_table_pdf_path = plot_nearest_centroid_table(
        nearest_summary, args.figure_dir
    )
    umap_table_png_path, umap_table_pdf_path = plot_umap_with_nearest_centroid_table(
        table, embedding, nearest_summary, args.figure_dir
    )
    ad_fraction_png_path, ad_fraction_pdf_path = plot_ad_fraction_bars(
        nearest_summary, args.figure_dir
    )
    matched_ad_fraction_png_path, matched_ad_fraction_pdf_path = (
        plot_ad_fraction_bars_matched_panel(nearest_summary, args.figure_dir)
    )
    pad_png_to_match_height(
        matched_ad_fraction_png_path,
        args.figure_dir / "psd_centroid_panel_a_umap_45hz.png",
    )
    umap_ad_fraction_png_path, umap_ad_fraction_pdf_path = plot_umap_with_ad_fraction_bars(
        table, embedding, nearest_summary, args.figure_dir
    )
    combined_png_path, combined_pdf_path = plot_embedding_and_closeness(
        table, embedding, args.figure_dir
    )
    umap_3d_png_path, umap_3d_pdf_path = plot_embedding_3d(
        table, embedding_3d, args.figure_dir
    )
    umap_3d_html_path = plot_embedding_3d_interactive(
        table, embedding_3d, args.figure_dir
    )

    print(f"Saved {png_path}")
    print(f"Saved {triptych_png_path}")
    print(f"Saved {centroid_png_path}")
    print(f"Saved {tsne_centroid_png_path}")
    print(f"Saved {centroid_table_png_path}")
    print(f"Saved {umap_table_png_path}")
    print(f"Saved {ad_fraction_png_path}")
    print(f"Saved {matched_ad_fraction_png_path}")
    print(f"Saved {umap_ad_fraction_png_path}")
    print(f"Saved {combined_png_path}")
    print(f"Saved {umap_3d_png_path}")
    print(f"Saved {umap_3d_html_path}")
    print(f"Saved {subject_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {nearest_summary_path}")
    print(f"Saved {tests_path}")
    print(f"Saved {tex_path}")
    print(summary.to_string(index=False))
    print(nearest_summary.to_string(index=False))
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
