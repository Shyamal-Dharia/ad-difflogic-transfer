"""Nearest-centroid transfer restricted to single bands or scalp regions.

Repeats the Section 3.9.2 centroid analysis on feature subsets, to show which
frequency bands and which scalp regions carry the genetic-risk pattern.

Clinical performance is leave-one-out within the clinical cohort, so the
participant being classified never contributes to its own centroid. Genetic-risk
participants are always held out entirely: centroids come only from clinical
CN/AD data, as in the primary analysis.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from train_helpers import BANDS, ROOT_DIR, load_source_dataset, load_target_dataset

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]
BAND_NAMES = list(BANDS)
REGIONS = {
    "Frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "Central": ["C3", "Cz", "C4"],
    "Parietal": ["P3", "Pz", "P4"],
    "Occipital": ["O1", "O2"],
    "Frontotemporal (F7/F8)": ["F7", "F8"],
}
GROUPS = ["N", "A+P-", "A+P+"]
CONTRASTS = [("A+P+", "N"), ("A+P+", "A+P-")]


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def subject_matrix(subjects):
    return np.stack([s["x"].mean(axis=0) for s in subjects])  # (n, 15, 5)


def evaluate(clinical, clinical_labels, genetic, genetic_groups, ch_idx, band_idx):
    c = clinical[:, ch_idx][:, :, band_idx].reshape(len(clinical), -1)
    g = genetic[:, ch_idx][:, :, band_idx].reshape(len(genetic), -1)
    mean, sd = c.mean(0), c.std(0) + 1e-12
    c, g = (c - mean) / sd, (g - mean) / sd

    # leave-one-out clinical accuracy
    correct = 0
    for i in range(len(c)):
        keep = np.ones(len(c), bool); keep[i] = False
        cn = c[keep & (clinical_labels == 0)].mean(0)
        ad = c[keep & (clinical_labels == 1)].mean(0)
        pred = int(np.linalg.norm(c[i] - ad) < np.linalg.norm(c[i] - cn))
        correct += pred == clinical_labels[i]
    # sensitivity/specificity balanced
    accs = []
    for lab in (0, 1):
        idx = clinical_labels == lab
        preds = []
        for i in np.where(idx)[0]:
            keep = np.ones(len(c), bool); keep[i] = False
            cn = c[keep & (clinical_labels == 0)].mean(0)
            ad = c[keep & (clinical_labels == 1)].mean(0)
            preds.append(int(np.linalg.norm(c[i] - ad) < np.linalg.norm(c[i] - cn)))
        accs.append(np.mean(np.array(preds) == lab))
    bal_acc = float(np.mean(accs))

    cn_full, ad_full = c[clinical_labels == 0].mean(0), c[clinical_labels == 1].mean(0)
    ad_nearest = np.linalg.norm(g - ad_full, axis=1) < np.linalg.norm(g - cn_full, axis=1)

    counts = {gr: [int(ad_nearest[genetic_groups == gr].sum()), int((genetic_groups == gr).sum())]
              for gr in GROUPS}
    raw = [fisher_exact([[counts[a][0], counts[a][1] - counts[a][0]],
                         [counts[b][0], counts[b][1] - counts[b][0]]]).pvalue for a, b in CONTRASTS]
    return bal_acc, counts, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT_DIR / "outputs/revision_2026/centroid_subsets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    src, src_tab = load_source_dataset(dataset_name="ALZ_FTD", clinical_task="cn_vs_ad",
                                       feature_kind="psd")
    tgt, tgt_tab = load_target_dataset("PEARL", feature_kind="psd")
    clinical = subject_matrix(src)
    clinical_labels = src_tab["label"].to_numpy()
    genetic = subject_matrix(tgt)
    genetic_groups = tgt_tab["group"].replace({"A+P−": "A+P-"}).to_numpy()

    rows = []
    all_idx = list(range(15))
    for band in BAND_NAMES + ["all bands"]:
        b_idx = all_idx[:5] if band == "all bands" else [BAND_NAMES.index(band)]
        b_idx = list(range(5)) if band == "all bands" else b_idx
        acc, counts, raw = evaluate(clinical, clinical_labels, genetic, genetic_groups, all_idx, b_idx)
        rows.append(dict(subset=f"Band: {band}", n_features=15 * len(b_idx),
                         clinical_bal_acc=acc,
                         **{f"{g}_prop": counts[g][0] / counts[g][1] for g in GROUPS},
                         **{f"{g}_n": f"{counts[g][0]}/{counts[g][1]}" for g in GROUPS},
                         p_APP_vs_N=raw[0], p_APP_vs_APM=raw[1]))
    for region, chans in REGIONS.items():
        c_idx = [CHANNELS.index(c) for c in chans]
        acc, counts, raw = evaluate(clinical, clinical_labels, genetic, genetic_groups,
                                    c_idx, list(range(5)))
        rows.append(dict(subset=f"Region: {region}", n_features=len(c_idx) * 5,
                         clinical_bal_acc=acc,
                         **{f"{g}_prop": counts[g][0] / counts[g][1] for g in GROUPS},
                         **{f"{g}_n": f"{counts[g][0]}/{counts[g][1]}" for g in GROUPS},
                         p_APP_vs_N=raw[0], p_APP_vs_APM=raw[1]))

    frame = pd.DataFrame(rows)
    for col in ["p_APP_vs_N", "p_APP_vs_APM"]:
        frame[col + "_holm"] = holm(frame[col].to_numpy())
    frame.to_csv(args.output_dir / "centroid_subset_results.csv", index=False)
    show = ["subset", "n_features", "clinical_bal_acc", "N_n", "A+P-_n", "A+P+_n",
            "A+P+_prop", "p_APP_vs_N", "p_APP_vs_APM"]
    print(frame[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {args.output_dir/'centroid_subset_results.csv'}")


if __name__ == "__main__":
    main()
