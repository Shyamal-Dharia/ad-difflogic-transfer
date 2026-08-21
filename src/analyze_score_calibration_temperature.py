"""Source calibration, shuffled-label score offset, and temperature sensitivity.

Addresses reviewer 1 major comments 2 and 7 and minor comment 6.

The GroupSum layer divides the class evidence by tau before the softmax, so the
stored epoch probability satisfies p_AD = sigmoid((z_AD - z_CN) / tau). The
relative AD evidence s_AD = z_AD - z_CN can therefore be recovered exactly as
tau * logit(p_AD), and any alternative temperature can be applied without
retraining.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from train_helpers import ROOT_DIR


DIFFLOGIC_DIR = ROOT_DIR / "outputs" / "difflogic_45hz"
TRUE_RUN = "benchmark_250k_psd"
SHUFFLED_RUN = "benchmark_250k_psd_shuffled_alz_labels"
TRAINED_TAU = 30.0
TAU_GRID = [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]
CONTRASTS = [("A+P-", "N"), ("A+P+", "N"), ("A+P+", "A+P-")]
GROUP_ORDER = ["N", "A+P-", "A+P+"]
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def auc_from_groups(target, reference):
    pairwise_differences = target[:, None] - reference[None, :]
    return (
        np.count_nonzero(pairwise_differences > 0)
        + 0.5 * np.count_nonzero(pairwise_differences == 0)
    ) / pairwise_differences.size


def bootstrap_mean_difference_ci(values_a, values_b, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    differences = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample_a = rng.choice(values_a, size=values_a.size, replace=True)
        sample_b = rng.choice(values_b, size=values_b.size, replace=True)
        differences[index] = sample_a.mean() - sample_b.mean()
    return np.quantile(differences, [0.025, 0.975])


def expected_calibration_error(y_true, y_score, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(y_score, edges[1:-1], right=False), 0, n_bins - 1)
    error = 0.0
    max_error = 0.0
    for bin_id in range(n_bins):
        mask = bin_index == bin_id
        if not mask.any():
            continue
        gap = abs(y_true[mask].mean() - y_score[mask].mean())
        error += mask.mean() * gap
        max_error = max(max_error, gap)
    return error, max_error


def reliability_curve(y_true, y_score, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(y_score, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_index == bin_id
        rows.append(
            {
                "bin_low": edges[bin_id],
                "bin_high": edges[bin_id + 1],
                "n_subjects": int(mask.sum()),
                "mean_predicted": float(y_score[mask].mean()) if mask.any() else np.nan,
                "observed_ad_rate": float(y_true[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_source_predictions(run_name):
    frames = []
    pattern = "seed_*/alz_test_subject_predictions_all_folds.csv"
    for path in sorted((DIFFLOGIC_DIR / run_name).glob(pattern)):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No source predictions under {run_name}")
    return pd.concat(frames, ignore_index=True)


def summarize_source_calibration(run_name, label):
    predictions = load_source_predictions(run_name)
    rows = []
    for seed, seed_frame in predictions.groupby("seed"):
        y_true = seed_frame["true_label"].to_numpy()
        y_score = seed_frame["mean_p_ad"].to_numpy()
        ece, mce = expected_calibration_error(y_true, y_score)
        rows.append(
            {
                "model": label,
                "seed": int(seed),
                "n_subjects": int(y_true.size),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_score >= 0.5),
                "auroc": roc_auc_score(y_true, y_score),
                "brier": float(np.mean((y_score - y_true) ** 2)),
                "ece": ece,
                "mce": mce,
                "mean_predicted": float(y_score.mean()),
                "observed_ad_rate": float(y_true.mean()),
                "score_sd_across_subjects": float(y_score.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def load_pearl_epoch_predictions(run_name):
    frames = []
    for path in sorted((DIFFLOGIC_DIR / run_name).glob("seed_*/fold_*/pearl_epoch_predictions.csv")):
        frames.append(pd.read_csv(path, usecols=["seed", "fold", "participant_id", "group", "p_ad"]))
    if not frames:
        raise FileNotFoundError(f"No PEARL epoch predictions under {run_name}")
    return pd.concat(frames, ignore_index=True)


def subject_scores_at_tau(epoch_predictions, tau):
    """Rescale epoch probabilities to a new temperature, then aggregate as in the paper.

    Aggregation order matches the training script: epochs to subject within each
    fold model, folds to a seed ensemble, then seeds to the reported score.
    """
    evidence = TRAINED_TAU * logit(epoch_predictions["p_ad"].to_numpy(dtype=np.float64))
    rescaled = epoch_predictions.assign(p_ad_tau=expit(evidence / tau))
    per_model = (
        rescaled.groupby(["seed", "fold", "participant_id", "group"], as_index=False)["p_ad_tau"]
        .mean()
    )
    per_seed = per_model.groupby(["seed", "participant_id", "group"], as_index=False)["p_ad_tau"].mean()
    return per_seed.groupby(["participant_id", "group"], as_index=False)["p_ad_tau"].mean()


def contrast_rows(subject_scores, score_column, label_fields):
    grouped = {
        group: frame[score_column].to_numpy()
        for group, frame in subject_scores.groupby("group")
    }
    rows = []
    for target_group, reference_group in CONTRASTS:
        target = grouped[target_group]
        reference = grouped[reference_group]
        ci_low, ci_high = bootstrap_mean_difference_ci(target, reference)
        rows.append(
            {
                **label_fields,
                "contrast": f"{target_group} vs {reference_group}",
                "mean_difference": target.mean() - reference.mean(),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ranking_auroc": auc_from_groups(target, reference),
                "p_uncorrected": mannwhitneyu(target, reference, alternative="two-sided").pvalue,
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_adjust(frame["p_uncorrected"].to_numpy())
    return frame


def group_summary_rows(subject_scores, score_column, label_fields):
    rows = []
    for group in GROUP_ORDER:
        values = subject_scores.loc[subject_scores["group"] == group, score_column].to_numpy()
        rows.append(
            {
                **label_fields,
                "group": group,
                "n_subjects": values.size,
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "median": np.median(values),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "outputs" / "revision_2026" / "score_calibration_temperature",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Source calibration")
    calibration = pd.concat(
        [
            summarize_source_calibration(TRUE_RUN, "True labels"),
            summarize_source_calibration(SHUFFLED_RUN, "Shuffled labels"),
        ],
        ignore_index=True,
    )
    calibration.to_csv(args.output_dir / "source_calibration_by_seed.csv", index=False)

    numeric_columns = [column for column in calibration.columns if column not in {"model", "seed"}]
    calibration_summary = calibration.groupby("model")[numeric_columns].agg(["mean", "std"])
    calibration_summary.to_csv(args.output_dir / "source_calibration_summary.csv")
    print(calibration_summary[["balanced_accuracy", "auroc", "brier", "ece", "mean_predicted"]].round(3))

    curves = []
    for run_name, label in [(TRUE_RUN, "True labels"), (SHUFFLED_RUN, "Shuffled labels")]:
        predictions = load_source_predictions(run_name)
        curve = reliability_curve(
            predictions["true_label"].to_numpy(),
            predictions["mean_p_ad"].to_numpy(),
        )
        curve.insert(0, "model", label)
        curves.append(curve)
    pd.concat(curves, ignore_index=True).to_csv(
        args.output_dir / "source_reliability_curve.csv", index=False
    )

    print("\nTemperature sensitivity on the genetic-risk cohort")
    epochs = load_pearl_epoch_predictions(TRUE_RUN)
    tau_summaries = []
    tau_contrasts = []
    for tau in TAU_GRID:
        scores = subject_scores_at_tau(epochs, tau)
        tau_summaries.append(group_summary_rows(scores, "p_ad_tau", {"tau": tau}))
        tau_contrasts.append(contrast_rows(scores, "p_ad_tau", {"tau": tau}))
    pd.concat(tau_summaries, ignore_index=True).to_csv(
        args.output_dir / "tau_group_summary.csv", index=False
    )
    tau_contrast_frame = pd.concat(tau_contrasts, ignore_index=True)
    tau_contrast_frame.to_csv(args.output_dir / "tau_contrasts.csv", index=False)
    print(tau_contrast_frame.round(4).to_string(index=False))

    print("\nTrue versus shuffled transfer scores")
    model_summaries = []
    model_contrasts = []
    for run_name, label in [(TRUE_RUN, "True labels"), (SHUFFLED_RUN, "Shuffled labels")]:
        scores = subject_scores_at_tau(load_pearl_epoch_predictions(run_name), TRAINED_TAU)
        model_summaries.append(group_summary_rows(scores, "p_ad_tau", {"model": label}))
        model_contrasts.append(contrast_rows(scores, "p_ad_tau", {"model": label}))
    pd.concat(model_summaries, ignore_index=True).to_csv(
        args.output_dir / "true_vs_shuffled_group_summary.csv", index=False
    )
    shuffled_frame = pd.concat(model_contrasts, ignore_index=True)
    shuffled_frame.to_csv(args.output_dir / "true_vs_shuffled_contrasts.csv", index=False)
    print(shuffled_frame.round(4).to_string(index=False))

    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
