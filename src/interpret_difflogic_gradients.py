import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from difflogic_model import make_difflogic_model, set_logic_connections
from train_helpers import (
    BANDS,
    ROOT_DIR,
    apply_thermometer_encoder,
    load_pearl_bandpower_dataset,
    stack_subjects,
)
from train_difflogic import parse_exclude_channels


OUTPUT_DIR = ROOT_DIR / "outputs/difflogic"


def parse_number_list(value):
    if value == "":
        return None

    return {int(item.strip()) for item in value.split(",") if item.strip()}


def resolve_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"

    return device


def load_encoder(path):
    encoder = np.load(path)
    return {
        "feature_min": encoder["feature_min"],
        "feature_max": encoder["feature_max"],
        "thresholds": encoder["thresholds"],
        "n_bins": int(encoder["n_bins"]),
    }


def load_model(checkpoint_path, model_size, input_dim, device, logic_mode):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "logic_connections" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} does not contain logic_connections. "
            "Rerun training with the updated train_difflogic.py before interpretation."
        )

    implementation = "python" if logic_mode == "hard" else None
    model = make_difflogic_model(
        model_size,
        input_dim=input_dim,
        device=device,
        implementation=implementation,
    )
    set_logic_connections(model, checkpoint["logic_connections"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if logic_mode == "soft":
        model.train()
    else:
        model.eval()

    return model


def gradient_x_input_relevance(model, x_batch):
    x_batch = x_batch.detach().clone().requires_grad_(True)
    logits = model(x_batch)
    ad_score = logits[:, 1] - logits[:, 0]
    model.zero_grad()
    ad_score.sum().backward()
    return x_batch.grad * x_batch


def integrated_gradient_relevance(model, x_batch, steps):
    baseline = torch.zeros_like(x_batch)
    total_gradient = torch.zeros_like(x_batch)

    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        x_step = (baseline + alpha * (x_batch - baseline)).detach().requires_grad_(True)
        logits = model(x_step)
        ad_score = logits[:, 1] - logits[:, 0]
        gradient = torch.autograd.grad(ad_score.sum(), x_step)[0]
        total_gradient += gradient

    return (x_batch - baseline) * total_gradient / steps


def compute_epoch_relevance(model, x_encoded, n_channels, n_bands, n_bins, args):
    signed_relevance = []
    absolute_relevance = []

    for start in range(0, len(x_encoded), args.batch_size):
        end = start + args.batch_size
        x_batch = torch.tensor(x_encoded[start:end], dtype=torch.float32, device=args.device)

        if args.method == "integrated_gradients":
            relevance = integrated_gradient_relevance(model, x_batch, args.ig_steps)
        else:
            relevance = gradient_x_input_relevance(model, x_batch)

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
    band_names = list(BANDS)
    rows = []

    for participant_id in np.unique(subject_ids):
        subject_mask = subject_ids == participant_id
        signed_subject = signed_relevance[subject_mask].mean(axis=0)
        absolute_subject = absolute_relevance[subject_mask].mean(axis=0)

        for channel_index, channel in enumerate(channel_names):
            for band_index, band in enumerate(band_names):
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


def summarize_relevance(subject_relevance, output_dir):
    seed_average = (
        subject_relevance
        .groupby(["participant_id", "group", "channel", "band"])
        .agg(
            signed_relevance=("signed_relevance", "mean"),
            absolute_relevance=("absolute_relevance", "mean"),
        )
        .reset_index()
    )

    group_summary = (
        seed_average
        .groupby(["group", "channel", "band"])
        .agg(
            mean_signed_relevance=("signed_relevance", "mean"),
            median_signed_relevance=("signed_relevance", "median"),
            mean_absolute_relevance=("absolute_relevance", "mean"),
            median_absolute_relevance=("absolute_relevance", "median"),
        )
        .reset_index()
    )

    signed_pivot = group_summary.pivot(
        index=["channel", "band"],
        columns="group",
        values="mean_signed_relevance",
    ).reset_index()
    signed_pivot["A+P+_minus_N"] = signed_pivot["A+P+"] - signed_pivot["N"]
    signed_pivot["A+P+_minus_A+P-"] = signed_pivot["A+P+"] - signed_pivot["A+P-"]

    absolute_pivot = group_summary.pivot(
        index=["channel", "band"],
        columns="group",
        values="mean_absolute_relevance",
    ).reset_index()
    absolute_pivot["A+P+_minus_N"] = absolute_pivot["A+P+"] - absolute_pivot["N"]
    absolute_pivot["A+P+_minus_A+P-"] = absolute_pivot["A+P+"] - absolute_pivot["A+P-"]

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_relevance.to_csv(output_dir / "pearl_gradient_relevance_all_models.csv", index=False)
    seed_average.to_csv(output_dir / "pearl_gradient_relevance_seed_average.csv", index=False)
    group_summary.to_csv(output_dir / "pearl_gradient_relevance_group_summary.csv", index=False)
    signed_pivot.to_csv(output_dir / "pearl_signed_relevance_group_contrasts.csv", index=False)
    absolute_pivot.to_csv(output_dir / "pearl_absolute_relevance_group_contrasts.csv", index=False)

    top_absolute = group_summary.sort_values("mean_absolute_relevance", ascending=False).head(20)
    top_app_n = signed_pivot.sort_values("A+P+_minus_N", ascending=False).head(20)
    top_app_apm = signed_pivot.sort_values("A+P+_minus_A+P-", ascending=False).head(20)

    print("\nTop absolute relevance features")
    print(top_absolute.to_string(index=False))
    print("\nTop signed A+P+ minus N relevance features")
    print(top_app_n.to_string(index=False))
    print("\nTop signed A+P+ minus A+P- relevance features")
    print(top_app_apm.to_string(index=False))


def interpret_run(args):
    args.device = resolve_device(args.device)
    if args.logic_mode == "hard":
        args.device = "cpu"

    run_dir = OUTPUT_DIR / args.run_name
    output_dir = run_dir / "interpretation" / f"{args.logic_mode}_{args.method}"
    selected_seeds = parse_number_list(args.seeds)
    selected_folds = parse_number_list(args.folds)
    exclude_channels = parse_exclude_channels(args.exclude_channels)
    pearl_subjects, pearl_table = load_pearl_bandpower_dataset(exclude_channels=exclude_channels)
    x_pearl, _, pearl_subject_ids = stack_subjects(
        pearl_subjects,
        pearl_table["participant_id"].to_numpy(),
    )
    channel_names = pearl_subjects[0]["channel_names"]
    n_channels = x_pearl.shape[1]
    n_bands = x_pearl.shape[2]
    tables = []

    for seed_dir in sorted(run_dir.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        if selected_seeds is not None and seed not in selected_seeds:
            continue

        for fold_dir in sorted(seed_dir.glob("fold_*")):
            fold = int(fold_dir.name.split("_")[1])
            if selected_folds is not None and fold not in selected_folds:
                continue

            encoder = load_encoder(fold_dir / "thermometer_encoder.npz")
            x_encoded = apply_thermometer_encoder(x_pearl, encoder)
            model = load_model(
                fold_dir / "best_model.pt",
                args.model_size,
                input_dim=x_encoded.shape[1],
                device=args.device,
                logic_mode=args.logic_mode,
            )
            signed_relevance, absolute_relevance = compute_epoch_relevance(
                model,
                x_encoded,
                n_channels,
                n_bands,
                encoder["n_bins"],
                args,
            )
            table = make_subject_relevance_table(
                signed_relevance,
                absolute_relevance,
                pearl_subject_ids,
                pearl_table,
                channel_names,
                seed,
                fold,
            )
            tables.append(table)
            print(f"interpreted seed {seed} fold {fold}")

    if len(tables) == 0:
        raise ValueError("No seed/fold outputs matched the requested filters.")

    subject_relevance = pd.concat(tables, ignore_index=True)
    summarize_relevance(subject_relevance, output_dir)
    print(f"\nSaved interpretation outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="medium")
    parser.add_argument("--model-size", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--exclude-channels", default="")
    parser.add_argument("--method", default="grad_x_input", choices=["grad_x_input", "integrated_gradients"])
    parser.add_argument("--logic-mode", default="soft", choices=["soft", "hard"])
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--folds", default="")
    return parser.parse_args()


if __name__ == "__main__":
    interpret_run(parse_args())
