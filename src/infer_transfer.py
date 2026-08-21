import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from difflogic_model import set_logic_connections
from train_difflogic import (
    DIFFLOGIC_MODEL_KINDS,
    make_epoch_predictions,
    make_model,
    make_subject_predictions,
    make_target_ensemble,
    parse_exclude_channels,
    predict_probabilities,
)
from train_helpers import (
    apply_minmax_scaler,
    apply_thermometer_encoder,
    load_target_dataset,
    stack_subjects,
)


def load_transform(path):
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def transform_target(x, fold_dir, preprocessing):
    if preprocessing == "thermometer":
        encoder = load_transform(fold_dir / "thermometer_encoder.npz")
        return apply_thermometer_encoder(x, encoder)

    scaler = load_transform(fold_dir / "minmax_scaler.npz")
    return apply_minmax_scaler(x, scaler)


def load_trained_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = SimpleNamespace(
        model_kind=checkpoint["model_kind"],
        model_size=checkpoint["model_size"],
        tau=checkpoint.get("tau", 30.0),
        device=device,
        dropout=checkpoint.get("dropout", 0.2),
        target_parameters=checkpoint.get("target_parameters", 250_000),
    )
    model = make_model(args, checkpoint["input_shape"])
    if args.model_kind in DIFFLOGIC_MODEL_KINDS:
        set_logic_connections(model, checkpoint["logic_connections"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def selected_number(path, selected):
    return selected is None or int(path.name.split("_")[1]) in selected


def infer_run(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    first_checkpoint_path = next(args.run_dir.glob("seed_*/fold_*/best_model.pt"), None)
    if first_checkpoint_path is None:
        raise FileNotFoundError(f"No fold checkpoints found in {args.run_dir}")
    first_checkpoint = torch.load(
        first_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    exclude_channels = parse_exclude_channels(first_checkpoint.get("exclude_channels", ""))
    feature_kind = first_checkpoint.get("feature_kind", "psd")
    hfd_name = first_checkpoint.get("hfd_name", "kmax_16")
    targets = {}

    for dataset_name in args.datasets:
        subjects, table = load_target_dataset(
            dataset_name,
            feature_kind=feature_kind,
            hfd_name=hfd_name,
            exclude_channels=exclude_channels,
        )
        if not subjects:
            raise ValueError(f"No {feature_kind.upper()} features found for {dataset_name}")
        targets[dataset_name] = (*stack_subjects(subjects, table["participant_id"]), table)

    for seed_dir in sorted(args.run_dir.glob("seed_*")):
        if not selected_number(seed_dir, args.seeds):
            continue
        seed = int(seed_dir.name.split("_")[1])
        seed_predictions = {dataset_name: [] for dataset_name in args.datasets}

        for fold_dir in sorted(seed_dir.glob("fold_*")):
            if not selected_number(fold_dir, args.folds):
                continue
            fold = int(fold_dir.name.split("_")[1])
            model, checkpoint = load_trained_model(fold_dir / "best_model.pt", device)

            for dataset_name, (x, y, subject_ids, table) in targets.items():
                x_transformed = transform_target(x, fold_dir, checkpoint["preprocessing"])
                probabilities = predict_probabilities(
                    model,
                    x_transformed,
                    args.batch_size,
                    device,
                )
                epoch_predictions = make_epoch_predictions(
                    "transfer",
                    fold,
                    seed,
                    probabilities,
                    y,
                    subject_ids,
                    table,
                )
                subject_predictions = make_subject_predictions(epoch_predictions)
                output_name = dataset_name.lower()
                epoch_predictions.to_csv(
                    fold_dir / f"{output_name}_epoch_predictions.csv",
                    index=False,
                )
                subject_predictions.to_csv(
                    fold_dir / f"{output_name}_subject_predictions.csv",
                    index=False,
                )
                seed_predictions[dataset_name].append(subject_predictions)

        for dataset_name, predictions in seed_predictions.items():
            if not predictions:
                continue
            output_name = dataset_name.lower()
            predictions = pd.concat(predictions, ignore_index=True)
            predictions.to_csv(
                seed_dir / f"{output_name}_subject_predictions_all_folds.csv",
                index=False,
            )
            make_target_ensemble(predictions).to_csv(
                seed_dir / f"{output_name}_subject_predictions_ensemble.csv",
                index=False,
            )
            print(f"{seed_dir.name} {dataset_name}: {len(predictions)} fold predictions")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["PEARL", "DS007427"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    infer_run(parse_args())
