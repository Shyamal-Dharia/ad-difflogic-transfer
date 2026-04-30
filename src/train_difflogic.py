import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from difflogic_model import get_logic_connections, make_difflogic_model
from train_helpers import (
    ROOT_DIR,
    encode_fold_for_difflogic,
    load_alz_c_vs_a_bandpower_dataset,
    load_pearl_bandpower_dataset,
    make_subject_stratified_folds,
    stack_subjects,
)


OUTPUT_DIR = ROOT_DIR / "outputs/difflogic"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_fold_list(folds):
    if folds == "all":
        return list(range(1, 11))

    return [int(fold) for fold in folds.split(",")]


def parse_exclude_channels(exclude_channels):
    if exclude_channels == "":
        return []

    return [channel.strip() for channel in exclude_channels.split(",") if channel.strip()]


def get_run_name(args):
    if args.output_name is not None:
        return args.output_name

    return args.model_size


def shuffle_alz_subject_labels(alz_subjects, alz_table, seed):
    rng = np.random.default_rng(seed)
    label_table = alz_table[["participant_id", "label", "group"]].copy()
    shuffled_index = rng.permutation(label_table.index.to_numpy())
    label_table["label"] = label_table.loc[shuffled_index, "label"].to_numpy()
    label_table["group"] = label_table.loc[shuffled_index, "group"].to_numpy()

    shuffled_labels = label_table.set_index("participant_id")

    for subject in alz_subjects:
        participant_id = subject["participant_id"]
        subject["label"] = int(shuffled_labels.loc[participant_id, "label"])
        subject["group"] = shuffled_labels.loc[participant_id, "group"]

    alz_table = alz_table.copy()
    alz_table["label"] = alz_table["participant_id"].map(shuffled_labels["label"]).astype(int)
    alz_table["group"] = alz_table["participant_id"].map(shuffled_labels["group"])
    return alz_subjects, alz_table


def make_loader(x, y, batch_size, shuffle):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def class_weights(y_train, device):
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch_index, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        total_examples += len(x)

        if max_batches is not None and batch_index >= max_batches:
            break

    return total_loss / total_examples


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(x)
            total_examples += len(x)

    return total_loss / total_examples


def predict_probabilities(model, x, batch_size, device):
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    probabilities = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probabilities.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    return np.concatenate(probabilities, axis=0)


def make_epoch_predictions(dataset, split, fold, seed, probabilities, y_true, subject_ids, subject_table):
    metadata = subject_table.set_index("participant_id")
    rows = []

    for index, subject_id in enumerate(subject_ids):
        subject = metadata.loc[subject_id]
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "seed": seed,
                "fold": fold,
                "participant_id": subject_id,
                "group": subject["group"],
                "true_label": int(y_true[index]),
                "p_control": probabilities[index, 0],
                "p_ad": probabilities[index, 1],
                "pred_label": int(probabilities[index, 1] >= 0.5),
            }
        )

    return pd.DataFrame(rows)


def make_subject_predictions(epoch_predictions):
    subject_predictions = (
        epoch_predictions
        .groupby(["dataset", "split", "seed", "fold", "participant_id", "group", "true_label"])
        .agg(
            n_epochs=("p_ad", "count"),
            mean_p_control=("p_control", "mean"),
            mean_p_ad=("p_ad", "mean"),
            fraction_epochs_predicted_ad=("pred_label", "mean"),
        )
        .reset_index()
    )
    subject_predictions["pred_label"] = (subject_predictions["mean_p_ad"] >= 0.5).astype(int)
    return subject_predictions


def compute_binary_metrics(subject_predictions):
    y_true = subject_predictions["true_label"].to_numpy()
    y_pred = subject_predictions["pred_label"].to_numpy()
    y_score = subject_predictions["mean_p_ad"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_score),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def save_encoder(encoder, output_path):
    np.savez_compressed(
        output_path,
        feature_min=encoder["feature_min"],
        feature_max=encoder["feature_max"],
        thresholds=encoder["thresholds"],
        n_bins=np.int64(encoder["n_bins"]),
    )


def train_fold(args, fold, alz_subjects, alz_table, pearl_subjects, pearl_table, folds):
    run_dir = OUTPUT_DIR / get_run_name(args) / "seed_{:03d}".format(args.seed)
    fold_dir = run_dir / "fold_{:02d}".format(fold["fold"])
    fold_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, train_subject_ids = stack_subjects(alz_subjects, fold["train_subject_ids"])
    x_val, y_val, val_subject_ids = stack_subjects(alz_subjects, fold["val_subject_ids"])
    x_test, y_test, test_subject_ids = stack_subjects(alz_subjects, fold["test_subject_ids"])
    x_pearl, y_pearl, pearl_subject_ids = stack_subjects(
        pearl_subjects,
        pearl_table["participant_id"].to_numpy(),
    )

    encoder, x_train, x_val, x_test, x_pearl = encode_fold_for_difflogic(
        x_train,
        x_val,
        x_test,
        x_pearl,
        n_bins=args.thermometer_bins,
    )
    save_encoder(encoder, fold_dir / "thermometer_encoder.npz")

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)

    input_dim = x_train.shape[1]
    model = make_difflogic_model(
        args.model_size,
        tau=args.tau,
        device=args.device,
        input_dim=input_dim,
    )
    model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(y_train, args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = np.inf
    best_epoch = 0
    bad_epochs = 0
    history_rows = []
    checkpoint_path = fold_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            args.device,
            max_batches=args.max_train_batches,
        )
        val_loss = evaluate_loss(model, val_loader, criterion, args.device)

        history_rows.append(
            {
                "fold": fold["fold"],
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            "fold {} epoch {} train_loss {:.4f} val_loss {:.4f}".format(
                fold["fold"],
                epoch,
                train_loss,
                val_loss,
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "model_size": args.model_size,
                    "run_name": get_run_name(args),
                    "input_dim": input_dim,
                    "logic_connections": get_logic_connections(model),
                    "exclude_channels": args.exclude_channels,
                    "shuffle_alz_labels": args.shuffle_alz_labels,
                    "thermometer_bins": args.thermometer_bins,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            break

    pd.DataFrame(history_rows).to_csv(fold_dir / "training_history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_probabilities = predict_probabilities(model, x_test, args.batch_size, args.device)
    pearl_probabilities = predict_probabilities(model, x_pearl, args.batch_size, args.device)

    test_epoch_predictions = make_epoch_predictions(
        "ALZ_FTD",
        "test",
        fold["fold"],
        args.seed,
        test_probabilities,
        y_test,
        test_subject_ids,
        alz_table,
    )
    pearl_epoch_predictions = make_epoch_predictions(
        "PEARL",
        "transfer",
        fold["fold"],
        args.seed,
        pearl_probabilities,
        y_pearl,
        pearl_subject_ids,
        pearl_table,
    )

    test_subject_predictions = make_subject_predictions(test_epoch_predictions)
    pearl_subject_predictions = make_subject_predictions(pearl_epoch_predictions)
    metrics = compute_binary_metrics(test_subject_predictions)
    metrics["seed"] = args.seed
    metrics["fold"] = fold["fold"]
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = best_val_loss
    metrics["input_dim"] = input_dim
    metrics["exclude_channels"] = args.exclude_channels
    metrics["shuffle_alz_labels"] = args.shuffle_alz_labels

    test_epoch_predictions.to_csv(fold_dir / "alz_test_epoch_predictions.csv", index=False)
    test_subject_predictions.to_csv(fold_dir / "alz_test_subject_predictions.csv", index=False)
    pearl_epoch_predictions.to_csv(fold_dir / "pearl_epoch_predictions.csv", index=False)
    pearl_subject_predictions.to_csv(fold_dir / "pearl_subject_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(fold_dir / "alz_test_metrics.csv", index=False)

    print(
        "fold {} best_epoch {} test_bal_acc {:.3f} test_auroc {:.3f}".format(
            fold["fold"],
            best_epoch,
            metrics["balanced_accuracy"],
            metrics["auroc"],
        )
    )

    return metrics, test_subject_predictions, pearl_subject_predictions


def make_pearl_ensemble(pearl_predictions):
    ensemble = (
        pearl_predictions
        .groupby(["participant_id", "group", "true_label"])
        .agg(
            n_folds=("fold", "count"),
            mean_p_control=("mean_p_control", "mean"),
            mean_p_ad=("mean_p_ad", "mean"),
            std_p_ad=("mean_p_ad", "std"),
            mean_fraction_epochs_predicted_ad=("fraction_epochs_predicted_ad", "mean"),
        )
        .reset_index()
    )
    ensemble["pred_label"] = (ensemble["mean_p_ad"] >= 0.5).astype(int)
    return ensemble


def run_training(args):
    set_seed(args.seed)
    run_dir = OUTPUT_DIR / get_run_name(args) / "seed_{:03d}".format(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    exclude_channels = parse_exclude_channels(args.exclude_channels)
    alz_subjects, alz_table = load_alz_c_vs_a_bandpower_dataset(
        exclude_channels=exclude_channels,
    )
    pearl_subjects, pearl_table = load_pearl_bandpower_dataset(
        exclude_channels=exclude_channels,
    )

    if args.shuffle_alz_labels:
        alz_subjects, alz_table = shuffle_alz_subject_labels(
            alz_subjects,
            alz_table,
            seed=args.seed,
        )

    run_config = {
        "model_size": args.model_size,
        "output_name": get_run_name(args),
        "seed": args.seed,
        "exclude_channels": exclude_channels,
        "shuffle_alz_labels": args.shuffle_alz_labels,
        "epochs": args.epochs,
        "patience": args.patience,
        "thermometer_bins": args.thermometer_bins,
    }
    with open(run_dir / "run_config.json", "w") as file:
        json.dump(run_config, file, indent=2)

    folds = make_subject_stratified_folds(alz_table, random_state=args.seed)
    selected_folds = set(parse_fold_list(args.folds))

    all_metrics = []
    all_test_predictions = []
    all_pearl_predictions = []

    for fold in folds:
        if fold["fold"] not in selected_folds:
            continue

        metrics, test_predictions, pearl_predictions = train_fold(
            args,
            fold,
            alz_subjects,
            alz_table,
            pearl_subjects,
            pearl_table,
            folds,
        )
        all_metrics.append(metrics)
        all_test_predictions.append(test_predictions)
        all_pearl_predictions.append(pearl_predictions)

    metrics_table = pd.DataFrame(all_metrics)
    test_predictions = pd.concat(all_test_predictions, ignore_index=True)
    pearl_predictions = pd.concat(all_pearl_predictions, ignore_index=True)
    pearl_ensemble = make_pearl_ensemble(pearl_predictions)

    metrics_table.to_csv(run_dir / "alz_test_metrics_all_folds.csv", index=False)
    test_predictions.to_csv(run_dir / "alz_test_subject_predictions_all_folds.csv", index=False)
    pearl_predictions.to_csv(run_dir / "pearl_subject_predictions_all_folds.csv", index=False)
    pearl_ensemble.to_csv(run_dir / "pearl_subject_predictions_ensemble.csv", index=False)

    print("\nSaved fold outputs to {}".format(run_dir))
    print("Excluded channels: {}".format(args.exclude_channels or "none"))
    print("Shuffled ALZ_FTD labels: {}".format(args.shuffle_alz_labels))
    print(metrics_table[["fold", "balanced_accuracy", "auroc", "best_epoch", "best_val_loss"]])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--exclude-channels", default="")
    parser.add_argument("--shuffle-alz-labels", action="store_true")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=30.0)
    parser.add_argument("--thermometer-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
