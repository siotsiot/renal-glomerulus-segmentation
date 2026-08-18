import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unext_train import SegmentationDataset, make_model, normalize_ext


def parse_args():
    parser = argparse.ArgumentParser(description="Threshold sweep evaluation for trained UNext 5-fold checkpoints.")
    parser.add_argument("--model_root", default="models/glomerulus_UNext_woDS", help="Directory containing fold_1...fold_N.")
    parser.add_argument("--img_dir", default=None, help="Override image directory. Defaults to checkpoint config.")
    parser.add_argument("--mask_dir", default=None, help="Override mask directory. Defaults to checkpoint config.")
    parser.add_argument("--img_ext", default=None, help="Override image extension. Defaults to checkpoint config.")
    parser.add_argument("--mask_ext", default=None, help="Override mask extension. Defaults to checkpoint config.")
    parser.add_argument("--folds", default=5, type=int, help="Number of fold directories to evaluate.")
    parser.add_argument("--batch_size", "-b", default=4, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--threshold_start", default=0.20, type=float)
    parser.add_argument("--threshold_end", default=0.70, type=float)
    parser.add_argument("--threshold_step", default=0.05, type=float)
    parser.add_argument("--eps", default=1e-7, type=float)
    return parser.parse_args()


def read_ids(csv_path):
    ids = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "id" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain an 'id' column.")
        for row in reader:
            if row["id"]:
                ids.append(row["id"])
    return ids


def thresholds_from_args(args):
    count = int(round((args.threshold_end - args.threshold_start) / args.threshold_step)) + 1
    values = [args.threshold_start + i * args.threshold_step for i in range(count)]
    return [round(value, 10) for value in values if value <= args.threshold_end + 1e-9]


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    if "config" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain config: {checkpoint_path}")
    return checkpoint


def config_for_eval(checkpoint_config, args):
    config = dict(checkpoint_config)
    if args.img_dir is not None:
        config["img_dir"] = args.img_dir
    if args.mask_dir is not None:
        config["mask_dir"] = args.mask_dir
    if args.img_ext is not None:
        config["img_ext"] = args.img_ext
    if args.mask_ext is not None:
        config["mask_ext"] = args.mask_ext

    config["img_ext"] = normalize_ext(config["img_ext"])
    config["mask_ext"] = normalize_ext(config["mask_ext"])
    config.setdefault("num_classes", 1)
    config.setdefault("input_channels", 3)
    config.setdefault("deep_supervision", False)
    config.setdefault("input_h", config.get("img_size", 800))
    config.setdefault("input_w", config.get("img_size", 800))
    return config


def update_binary_stats(stats, logits, targets, thresholds):
    probs = torch.sigmoid(logits).detach().cpu()
    targets = (targets.detach().cpu() > 0.5)

    if probs.ndim == 3:
        probs = probs.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    for threshold in thresholds:
        pred = probs > threshold
        tp = torch.logical_and(pred, targets).sum().item()
        fp = torch.logical_and(pred, torch.logical_not(targets)).sum().item()
        fn = torch.logical_and(torch.logical_not(pred), targets).sum().item()
        stats[threshold]["tp"] += float(tp)
        stats[threshold]["fp"] += float(fp)
        stats[threshold]["fn"] += float(fn)


def stats_to_metrics(stats, eps):
    rows = []
    for threshold, counts in stats.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        rows.append(
            OrderedDict(
                [
                    ("threshold", threshold),
                    ("dice", dice),
                    ("iou", iou),
                    ("precision", precision),
                    ("recall", recall),
                    ("tp", tp),
                    ("fp", fp),
                    ("fn", fn),
                ]
            )
        )
    return rows


def write_rows(csv_path, rows):
    if not rows:
        raise ValueError(f"No rows to write for {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_fold(args, fold_index, thresholds, device):
    fold_dir = os.path.join(args.model_root, f"fold_{fold_index}")
    checkpoint_path = os.path.join(fold_dir, "best_checkpoint.pth")
    val_ids_path = os.path.join(fold_dir, "val_ids.csv")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(val_ids_path):
        raise FileNotFoundError(f"val_ids.csv not found: {val_ids_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    config = config_for_eval(checkpoint["config"], args)
    val_ids = read_ids(val_ids_path)

    dataset = SegmentationDataset(
        img_ids=val_ids,
        img_dir=config["img_dir"],
        mask_dir=config["mask_dir"],
        img_ext=config["img_ext"],
        mask_ext=config["mask_ext"],
        num_classes=config["num_classes"],
        input_h=config["input_h"],
        input_w=config["input_w"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    model = make_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = {threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for threshold in thresholds}
    with torch.no_grad():
        for images, targets, _ in tqdm(loader, desc=f"fold_{fold_index}", leave=False):
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[-1]
            update_binary_stats(stats, logits, targets, thresholds)

    rows = stats_to_metrics(stats, args.eps)
    for row in rows:
        row["fold"] = fold_index
        row.move_to_end("fold", last=False)

    fold_csv_path = os.path.join(args.model_root, f"threshold_sweep_fold_{fold_index}.csv")
    write_rows(fold_csv_path, rows)

    best_row = max(rows, key=lambda row: row["dice"])
    return rows, OrderedDict(
        [
            ("fold", fold_index),
            ("best_threshold", best_row["threshold"]),
            ("best_dice", best_row["dice"]),
            ("best_iou", best_row["iou"]),
            ("best_precision", best_row["precision"]),
            ("best_recall", best_row["recall"]),
            ("checkpoint_path", checkpoint_path),
            ("val_count", len(val_ids)),
        ]
    )


def build_mean_std_rows(all_rows, thresholds):
    rows = []
    for threshold in thresholds:
        threshold_rows = [row for row in all_rows if row["threshold"] == threshold]
        out = OrderedDict([("threshold", threshold)])
        for metric in ["dice", "iou", "precision", "recall"]:
            values = np.array([float(row[metric]) for row in threshold_rows], dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(out)
    return rows


def main():
    args = parse_args()
    thresholds = thresholds_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows = []
    best_rows = []
    for fold_index in range(1, args.folds + 1):
        fold_rows, best_row = evaluate_fold(args, fold_index, thresholds, device)
        all_rows.extend(fold_rows)
        best_rows.append(best_row)
        print(
            f"fold_{fold_index}: best_threshold={best_row['best_threshold']:.2f}, "
            f"best_dice={best_row['best_dice']:.6f}"
        )

    write_rows(os.path.join(args.model_root, "threshold_best_by_fold.csv"), best_rows)
    mean_std_rows = build_mean_std_rows(all_rows, thresholds)
    write_rows(os.path.join(args.model_root, "threshold_sweep_mean_std.csv"), mean_std_rows)
    print(f"saved: {os.path.join(args.model_root, 'threshold_best_by_fold.csv')}")
    print(f"saved: {os.path.join(args.model_root, 'threshold_sweep_mean_std.csv')}")


if __name__ == "__main__":
    main()
