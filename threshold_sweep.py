import argparse
import csv
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unext_train import (
    ARCHS,
    SegmentationDataset,
    make_model,
    normalize_ext,
    target_to_binary_mask,
)


DEFAULT_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def parse_args():
    parser = argparse.ArgumentParser(description="Threshold sweep evaluation for saved segmentation folds.")
    parser.add_argument("--model_dir", default="models/unet_512_aug_5fold", help="directory containing fold_N folders")
    parser.add_argument("--folds", default=5, type=int, help="number of folds to evaluate")
    parser.add_argument("--checkpoint_name", default="best_checkpoint.pth", help="checkpoint filename inside each fold folder")
    parser.add_argument("--batch_size", default=1, type=int, help="validation batch size")
    parser.add_argument("--num_workers", default=0, type=int, help="DataLoader workers")
    parser.add_argument(
        "--output_dir",
        default="models/unet_512_aug_5fold/threshold_sweep",
        help="directory to write threshold sweep CSV files",
    )
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        nargs="+",
        type=float,
        help="thresholds to evaluate, e.g. --thresholds 0.2 0.25 0.3 0.35 0.4 0.45 0.5",
    )
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def read_id_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if "id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an 'id' column.")
        return [row["id"] for row in reader if row.get("id")]


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return OrderedDict((key.removeprefix("module."), value) for key, value in state_dict.items())


def load_checkpoint_model(config, checkpoint_path, device):
    model = make_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} must contain a 'model_state_dict' entry.")
    state_dict = strip_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_val_loader(config, val_ids, batch_size, num_workers):
    img_ext = normalize_ext(config.get("img_ext", ".png"))
    mask_ext = normalize_ext(config.get("mask_ext", ".png"))
    dataset = SegmentationDataset(
        img_ids=val_ids,
        img_dir=config["img_dir"],
        mask_dir=config["mask_dir"],
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config["num_classes"],
        input_h=config["input_h"],
        input_w=config["input_w"],
        augment=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def sample_metrics_from_masks(pred_mask, target_mask, eps=1e-7):
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)

    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, np.logical_not(target)).sum())
    fn = float(np.logical_and(np.logical_not(pred), target).sum())

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return dice, iou, precision, recall


def get_batch_logits(model, images, deep_supervision):
    outputs = model(images)
    if deep_supervision:
        return outputs[-1]
    return outputs


def evaluate_fold_thresholds(model, val_loader, thresholds, config, device, fold):
    threshold_sample_rows = {threshold: [] for threshold in thresholds}

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc=f"fold_{fold} threshold sweep", leave=False)
        for images, targets, meta in val_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            logits = get_batch_logits(model, images, config.get("deep_supervision", False)).detach().cpu()
            targets = targets.detach().cpu()

            batch_ids = meta["id"]
            for sample_index in range(logits.size(0)):
                sample_id = batch_ids[sample_index] if isinstance(batch_ids, (list, tuple)) else str(batch_ids)
                target_mask = target_to_binary_mask(targets[sample_index])
                probs = torch.sigmoid(logits[sample_index])
                gt_fg_ratio = float(target_mask.mean())

                for threshold in thresholds:
                    output = probs
                    if output.ndim == 3 and output.shape[0] > 1:
                        pred_mask = (torch.argmax(output, dim=0).numpy() > 0).astype(np.uint8)
                    else:
                        if output.ndim == 3:
                            output = output[0]
                        pred_mask = (output.numpy() >= threshold).astype(np.uint8)

                    dice, iou, precision, recall = sample_metrics_from_masks(pred_mask, target_mask)
                    threshold_sample_rows[threshold].append(
                        {
                            "fold": fold,
                            "id": sample_id,
                            "threshold": f"{threshold:.2f}",
                            "dice": dice,
                            "iou": iou,
                            "precision": precision,
                            "recall": recall,
                            "gt_fg_ratio": gt_fg_ratio,
                            "pred_fg_ratio": float(pred_mask.mean()),
                        }
                    )
            pbar.update(1)
        pbar.close()

    threshold_rows = []
    for threshold in thresholds:
        sample_rows = threshold_sample_rows[threshold]
        threshold_rows.append(
            {
                "fold": fold,
                "threshold": f"{threshold:.2f}",
                "dice": float(np.mean([row["dice"] for row in sample_rows])),
                "iou": float(np.mean([row["iou"] for row in sample_rows])),
                "precision": float(np.mean([row["precision"] for row in sample_rows])),
                "recall": float(np.mean([row["recall"] for row in sample_rows])),
            }
        )

    best_row = max(threshold_rows, key=lambda row: row["dice"])
    best_threshold = float(best_row["threshold"])
    return threshold_rows, best_row, threshold_sample_rows[best_threshold]


def summarize_best_rows(best_rows):
    summary = {}
    for metric in ["dice", "iou", "precision", "recall"]:
        values = np.array([float(row[f"best_{metric}"]) for row in best_rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std())
    return summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = [round(float(threshold), 10) for threshold in args.thresholds]

    all_threshold_rows = []
    best_rows = []
    best_sample_rows = []

    for fold in range(1, args.folds + 1):
        fold_dir = os.path.join(args.model_dir, f"fold_{fold}")
        config_path = os.path.join(fold_dir, "config.json")
        val_ids_path = os.path.join(fold_dir, "val_ids.csv")
        checkpoint_path = os.path.join(fold_dir, args.checkpoint_name)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config: {config_path}")
        if not os.path.exists(val_ids_path):
            raise FileNotFoundError(f"Missing validation IDs: {val_ids_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        config = read_json(config_path)
        if config.get("arch") not in ARCHS:
            raise ValueError(f"Unsupported arch in {config_path}: {config.get('arch')}")

        val_ids = read_id_csv(val_ids_path)
        model = load_checkpoint_model(config, checkpoint_path, device)
        val_loader = make_val_loader(config, val_ids, args.batch_size, args.num_workers)

        threshold_rows, best_row, sample_rows = evaluate_fold_thresholds(
            model=model,
            val_loader=val_loader,
            thresholds=thresholds,
            config=config,
            device=device,
            fold=fold,
        )

        all_threshold_rows.extend(threshold_rows)
        best_sample_rows.extend(sample_rows)
        best_rows.append(
            {
                "fold": fold,
                "best_threshold": best_row["threshold"],
                "best_dice": best_row["dice"],
                "best_iou": best_row["iou"],
                "best_precision": best_row["precision"],
                "best_recall": best_row["recall"],
            }
        )
        print(f"fold_{fold}: best_threshold={float(best_row['threshold']):.2f}, best_dice={best_row['dice']:.6f}")

    write_csv(
        os.path.join(args.output_dir, "fold_threshold_results.csv"),
        all_threshold_rows,
        ["fold", "threshold", "dice", "iou", "precision", "recall"],
    )
    write_csv(
        os.path.join(args.output_dir, "fold_best_thresholds.csv"),
        best_rows,
        ["fold", "best_threshold", "best_dice", "best_iou", "best_precision", "best_recall"],
    )
    write_csv(
        os.path.join(args.output_dir, "sample_metrics_best_threshold.csv"),
        best_sample_rows,
        ["fold", "id", "threshold", "dice", "iou", "precision", "recall", "gt_fg_ratio", "pred_fg_ratio"],
    )

    summary = summarize_best_rows(best_rows)
    write_csv(
        os.path.join(args.output_dir, "mean_std_summary.csv"),
        [summary],
        [
            "dice_mean",
            "dice_std",
            "iou_mean",
            "iou_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
        ],
    )

    print("\nFinal 5-fold result:")
    print(f"Dice: {summary['dice_mean']:.6f} ± {summary['dice_std']:.6f}")
    print(f"IoU: {summary['iou_mean']:.6f} ± {summary['iou_std']:.6f}")
    print(f"Precision: {summary['precision_mean']:.6f} ± {summary['precision_std']:.6f}")
    print(f"Recall: {summary['recall_mean']:.6f} ± {summary['recall_std']:.6f}")


if __name__ == "__main__":
    main()
