"""Future full-run entry point implementing the locked clean protocol.

This module is intentionally not invoked by smoke tests. When explicitly run in
the future, one invocation trains one registered model on one existing fold for
exactly 35 epochs in a brand-new run directory.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import COMMON_TRAINING_CONFIG
from dataset import GlomerulusSegmentationDataset
from losses import WeightedBCEDiceLoss
from metrics import fixed_threshold_sample_macro
from provenance import build_new_run_layout, resolve_primary_checkpoint, validate_run_manifest
from registry import MODEL_REGISTRY, get_model, model_config


PRODUCTION_SCOPES = ("fold1_only", "all_folds_locked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked one-model/one-fold clean experiment")
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--production-scope",
        choices=PRODUCTION_SCOPES,
        help=(
            "required for a non-dry production run; fold1_only preserves the original "
            "Fold 1 authorization and all_folds_locked authorizes validated Folds 1-5"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate policy and collision protection without creating files or training",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_snapshot_info() -> dict[str, object]:
    repository = Path(__file__).resolve().parent.parent
    base_command = ["git", "-c", f"safe.directory={repository.as_posix()}"]
    commit = subprocess.run(
        [*base_command, "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_diff = subprocess.run(
        [*base_command, "diff", "--quiet", "HEAD", "--", "uanv_experiment"],
        cwd=repository,
        check=False,
    )
    untracked = subprocess.run(
        [*base_command, "ls-files", "--others", "--exclude-standard", "--", "uanv_experiment"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_diff.returncode != 0 or untracked:
        raise RuntimeError(
            "uanv_experiment source differs from the committed snapshot; "
            "commit a new source snapshot before production"
        )
    return {
        "git_commit": commit,
        "uanv_experiment_matches_commit": True,
        "untracked_source_file_count": 0,
    }


def write_new_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def write_new_json(path: Path, value: object) -> None:
    write_new_text(path, json.dumps(value, indent=2))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_determinism(seed: int) -> None:
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id"]:
            raise AssertionError("fold CSV schema must contain only an id column")
        values = [row["id"].strip() for row in reader]
    if len(values) != len(set(values)):
        raise AssertionError("duplicate fold assignment")
    return values


def validate_production_scope(args: argparse.Namespace) -> None:
    if args.production_scope is None:
        if args.dry_run:
            return
        raise PermissionError("a locked --production-scope is required for production")
    if args.production_scope == "fold1_only" and args.fold != 1:
        raise PermissionError(
            "production scope fold1_only authorizes Fold 1 only; use --fold 1"
        )
    if args.production_scope == "all_folds_locked" and args.fold not in range(1, 6):
        raise PermissionError("production scope all_folds_locked authorizes Folds 1-5 only")


def validate_locked_protocol() -> None:
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_common = {
        "input_size": [512, 512],
        "batch_size": 2,
        "epochs": 35,
        "loss": "WeightedBCEDiceLoss",
        "bce_weight": 0.5,
        "dice_weight": 1.0,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "scheduler": "CosineAnnealingLR",
        "minimum_learning_rate": 0.00001,
        "augmentation": True,
        "fold_source": "historical_fold_csv_direct_read",
        "seed": 42,
        "num_workers": 0,
        "cublas_workspace_config": ":4096:8",
        "strict_deterministic_algorithms": True,
    }
    expected_primary = {
        "primary_checkpoint_policy": "fixed_epoch_35",
        "checkpoint_filename": "epoch_35_checkpoint.pth",
        "evaluation_checkpoint": "epoch_35_checkpoint.pth",
        "threshold": 0.5,
        "comparison_operator": ">=",
        "aggregation": "sample_macro",
        "metrics_output": "primary_metrics_threshold_0.5.csv",
    }
    if protocol.get("authorized_production_scopes") != list(PRODUCTION_SCOPES):
        raise AssertionError("protocol production scopes do not match the locked runner")
    if protocol.get("model_labels") != list(MODEL_REGISTRY):
        raise AssertionError("protocol model labels do not match the locked registry")
    if protocol.get("common_training") != expected_common:
        raise AssertionError("locked common training protocol mismatch")
    if protocol.get("primary") != expected_primary:
        raise AssertionError("locked primary evaluation protocol mismatch")


def validate_fold_assignments(fold_root: Path, fold: int) -> dict[str, str]:
    fold_dir = fold_root / f"fold_{fold}"
    paths = {name: fold_dir / f"{name}_ids.csv" for name in ("paired", "train", "val")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"validated fold CSV missing: {missing}")
    ids = {name: read_ids(path) for name, path in paths.items()}
    if set(ids["train"]) & set(ids["val"]):
        raise AssertionError("train and validation folds overlap")
    if set(ids["paired"]) != set(ids["train"]) | set(ids["val"]):
        raise AssertionError("paired fold IDs do not equal train/validation partition")
    return {name: sha256(path) for name, path in paths.items()}


def make_loaders(args: argparse.Namespace, seed: int):
    fold_dir = args.fold_root / f"fold_{args.fold}"
    train_ids = read_ids(fold_dir / "train_ids.csv")
    validation_ids = read_ids(fold_dir / "val_ids.csv")
    if set(train_ids) & set(validation_ids):
        raise AssertionError("train and validation folds overlap")
    common = {
        "image_dir": args.image_dir,
        "mask_dir": args.mask_dir,
        "image_extension": ".png",
        "mask_extension": ".tiff",
        "height": 512,
        "width": 512,
    }
    train_dataset = GlomerulusSegmentationDataset(train_ids, augment=True, **common)
    validation_dataset = GlomerulusSegmentationDataset(validation_ids, augment=False, **common)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=seed_worker,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=seed_worker,
    )
    fold_hashes = {
        name: sha256(fold_dir / f"{name}_ids.csv")
        for name in ("paired", "train", "val")
    }
    return train_loader, validation_loader, fold_hashes, len(train_ids), len(validation_ids)


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        if not all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            raise FloatingPointError("non-finite training gradient")
        optimizer.step()
        total_loss += float(loss.item()) * images.shape[0]
        total_samples += images.shape[0]
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    metric_sums = {name: 0.0 for name in ("dice", "iou", "precision", "recall")}
    valid_counts = {name: 0 for name in metric_sums}
    empty_count = 0
    empty_false_positive_count = 0
    empty_predicted_ratio_sum = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        report = fixed_threshold_sample_macro(logits, targets, threshold=threshold)
        batch_size = images.shape[0]
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        for name, value in report["metrics"].items():
            count = report["metric_valid_sample_counts"][name]
            if value is not None:
                metric_sums[name] += float(value) * count
            valid_counts[name] += count
        empty = report["empty_gt_summary"]
        empty_count += empty["empty_gt_sample_count"]
        empty_false_positive_count += empty["empty_gt_false_positive_count"]
        if empty["empty_gt_mean_predicted_foreground_ratio"] is not None:
            empty_predicted_ratio_sum += (
                empty["empty_gt_mean_predicted_foreground_ratio"]
                * empty["empty_gt_sample_count"]
            )
    metrics = {
        name: (metric_sums[name] / valid_counts[name] if valid_counts[name] else None)
        for name in metric_sums
    }
    return {
        "loss": loss_sum / sample_count,
        **metrics,
        "sample_count": sample_count,
        "recall_valid_sample_count": valid_counts["recall"],
        "empty_gt_sample_count": empty_count,
        "empty_gt_false_positive_count": empty_false_positive_count,
        "empty_gt_mean_predicted_foreground_ratio": (
            empty_predicted_ratio_sum / empty_count if empty_count else None
        ),
    }


def save_checkpoint_atomic(path: Path, payload: dict[str, object], allow_active_run_replacement: bool) -> None:
    if path.exists() and not allow_active_run_replacement:
        raise FileExistsError(f"refusing checkpoint overwrite: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary checkpoint exists: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model, optimizer, scheduler, epoch, model_label, config):
    return {
        "epoch": epoch,
        "model_label": model_label,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }


def write_metrics_csv_new(path: Path, row: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    validate_production_scope(args)
    validate_locked_protocol()
    layout = build_new_run_layout(args.run_dir.resolve())
    fold_hashes = validate_fold_assignments(args.fold_root, args.fold)
    config = model_config(args.model)
    manifest = {
        "schema": "uanv_clean_run_manifest_v1",
        "model_label": args.model,
        "fold": args.fold,
        "primary_checkpoint_policy": "fixed_epoch_35",
        "primary_checkpoint": str(layout.primary_checkpoint),
        "best_validation_checkpoint": str(layout.best_val_checkpoint),
        "primary_threshold": 0.5,
        "primary_comparison_operator": ">=",
        "primary_aggregation": "sample_macro",
        "best_validation_primary_eligible": False,
        "threshold_sweep_primary_eligible": False,
    }
    validate_run_manifest(manifest, layout)
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "run_directory_created": False,
            "full_training_started": False,
            "layout": layout.as_dict(),
            "manifest": manifest,
            "fold_csv_sha256": fold_hashes,
        }, indent=2))
        return

    snapshot = source_snapshot_info()
    manifest["authorized_production_scope"] = args.production_scope
    manifest["source_snapshot"] = snapshot
    configure_determinism(int(COMMON_TRAINING_CONFIG["kfold_seed"]))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    layout.run_directory.mkdir(parents=True, exist_ok=False)
    write_new_json(layout.path("config.json"), config)
    write_new_text(layout.path("command.txt"), " ".join(sys.argv) + "\n")
    write_new_json(layout.path("run_manifest.json"), manifest)
    write_new_json(layout.path("run_status.json"), {"status": "running", "started_at_utc": utc_now()})

    try:
        train_loader, validation_loader, fold_hashes, train_count, val_count = make_loaders(args, 42)
        manifest.update({
            "fold_csv_sha256": fold_hashes,
            "train_count": train_count,
            "validation_count": val_count,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
            "source_sha256": {
                "registry.py": sha256(Path(__file__).with_name("registry.py")),
                "metrics.py": sha256(Path(__file__).with_name("metrics.py")),
                "dataset.py": sha256(Path(__file__).with_name("dataset.py")),
                "run_experiment.py": sha256(Path(__file__)),
            },
        })
        layout.path("run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        model = get_model(args.model).to(device)
        criterion = WeightedBCEDiceLoss(bce_weight=0.5, dice_weight=1.0).to(device)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = CosineAnnealingLR(optimizer, T_max=35, eta_min=0.00001)
        best_dice = float("-inf")
        log_rows = []
        for epoch in range(1, 36):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            validation = evaluate(model, validation_loader, criterion, device, threshold=0.5)
            scheduler.step()
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_dice_threshold_0.5": validation["dice"],
                "validation_iou_threshold_0.5": validation["iou"],
            }
            log_rows.append(row)
            payload = checkpoint_payload(model, optimizer, scheduler, epoch, args.model, config)
            if float(validation["dice"]) > best_dice:
                best_dice = float(validation["dice"])
                save_checkpoint_atomic(layout.best_val_checkpoint, payload, allow_active_run_replacement=True)
            if epoch == 35:
                save_checkpoint_atomic(layout.primary_checkpoint, payload, allow_active_run_replacement=False)

        with layout.path("log.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
            writer.writeheader()
            writer.writerows(log_rows)
        write_new_text(layout.path("epoch_35_checkpoint_sha256.txt"), sha256(layout.primary_checkpoint) + "\n")
        write_new_text(layout.path("best_val_checkpoint_sha256.txt"), sha256(layout.best_val_checkpoint) + "\n")

        primary_path = resolve_primary_checkpoint(layout.path("run_manifest.json"))
        checkpoint = torch.load(primary_path, map_location=device, weights_only=False)
        if checkpoint.get("epoch") != 35:
            raise AssertionError("primary checkpoint payload is not epoch 35")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        primary_metrics = evaluate(model, validation_loader, criterion, device, threshold=0.5)
        primary_metrics.update({
            "checkpoint": "epoch_35_checkpoint.pth",
            "checkpoint_policy": "fixed_epoch_35",
            "threshold": 0.5,
            "comparison_operator": ">=",
            "aggregation": "sample_macro",
        })
        write_metrics_csv_new(layout.path("primary_metrics_threshold_0.5.csv"), primary_metrics)

        with layout.path("exploratory_threshold_metrics.csv").open("x", encoding="utf-8", newline="") as handle:
            rows = []
            for threshold in np.arange(0.20, 0.701, 0.05):
                metrics = evaluate(model, validation_loader, criterion, device, threshold=float(threshold))
                rows.append({"checkpoint": "epoch_35_checkpoint.pth", "threshold": float(threshold), **metrics})
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        layout.path("run_status.json").write_text(json.dumps({
            "status": "complete",
            "completed_at_utc": utc_now(),
            "primary_checkpoint_policy": "fixed_epoch_35",
        }, indent=2), encoding="utf-8")
    except Exception as exc:
        layout.path("run_status.json").write_text(json.dumps({
            "status": "failed",
            "failed_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
