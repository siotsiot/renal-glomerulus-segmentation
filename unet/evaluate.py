import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from .dataset import GlomerulusDataset
    from .model import UNet
except ImportError:
    from dataset import GlomerulusDataset
    from model import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UNet checkpoint for glomerulus binary segmentation.")

    parser.add_argument("--images_dir", type=str, default="./data/images")
    parser.add_argument("--masks_dir", type=str, default="./data/masks")
    parser.add_argument("--image_ext", type=str, default=".png")
    parser.add_argument("--mask_ext", type=str, default=".tiff")
    parser.add_argument("--resize", type=int, default=512, help="Resize target H/W. Use 512 for current baseline.")
    parser.add_argument("--foreground_value", type=int, default=0, help="Mask polarity foreground value.")

    parser.add_argument("--checkpoint", type=str, default="./weights/unet_glom_dice.pth")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-7)

    parser.add_argument("--output_dir", type=str, default="./eval_outputs/unet_eval")
    parser.add_argument(
        "--vis_mode",
        type=str,
        default="first",
        choices=["first", "best", "worst", "random", "mixed", "all"],
        help="Visualization selection strategy.",
    )
    parser.add_argument("--max_vis", type=int, default=20, help="Max visualizations for non-'all' modes.")
    parser.add_argument("--vis_best", type=int, default=0, help="Used when vis_mode='mixed'.")
    parser.add_argument("--vis_worst", type=int, default=0, help="Used when vis_mode='mixed'.")
    parser.add_argument("--vis_random", type=int, default=0, help="Used when vis_mode='mixed'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random visualization sampling.")
    return parser.parse_args()


def sorted_stems_from_dataset(dataset):
    stems = [img_path.stem for img_path, _ in dataset.samples]
    stems = sorted(stems)
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate stems detected. Stem-based matching requires unique stems.")
    return stems


def split_stems(stems_sorted, split, test_size, random_state):
    if split == "all":
        return stems_sorted

    train_stems, val_stems = train_test_split(
        stems_sorted,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_stems if split == "train" else val_stems


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    metadata = {}

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            tensor_values_only = all(torch.is_tensor(v) for v in ckpt.values())
            if tensor_values_only:
                state_dict = ckpt
            else:
                raise ValueError("Unsupported checkpoint dict format.")

        for key in ("epoch", "best_val_loss", "val_loss", "best_dice", "best_iou"):
            if key in ckpt:
                metadata[key] = ckpt[key]
    else:
        raise ValueError("Unsupported checkpoint format. Expected a state_dict or checkpoint dict.")

    # Handle potential DataParallel checkpoint keys.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[len("module."):]] = value
        else:
            cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    return metadata, missing, unexpected


def compute_metrics(pred_bin, gt_bin, eps):
    # pred_bin / gt_bin: uint8 or bool arrays with {0,1}
    pred = pred_bin.astype(np.float32)
    gt = gt_bin.astype(np.float32)

    tp = float((pred * gt).sum())
    fp = float((pred * (1.0 - gt)).sum())
    fn = float(((1.0 - pred) * gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())

    dice = (2.0 * tp + eps) / (pred_sum + gt_sum + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def to_uint8_grayscale(image_chw):
    # image_chw: [1, H, W] float(0~1)
    image = image_chw[0]
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def make_visualization_grid(image_u8, gt_bin, pred_bin):
    h, w = image_u8.shape
    image_rgb = np.stack([image_u8, image_u8, image_u8], axis=-1)

    gt_panel = np.zeros((h, w, 3), dtype=np.uint8)
    gt_panel[..., 1] = gt_bin * 255  # green

    pred_panel = np.zeros((h, w, 3), dtype=np.uint8)
    pred_panel[..., 2] = pred_bin * 255  # red in BGR context after save conversion

    overlay = image_rgb.astype(np.float32)
    alpha = 0.45
    overlay[gt_bin == 1] = (1 - alpha) * overlay[gt_bin == 1] + alpha * np.array([0, 255, 0], dtype=np.float32)
    overlay[pred_bin == 1] = (1 - alpha) * overlay[pred_bin == 1] + alpha * np.array([255, 0, 0], dtype=np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    top = np.concatenate([image_rgb, gt_panel], axis=1)
    bottom = np.concatenate([pred_panel, overlay], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    return grid


def select_visualization_records(records, args):
    if not records:
        return []

    random.seed(args.seed)
    n = len(records)

    if args.vis_mode == "all":
        return records

    if args.vis_mode == "first":
        return records[: min(args.max_vis, n)]

    if args.vis_mode == "best":
        ranked = sorted(records, key=lambda x: x["dice"], reverse=True)
        return ranked[: min(args.max_vis, n)]

    if args.vis_mode == "worst":
        ranked = sorted(records, key=lambda x: x["dice"])
        return ranked[: min(args.max_vis, n)]

    if args.vis_mode == "random":
        k = min(args.max_vis, n)
        return random.sample(records, k)

    # mixed
    best_k = args.vis_best
    worst_k = args.vis_worst
    random_k = args.vis_random

    if best_k == 0 and worst_k == 0 and random_k == 0:
        # fallback: split max_vis roughly into thirds
        best_k = args.max_vis // 3
        worst_k = args.max_vis // 3
        random_k = args.max_vis - best_k - worst_k

    ranked_best = sorted(records, key=lambda x: x["dice"], reverse=True)
    ranked_worst = sorted(records, key=lambda x: x["dice"])

    selected = []
    selected.extend(ranked_best[: min(best_k, n)])
    selected.extend(ranked_worst[: min(worst_k, n)])

    used_stems = {r["stem"] for r in selected}
    pool = [r for r in records if r["stem"] not in used_stems]
    if random_k > 0 and pool:
        selected.extend(random.sample(pool, min(random_k, len(pool))))

    # Deduplicate while preserving order.
    dedup = []
    seen = set()
    for rec in selected:
        if rec["stem"] in seen:
            continue
        dedup.append(rec)
        seen.add(rec["stem"])
    return dedup


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    dataset = GlomerulusDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        image_ext=args.image_ext,
        mask_ext=args.mask_ext,
        resize_hw=(args.resize, args.resize),
        transform=None,
        foreground_value=args.foreground_value,
        debug=False,
        debug_max_prints=0,
    )

    stems_sorted = sorted_stems_from_dataset(dataset)
    selected_stems = split_stems(stems_sorted, args.split, args.test_size, args.random_state)

    stem_to_index = {img_path.stem: idx for idx, (img_path, _) in enumerate(dataset.samples)}
    eval_indices = [stem_to_index[stem] for stem in selected_stems]

    model = UNet(in_channels=1, out_channels=1).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    metadata, missing, unexpected = load_checkpoint(model, str(checkpoint_path), device)
    model.eval()

    print(f"[Eval] checkpoint: {checkpoint_path}")
    if metadata:
        print(f"[Eval] checkpoint metadata: {metadata}")
    else:
        print("[Eval] checkpoint metadata: unavailable (state_dict only)")
    if missing or unexpected:
        print(f"[Eval] state_dict mismatch - missing={len(missing)}, unexpected={len(unexpected)}")

    print(
        f"[Eval] split={args.split}, total_samples={len(dataset)}, "
        f"evaluated_samples={len(eval_indices)}, threshold={args.threshold}"
    )

    records = []
    with torch.no_grad():
        for idx in tqdm(eval_indices, desc="Evaluating"):
            image_tensor, mask_tensor = dataset[idx]  # [1,H,W], [1,H,W]
            stem = dataset.samples[idx][0].stem

            image_batch = image_tensor.unsqueeze(0).to(device)  # [1,1,H,W]
            gt_batch = mask_tensor.unsqueeze(0).to(device)       # [1,1,H,W]

            logits = model(image_batch)
            probs = torch.sigmoid(logits)
            pred = (probs >= args.threshold).float()

            gt_np = gt_batch.squeeze().cpu().numpy().astype(np.uint8)
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)

            metrics = compute_metrics(pred_np, gt_np, eps=args.eps)

            record = {
                "stem": stem,
                "index": idx,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "gt_fg_ratio": float(gt_np.mean()),
                "pred_fg_ratio": float(pred_np.mean()),
                "logits_mean": float(logits.mean().item()),
                "probs_mean": float(probs.mean().item()),
                "image_tensor": image_tensor.cpu().numpy(),
                "gt_mask": gt_np,
                "pred_mask": pred_np,
            }
            records.append(record)

    if not records:
        raise RuntimeError("No evaluation records were generated.")

    mean_dice = float(np.mean([r["dice"] for r in records]))
    mean_iou = float(np.mean([r["iou"] for r in records]))
    mean_precision = float(np.mean([r["precision"] for r in records]))
    mean_recall = float(np.mean([r["recall"] for r in records]))

    sample_csv_path = output_dir / "sample_metrics.csv"
    with open(sample_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stem",
                "index",
                "dice",
                "iou",
                "precision",
                "recall",
                "gt_fg_ratio",
                "pred_fg_ratio",
                "logits_mean",
                "probs_mean",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r["stem"],
                    r["index"],
                    f"{r['dice']:.8f}",
                    f"{r['iou']:.8f}",
                    f"{r['precision']:.8f}",
                    f"{r['recall']:.8f}",
                    f"{r['gt_fg_ratio']:.8f}",
                    f"{r['pred_fg_ratio']:.8f}",
                    f"{r['logits_mean']:.8f}",
                    f"{r['probs_mean']:.8f}",
                ]
            )

    selected_for_vis = select_visualization_records(records, args)
    for r in selected_for_vis:
        image_u8 = to_uint8_grayscale(r["image_tensor"])
        grid = make_visualization_grid(image_u8, r["gt_mask"], r["pred_mask"])
        filename = f"{r['stem']}_dice_{r['dice']:.4f}.png"
        save_path = vis_dir / filename
        cv2.imwrite(str(save_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("UNet Evaluation Summary\n")
        f.write("=======================\n")
        f.write(f"checkpoint_path: {checkpoint_path}\n")
        if metadata:
            f.write(f"checkpoint_metadata: {metadata}\n")
        else:
            f.write("checkpoint_metadata: unavailable (state_dict only)\n")
        f.write(f"split: {args.split}\n")
        f.write(f"evaluated_samples: {len(records)}\n")
        f.write(f"threshold: {args.threshold}\n")
        f.write(f"eps: {args.eps}\n")
        f.write("\n")
        f.write(f"mean_dice: {mean_dice:.6f}\n")
        f.write(f"mean_iou: {mean_iou:.6f}\n")
        f.write(f"mean_precision: {mean_precision:.6f}\n")
        f.write(f"mean_recall: {mean_recall:.6f}\n")
        f.write("\n")
        f.write("Metric formulas:\n")
        f.write("Dice=(2TP+eps)/(2TP+FP+FN+eps)\n")
        f.write("IoU=(TP+eps)/(TP+FP+FN+eps)\n")
        f.write("Precision=(TP+eps)/(TP+FP+eps)\n")
        f.write("Recall=(TP+eps)/(TP+FN+eps)\n")
        f.write("\n")
        f.write(f"sample_csv: {sample_csv_path}\n")
        f.write(f"visualization_dir: {vis_dir}\n")

    print("[Eval] done")
    print(f"[Eval] checkpoint path: {checkpoint_path}")
    print(f"[Eval] number of evaluated samples: {len(records)}")
    print(f"[Eval] threshold: {args.threshold}")
    print(
        "[Eval] mean metrics - "
        f"Dice: {mean_dice:.6f}, IoU: {mean_iou:.6f}, "
        f"Precision: {mean_precision:.6f}, Recall: {mean_recall:.6f}"
    )
    print(
        "[Eval] metric formulas - "
        "Dice=(2TP+eps)/(2TP+FP+FN+eps), "
        "IoU=(TP+eps)/(TP+FP+FN+eps), "
        "Precision=(TP+eps)/(TP+FP+eps), "
        "Recall=(TP+eps)/(TP+FN+eps)"
    )
    print(f"[Eval] sample-wise CSV: {sample_csv_path}")
    print(f"[Eval] visualization dir: {vis_dir}")
    print(f"[Eval] summary file: {summary_path}")


if __name__ == "__main__":
    main()
