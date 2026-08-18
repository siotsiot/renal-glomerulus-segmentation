import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def split_2x2_panels(img):
    h, w = img.shape[:2]

    h2 = h // 2
    w2 = w // 2

    original = img[:h2, :w2]
    gt_panel = img[:h2, w2:]
    pred_panel = img[h2:, :w2]
    overlay_panel = img[h2:, w2:]

    return original, gt_panel, pred_panel, overlay_panel


def panel_to_binary_mask(panel, threshold=10):
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8)
    return mask


def compute_metrics(gt, pred, eps=1e-7):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp_pixels": int(tp),
        "fp_pixels": int(fp),
        "fn_pixels": int(fn),
    }


def get_boundary(mask):
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)

    return (dilated - eroded).astype(np.uint8)


def compute_boundary_dice(gt, pred, eps=1e-7):
    gt_b = get_boundary(gt).astype(bool)
    pred_b = get_boundary(pred).astype(bool)

    tp = np.logical_and(gt_b, pred_b).sum()
    fp = np.logical_and(~gt_b, pred_b).sum()
    fn = np.logical_and(gt_b, ~pred_b).sum()

    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def count_extra_prediction_components(gt, pred, min_area=50):
    pred_uint8 = (pred > 0).astype(np.uint8)
    gt_bool = gt.astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        pred_uint8,
        connectivity=8
    )

    extra_count = 0

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        component = labels == label_id
        overlap = np.logical_and(component, gt_bool).sum()
        overlap_ratio = overlap / (area + 1e-7)

        if overlap_ratio < 0.1:
            extra_count += 1

    return extra_count


def classify_error(metrics, boundary_dice, extra_components):
    dice = metrics["dice"]
    precision = metrics["precision"]
    recall = metrics["recall"]

    if dice >= 0.90:
        return "success"

    if extra_components > 0 and precision < 0.85:
        return "annotation_mismatch_candidate_or_false_positive"

    if precision < 0.75 and recall >= 0.80:
        return "over_segmentation_or_false_positive"

    if recall < 0.75 and precision >= 0.80:
        return "under_segmentation_or_false_negative"

    if boundary_dice < 0.50 and dice >= 0.70:
        return "boundary_ambiguity"

    if dice < 0.60:
        return "severe_failure"

    return "moderate_failure"


def annotate_image(img, lines):
    annotated = img.copy()

    box_h = 35 + 30 * len(lines)
    cv2.rectangle(
        annotated,
        (0, 0),
        (annotated.shape[1], box_h),
        (0, 0, 0),
        -1
    )

    y = 30
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y += 30

    return annotated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--overlay_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_panels", type=int, default=4)
    parser.add_argument("--gt_panel_index", type=int, default=1)
    parser.add_argument("--pred_panel_index", type=int, default=2)
    parser.add_argument("--mask_threshold", type=int, default=10)
    parser.add_argument("--min_component_area", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()

    overlay_dir = Path(args.overlay_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    analyzed_dir = output_dir / "analyzed_overlays"
    candidate_dir = output_dir / "selected_candidates"

    analyzed_dir.mkdir(exist_ok=True)
    candidate_dir.mkdir(exist_ok=True)

    type_names = [
        "success",
        "moderate_failure",
        "severe_failure",
        "boundary_ambiguity",
        "over_segmentation_or_false_positive",
        "under_segmentation_or_false_negative",
        "annotation_mismatch_candidate_or_false_positive",
    ]

    type_dirs = {}
    for t in type_names:
        d = analyzed_dir / t
        d.mkdir(parents=True, exist_ok=True)
        type_dirs[t] = d

    png_paths = sorted(overlay_dir.glob("*.png"))

    if len(png_paths) == 0:
        raise RuntimeError(f"PNG 파일이 없습니다: {overlay_dir}")

    rows = []

    for path in tqdm(png_paths, desc=f"Analyzing {overlay_dir}"):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img is None:
            print(f"[Skip] 읽기 실패: {path}")
            continue

        original_panel, gt_panel, pred_panel, overlay_panel = split_2x2_panels(img)

        gt = panel_to_binary_mask(gt_panel, threshold=args.mask_threshold)
        pred = panel_to_binary_mask(pred_panel, threshold=args.mask_threshold)

        metrics = compute_metrics(gt, pred)
        boundary_dice = compute_boundary_dice(gt, pred)
        extra_components = count_extra_prediction_components(
            gt,
            pred,
            min_area=args.min_component_area
        )

        error_type = classify_error(metrics, boundary_dice, extra_components)

        row = {
            "filename": path.name,
            "path": str(path),
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "boundary_dice": boundary_dice,
            "tp_pixels": metrics["tp_pixels"],
            "fp_pixels": metrics["fp_pixels"],
            "fn_pixels": metrics["fn_pixels"],
            "extra_pred_components": extra_components,
            "error_type": error_type,
        }

        rows.append(row)

        lines = [
            path.name,
            f"type={error_type}",
            f"Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, "
            f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"BoundaryDice={boundary_dice:.4f}",
        ]

        annotated = annotate_image(img, lines)

        save_name = (
            f"dice_{metrics['dice']:.4f}_"
            f"p_{metrics['precision']:.4f}_"
            f"r_{metrics['recall']:.4f}_"
            f"{path.name}"
        )

        cv2.imwrite(str(type_dirs[error_type] / save_name), annotated)

    df = pd.DataFrame(rows)

    result_csv = output_dir / "overlay_failure_analysis.csv"
    df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    summary = df.groupby("error_type").agg(
        count=("filename", "count"),
        mean_dice=("dice", "mean"),
        mean_iou=("iou", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_boundary_dice=("boundary_dice", "mean"),
    ).reset_index()

    summary_csv = output_dir / "failure_type_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    df.sort_values("dice", ascending=False).head(args.top_k).to_csv(
        candidate_dir / "success_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    df.sort_values("dice", ascending=True).head(args.top_k).to_csv(
        candidate_dir / "worst_dice_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    df.sort_values("precision", ascending=True).head(args.top_k).to_csv(
        candidate_dir / "false_positive_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    df.sort_values("recall", ascending=True).head(args.top_k).to_csv(
        candidate_dir / "false_negative_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    df.sort_values("boundary_dice", ascending=True).head(args.top_k).to_csv(
        candidate_dir / "boundary_ambiguity_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    df[df["extra_pred_components"] > 0].sort_values("dice", ascending=True).head(args.top_k).to_csv(
        candidate_dir / "annotation_mismatch_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n분석 완료")
    print(f"결과 CSV: {result_csv}")
    print(f"요약 CSV: {summary_csv}")
    print(f"분석 이미지 폴더: {analyzed_dir}")
    print(f"후보 CSV 폴더: {candidate_dir}")
    print("\n유형별 요약")
    print(summary)


if __name__ == "__main__":
    main()