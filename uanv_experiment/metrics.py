"""Explicit fixed-threshold, sample-macro segmentation metrics.

Undefined empty-mask cases are handled by named policy branches rather than by
adding epsilon to every denominator.
"""

from __future__ import annotations

import torch


EMPTY_MASK_POLICY = {
    "dice": "gt_empty_and_pred_empty=1; gt_empty_and_pred_nonempty=0; otherwise_standard",
    "iou": "gt_empty_and_pred_empty=1; gt_empty_and_pred_nonempty=0; otherwise_standard",
    "precision": "pred_empty_and_gt_empty=1; pred_empty_and_gt_nonempty=0; otherwise_standard",
    "recall": "gt_empty=NaN; exclude_NaN_from_sample_macro",
}


@torch.no_grad()
def per_sample_binary_metrics(
    predictions: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return one metric value per sample, with NaN recall for empty GT."""
    if predictions.shape != target.shape:
        raise AssertionError(f"prediction/target shape mismatch: {predictions.shape} vs {target.shape}")
    if predictions.ndim != 4:
        raise AssertionError("predictions and target must be BCHW tensors")

    predictions = predictions.bool().reshape(predictions.shape[0], -1)
    target = target.bool().reshape(target.shape[0], -1)
    true_positive = (predictions & target).sum(dim=1, dtype=torch.float64)
    false_positive = (predictions & ~target).sum(dim=1, dtype=torch.float64)
    false_negative = (~predictions & target).sum(dim=1, dtype=torch.float64)
    predicted_positive = predictions.sum(dim=1, dtype=torch.float64)
    target_positive = target.sum(dim=1, dtype=torch.float64)

    gt_empty = target_positive == 0
    prediction_empty = predicted_positive == 0
    both_empty = gt_empty & prediction_empty

    dice = torch.empty_like(true_positive)
    iou = torch.empty_like(true_positive)
    dice[both_empty] = 1.0
    iou[both_empty] = 1.0
    empty_gt_false_positive = gt_empty & ~prediction_empty
    dice[empty_gt_false_positive] = 0.0
    iou[empty_gt_false_positive] = 0.0
    nonempty_gt = ~gt_empty
    dice[nonempty_gt] = (
        2.0 * true_positive[nonempty_gt]
        / (2.0 * true_positive[nonempty_gt] + false_positive[nonempty_gt] + false_negative[nonempty_gt])
    )
    iou[nonempty_gt] = (
        true_positive[nonempty_gt]
        / (true_positive[nonempty_gt] + false_positive[nonempty_gt] + false_negative[nonempty_gt])
    )

    precision = torch.empty_like(true_positive)
    precision[both_empty] = 1.0
    pred_empty_gt_nonempty = prediction_empty & nonempty_gt
    precision[pred_empty_gt_nonempty] = 0.0
    nonempty_prediction = ~prediction_empty
    precision[nonempty_prediction] = (
        true_positive[nonempty_prediction]
        / (true_positive[nonempty_prediction] + false_positive[nonempty_prediction])
    )

    recall = torch.full_like(true_positive, float("nan"))
    recall[nonempty_gt] = (
        true_positive[nonempty_gt]
        / (true_positive[nonempty_gt] + false_negative[nonempty_gt])
    )
    predicted_foreground_ratio = predicted_positive / predictions.shape[1]

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "gt_empty": gt_empty,
        "prediction_empty": prediction_empty,
        "predicted_foreground_ratio": predicted_foreground_ratio,
    }


def _nan_excluding_mean(values: torch.Tensor) -> tuple[float | None, int]:
    valid = ~torch.isnan(values)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return None, 0
    return float(values[valid].mean().item()), valid_count


@torch.no_grad()
def fixed_threshold_sample_macro(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, object]:
    if logits.shape != target.shape:
        raise AssertionError(f"logits/target shape mismatch: {logits.shape} vs {target.shape}")
    predictions = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    per_sample = per_sample_binary_metrics(predictions, truth)

    macro: dict[str, float | None] = {}
    valid_counts: dict[str, int] = {}
    for name in ("dice", "iou", "precision", "recall"):
        macro[name], valid_counts[name] = _nan_excluding_mean(per_sample[name])

    gt_empty = per_sample["gt_empty"]
    prediction_empty = per_sample["prediction_empty"]
    empty_count = int(gt_empty.sum().item())
    empty_ratios = per_sample["predicted_foreground_ratio"][gt_empty]
    empty_summary = {
        "empty_gt_sample_count": empty_count,
        "empty_gt_false_positive_count": int((gt_empty & ~prediction_empty).sum().item()),
        "empty_gt_mean_predicted_foreground_ratio": (
            float(empty_ratios.mean().item()) if empty_count else None
        ),
    }

    finite_where_defined = True
    for name in ("dice", "iou", "precision", "recall"):
        values = per_sample[name]
        defined = ~torch.isnan(values)
        if defined.any() and not torch.isfinite(values[defined]).all().item():
            finite_where_defined = False

    return {
        "threshold": threshold,
        "comparison_operator": ">=",
        "aggregation": "mean_of_per_sample_metrics",
        "sample_count": int(logits.shape[0]),
        "empty_mask_policy": EMPTY_MASK_POLICY,
        "metrics": macro,
        "metric_valid_sample_counts": valid_counts,
        "empty_gt_summary": empty_summary,
        "all_finite_where_defined": finite_where_defined,
    }
