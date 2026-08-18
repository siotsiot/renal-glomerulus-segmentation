import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity-check image/mask pairs before segmentation training."
    )
    parser.add_argument("--img_dir", required=True, help="Directory containing input images.")
    parser.add_argument("--mask_dir", required=True, help="Directory containing masks.")
    parser.add_argument("--img_ext", default=".png", help="Image extension, e.g. .png")
    parser.add_argument("--mask_ext", default=".png", help="Mask extension, e.g. .png or .tiff")
    parser.add_argument("--out_dir", required=True, help="Directory for CSV reports and overlays.")
    return parser.parse_args()


def normalize_ext(ext):
    ext = str(ext).strip()
    if not ext:
        raise ValueError("Extension must not be empty.")
    return ext if ext.startswith(".") else f".{ext}"


def collect_by_stem(directory, extension):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    paths = sorted(directory.glob(f"*{extension}"))
    by_stem = {}
    duplicates = []
    for path in paths:
        if path.stem in by_stem:
            duplicates.append(path.stem)
        by_stem[path.stem] = path
    return by_stem, duplicates


def write_stem_csv(path, stems, column_name="stem"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([column_name])
        for stem in stems:
            writer.writerow([stem])


def write_dict_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def shape_hw(array):
    return tuple(array.shape[:2])


def mask_foreground(mask):
    if mask.ndim == 2:
        return mask > 0
    return np.any(mask > 0, axis=2)


def unique_values(mask):
    if mask.ndim == 2:
        values = np.unique(mask)
        return [int(v) if np.issubdtype(values.dtype, np.integer) else float(v) for v in values]

    flattened = mask.reshape(-1, mask.shape[-1])
    values = np.unique(flattened, axis=0)
    return [tuple(int(v) for v in row) for row in values]


def unique_preview(values, limit=30):
    shown = values[:limit]
    preview = ";".join(str(value) for value in shown)
    if len(values) > limit:
        preview += f";...(+{len(values) - limit})"
    return preview


def scalar_unique_set(values):
    if values and isinstance(values[0], tuple):
        return set()
    return set(values)


def positive_label_count(values):
    if values and isinstance(values[0], tuple):
        return 0
    return sum(1 for value in values if value > 0)


def foreground_ratio(binary_mask):
    return float(binary_mask.mean()) if binary_mask.size else 0.0


def connected_components(binary_mask):
    if not np.any(binary_mask):
        return 0
    mask_uint8 = binary_mask.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_uint8, connectivity=8)
    return int(num_labels - 1)


def to_display_gray(array):
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        if array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        else:
            array = array[:, :, 0]

    array = array.astype(np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value > min_value:
        array = (array - min_value) / (max_value - min_value) * 255.0
    else:
        array = np.zeros_like(array, dtype=np.float32)
    return array.astype(np.uint8)


def make_overlay(image, binary_mask, alpha=0.45):
    gray = to_display_gray(image)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(base)
    color[:, :, 2] = 255
    overlay = base.copy()
    overlay[binary_mask] = cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0)[binary_mask]
    return overlay


def save_overlay_panel(path, image, mask, binary_mask):
    image_panel = cv2.cvtColor(to_display_gray(image), cv2.COLOR_GRAY2BGR)
    mask_panel = cv2.cvtColor(to_display_gray(mask), cv2.COLOR_GRAY2BGR)
    overlay_panel = make_overlay(image, binary_mask)

    h, w = image_panel.shape[:2]
    mask_panel = cv2.resize(mask_panel, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay_panel = cv2.resize(overlay_panel, (w, h), interpolation=cv2.INTER_NEAREST)

    panel = np.concatenate([image_panel, mask_panel, overlay_panel], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def summarize_numbers(values):
    if not values:
        return {"min": math.nan, "max": math.nan, "mean": math.nan}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def infer_mask_structure(global_scalar_values, unique_counts, positive_label_counts, component_counts, has_color_masks):
    if has_color_masks:
        return "semantic multi-class candidate"

    values = set(global_scalar_values)
    if values == {0, 255} or values == {0} or values == {255}:
        return "0/255 binary"
    if values == {0, 1} or values == {1}:
        return "0/1 binary"

    positive_values = {value for value in values if value > 0}
    max_unique_count = max(unique_counts) if unique_counts else 0
    max_positive_value = max(positive_values) if positive_values else 0

    if max_unique_count > 10 or max_positive_value > 255:
        return "instance label"

    non_empty_pairs = [
        (positive_count, component_count)
        for positive_count, component_count in zip(positive_label_counts, component_counts)
        if positive_count > 0 and component_count > 0
    ]
    if non_empty_pairs:
        instance_like = [
            positive_count == component_count
            for positive_count, component_count in non_empty_pairs
            if positive_count > 1
        ]
        if instance_like and sum(instance_like) / len(instance_like) >= 0.8:
            return "instance label"

    return "semantic multi-class candidate"


def main():
    args = parse_args()
    img_ext = normalize_ext(args.img_ext)
    mask_ext = normalize_ext(args.mask_ext)
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlays"

    img_by_stem, img_duplicates = collect_by_stem(args.img_dir, img_ext)
    mask_by_stem, mask_duplicates = collect_by_stem(args.mask_dir, mask_ext)

    img_ids = set(img_by_stem)
    mask_ids = set(mask_by_stem)
    paired_ids = sorted(img_ids & mask_ids)
    missing_masks = sorted(img_ids - mask_ids)
    extra_masks = sorted(mask_ids - img_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_stem_csv(out_dir / "missing_masks.csv", missing_masks)
    write_stem_csv(out_dir / "extra_masks.csv", extra_masks)
    write_stem_csv(out_dir / "paired_ids.csv", paired_ids)

    print(f"image_count: {len(img_ids)}")
    print(f"mask_count: {len(mask_ids)}")
    print(f"paired_ids_count: {len(paired_ids)}")
    print(f"missing_masks_count: {len(missing_masks)}")
    print(f"extra_masks_count: {len(extra_masks)}")

    if img_duplicates:
        print(f"warning: duplicate image stems detected: {len(img_duplicates)}")
    if mask_duplicates:
        print(f"warning: duplicate mask stems detected: {len(mask_duplicates)}")

    per_mask_rows = []
    empty_rows = []
    shape_mismatch_rows = []
    all_scalar_values = set()
    unique_counts = []
    positive_label_counts = []
    foreground_ratios = []
    component_counts = []
    has_color_masks = False

    for index, stem in enumerate(paired_ids):
        img_path = img_by_stem[stem]
        mask_path = mask_by_stem[stem]

        image = read_image(img_path)
        mask = read_image(mask_path)
        binary_mask = mask_foreground(mask)
        values = unique_values(mask)

        if mask.ndim == 3:
            has_color_masks = True
        else:
            all_scalar_values.update(scalar_unique_set(values))

        unique_count = len(values)
        positive_count = positive_label_count(values)
        fg_ratio = foreground_ratio(binary_mask)
        cc_count = connected_components(binary_mask)

        unique_counts.append(unique_count)
        positive_label_counts.append(positive_count)
        foreground_ratios.append(fg_ratio)
        component_counts.append(cc_count)

        image_hw = shape_hw(image)
        mask_hw = shape_hw(mask)
        mismatch = image_hw != mask_hw

        if mismatch:
            shape_mismatch_rows.append(
                {
                    "stem": stem,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "image_shape": str(tuple(image.shape)),
                    "mask_shape": str(tuple(mask.shape)),
                }
            )

        if fg_ratio == 0.0:
            empty_rows.append(
                {
                    "stem": stem,
                    "mask_path": str(mask_path),
                    "mask_shape": str(tuple(mask.shape)),
                    "unique_values": unique_preview(values),
                }
            )

        per_mask_rows.append(
            {
                "stem": stem,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "image_shape": str(tuple(image.shape)),
                "mask_shape": str(tuple(mask.shape)),
                "shape_mismatch": mismatch,
                "unique_count": unique_count,
                "positive_label_count": positive_count,
                "unique_values_preview": unique_preview(values),
                "foreground_ratio_mask_gt_0": f"{fg_ratio:.10f}",
                "connected_components_mask_gt_0": cc_count,
            }
        )

        print(
            "mask_stats: "
            f"stem={stem}, "
            f"unique_count={unique_count}, "
            f"positive_label_count={positive_count}, "
            f"foreground_ratio={fg_ratio:.10f}, "
            f"connected_components={cc_count}"
        )

        if index < 20:
            if image_hw != mask_hw:
                resized_mask = cv2.resize(
                    binary_mask.astype(np.uint8),
                    (image_hw[1], image_hw[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                resized_mask = binary_mask
            save_overlay_panel(overlay_dir / f"{index + 1:02d}_{stem}.png", image, mask, resized_mask)

    write_dict_csv(
        out_dir / "per_mask_stats.csv",
        per_mask_rows,
        [
            "stem",
            "image_path",
            "mask_path",
            "image_shape",
            "mask_shape",
            "shape_mismatch",
            "unique_count",
            "positive_label_count",
            "unique_values_preview",
            "foreground_ratio_mask_gt_0",
            "connected_components_mask_gt_0",
        ],
    )
    write_dict_csv(out_dir / "empty_masks.csv", empty_rows, ["stem", "mask_path", "mask_shape", "unique_values"])
    write_dict_csv(
        out_dir / "shape_mismatch.csv",
        shape_mismatch_rows,
        ["stem", "image_path", "mask_path", "image_shape", "mask_shape"],
    )

    unique_summary = summarize_numbers(unique_counts)
    fg_summary = summarize_numbers(foreground_ratios)
    cc_summary = summarize_numbers(component_counts)

    if has_color_masks:
        print("global_unique_values: color/tuple masks detected; see per_mask_stats.csv previews")
    else:
        print(f"global_unique_values: {sorted(all_scalar_values)}")

    print(
        "unique_value_count_per_mask: "
        f"min={unique_summary['min']:.4f}, "
        f"max={unique_summary['max']:.4f}, "
        f"mean={unique_summary['mean']:.4f}"
    )
    print(
        "foreground_ratio_mask_gt_0: "
        f"min={fg_summary['min']:.10f}, "
        f"max={fg_summary['max']:.10f}, "
        f"mean={fg_summary['mean']:.10f}"
    )
    print(f"empty_mask_count: {len(empty_rows)}")
    print(
        "connected_components_mask_gt_0: "
        f"min={cc_summary['min']:.4f}, "
        f"max={cc_summary['max']:.4f}, "
        f"mean={cc_summary['mean']:.4f}"
    )
    print(f"shape_mismatch_count: {len(shape_mismatch_rows)}")

    inferred = infer_mask_structure(
        all_scalar_values,
        unique_counts,
        positive_label_counts,
        component_counts,
        has_color_masks,
    )
    print(f"inferred_mask_structure: {inferred}")

    print("recommended_training_setting:")
    print("  - If the goal is whole glomerulus region segmentation: num_classes=1, mask_mode=binary")
    print("  - If positive pixel values have separate class meanings: inspect multiclass training")
    print(f"reports_written_to: {out_dir}")
    print(f"overlay_samples_written_to: {overlay_dir}")


if __name__ == "__main__":
    main()
