import argparse
import shutil
from pathlib import Path

import pandas as pd


def find_annotated_image(csv_path, row):
    """
    analyze_val_overlays.py가 저장한 annotated image를 찾는다.

    구조:
    failure_analysis_final/
      model/
        fold/
          epoch/
            overlay_failure_analysis.csv
            analyzed_overlays/
              error_type/
                dice_..._파일명.png
    """
    base_dir = csv_path.parent
    error_type = row["error_type"]
    filename = row["filename"]

    search_dir = base_dir / "analyzed_overlays" / error_type

    if not search_dir.exists():
        return None

    matches = list(search_dir.glob(f"*{filename}"))

    if len(matches) == 0:
        return None

    return matches[0]


def add_case(selected_rows, row, category):
    row_dict = row.to_dict()
    row_dict["selected_category"] = category
    selected_rows.append(row_dict)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="./failure_analysis_final",
        help="failure_analysis_final 폴더"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./failure_analysis_selected",
        help="대표 사례 저장 폴더"
    )

    parser.add_argument(
        "--epoch_filter",
        type=str,
        default="epoch_035",
        help="분석할 epoch 이름. 예: epoch_035. 전체 epoch를 보려면 all"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="유형별로 몇 장씩 고를지"
    )

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(root.rglob("overlay_failure_analysis.csv"))

    if args.epoch_filter.lower() != "all":
        csv_paths = [p for p in csv_paths if p.parent.name == args.epoch_filter]

    if len(csv_paths) == 0:
        raise RuntimeError("분석 CSV를 찾지 못했습니다.")

    all_dfs = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        # 경로 정보 추가
        parts = csv_path.parts

        # 대략 failure_analysis_final/model/fold/epoch/overlay_failure_analysis.csv 구조
        model_name = csv_path.parents[2].name
        fold_name = csv_path.parents[1].name
        epoch_name = csv_path.parents[0].name

        df["model"] = model_name
        df["fold"] = fold_name
        df["epoch"] = epoch_name
        df["csv_path"] = str(csv_path)

        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    selected_rows = []

    # 1. 성공 사례: Dice 높은 것
    success = df_all.sort_values("dice", ascending=False).head(args.top_k)
    for _, row in success.iterrows():
        add_case(selected_rows, row, "01_success_high_dice")

    # 2. Boundary ambiguity: error_type 기준 우선, 없으면 boundary_dice 낮은 것
    boundary = df_all[df_all["error_type"] == "boundary_ambiguity"]
    if len(boundary) == 0:
        boundary = df_all[(df_all["dice"] >= 0.70) & (df_all["boundary_dice"] < 0.20)]

    boundary = boundary.sort_values("boundary_dice", ascending=True).head(args.top_k)
    for _, row in boundary.iterrows():
        add_case(selected_rows, row, "02_boundary_ambiguity")

    # 3. Under-segmentation / False negative: recall 낮고 precision 높은 것
    under = df_all[
        (df_all["error_type"] == "under_segmentation_or_false_negative")
        | ((df_all["recall"] < 0.80) & (df_all["precision"] >= 0.85))
    ]
    under = under.sort_values(["recall", "precision"], ascending=[True, False]).head(args.top_k)
    for _, row in under.iterrows():
        add_case(selected_rows, row, "03_under_segmentation_false_negative")

    # 4. Over-segmentation / False positive: precision 낮고 recall 높은 것
    over = df_all[
        (df_all["error_type"] == "over_segmentation_or_false_positive")
        | ((df_all["precision"] < 0.80) & (df_all["recall"] >= 0.80))
    ]
    over = over.sort_values(["precision", "recall"], ascending=[True, False]).head(args.top_k)
    for _, row in over.iterrows():
        add_case(selected_rows, row, "04_over_segmentation_false_positive")

    # 5. Annotation mismatch 후보
    mismatch = df_all[
        (df_all["error_type"] == "annotation_mismatch_candidate_or_false_positive")
        | (df_all["extra_pred_components"] > 0)
    ]
    mismatch = mismatch.sort_values("dice", ascending=True).head(args.top_k)
    for _, row in mismatch.iterrows():
        add_case(selected_rows, row, "05_annotation_mismatch_candidate")

    # 6. Worst Dice
    worst = df_all.sort_values("dice", ascending=True).head(args.top_k)
    for _, row in worst.iterrows():
        add_case(selected_rows, row, "06_worst_dice")

    selected_df = pd.DataFrame(selected_rows)

    # 중복 제거
    selected_df = selected_df.drop_duplicates(
        subset=["model", "fold", "epoch", "filename", "selected_category"]
    )

    summary_csv = output_dir / "selected_cases_summary.csv"
    selected_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    copied_count = 0

    for idx, row in selected_df.iterrows():
        csv_path = Path(row["csv_path"])
        image_path = find_annotated_image(csv_path, row)

        category = row["selected_category"]
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        if image_path is None:
            print(f"[Warning] annotated image를 찾지 못함: {row['filename']}")
            continue

        save_name = (
            f"{row['model']}_{row['fold']}_{row['epoch']}_"
            f"dice_{row['dice']:.4f}_"
            f"p_{row['precision']:.4f}_"
            f"r_{row['recall']:.4f}_"
            f"{row['filename']}"
        )

        shutil.copy2(image_path, category_dir / save_name)
        copied_count += 1

    print("\n대표 사례 추출 완료")
    print(f"저장 폴더: {output_dir}")
    print(f"요약 CSV: {summary_csv}")
    print(f"복사된 이미지 수: {copied_count}")

    print("\n선택된 사례 요약")
    print(selected_df[
        [
            "selected_category",
            "model",
            "fold",
            "epoch",
            "filename",
            "dice",
            "iou",
            "precision",
            "recall",
            "boundary_dice",
            "error_type",
        ]
    ])


if __name__ == "__main__":
    main()