import argparse
import subprocess
import sys
from pathlib import Path


def find_epoch_dirs(models_root: Path, target_models=None):
    epoch_dirs = []

    if target_models:
        model_dirs = [models_root / model_name for model_name in target_models]
    else:
        model_dirs = [p for p in models_root.iterdir() if p.is_dir()]

    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"[Skip] 모델 폴더 없음: {model_dir}")
            continue

        for fold_dir in sorted(model_dir.glob("fold_*")):
            val_overlay_dir = fold_dir / "val_overlays"

            if not val_overlay_dir.exists():
                continue

            for epoch_dir in sorted(val_overlay_dir.glob("epoch_*")):
                png_files = list(epoch_dir.glob("*.png"))

                if epoch_dir.is_dir() and len(png_files) > 0:
                    epoch_dirs.append(epoch_dir)

    return epoch_dirs


def parse_epoch_dir(epoch_dir: Path):
    """
    예:
    models/bottleneck_512_aug_5fold/fold_1/val_overlays/epoch_001
    """
    epoch_name = epoch_dir.name
    fold_name = epoch_dir.parent.parent.name
    model_name = epoch_dir.parent.parent.parent.name

    return model_name, fold_name, epoch_name


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models_root",
        type=str,
        default="./models",
        help="models 폴더 경로"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="./failure_analysis_all",
        help="분석 결과 저장 폴더"
    )

    parser.add_argument(
        "--analyzer_script",
        type=str,
        default="./analyze_val_overlays.py",
        help="단일 epoch overlay 분석 스크립트"
    )

    parser.add_argument(
        "--target_models",
        nargs="*",
        default=None,
        help="분석할 모델 폴더명. 비워두면 models 아래 전체 분석"
    )

    args = parser.parse_args()

    models_root = Path(args.models_root)
    output_root = Path(args.output_root)
    analyzer_script = Path(args.analyzer_script)

    if not models_root.exists():
        raise FileNotFoundError(f"models 폴더가 없습니다: {models_root}")

    if not analyzer_script.exists():
        raise FileNotFoundError(f"분석 스크립트가 없습니다: {analyzer_script}")

    epoch_dirs = find_epoch_dirs(models_root, args.target_models)

    if len(epoch_dirs) == 0:
        raise RuntimeError("분석할 epoch overlay 폴더를 찾지 못했습니다.")

    print(f"찾은 epoch overlay 폴더 수: {len(epoch_dirs)}")

    for epoch_dir in epoch_dirs:
        model_name, fold_name, epoch_name = parse_epoch_dir(epoch_dir)

        output_dir = output_root / model_name / fold_name / epoch_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n==============================")
        print(f"Model : {model_name}")
        print(f"Fold  : {fold_name}")
        print(f"Epoch : {epoch_name}")
        print(f"Input : {epoch_dir}")
        print(f"Output: {output_dir}")

        cmd = [
            sys.executable,
            str(analyzer_script),
            "--overlay_dir",
            str(epoch_dir),
            "--output_dir",
            str(output_dir),
        ]

        subprocess.run(cmd, check=True)

    print("\n전체 분석 완료")
    print(f"결과 저장 위치: {output_root}")


if __name__ == "__main__":
    main()