# U-Net-Based Glomerulus Segmentation on Renal Histopathology Images

## Overview

이 repository는 신장 병리 이미지에서 사구체(glomerulus) 영역을 binary semantic segmentation하기 위해 작성한 학부 졸업논문 연구용 코드입니다.

주요 목적은 실제 병원 병리 이미지 데이터셋을 대상으로 U-Net 계열 segmentation model을 학습하고, 동일한 실험 조건에서 정량 평가와 정성 분석을 수행하는 것입니다.

의료 이미지 데이터, mask, checkpoint, trained weight, overlay image, 환자 정보 및 병원 관련 비공개 정보는 이 repository에 포함하지 않습니다.

## Research Objective

본 연구의 목표는 소규모 실제 병원 병리 이미지 데이터셋에서 U-Net 계열 모델들의 성능을 동일 조건에서 비교하는 것입니다.

특히 binary mask 기반 사구체 segmentation task에서 model architecture, threshold selection, cross validation 결과가 Dice, IoU, Precision, Recall에 어떤 차이를 만드는지 확인하는 데 초점을 두었습니다.

## Compared Models

- U-Net
- U-NeXt
- U-NeXt + BottleneckRefineBlock

## Experimental Workflow

1. image/mask preprocessing
2. image-mask pair 구성 및 sanity check
3. 5-fold cross validation split
4. model training
5. threshold sweep
6. quantitative evaluation
7. overlay-based qualitative analysis
8. failure case analysis

## Repository Structure

```text
.
├── unext_train.py                 # U-NeXt, U-Net variant, training, 5-fold CV, loss/metric utilities
├── threshold_sweep.py             # saved fold checkpoint 기반 threshold sweep
├── eval_threshold_sweep.py        # U-NeXt checkpoint threshold evaluation script
├── analyze_val_overlays.py        # validation overlay 기반 qualitative failure analysis
├── batch_analyze_all_overlays.py  # 여러 fold/epoch overlay batch analysis
├── select_representative_cases.py # 대표 success/failure case 선택
├── sanity_check_dataset.py        # image/mask pair, mask statistics, overlay sanity check
├── unet/
│   ├── model.py                   # U-Net architecture
│   ├── dataset.py                 # GlomerulusDataset
│   ├── loss.py                    # BCE-Dice loss
│   ├── train.py                   # baseline U-Net training script
│   └── evaluate.py                # U-Net evaluation script
└── README.md
```

현재 public repository에는 연구 재현에 필요한 source code 중심으로 정리하는 것을 목표로 합니다. 실제 실험 결과물, private dataset, checkpoint, overlay image는 공개 대상에서 제외합니다.

## Dataset Format

실제 데이터는 포함하지 않습니다. 코드는 아래와 같은 directory structure를 가정합니다.

```text
data/
├── images/
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
└── masks/
    ├── sample_001.tiff
    ├── sample_002.tiff
    └── ...
```

- `data/images/*.png`: input renal histopathology image
- `data/masks/*.tiff`: corresponding binary segmentation mask
- image와 mask는 같은 file stem을 가져야 합니다.

## How to Run

아래 command는 실행 예시입니다. 실제 환경에서는 dataset path, image size, batch size, model name, checkpoint path 등을 수정해야 할 수 있습니다.

### Training

```bash
python unext_train.py \
  --name glomerulus_UNext_woDS \
  --arch UNext \
  --img_dir data/images \
  --mask_dir data/masks \
  --img_ext .png \
  --mask_ext .tiff \
  --img_size 512 \
  --kfolds 5 \
  --run_all_folds True
```

### Threshold Sweep

```bash
python threshold_sweep.py \
  --model_dir models/unet_512_aug_5fold \
  --folds 5 \
  --checkpoint_name best_checkpoint.pth \
  --output_dir models/unet_512_aug_5fold/threshold_sweep
```

### Qualitative Overlay Analysis

```bash
python analyze_val_overlays.py \
  --overlay_dir models/unet_512_aug_5fold/fold_1/val_overlays/epoch_035 \
  --output_dir failure_analysis/unet_fold1_epoch035
```

## Evaluation Metrics

- Dice: prediction mask와 ground-truth mask의 overlap을 평가하는 metric입니다.
- IoU: prediction mask와 ground-truth mask의 intersection over union을 계산합니다.
- Precision: prediction foreground 중 실제 foreground인 비율을 나타냅니다.
- Recall: ground-truth foreground 중 model이 찾아낸 비율을 나타냅니다.

## My Contribution

- 연구 데이터에 맞는 image-mask pair 구성
- binary mask preprocessing 및 mask format 확인
- U-Net baseline 추가 또는 정리
- U-Net, U-NeXt, U-NeXt + BottleneckRefineBlock 비교 실험
- 5-fold cross validation 수행
- threshold sweep 기반 evaluation
- Dice, IoU, Precision, Recall 계산
- overlay 기반 정성 분석 및 failure case analysis

## Code Base Notice

이 repository의 코드는 기존 U-Net / U-NeXt 기반 구현을 참고하고, 학부 졸업논문 연구 목적에 맞게 수정 및 확장한 것입니다.

주요 수정 방향은 신장 병리 이미지의 사구체 binary semantic segmentation task에 맞춘 dataset loading, mask preprocessing, cross validation, threshold sweep, quantitative evaluation, overlay-based qualitative analysis입니다.

## Data and Privacy Notice

이 repository에는 다음 항목을 포함하지 않습니다.

- 원본 의료 이미지
- segmentation mask
- 환자 정보 또는 식별 가능 정보
- 병원 관련 비공개 정보
- checkpoint
- trained weight
- prediction image
- overlay image
- private dataset 기반 result image

사용한 데이터셋은 연구 목적으로 제공된 비공개 의료 데이터이며, public GitHub를 통해 공개 배포하지 않습니다.

## Limitations

- 소규모 데이터셋을 기반으로 한 실험입니다.
- 단일 기관 데이터일 가능성이 있어 일반화 성능에는 추가 검증이 필요합니다.
- 병리 이미지 annotation에는 boundary ambiguity가 존재할 수 있습니다.
- BottleneckRefineBlock 적용에 따른 개선 폭은 제한적일 수 있습니다.
- 공개 repository에는 private dataset과 trained checkpoint가 포함되지 않으므로, 동일한 수치 재현에는 별도 데이터 준비가 필요합니다.

## License and Third-Party Code

This repository is published solely for non-commercial research or evaluation.
Unless otherwise identified in the third-party notices, its distribution and
use are subject to the [NVIDIA Source Code License for SegFormer](LICENSE).
This restriction is preserved because the UNeXt-derived implementation
contains SegFormer-derived components.

The UNeXt implementation is adapted from the official
[UNeXt-pytorch](https://github.com/jeya-maria-jose/UNeXt-pytorch) repository,
which is distributed under the MIT License. UNeXt upstream also acknowledges
code blocks and helper functions from
[pytorch-nested-unet](https://github.com/4uiiurz1/pytorch-nested-unet),
[SegFormer](https://github.com/NVlabs/SegFormer), and
[AS-MLP](https://github.com/svip-lab/AS-MLP).

Lovász hinge support uses the separately installed
[LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax) implementation;
its source code is not included in this repository.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSES/](LICENSES/) for component mappings, copyright notices, and complete
license texts. Publication of this research code does not grant permission to
use the private clinical dataset, annotations, checkpoints, or derived patient
materials; none of those materials is included here.

## Citation

If this repository supports academic work, please cite the relevant original
methods:

- Valanarasu, J. M. J., and Patel, V. M. (2022).
  [UNeXt: MLP-based Rapid Medical Image Segmentation Network](https://arxiv.org/abs/2203.04967).
- Ronneberger, O., Fischer, P., and Brox, T. (2015).
  [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
- Berman, M., Rannen Triki, A., and Blaschko, M. B. (2018).
  [The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-over-Union Measure in Neural Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_A_Tractable_Surrogate_for_the_Optimization_CVPR_2018_paper.html).
- Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., and Luo, P. (2021).
  [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html).
