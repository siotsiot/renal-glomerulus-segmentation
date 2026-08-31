# Renal Glomerulus Segmentation

This repository contains two clearly separated generations of code for binary
glomerulus segmentation in renal histopathology images.

The primary, current experiment is the controlled four-architecture Internal
comparison prepared for the KoSAIM abstract. Its frozen production source is
under `uanv_experiment/`. The older root scripts and `unet/` directory are
retained as the graduation-thesis legacy pipeline. The two pipelines do not
share one evaluator or one checkpoint-selection policy and must not be treated
as interchangeable.

No clinical images, masks, annotations, sample identifiers, fold-assignment
CSVs, checkpoints, or patient information are distributed in this repository.

## Current KoSAIM experiment scope

The current public scope is limited to the clean Internal experiment:

- binary semantic segmentation;
- 361 image-mask pairs from one internal clinical dataset;
- four architectures trained on the same five prevalidated folds;
- direct RGB resize to 512×512;
- fixed 35-epoch training;
- epoch-35 checkpoint as the primary checkpoint;
- threshold 0.5 with the `>=` comparison operator;
- sample-macro Dice, IoU, precision, and recall;
- explicit empty-ground-truth metric handling.

External generalization experiments and subsequent NEPTUNE, Gallego,
appearance-augmentation, spatial-augmentation, and sliding-window work are not
part of this repository release.

## Architectures

The model labels accepted by the production runner are defined by
`uanv_experiment/registry.py`:

| CLI label | Implementation |
|---|---|
| `vanilla_unet` | `VanillaUNet` |
| `unext` | `UNext` |
| `unext_legacy_brb` | `UNext` with the legacy residual refinement block |
| `uanv_paper_inspired_attention_unet` | `UANVPaperInspiredAttentionUNet` |

The paper-inspired attention model is an independent, paper-guided
implementation. It is not the original UANV authors' official implementation.

## Repository structure

```text
.
├── uanv_experiment/              # current clean Internal experiment
│   ├── architectures/
│   ├── config.py                 # frozen common protocol settings
│   ├── dataset.py                # RGB image and binary mask input pipeline
│   ├── losses.py                 # weighted BCE-Dice loss
│   ├── metrics.py                # fixed-threshold sample-macro metrics
│   ├── protocol.json             # machine-readable locked protocol
│   ├── provenance.py             # run-layout and checkpoint-policy guards
│   ├── registry.py               # four-architecture registry
│   └── run_experiment.py         # one-model/one-fold production runner
├── folds/
│   └── README.md                 # fold schema only; no private IDs
├── unext_train.py                # graduation-thesis legacy pipeline
├── unet/                         # graduation-thesis legacy U-Net pipeline
├── threshold_sweep.py            # legacy analysis
├── eval_threshold_sweep.py       # legacy analysis
└── remaining root analysis files # legacy qualitative utilities
```

## Environment setup

The recorded production environment used Python 3.11.9, PyTorch
2.7.1+cu128, and CUDA 12.8. Exact production versions of NumPy, OpenCV, and
`timm` were not recorded in the run manifests, so they are listed without
unsupported exact pins.

Create an isolated environment and install the small dependency set:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU reproduction, install the PyTorch build appropriate for the local CUDA
environment following the official PyTorch installation instructions.

## Dataset contract

The production runner expects user-supplied data and fold files:

```text
data/
├── images/
│   └── <id>.png
└── masks/
    └── <id>.tiff

folds/
├── fold_1/
│   ├── paired_ids.csv
│   ├── train_ids.csv
│   └── val_ids.csv
├── fold_2/
│   └── ...
└── fold_5/
    └── ...
```

Each CSV has exactly one column named `id`. Image and mask stems must match the
IDs. The audited Internal masks contained values 0 and 255; the production
foreground rule is `raw_mask > 0`. See `folds/README.md` for the schema and
aggregate fold sizes.

## Validate the CLI and fold contract

`--dry-run` validates the locked policy, fold CSV structure, and output-path
collision protection without loading the image dataset or starting training:

```powershell
python .\uanv_experiment\run_experiment.py `
  --model vanilla_unet `
  --fold 1 `
  --production-scope all_folds_locked `
  --image-dir .\data\images `
  --mask-dir .\data\masks `
  --fold-root .\folds `
  --run-dir .\runs\fold_1\vanilla_unet `
  --device cuda `
  --dry-run
```

## Training command

One invocation runs exactly one registered model on one existing fold. The
runner refuses to overwrite an existing run directory and requires the
`uanv_experiment/` source to match a clean committed Git snapshot for a
non-dry production run.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'

python .\uanv_experiment\run_experiment.py `
  --model vanilla_unet `
  --fold 1 `
  --production-scope all_folds_locked `
  --image-dir .\data\images `
  --mask-dir .\data\masks `
  --fold-root .\folds `
  --run-dir .\runs\fold_1\vanilla_unet `
  --device cuda
```

Repeat the command with the four registry labels and folds 1–5 while reusing
the same fold CSVs for every architecture.

## Primary evaluation

There is no separate evaluation-only CLI in the frozen production snapshot.
After epoch 35, the command above reloads `epoch_35_checkpoint.pth`, evaluates
the validation fold at threshold 0.5 using `>=`, and writes:

```text
runs/<fold>/<model>/primary_metrics_threshold_0.5.csv
```

It also writes an explicitly secondary exploratory threshold sweep. The
best-validation checkpoint and threshold sweep are not primary-result eligible
under `uanv_experiment/protocol.json`.

## Five-fold protocol and reproducibility scope

The four models used identical assignments within each fold:

| Fold | Training | Validation | Total |
|---:|---:|---:|---:|
| 1 | 288 | 73 | 361 |
| 2 | 289 | 72 | 361 |
| 3 | 289 | 72 | 361 |
| 4 | 289 | 72 | 361 |
| 5 | 289 | 72 | 361 |

The original fold CSVs are not public because they contain identifiers from a
non-public clinical dataset. Consequently, this repository publishes the
training/evaluation implementation and locked experimental protocol, but does
not claim that the exact original split can be reconstructed from public files.
Patient/group independence of the historical assignments was not verifiable.

## Data and code availability

The clinical dataset is not publicly distributed. The public code can be used
with appropriately authorized data prepared according to the documented image,
mask, and fold contracts. Checkpoints, run directories, raw metrics artifacts,
and private identifiers are not included.

`PROVENANCE_MANIFEST.json` records only production source hashes and source
snapshot identifiers. It contains no sample identifiers or workstation paths.

The frozen `uanv_experiment/protocol.json` retains a relative reference to a
historical parity source. That file is not distributed here, the reference is
not used by the production training path, and it is preserved only so the
published production source remains byte-identical to the audited snapshot.

## Graduation-thesis legacy pipeline

The pre-existing root scripts and `unet/` directory are preserved for history:

- `unext_train.py` contains the earlier U-Net/UNeXt/legacy-BRB integrated
  training workflow and dynamically generated cross-validation splits;
- `unet/` contains the earlier standalone U-Net workflow;
- threshold-sweep, overlay, representative-case, and sanity-check scripts are
  earlier exploratory and qualitative utilities.

These files remain available at their original paths, but they are not the
production source for the clean KoSAIM four-model comparison.

## License and upstream attribution

The existing project `LICENSE`, files under `LICENSES/`, and
`THIRD_PARTY_NOTICES.md` remain in effect. The UNeXt lineage, SegFormer-related
components acknowledged by upstream UNeXt, `timm`, and the paper-guided UANV
concept are described in `THIRD_PARTY_NOTICES.md`.

Human review is still required for the attribution and redistribution status
of the independently implemented attention/UANV-related source before release.
No statement in this README grants access to or redistribution rights for the
clinical data.

## Citation

Citation details for the KoSAIM abstract will be added when the final
bibliographic record is available.

Method references:

- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for
  Biomedical Image Segmentation. MICCAI 2015.
- Valanarasu JMJ, Patel VM. UNeXt: MLP-based Rapid Medical Image Segmentation
  Network. MICCAI 2022.
