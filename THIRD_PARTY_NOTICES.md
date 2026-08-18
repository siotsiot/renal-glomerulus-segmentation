# Third-Party Notices

This repository is intended solely for non-commercial research or evaluation.
Unless a component is identified otherwise below, distribution and use of this
repository are subject to the NVIDIA Source Code License for SegFormer in
[LICENSE](LICENSE). Third-party license texts are preserved in
[LICENSES/](LICENSES/).

## UNeXt-pytorch

Affected code: unext_train.py, including shiftmlp, shiftedBlock, DWConv,
OverlapPatchEmbed, UNext, UNext_S, BCEDiceLoss, LovaszHingeLoss, and related
training and metric utilities, with project-specific modifications.

Upstream: https://github.com/jeya-maria-jose/UNeXt-pytorch

Copyright (c) 2022 Jeya Maria Jose

License: MIT License — [LICENSES/UNEXT-MIT.txt](LICENSES/UNEXT-MIT.txt)

The upstream UNeXt repository acknowledges code blocks and helper functions
from pytorch-nested-unet, SegFormer, and AS-MLP. The relevant upstream notices
are preserved below. This repository contains modifications for renal
glomerulus binary segmentation, dataset handling, cross-validation, threshold
evaluation, and qualitative analysis.

## SegFormer

Affected code: the UNeXt-derived implementation in unext_train.py, in
particular OverlapPatchEmbed, DWConv, weight-initialization logic, and related
block scaffolding.

Upstream: https://github.com/NVlabs/SegFormer

Copyright (c) 2021, NVIDIA Corporation. All rights reserved.

License: NVIDIA Source Code License for SegFormer — [LICENSE](LICENSE)

The license permits use and distribution only non-commercially, meaning for
research or evaluation purposes, and requires preservation of its license and
notices.

## AS-MLP

Relationship: acknowledged by the upstream UNeXt project as a source of
certain code blocks or helper functions.

Upstream: https://github.com/svip-lab/AS-MLP

Copyright (c) 2021 SVIP Lab

License: MIT License — [LICENSES/AS-MLP-MIT.txt](LICENSES/AS-MLP-MIT.txt)

## pytorch-nested-unet

Relationship: acknowledged by the upstream UNeXt project as a source of
certain code blocks or helper functions.

Upstream: https://github.com/4uiiurz1/pytorch-nested-unet

Copyright (c) 2018 Takato Kimura

License: MIT License —
[LICENSES/PYTORCH-NESTED-UNET-MIT.txt](LICENSES/PYTORCH-NESTED-UNET-MIT.txt)

## LovaszSoftmax

Affected code: unext_train.py optionally imports lovasz_hinge and provides a
small integration wrapper. LovaszSoftmax source code is not bundled in this
repository and must be installed separately if that loss is used.

Upstream: https://github.com/bermanmaxim/LovaszSoftmax

Copyright (c) 2018 Maxim Berman

License: MIT License —
[LICENSES/LOVASZSOFTMAX-MIT.txt](LICENSES/LOVASZSOFTMAX-MIT.txt)

## timm / PyTorch Image Models

Affected code: unext_train.py imports DropPath, to_2tuple, and trunc_normal_
from timm and contains a local DropPath fallback implementation.

Upstream: https://github.com/huggingface/pytorch-image-models

Copyright 2019 Ross Wightman

License: Apache License 2.0 —
[LICENSES/TIMM-APACHE-2.0.txt](LICENSES/TIMM-APACHE-2.0.txt)

## U-Net Architecture

The U-Net implementations in unet/model.py and unext_train.py follow the
architecture described by Ronneberger, Fischer, and Brox (2015). The original
paper is cited in the repository README.
