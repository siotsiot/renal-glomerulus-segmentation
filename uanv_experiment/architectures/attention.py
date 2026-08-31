"""Paper-guided locational attention used by the independent reimplementation.

Paper-specified operations:
* 3x3 convolution of the encoder (low-level) feature.
* global pooling and 1x1 convolution of the decoder (high-level) feature.
* element-wise multiplication.
* upsampling and residual addition to the original encoder feature.

Implementation choices approved for v1:
* low projection keeps the encoder channel count and uses stride=2, padding=1.
* global pooling is AdaptiveAvgPool2d(1).
* the high projection maps to the encoder channel count.
* no sigmoid, softmax, extra activation, or normalization is applied.
* bilinear interpolation uses align_corners=False.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ShapeTrace = Dict[str, Tuple[int, ...]]


class LocationalAttentionBlock(nn.Module):
    """Fuse one encoder skip with a decoder feature using paper-guided attention."""

    def __init__(self, encoder_channels: int, decoder_channels: int, stage_name: str) -> None:
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.stage_name = stage_name

        self.low_projection = nn.Conv2d(
            encoder_channels,
            encoder_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.high_pool = nn.AdaptiveAvgPool2d(1)
        self.high_projection = nn.Conv2d(decoder_channels, encoder_channels, kernel_size=1)

    @staticmethod
    def _shape(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(int(value) for value in tensor.shape)

    def forward(
        self,
        encoder_feature: torch.Tensor,
        decoder_feature: torch.Tensor,
        return_trace: bool = False,
    ):
        if encoder_feature.ndim != 4 or decoder_feature.ndim != 4:
            raise AssertionError(
                f"{self.stage_name}: encoder and decoder features must both be BCHW tensors; "
                f"got {encoder_feature.ndim}D and {decoder_feature.ndim}D"
            )
        if encoder_feature.shape[0] != decoder_feature.shape[0]:
            raise AssertionError(f"{self.stage_name}: encoder/decoder batch sizes differ")
        if encoder_feature.shape[1] != self.encoder_channels:
            raise AssertionError(
                f"{self.stage_name}: expected {self.encoder_channels} encoder channels, "
                f"got {encoder_feature.shape[1]}"
            )
        if decoder_feature.shape[1] != self.decoder_channels:
            raise AssertionError(
                f"{self.stage_name}: expected {self.decoder_channels} decoder channels, "
                f"got {decoder_feature.shape[1]}"
            )

        encoder_height, encoder_width = encoder_feature.shape[-2:]
        decoder_height, decoder_width = decoder_feature.shape[-2:]
        if encoder_height != decoder_height * 2 or encoder_width != decoder_width * 2:
            raise AssertionError(
                f"{self.stage_name}: approved v1 expects the encoder skip to be exactly 2x the "
                f"decoder spatial size; got encoder={encoder_feature.shape[-2:]}, "
                f"decoder={decoder_feature.shape[-2:]}"
            )

        low_projection = self.low_projection(encoder_feature)
        if low_projection.shape[-2:] != decoder_feature.shape[-2:]:
            raise AssertionError(
                f"{self.stage_name}: stride-2 low projection did not match decoder resolution; "
                f"low={low_projection.shape[-2:]}, decoder={decoder_feature.shape[-2:]}"
            )

        high_projection = self.high_projection(self.high_pool(decoder_feature))
        expected_high_shape = (encoder_feature.shape[0], self.encoder_channels, 1, 1)
        if tuple(high_projection.shape) != expected_high_shape:
            raise AssertionError(
                f"{self.stage_name}: high projection shape {tuple(high_projection.shape)} "
                f"does not match expected {expected_high_shape}"
            )

        multiplied_attention = low_projection * high_projection
        if multiplied_attention.shape != low_projection.shape:
            raise AssertionError(f"{self.stage_name}: unintended broadcasting changed attention shape")

        upsampled_attention = F.interpolate(
            multiplied_attention,
            size=encoder_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        corrected_skip = encoder_feature + upsampled_attention
        if corrected_skip.shape != encoder_feature.shape:
            raise AssertionError(f"{self.stage_name}: corrected skip does not match original encoder shape")

        if not return_trace:
            return corrected_skip

        trace: ShapeTrace = OrderedDict(
            [
                ("encoder_feature_E", self._shape(encoder_feature)),
                ("high_level_feature_G", self._shape(decoder_feature)),
                ("low_projection_L", self._shape(low_projection)),
                ("high_projection_H", self._shape(high_projection)),
                ("multiplied_attention_A", self._shape(multiplied_attention)),
                ("upsampled_attention_A", self._shape(upsampled_attention)),
                ("corrected_skip", self._shape(corrected_skip)),
            ]
        )
        return corrected_skip, trace
