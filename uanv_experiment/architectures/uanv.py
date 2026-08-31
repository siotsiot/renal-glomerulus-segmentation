"""Independent paper-guided UANV-style attention U-Net.

Independent paper-guided reimplementation.
This is not the authors' official UANV implementation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch

from .attention import LocationalAttentionBlock, ShapeTrace
from .unet import VanillaUNet


class UANVPaperInspiredAttentionUNet(VanillaUNet):
    """VanillaUNet with one approved v1 attention block on each of four skips."""

    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            base_channels=base_channels,
        )

        self.attention_stage4 = LocationalAttentionBlock(
            encoder_channels=base_channels * 8,
            decoder_channels=base_channels * 16,
            stage_name="attention_stage4",
        )
        self.attention_stage3 = LocationalAttentionBlock(
            encoder_channels=base_channels * 4,
            decoder_channels=base_channels * 8,
            stage_name="attention_stage3",
        )
        self.attention_stage2 = LocationalAttentionBlock(
            encoder_channels=base_channels * 2,
            decoder_channels=base_channels * 4,
            stage_name="attention_stage2",
        )
        self.attention_stage1 = LocationalAttentionBlock(
            encoder_channels=base_channels,
            decoder_channels=base_channels * 2,
            stage_name="attention_stage1",
        )

    @staticmethod
    def _apply_attention(
        block: LocationalAttentionBlock,
        encoder_feature: torch.Tensor,
        decoder_feature: torch.Tensor,
        return_attention_shapes: bool,
    ):
        return block(
            encoder_feature,
            decoder_feature,
            return_trace=return_attention_shapes,
        )

    def forward(self, x: torch.Tensor, return_attention_shapes: bool = False):
        if x.ndim != 4:
            raise AssertionError(f"input must be BCHW, got {x.ndim}D")
        if x.shape[1] != self.input_channels:
            raise AssertionError(f"expected {self.input_channels} input channels, got {x.shape[1]}")
        if x.shape[-2] % 16 != 0 or x.shape[-1] % 16 != 0:
            raise AssertionError(f"input spatial dimensions must be divisible by 16, got {x.shape[-2:]}")

        traces: Dict[str, ShapeTrace] = OrderedDict()

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        stage4 = self._apply_attention(self.attention_stage4, e4, b, return_attention_shapes)
        if return_attention_shapes:
            e4_corrected, traces["attention_stage4"] = stage4
        else:
            e4_corrected = stage4
        d4 = self.match_size(self.up4(b), e4_corrected)
        d4 = self.dec4(torch.cat([d4, e4_corrected], dim=1))

        stage3 = self._apply_attention(self.attention_stage3, e3, d4, return_attention_shapes)
        if return_attention_shapes:
            e3_corrected, traces["attention_stage3"] = stage3
        else:
            e3_corrected = stage3
        d3 = self.match_size(self.up3(d4), e3_corrected)
        d3 = self.dec3(torch.cat([d3, e3_corrected], dim=1))

        stage2 = self._apply_attention(self.attention_stage2, e2, d3, return_attention_shapes)
        if return_attention_shapes:
            e2_corrected, traces["attention_stage2"] = stage2
        else:
            e2_corrected = stage2
        d2 = self.match_size(self.up2(d3), e2_corrected)
        d2 = self.dec2(torch.cat([d2, e2_corrected], dim=1))

        stage1 = self._apply_attention(self.attention_stage1, e1, d2, return_attention_shapes)
        if return_attention_shapes:
            e1_corrected, traces["attention_stage1"] = stage1
        else:
            e1_corrected = stage1
        d1 = self.match_size(self.up1(d2), e1_corrected)
        d1 = self.dec1(torch.cat([d1, e1_corrected], dim=1))

        logits = self.final(d1)
        if logits.shape[0] != x.shape[0] or logits.shape[-2:] != x.shape[-2:]:
            raise AssertionError(
                f"output batch/spatial shape must match input; input={tuple(x.shape)}, output={tuple(logits.shape)}"
            )

        if return_attention_shapes:
            return logits, traces
        return logits
