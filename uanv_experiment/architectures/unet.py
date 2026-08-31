"""Independent copy of the integrated experiment's Vanilla U-Net baseline.

The architecture is intentionally kept equivalent to ``VanillaUNet`` in the
historical integrated trainer.  Existing baseline source files are not imported
or modified at runtime, so this experiment remains self-contained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaUNet(nn.Module):
    """Four-level U-Net used as the architecture-matched baseline."""

    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.base_channels = base_channels

        self.enc1 = self.conv_block(input_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    @staticmethod
    def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def match_size(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == skip.shape[-2:]:
            return x
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.match_size(self.up4(b), e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.match_size(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.match_size(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.match_size(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)
