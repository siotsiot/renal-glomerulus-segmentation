"""Loss copied from the historical integrated baseline for parity."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 1.0, smooth: float = 1e-5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probabilities = torch.sigmoid(logits)
        batch_size = target.size(0)
        probabilities = probabilities.view(batch_size, -1)
        target = target.view(batch_size, -1)
        intersection = probabilities * target
        dice = (2.0 * intersection.sum(1) + self.smooth) / (
            probabilities.sum(1) + target.sum(1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss
