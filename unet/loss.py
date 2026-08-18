import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)

        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        target = target.view(-1)
        intersection = (probs * target).sum()
        dice = 1 - (2.0 * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth)

        return self.bce_weight * bce + self.dice_weight * dice
