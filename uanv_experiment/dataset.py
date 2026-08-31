"""Independent real-data loader matching the recovered integrated trainer.

Source data is treated as read-only. The foreground convention is explicit:
raw mask values greater than zero are foreground. Raw masks are asserted to
contain only the values observed during the read-only dataset audit: 0 and 255.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GlomerulusSegmentationDataset(Dataset):
    """Load paired images/masks without modifying or copying source files."""

    FOREGROUND_RULE = "raw_mask > 0"
    ALLOWED_RAW_MASK_VALUES = frozenset({0, 255})

    def __init__(
        self,
        sample_ids: Sequence[str],
        image_dir: str | Path,
        mask_dir: str | Path,
        image_extension: str = ".png",
        mask_extension: str = ".tiff",
        height: int = 512,
        width: int = 512,
        augment: bool = False,
    ) -> None:
        self.sample_ids = list(sample_ids)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        self.height = int(height)
        self.width = int(width)
        self.augment = bool(augment)
        if not self.sample_ids:
            raise ValueError("sample_ids must not be empty")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("a requested paired image could not be decoded")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        interpolation = cv2.INTER_AREA
        if image.shape[0] < self.height or image.shape[1] < self.width:
            interpolation = cv2.INTER_LINEAR
        image = cv2.resize(image, (self.width, self.height), interpolation=interpolation)
        return image.astype(np.float32) / 255.0

    def _read_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError("a requested paired mask could not be decoded")
        raw_values = {int(value) for value in np.unique(mask)}
        if not raw_values.issubset(self.ALLOWED_RAW_MASK_VALUES):
            raise AssertionError(
                "raw mask contains a value outside the audited set {0, 255}; "
                "foreground conversion was stopped"
            )
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.float32)[None, ...]

    @staticmethod
    def _augment_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=2)
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=1)
        rotations = int(np.random.randint(0, 4))
        if rotations:
            image = np.rot90(image, k=rotations, axes=(0, 1))
            mask = np.rot90(mask, k=rotations, axes=(1, 2))
        if np.random.rand() < 0.8:
            contrast = float(np.random.uniform(0.9, 1.1))
            brightness = float(np.random.uniform(-0.05, 0.05))
            image = np.clip(image * contrast + brightness, 0.0, 1.0)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[index]
        image = self._read_image(self.image_dir / f"{sample_id}{self.image_extension}")
        mask = self._read_mask(self.mask_dir / f"{sample_id}{self.mask_extension}")
        if self.augment:
            image, mask = self._augment_pair(image, mask)
        image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        return torch.from_numpy(image), torch.from_numpy(mask)
