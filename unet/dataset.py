from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GlomerulusDataset(Dataset):
    def __init__(
        self,
        images_dir="./data/images",
        masks_dir="./data/masks",
        image_ext=".png",
        mask_ext=".tiff",
        resize_hw=(512, 512),
        transform=None,
        foreground_value=0,
        debug=False,
        debug_max_prints=5,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        if resize_hw is None:
            self.resize_hw = None
        elif isinstance(resize_hw, int):
            self.resize_hw = (resize_hw, resize_hw)
        else:
            self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
        self.transform = transform
        self.foreground_value = foreground_value
        self.debug = debug
        self.debug_max_prints = debug_max_prints
        self._debug_print_count = 0

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        image_paths = sorted(self.images_dir.glob(f"*{self.image_ext}"))
        if not image_paths:
            raise RuntimeError(f"No images found in {self.images_dir} with extension {self.image_ext}")

        self.samples = []
        missing_masks = []

        for image_path in image_paths:
            stem = image_path.stem
            mask_path = self.masks_dir / f"{stem}{self.mask_ext}"

            # Common fallback for .tif/.tiff mismatch
            if not mask_path.exists() and self.mask_ext.lower() == ".tiff":
                tif_mask_path = self.masks_dir / f"{stem}.tif"
                if tif_mask_path.exists():
                    mask_path = tif_mask_path

            if mask_path.exists():
                self.samples.append((image_path, mask_path))
            else:
                missing_masks.append(stem)

        if missing_masks:
            preview = ", ".join(missing_masks[:10])
            raise FileNotFoundError(
                f"Matching masks not found for {len(missing_masks)} image(s). "
                f"Expected mask stem match in {self.masks_dir}. Example(s): {preview}"
            )

        if self.debug and self.samples:
            first_mask_path = self.samples[0][1]
            first_mask = cv2.imread(str(first_mask_path), cv2.IMREAD_UNCHANGED)
            if first_mask is not None:
                if first_mask.ndim == 3:
                    first_mask = cv2.cvtColor(first_mask, cv2.COLOR_BGR2GRAY)
                print(
                    f"[Dataset Debug] first mask={first_mask_path.name}, "
                    f"raw_unique={np.unique(first_mask).tolist()}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        if raw_mask.ndim == 3:
            raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)

        if self.resize_hw is not None:
            target_h, target_w = self.resize_hw
            image_interp = cv2.INTER_AREA if image.shape[0] >= target_h and image.shape[1] >= target_w else cv2.INTER_LINEAR
            image = cv2.resize(image, (target_w, target_h), interpolation=image_interp)
            raw_mask = cv2.resize(raw_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        # Glomerulus=0, Background=255 polarity
        mask = (raw_mask == self.foreground_value).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.debug and self._debug_print_count < self.debug_max_prints:
            print(
                f"[Dataset Debug] idx={idx}, image_shape={tuple(image.shape)}, "
                f"mask_shape={tuple(mask.shape)}, raw_mask_unique={np.unique(raw_mask).tolist()}, "
                f"bin_mask_unique={torch.unique(mask).cpu().tolist()}"
            )
            self._debug_print_count += 1

        return image, mask


# Backward compatibility for existing imports
SpineDataset = GlomerulusDataset
