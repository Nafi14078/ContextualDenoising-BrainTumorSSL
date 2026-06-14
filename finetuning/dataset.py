"""
finetuning/dataset.py
────────────────────────────────────────────────────────────────────────────────
Dataset for BraTS-PED 2D segmentation fine-tuning.

Returns:
  image : (4, H, W)  float32   — 4 modalities, normalised
  mask  : (H, W)     long      — label {0,1,2,3}

Augmentation (train only, applied identically to image and mask):
  • Random crop 192×192
  • Random horizontal + vertical flip
  • Random 90° rotation
  • Random intensity scale/shift per modality (image only)
  • Random Gaussian noise (image only, σ ~ U[0, 0.02])
────────────────────────────────────────────────────────────────────────────────
"""

import json
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class BraTSPEDSliceDataset(Dataset):

    def __init__(self,
                 slices_dir:  str,
                 split:       str   = "train",
                 patch_size:  int   = 192,
                 augment:     bool  = True,
                 num_classes: int   = 4):
        self.img_dir    = Path(slices_dir) / split / "images"
        self.mask_dir   = Path(slices_dir) / split / "masks"
        self.patch_size = patch_size
        self.augment    = augment and (split == "train")
        self.num_classes = num_classes

        meta_path = Path(slices_dir) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        self.fnames = meta[split]

        print(f"[Dataset BraTS-PED] {split}: {len(self.fnames)} slices")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]

        image = torch.from_numpy(
            np.load(self.img_dir / fname)).float()          # (4, H, W)
        mask  = torch.from_numpy(
            np.load(self.mask_dir / fname)).long()          # (H, W)

        if self.augment:
            image, mask = self._augment(image, mask)
        else:
            image, mask = self._centre_crop(image, mask)

        return {"image": image, "mask": mask}

    # ── augmentation ─────────────────────────────────────────────────────────

    def _augment(self, image, mask):
        H, W = image.shape[1], image.shape[2]
        P    = self.patch_size

        # Random crop
        top  = random.randint(0, H - P)
        left = random.randint(0, W - P)
        image = image[:, top:top+P, left:left+P]
        mask  = mask[   top:top+P, left:left+P]

        # Random flips
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k=k, dims=[1, 2])
            mask  = torch.rot90(mask.unsqueeze(0), k=k,
                                dims=[1, 2]).squeeze(0)

        # Intensity jitter (image only, per modality)
        for m in range(image.shape[0]):
            scale = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.05, 0.05)
            image[m] = (image[m] * scale + shift).clamp(0.0, 1.0)

        # Gaussian noise (image only)
        sigma = random.uniform(0.0, 0.02)
        image = (image + torch.randn_like(image) * sigma).clamp(0.0, 1.0)

        return image, mask

    def _centre_crop(self, image, mask):
        H, W = image.shape[1], image.shape[2]
        P    = self.patch_size
        top  = (H - P) // 2
        left = (W - P) // 2
        return (image[:, top:top+P, left:left+P],
                mask[   top:top+P, left:left+P])


def get_finetune_loaders(slices_dir:  str,
                         patch_size:  int = 192,
                         batch_size:  int = 8,
                         num_workers: int = 2):
    from torch.utils.data import DataLoader

    train_ds = BraTSPEDSliceDataset(
        slices_dir, "train", patch_size, augment=True)
    val_ds   = BraTSPEDSliceDataset(
        slices_dir, "val",   patch_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,  num_workers=num_workers,
        pin_memory=True, drop_last=True)
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True)

    return train_loader, val_loader
