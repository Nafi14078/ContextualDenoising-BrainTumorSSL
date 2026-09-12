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

NOTE ON MULTI-GPU:
  This file needs NO changes to support multiple GPUs. nn.DataParallel
  splits whichever batch this DataLoader produces evenly across GPUs at
  the *model* level, so all the multi-GPU logic lives in train_finetune.py.
  The only thing that matters here is feeding the GPUs fast enough — see
  num_workers / persistent_workers / prefetch_factor below.
────────────────────────────────────────────────────────────────────────────────
"""

import json
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
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

    def compute_et_sample_weights(self, et_oversample_factor: float = 5.0) -> list:
        """
        Scans every training mask ONCE (fast — just reading small uint8
        .npy arrays, no image loading) and returns a per-sample weight list:
        `et_oversample_factor` for slices containing any ET (label 3)
        pixels, 1.0 for everything else. Feed this into a
        torch.utils.data.WeightedRandomSampler to oversample ET-positive
        slices during training — ET is your weakest, most imbalanced
        region, and most slices in this dataset contain little/no ET.
        """
        weights = []
        n_et_positive = 0
        for fname in self.fnames:
            mask = np.load(self.mask_dir / fname, mmap_mode='r')
            is_et = bool((mask == 3).any())
            weights.append(et_oversample_factor if is_et else 1.0)
            n_et_positive += is_et

        print(f"[Dataset BraTS-PED] {n_et_positive}/{len(self.fnames)} "
              f"train slices contain ET — weighted {et_oversample_factor}x "
              f"vs 1.0x for the rest")
        return weights

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
        top  = random.randint(0, max(H - P, 0))
        left = random.randint(0, max(W - P, 0))
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
        top  = max((H - P) // 2, 0)
        left = max((W - P) // 2, 0)
        return (image[:, top:top+P, left:left+P],
                mask[   top:top+P, left:left+P])


def get_finetune_loaders(slices_dir:  str,
                         patch_size:  int = 192,
                         batch_size:  int = 8,
                         num_workers: int = 4,
                         oversample_et: bool = False,
                         et_oversample_factor: float = 5.0):
    """
    Build train/val DataLoaders.

    num_workers default bumped 2 -> 4: with 2 GPUs training in parallel via
    DataParallel, the model consumes batches roughly 2x faster, so the
    CPU-side data pipeline needs more worker processes to avoid GPUs sitting
    idle waiting for data. persistent_workers + prefetch_factor keep the
    workers warm between epochs instead of respawning every epoch.

    oversample_et: if True, builds a WeightedRandomSampler that samples
      ET-positive train slices `et_oversample_factor`x more often than
      other slices — most slices in this dataset have little/no ET, which
      dilutes the training signal for your weakest, most imbalanced
      region. Mutually exclusive with `shuffle=True` (a sampler replaces
      plain shuffling; it still draws a random slice each time, just with
      non-uniform probability), so shuffle is dropped automatically when
      this is on. `drop_last=True` is kept either way for stable batch
      shapes under multi-GPU.
    """
    train_ds = BraTSPEDSliceDataset(
        slices_dir, "train", patch_size, augment=True)
    val_ds   = BraTSPEDSliceDataset(
        slices_dir, "val",   patch_size, augment=False)

    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = 4

    if oversample_et:
        sample_weights = train_ds.compute_et_sample_weights(et_oversample_factor)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler, drop_last=True, **common_kwargs)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, drop_last=True, **common_kwargs)

    val_loader = DataLoader(
        val_ds,   batch_size=batch_size,
        shuffle=False, **common_kwargs)

    return train_loader, val_loader