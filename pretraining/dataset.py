"""
pretraining/dataset.py
────────────────────────────────────────────────────────────────────────────────
Dataset for STS-UVD pretraining.

Each sample returned is a SLICE WINDOW:
  - A stack of N consecutive axial slices centred on a target slice.
  - Shape: (N, 4, H, W)  — N slices, 4 modalities, patch H×W

The DataLoader collates these into (B, N, 4, H, W).
The STS module then uses slice index as the "temporal" axis.

Augmentation (train only):
  - Random crop to patch_size × patch_size
  - Random horizontal flip
  - Random 90° rotation
  - Gaussian noise jitter (simulate extra noise for pretraining signal)
────────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class BraTS2021SliceWindowDataset(Dataset):
    """
    Returns a window of N consecutive slices from the same subject.
    Slices are loaded from preprocessed .npy files (shape: 4, H, W).

    The dataset indexes into the *central* slice; neighbours are found
    by loading sibling files (same subject, adjacent indices).
    """

    def __init__(self,
                 slices_dir:  str,
                 split:       str   = "train",
                 N:           int   = 7,
                 patch_size:  int   = 192,
                 augment:     bool  = True):
        """
        Args:
            slices_dir : root dir produced by preprocess_brats2021.py
            split      : "train" or "val"
            N          : number of slices in each window (must be odd)
            patch_size : random crop size
            augment    : apply augmentation (train only)
        """
        assert N % 2 == 1, "N must be odd so there is a clear central slice"
        self.slices_dir = Path(slices_dir) / split
        self.split      = split
        self.N          = N
        self.half       = N // 2
        self.patch_size = patch_size
        self.augment    = augment and (split == "train")

        # Load metadata to know all slice file names
        meta_path = Path(slices_dir) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        all_fnames = meta[split]   # list of "SubjID_slice_NNN.npy"

        # Build subject → sorted slice list mapping
        self.subject_slices = self._group_by_subject(all_fnames)

        # Build flat index: list of (subject_id, central_slice_local_idx)
        self.index = self._build_index()

        print(f"[Dataset] {split}: "
              f"{len(self.index)} windows from "
              f"{len(self.subject_slices)} subjects")

    # ── private ──────────────────────────────────────────────────────────────

    def _group_by_subject(self, fnames: List[str]) -> dict:
        """
        Group filenames by subject ID.
        Filename format: BraTS2021_XXXXX_slice_NNN.npy
        Subject ID     : BraTS2021_XXXXX
        """
        groups = {}
        for fname in fnames:
            # Extract subject ID: everything before "_slice_"
            subj_id = fname.split("_slice_")[0]
            groups.setdefault(subj_id, []).append(fname)
        # Sort slices within each subject by slice index
        for subj_id in groups:
            groups[subj_id] = sorted(groups[subj_id])
        return groups

    def _build_index(self) -> List[Tuple[str, int]]:
        """
        Create (subject_id, local_slice_idx) pairs for valid central slices.
        A central slice is valid if it has half neighbours on each side.
        """
        index = []
        for subj_id, fnames in self.subject_slices.items():
            n_slices = len(fnames)
            for local_idx in range(self.half, n_slices - self.half):
                index.append((subj_id, local_idx))
        return index

    def _load_slice(self, subj_id: str, local_idx: int) -> torch.Tensor:
        """Load one .npy file → float32 tensor (4, H, W)."""
        fname = self.subject_slices[subj_id][local_idx]
        arr   = np.load(self.slices_dir / fname)        # (4, H, W)
        return torch.from_numpy(arr).float()

    def _random_crop_params(self, h: int, w: int):
        """Compute a random crop box once and reuse for all slices in window."""
        top  = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        return top, left

    def _apply_augment(self, window: torch.Tensor,
                       crop_top: int, crop_left: int,
                       hflip: bool, rot: int) -> torch.Tensor:
        """
        Apply identical spatial transform to all slices in the window.
        window shape: (N, 4, H, W)
        Returns: (N, 4, patch_size, patch_size)
        """
        out = []
        for i in range(self.N):
            sl = window[i]   # (4, H, W)
            # Crop
            sl = sl[:, crop_top:crop_top + self.patch_size,
                       crop_left:crop_left + self.patch_size]
            # Horizontal flip
            if hflip:
                sl = TF.hflip(sl)
            # 90° rotation
            if rot > 0:
                sl = torch.rot90(sl, k=rot, dims=[1, 2])
            out.append(sl)
        return torch.stack(out, dim=0)   # (N, 4, P, P)

    def _add_noise_jitter(self, window: torch.Tensor) -> torch.Tensor:
        """
        Add slight Gaussian noise to simulate extra corruption.
        This is the 'noisy input' for the denoiser — the target is the
        original window (acting as a pseudo-clean reference).
        σ sampled uniformly from [0.01, 0.05] each batch.
        """
        sigma = random.uniform(0.01, 0.05)
        noise = torch.randn_like(window) * sigma
        return (window + noise).clamp(0.0, 1.0)

    # ── public ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        subj_id, central_local = self.index[idx]

        # Load N consecutive slices
        slices = []
        for offset in range(-self.half, self.half + 1):
            sl = self._load_slice(subj_id, central_local + offset)
            slices.append(sl)
        window = torch.stack(slices, dim=0)   # (N, 4, H, W)

        # Augmentation — same transform applied to all N slices
        if self.augment:
            H, W    = window.shape[2], window.shape[3]
            top, left = self._random_crop_params(H, W)
            hflip     = random.random() > 0.5
            rot       = random.randint(0, 3)
            window    = self._apply_augment(window, top, left, hflip, rot)
        else:
            # Centre crop for validation
            H, W = window.shape[2], window.shape[3]
            top  = (H - self.patch_size) // 2
            left = (W - self.patch_size) // 2
            window_out = []
            for i in range(self.N):
                sl = window[i, :,
                            top:top + self.patch_size,
                            left:left + self.patch_size]
                window_out.append(sl)
            window = torch.stack(window_out, dim=0)

        # Clean window = original (before noise), noisy = with jitter
        clean  = window.clone()
        noisy  = self._add_noise_jitter(window)

        return {
            "noisy":    noisy,    # (N, 4, P, P) — model input
            "clean":    clean,    # (N, 4, P, P) — reconstruction target
            "central":  self.half # index of central slice in the window
        }


def get_pretrain_loaders(slices_dir: str,
                         N:          int = 7,
                         patch_size: int = 192,
                         batch_size: int = 4,
                         num_workers: int = 2):
    """Convenience function returning (train_loader, val_loader)."""
    from torch.utils.data import DataLoader

    train_ds = BraTS2021SliceWindowDataset(
        slices_dir, split="train", N=N,
        patch_size=patch_size, augment=True)
    val_ds   = BraTS2021SliceWindowDataset(
        slices_dir, split="val", N=N,
        patch_size=patch_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True)
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True)

    return train_loader, val_loader
