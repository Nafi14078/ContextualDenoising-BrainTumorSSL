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
                 slices_dir:   str,
                 split:        str   = "train",
                 N:            int   = 7,
                 patch_size:   int   = 192,
                 augment:      bool  = True,
                 max_subjects: int   = None,
                 subset_seed:  int   = 42):
        """
        Args:
            slices_dir   : root dir produced by preprocess_brats2021.py
            split        : "train" or "val"
            N            : number of slices in each window (must be odd)
            patch_size   : random crop size
            augment      : apply augmentation (train only)
            max_subjects : if set, randomly keep only this many subjects.
                           Speeds up training dramatically since SSL
                           pretraining needs diversity, not volume —
                           150-200 subjects is typically plenty.
            subset_seed  : fixed seed so the same subjects are kept
                           across runs (reproducible subsampling)
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
        full_subject_slices  = self.subject_slices   # keep reference for verification

        # ── Optional subject-level subsampling for faster pretraining ──
        self.subsample_report = None
        if max_subjects is not None and len(self.subject_slices) > max_subjects:
            all_ids = sorted(self.subject_slices.keys())
            rng     = random.Random(subset_seed)
            kept_ids = set(rng.sample(all_ids, max_subjects))

            subsampled_slices = {
                k: v for k, v in full_subject_slices.items() if k in kept_ids
            }

            # ── Verify the subsample is distributionally representative ──
            self.subsample_report = self._verify_subsample(
                full_subject_slices, subsampled_slices, split)

            self.subject_slices = subsampled_slices

        # Build flat index: list of (subject_id, central_slice_local_idx)
        self.index = self._build_index()

        print(f"[Dataset] {split}: "
              f"{len(self.index)} windows from "
              f"{len(self.subject_slices)} subjects"
              + (f" (subsampled from {len(all_fnames)} total slices)"
                 if max_subjects is not None else ""))

    # ── subsample verification ──────────────────────────────────────────────

    def _verify_subsample(self,
                          full_slices: dict,
                          subset_slices: dict,
                          split: str) -> dict:
        """
        Compares the slices-per-subject distribution of the full dataset
        vs. the subsampled subset, to check whether random subsampling
        accidentally introduced a skew (e.g. toward subjects with more
        or fewer slices, which often correlates with brain/tumor size).

        Prints a summary table and, if scipy is available, runs a
        two-sample Kolmogorov-Smirnov test — a standard way to check
        whether two samples could plausibly come from the same
        underlying distribution. A high p-value (> 0.05) means there's
        no statistically significant evidence the subset distribution
        differs from the full dataset's.

        Returns a dict report (also useful to log/cite in a thesis
        methodology section instead of just asserting "randomly sampled").
        """
        full_counts   = np.array([len(v) for v in full_slices.values()])
        subset_counts = np.array([len(v) for v in subset_slices.values()])

        report = {
            "split":              split,
            "full_n_subjects":    len(full_counts),
            "subset_n_subjects":  len(subset_counts),
            "full_mean_slices":   float(full_counts.mean()),
            "subset_mean_slices": float(subset_counts.mean()),
            "full_std_slices":    float(full_counts.std()),
            "subset_std_slices":  float(subset_counts.std()),
            "full_min_slices":    int(full_counts.min()),
            "subset_min_slices":  int(subset_counts.min()),
            "full_max_slices":    int(full_counts.max()),
            "subset_max_slices":  int(subset_counts.max()),
        }

        print(f"\n[Subsample Verification — {split}]")
        print(f"  {'Metric':<22} {'Full':>12} {'Subset':>12}")
        print(f"  {'-'*22} {'-'*12} {'-'*12}")
        print(f"  {'N subjects':<22} {report['full_n_subjects']:>12} "
              f"{report['subset_n_subjects']:>12}")
        print(f"  {'Mean slices/subj':<22} {report['full_mean_slices']:>12.1f} "
              f"{report['subset_mean_slices']:>12.1f}")
        print(f"  {'Std slices/subj':<22} {report['full_std_slices']:>12.1f} "
              f"{report['subset_std_slices']:>12.1f}")
        print(f"  {'Min slices/subj':<22} {report['full_min_slices']:>12} "
              f"{report['subset_min_slices']:>12}")
        print(f"  {'Max slices/subj':<22} {report['full_max_slices']:>12} "
              f"{report['subset_max_slices']:>12}")

        try:
            from scipy.stats import ks_2samp
            ks_stat, ks_pvalue = ks_2samp(full_counts, subset_counts)
            report["ks_statistic"] = float(ks_stat)
            report["ks_pvalue"]    = float(ks_pvalue)

            verdict = ("representative (no significant difference)"
                      if ks_pvalue > 0.05
                      else "POTENTIALLY SKEWED (distributions differ)")

            print(f"  {'KS statistic':<22} {ks_stat:>12.4f}")
            print(f"  {'KS p-value':<22} {ks_pvalue:>12.4f}")
            print(f"  → Verdict: {verdict}")

            if ks_pvalue <= 0.05:
                print(f"  ⚠️  WARNING: subsample distribution differs "
                      f"significantly from full dataset.")
                print(f"     Consider increasing max_subjects or trying "
                      f"a different subset_seed.")
        except ImportError:
            print(f"  [scipy not installed — skipping KS test. "
                  f"pip install scipy for full statistical verification]")
            report["ks_statistic"] = None
            report["ks_pvalue"]    = None

        print()
        return report

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
            "central":  self.half,# index of central slice in the window
            "index":    idx       # STABLE identity of this (subject, slice)
                                   # pair — same value across every epoch,
                                   # regardless of DataLoader shuffling.
                                   # Used by ReferenceCache so the recurrent
                                   # L1 reference correctly tracks "this
                                   # exact slice's previous-epoch denoised
                                   # output", matching the paper's design
                                   # (Section 4), instead of "whatever
                                   # happened to be at this batch position."
        }


def get_pretrain_loaders(slices_dir:        str,
                         N:                 int = 7,
                         patch_size:        int = 192,
                         batch_size:        int = 4,
                         num_workers:       int = 2,
                         max_train_subjects: int = None,
                         max_val_subjects:   int = None):
    """
    Convenience function returning (train_loader, val_loader).

    max_train_subjects / max_val_subjects : optional subject-count caps
    for faster pretraining. SSL pretraining needs diversity, not volume —
    150-200 train subjects / 20-25 val subjects is typically plenty to
    learn strong generic denoising features that transfer well.
    """
    from torch.utils.data import DataLoader

    train_ds = BraTS2021SliceWindowDataset(
        slices_dir, split="train", N=N,
        patch_size=patch_size, augment=True,
        max_subjects=max_train_subjects)
    val_ds   = BraTS2021SliceWindowDataset(
        slices_dir, split="val", N=N,
        patch_size=patch_size, augment=False,
        max_subjects=max_val_subjects)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True)
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True)

    return train_loader, val_loader