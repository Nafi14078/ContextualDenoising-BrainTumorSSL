"""
pretraining/reference_cache.py
────────────────────────────────────────────────────────────────────────────────
Disk-backed recurrent reference cache — implements the paper's Section 4
recurrent optimization:

    "we update the reference frame I_ref_t recurrently using the output
     denoised frame I_t from the previous epoch."

Why this exists / what it fixes vs. a naive dict-based cache
──────────────────────────────────────────────────────────────
A first attempt at this stored one float32 tensor per training sample in a
plain Python dict (47,115 samples x ~576KB = ~26GB), which exhausted RAM on
Kaggle. It also had a correctness bug: since training uses a fresh random
crop every epoch, "previous epoch's denoised output" referred to a
DIFFERENT spatial crop than the current epoch's crop — the L1 comparison
wasn't even spatially aligned.

This version fixes both:
  1. MEMORY  — values live in a `float16` `numpy.memmap` file on disk, not
     in a Python dict in RAM. The OS pages it in/out as needed. Total disk
     footprint for N samples of shape (C, P, P) is N*C*P*P*2 bytes (e.g.
     ~13GB for 47,115 samples at patch_size=192 — fine as a disk file,
     never fully resident in RAM).
  2. ALIGNMENT — this cache assumes the caller uses a per-sample-index
     DETERMINISTIC crop/flip/rotation (see dataset.py's `_augment_params`,
     seeded by `idx`) so "sample idx's patch" refers to the same physical
     region of the volume on every epoch. Only pixel noise jitter differs
     epoch to epoch, which is orthogonal to spatial alignment. Without
     that change, this cache would silently reintroduce the misalignment
     bug — it does NOT re-derive or check crop coordinates itself.

A companion boolean `valid` memmap tracks which entries have been written
at least once, so epoch 0 (and any sample not yet seen) correctly falls
back to the noisy central slice, matching the paper's own bootstrap
behaviour ("On epoch 0, reference = original noisy central slice.").
────────────────────────────────────────────────────────────────────────────────
"""

import json
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


class RecurrentReferenceCache:
    """
    Disk-backed, per-sample-index cache of "previous epoch's denoised
    central slice", used as the recurrent L1 reference I_ref_t.

    Usage:
        cache = RecurrentReferenceCache(
            path=ckpt_dir / "reference_cache",
            num_samples=len(train_loader.dataset),
            shape=(4, patch_size, patch_size))

        # ── at read time (building the reference for L1) ──
        ref_vals, ref_valid = cache.get_batch(batch["index"])
        reference = torch.where(ref_valid, ref_vals, noisy_central)

        # ── at write time (after computing this epoch's denoised output) ──
        cache.set_batch(batch["index"], denoised_central.detach())
    """

    def __init__(self,
                 path:        "str | Path",
                 num_samples: int,
                 shape:       Tuple[int, int, int],
                 dtype=np.float16):
        """
        Args:
            path        : base path (no extension) for the cache files.
                          Creates `<path>.dat`, `<path>.valid`, `<path>.meta.json`.
            num_samples : total number of dataset samples (rows in the cache).
            shape       : (C, H, W) shape of one cached tensor (should match
                          the denoised central slice, e.g. (4, patch, patch)).
            dtype       : storage dtype. float16 halves disk usage vs float32;
                          values are cast back to float32 on read since the
                          rest of the pipeline (AMP aside) expects float32.
        """
        self.path        = Path(path)
        self.num_samples = num_samples
        self.shape        = tuple(shape)
        self.dtype        = dtype
        self.full_shape    = (num_samples,) + self.shape

        self.data_path  = Path(str(self.path) + ".dat")
        self.valid_path = Path(str(self.path) + ".valid")
        self.meta_path  = Path(str(self.path) + ".meta.json")

        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_or_validate_files()

        self.data = np.memmap(self.data_path, dtype=self.dtype,
                              mode="r+", shape=self.full_shape)
        self.valid = np.memmap(self.valid_path, dtype=np.bool_,
                               mode="r+", shape=(num_samples,))

    # ── setup ────────────────────────────────────────────────────────────────

    def _init_or_validate_files(self):
        meta = {
            "num_samples": self.num_samples,
            "shape":       list(self.shape),
            "dtype":       np.dtype(self.dtype).name,
        }

        if self.data_path.exists() and self.valid_path.exists() and self.meta_path.exists():
            with open(self.meta_path) as f:
                existing_meta = json.load(f)
            if existing_meta != meta:
                warnings.warn(
                    f"[RecurrentReferenceCache] Existing cache at {self.path} "
                    f"has metadata {existing_meta}, but this run expects "
                    f"{meta}. This usually means max_train_subjects, N, or "
                    f"patch_size changed since the cache was created. "
                    f"Resetting the cache — the recurrent reference will "
                    f"restart from the noisy-central-slice bootstrap for "
                    f"a few epochs.")
                self._create_files(meta)
            # else: existing files are compatible — reuse them as-is
            #       (this is what makes --resume "just work": the cache
            #       already holds whatever was written before the run
            #       was interrupted, keyed by sample index).
        else:
            self._create_files(meta)

    def _create_files(self, meta: dict):
        mm = np.memmap(self.data_path, dtype=self.dtype, mode="w+",
                       shape=self.full_shape)
        mm.flush()
        del mm

        v = np.memmap(self.valid_path, dtype=np.bool_, mode="w+",
                      shape=(self.num_samples,))
        v[:] = False
        v.flush()
        del v

        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

    # ── read / write ─────────────────────────────────────────────────────────

    def get_batch(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            indices : (B,) LongTensor of sample indices (dataset's stable
                      "index" field — NOT the batch position).
        Returns:
            values : (B, C, H, W) float32 tensor — cached previous-epoch
                     denoised output, or garbage where valid[i] is False
                     (caller MUST gate on `valid`, e.g. via torch.where).
            valid  : (B, 1, 1, 1) bool tensor — True where a real cached
                     value exists for that index.
        """
        idx_np = indices.detach().cpu().numpy()
        vals   = np.asarray(self.data[idx_np], dtype=np.float32)
        val_ok = np.asarray(self.valid[idx_np])

        values = torch.from_numpy(vals)
        valid  = torch.from_numpy(val_ok).view(-1, 1, 1, 1)
        return values, valid

    def set_batch(self, indices: torch.Tensor, values: torch.Tensor):
        """
        Args:
            indices : (B,) LongTensor of sample indices.
            values  : (B, C, H, W) tensor — this epoch's denoised central
                     slice, to become next epoch's reference for these
                     samples. Should already be detached from the graph.
        """
        idx_np  = indices.detach().cpu().numpy()
        vals_np = values.detach().cpu().numpy().astype(self.dtype)

        self.data[idx_np]  = vals_np
        self.valid[idx_np] = True
        # Flush periodically rather than every call is possible, but for
        # a Kaggle-scale dataset flushing every batch is cheap relative to
        # a training step and guarantees no lost writes if the kernel
        # is killed mid-epoch.
        self.data.flush()
        self.valid.flush()

    def fraction_populated(self) -> float:
        """Debug helper: fraction of samples that have a cached value."""
        return float(np.asarray(self.valid).mean())