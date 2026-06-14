"""
preprocess_brats2021.py
────────────────────────────────────────────────────────────────────────────────
Converts BraTS2021 3D NIfTI volumes → 2D axial .npy slices for pretraining.

Expected BraTS2021 folder layout:
  brats2021_root/
    BraTS2021_00000/
      BraTS2021_00000_t1.nii.gz
      BraTS2021_00000_t1ce.nii.gz
      BraTS2021_00000_t2.nii.gz
      BraTS2021_00000_flair.nii.gz
      BraTS2021_00000_seg.nii.gz   ← not used in pretraining

Output layout (saved to output_dir):
  slices/
    train/
      BraTS2021_00000_slice_045.npy   shape: (4, H, W)  float32 normalised
      ...
    val/
      ...
    metadata.json    ← slice counts, min/max per modality, split info

Run on Kaggle:
  python preprocess_brats2021.py \
    --root /kaggle/input/brats2021/BraTS2021_Training_Data \
    --out  /kaggle/working/slices \
    --skip 10 \
    --min_brain 0.05
────────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import argparse
import random
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm


# ── helpers ──────────────────────────────────────────────────────────────────

def load_volume(path: Path) -> np.ndarray:
    """Load a NIfTI file and return float32 numpy array (H, W, D)."""
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def normalise_volume(vol: np.ndarray) -> np.ndarray:
    """
    Z-score normalise within the brain mask (non-zero voxels).
    Clips to [-5, 5] then rescales to [0, 1] for stable training.
    """
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    mu  = vol[mask].mean()
    std = vol[mask].std() + 1e-8
    vol = (vol - mu) / std
    vol = np.clip(vol, -5.0, 5.0)
    vol = (vol + 5.0) / 10.0       # → [0, 1]
    return vol.astype(np.float32)


def brain_ratio(slice_2d: np.ndarray) -> float:
    """Fraction of non-zero pixels — proxy for 'is this a useful slice'."""
    return (slice_2d > 0).mean()


def extract_slices(subject_dir: Path,
                   modalities: list,
                   skip: int,
                   min_brain: float,
                   axis: int = 2):
    """
    Returns list of 2D arrays, each shaped (len(modalities), H, W).
    Only slices passing the brain-content threshold are returned.
    """
    volumes = []
    for mod in modalities:
        candidates = list(subject_dir.glob(f"*_{mod}.nii.gz"))
        if not candidates:
            raise FileNotFoundError(
                f"Missing modality '{mod}' in {subject_dir}")
        vol = load_volume(candidates[0])
        vol = normalise_volume(vol)
        volumes.append(vol)         # each: (H, W, D)

    D = volumes[0].shape[axis]      # depth along chosen axis
    slices_out = []

    for z in range(skip, D - skip):
        # Extract 2D slice from each modality
        if axis == 2:
            stack = np.stack([v[:, :, z] for v in volumes], axis=0)  # (M,H,W)
        elif axis == 1:
            stack = np.stack([v[:, z, :] for v in volumes], axis=0)
        else:
            stack = np.stack([v[z, :, :] for v in volumes], axis=0)

        # Quality filter: check first modality (T1)
        if brain_ratio(stack[0]) < min_brain:
            continue

        slices_out.append(stack)

    return slices_out   # list of (M, H, W) float32 arrays


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)
    np.random.seed(42)

    root       = Path(args.root)
    out_dir    = Path(args.out)
    modalities = ["t1", "t1ce", "t2", "flair"]
    axis       = 2          # axial
    skip       = args.skip
    min_brain  = args.min_brain
    val_frac   = 0.1

    # Find all subject directories
    subjects = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Found {len(subjects)} subjects")

    # Train / val split at subject level (important: no data leakage)
    random.shuffle(subjects)
    n_val   = max(1, int(len(subjects) * val_frac))
    val_set = set(s.name for s in subjects[:n_val])

    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    metadata = {"train": [], "val": [], "modalities": modalities,
                "axis": axis, "skip": skip, "min_brain": min_brain}

    total_train, total_val = 0, 0

    for subj in tqdm(subjects, desc="Processing subjects"):
        split = "val" if subj.name in val_set else "train"
        try:
            slices = extract_slices(subj, modalities, skip, min_brain, axis)
        except FileNotFoundError as e:
            print(f"  [WARN] {e} — skipping")
            continue

        for idx, sl in enumerate(slices):
            fname = f"{subj.name}_slice_{idx:03d}.npy"
            np.save(out_dir / split / fname, sl)
            metadata[split].append(fname)

        if split == "train":
            total_train += len(slices)
        else:
            total_val += len(slices)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Done")
    print(f"  Train slices : {total_train}")
    print(f"  Val slices   : {total_val}")
    print(f"  Saved to     : {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      required=True,
                        help="Path to BraTS2021 training root")
    parser.add_argument("--out",       required=True,
                        help="Output directory for .npy slices")
    parser.add_argument("--skip",      type=int,   default=10,
                        help="Skip N slices at each end (background)")
    parser.add_argument("--min_brain", type=float, default=0.05,
                        help="Min brain pixel ratio to keep a slice")
    args = parser.parse_args()
    main(args)
