"""
preprocess_bratsped.py
────────────────────────────────────────────────────────────────────────────────
Converts BraTS-PED 3D NIfTI volumes → 2D axial .npy slice pairs:
  - image:  (4, H, W)  float32  normalised
  - mask:   (H, W)     uint8    label map {0,1,2,3}

BraTS-PED label convention (same as BraTS2021):
  0 = background
  1 = NCR/NET  (necrotic core)
  2 = ED       (peritumoral edema)
  3 = ET       (enhancing tumor)

Output layout:
  slices/
    train/
      images/  BraTS_PED_00001_slice_045.npy   shape: (4, H, W)
      masks/   BraTS_PED_00001_slice_045.npy   shape: (H, W)
    val/
      images/  ...
      masks/   ...
    metadata.json

Strategy: keep ALL slices that contain at least min_tumor_ratio tumor voxels
          PLUS a balanced sample of non-tumor slices (1:1 ratio).
          This avoids extreme class imbalance at the slice level.
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


def load_volume(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def load_mask(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.uint8)


def normalise_volume(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    mu  = vol[mask].mean()
    std = vol[mask].std() + 1e-8
    vol = (vol - mu) / std
    vol = np.clip(vol, -5.0, 5.0)
    vol = (vol + 5.0) / 10.0
    return vol.astype(np.float32)


def extract_subject(subject_dir: Path, modalities: list,
                    skip: int, min_brain: float,
                    min_tumor: float, axis: int = 2):
    """
    Returns:
        tumor_slices   : list of (image, mask) tuples — contain tumor
        normal_slices  : list of (image, mask) tuples — no tumor
    """
    volumes = []
    for mod in modalities:
        candidates = list(subject_dir.glob(f"*_{mod}.nii.gz"))
        if not candidates:
            raise FileNotFoundError(f"Missing '{mod}' in {subject_dir}")
        vol = load_volume(candidates[0])
        vol = normalise_volume(vol)
        volumes.append(vol)

    seg_candidates = list(subject_dir.glob("*_seg.nii.gz"))
    if not seg_candidates:
        raise FileNotFoundError(f"Missing seg in {subject_dir}")
    seg_vol = load_mask(seg_candidates[0])

    D = volumes[0].shape[axis]
    tumor_slices, normal_slices = [], []

    for z in range(skip, D - skip):
        if axis == 2:
            img  = np.stack([v[:, :, z] for v in volumes], axis=0)
            mask = seg_vol[:, :, z]
        elif axis == 1:
            img  = np.stack([v[:, z, :] for v in volumes], axis=0)
            mask = seg_vol[:, z, :]
        else:
            img  = np.stack([v[z, :, :] for v in volumes], axis=0)
            mask = seg_vol[z, :, :]

        brain_frac = (img[0] > 0).mean()
        if brain_frac < min_brain:
            continue

        tumor_frac = (mask > 0).mean()
        if tumor_frac >= min_tumor:
            tumor_slices.append((img, mask))
        else:
            normal_slices.append((img, mask))

    return tumor_slices, normal_slices


def main(args):
    random.seed(42)
    np.random.seed(42)

    root       = Path(args.root)
    out_dir    = Path(args.out)
    modalities = ["t1", "t1ce", "t2", "flair"]
    skip       = args.skip
    min_brain  = 0.05
    min_tumor  = args.min_tumor
    val_frac   = 0.15

    subjects = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Found {len(subjects)} subjects")

    random.shuffle(subjects)
    n_val   = max(1, int(len(subjects) * val_frac))
    val_set = set(s.name for s in subjects[:n_val])

    for split in ["train", "val"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    metadata = {"train": [], "val": [], "modalities": modalities}
    counters = {"train": {"tumor": 0, "normal": 0},
                "val":   {"tumor": 0, "normal": 0}}

    for subj in tqdm(subjects, desc="Processing BraTS-PED"):
        split = "val" if subj.name in val_set else "train"
        try:
            tumor_sl, normal_sl = extract_subject(
                subj, modalities, skip, min_brain, min_tumor)
        except FileNotFoundError as e:
            print(f"  [WARN] {e} — skipping")
            continue

        # Balance: keep all tumor slices, sample equal # of normal slices
        n_keep_normal = min(len(normal_sl), len(tumor_sl))
        normal_sl     = random.sample(normal_sl, n_keep_normal)
        all_slices    = tumor_sl + normal_sl

        for idx, (img, mask) in enumerate(all_slices):
            fname = f"{subj.name}_slice_{idx:03d}.npy"
            np.save(out_dir / split / "images" / fname, img)
            np.save(out_dir / split / "masks"  / fname, mask)
            metadata[split].append(fname)

        counters[split]["tumor"]  += len(tumor_sl)
        counters[split]["normal"] += n_keep_normal

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Done")
    for split in ["train", "val"]:
        t = counters[split]["tumor"]
        n = counters[split]["normal"]
        print(f"  {split}: {t} tumor slices + {n} normal slices = {t+n} total")
    print(f"  Saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      required=True)
    parser.add_argument("--out",       required=True)
    parser.add_argument("--skip",      type=int,   default=10)
    parser.add_argument("--min_tumor", type=float, default=0.01)
    args = parser.parse_args()
    main(args)
