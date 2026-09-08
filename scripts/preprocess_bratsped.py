"""
preprocess_bratsped.py  (ALL-BRAIN VERSION + RESUMABLE)
────────────────────────────────────────────────────────────────────────────────
Converts BraTS-PED 3D NIfTI volumes → 2D axial .npy slice pairs:
  - image : (4, H, W)  float16  normalised
  - mask  : (H, W)     uint8    label map {0,1,2,3}

BraTS-PED 2023 filename convention:
  BraTS-PED-00081-000-t1n.nii.gz   ← T1 native
  BraTS-PED-00081-000-t1c.nii.gz   ← T1 contrast
  BraTS-PED-00081-000-t2w.nii.gz   ← T2 weighted
  BraTS-PED-00081-000-t2f.nii.gz   ← T2 FLAIR
  BraTS-PED-00081-000-seg.nii.gz   ← segmentation mask

BraTS-PED label convention:
  0 = background
  1 = NCR/NET  (necrotic core)
  2 = ED       (peritumoral edema)
  3 = ET       (enhancing tumor)

SLICE STRATEGY (updated):
  Keep ALL slices that contain any brain tissue (brain_frac >= min_brain).
  Keep only --bg_per_subject pure-background slices per subject.
  This maximizes useful training data — every brain slice contributes
  anatomy, tumor boundaries, and spatial context, regardless of whether
  it contains labeled tumor pixels.

Output layout:
  slices/
    train/
      images/  BraTS-PED-00081-000_slice_045.npy   (4, H, W) float16
      masks/   BraTS-PED-00081-000_slice_045.npy   (H, W)    uint8
    val/
      images/  ...
      masks/   ...
    metadata.json

Resume: skips already-processed subjects on restart.

Run:
    python scripts/preprocess_bratsped.py \
        --root  "D:\\...\\ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData" \
        --out   "D:\\...\\bratsped_slices" \
        --skip  10 \
        --bg_per_subject 5
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


# ── helpers ───────────────────────────────────────────────────────────────────

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


def subject_already_done(out_dir: Path, split: str,
                         subj_name: str) -> bool:
    """Check if subject already processed — for resume support."""
    img_dir = out_dir / split / "images"
    if not img_dir.exists():
        return False
    return any(img_dir.glob(f"{subj_name}_slice_*.npy"))


def save_metadata_atomic(out_dir: Path, metadata: dict):
    """Write metadata.json atomically to survive crashes mid-run."""
    tmp   = out_dir / "metadata.json.tmp"
    final = out_dir / "metadata.json"
    with open(tmp, "w") as f:
        json.dump(metadata, f, indent=2)
    os.replace(tmp, final)


# ── core extraction ───────────────────────────────────────────────────────────

def extract_subject(subject_dir: Path,
                    modalities:  list,
                    skip:        int,
                    min_brain:   float,
                    axis:        int = 2):
    """
    Extracts all 2D axial slices from one subject volume.

    Returns:
        brain_slices      : list of (img, mask) — slices with brain content
                            (brain_frac >= min_brain). ALL of these are kept.
        background_slices : list of (img, mask) — pure background slices
                            (brain_frac < min_brain). Only a few are kept.

    This is more inclusive than the previous tumor-only strategy:
      - Slices near tumor boundaries (small/no tumor pixels but brain present)
        are now kept — they teach the model what "no tumor" looks like in
        context, which is critical for reducing false positives.
      - Slices with large tumors and slices with zero visible tumor are both
        included — the model sees the full spectrum of tumor involvement.
    """
    volumes = []
    for mod in modalities:
        candidates = list(subject_dir.glob(f"*-{mod}.nii.gz"))
        if not candidates:
            raise FileNotFoundError(
                f"Missing modality '{mod}' in {subject_dir}\n"
                f"  Files found: {[f.name for f in subject_dir.iterdir()][:5]}")
        vol = load_volume(candidates[0])
        vol = normalise_volume(vol)
        volumes.append(vol)

    seg_candidates = list(subject_dir.glob("*-seg.nii.gz"))
    if not seg_candidates:
        raise FileNotFoundError(
            f"Missing seg mask in {subject_dir}\n"
            f"  Files found: {[f.name for f in subject_dir.iterdir()][:5]}")
    seg_vol = load_mask(seg_candidates[0])

    D = volumes[0].shape[axis]
    brain_slices, background_slices = [], []

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

        if brain_frac >= min_brain:
            brain_slices.append((img, mask))      # ← keep ALL brain slices
        else:
            background_slices.append((img, mask)) # ← keep only a few

    return brain_slices, background_slices


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)
    np.random.seed(42)

    root             = Path(args.root)
    out_dir          = Path(args.out)
    modalities       = ["t1n", "t1c", "t2w", "t2f"]
    skip             = args.skip
    min_brain        = 0.05
    val_frac         = 0.15
    bg_per_subject   = args.bg_per_subject
    checkpoint_every = args.checkpoint_every

    subjects = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Found {len(subjects)} subjects")
    print(f"Strategy: ALL brain slices (brain_frac >= {min_brain}) + "
          f"{bg_per_subject} background slices per subject")

    # Deterministic patient-level train/val split
    subjects_for_split = subjects.copy()
    random.shuffle(subjects_for_split)
    n_val   = max(1, int(len(subjects_for_split) * val_frac))
    val_set = set(s.name for s in subjects_for_split[:n_val])

    print(f"  Train subjects: {len(subjects) - n_val} | "
          f"Val subjects: {n_val}")

    for split in ["train", "val"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    # Load or initialise metadata
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        print(f"✓ Resuming — already recorded: "
              f"{len(metadata['train'])} train, "
              f"{len(metadata['val'])} val slices")
    else:
        metadata = {"train": [], "val": [], "modalities": modalities}

    counters = {
        "train": {"brain": 0, "background": 0},
        "val":   {"brain": 0, "background": 0}
    }
    skipped_done  = 0
    processed_run = 0
    corrupted     = 0

    for subj in tqdm(subjects, desc="Processing BraTS-PED"):
        split = "val" if subj.name in val_set else "train"

        # Resume check — skip if already done
        if subject_already_done(out_dir, split, subj.name):
            skipped_done += 1
            continue

        try:
            brain_sl, bg_sl = extract_subject(
                subj, modalities, skip, min_brain)
        except FileNotFoundError as e:
            corrupted += 1
            print(f"  [WARN] {e} — skipping")
            continue

        # Keep ALL brain slices
        # Keep only bg_per_subject background slices (avoid flooding dataset)
        n_bg_keep = min(bg_per_subject, len(bg_sl))
        bg_kept   = random.sample(bg_sl, n_bg_keep) if n_bg_keep > 0 else []
        all_slices = brain_sl + bg_kept

        for idx, (img, mask) in enumerate(all_slices):
            fname = f"{subj.name}_slice_{idx:03d}.npy"
            np.save(out_dir / split / "images" / fname,
                    img.astype(np.float16))
            np.save(out_dir / split / "masks"  / fname, mask)
            metadata[split].append(fname)

        counters[split]["brain"]      += len(brain_sl)
        counters[split]["background"] += n_bg_keep
        processed_run += 1

        # Incremental metadata save every N subjects
        if processed_run % checkpoint_every == 0:
            save_metadata_atomic(out_dir, metadata)

    # Final metadata save
    save_metadata_atomic(out_dir, metadata)

    print(f"\n✓ Done")
    print(f"  Subjects skipped (already done) : {skipped_done}")
    print(f"  Subjects processed this run     : {processed_run}")
    print(f"  Subjects corrupted/missing      : {corrupted}")
    for split in ["train", "val"]:
        b = counters[split]["brain"]
        g = counters[split]["background"]
        print(f"  {split}: {b} brain + {g} background = {b+g} total slices")
    print(f"  Saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      required=True,
                        help="Path to BraTS-PED training root")
    parser.add_argument("--out",       required=True,
                        help="Output directory for .npy slices")
    parser.add_argument("--skip",      type=int, default=10,
                        help="Skip N slices from each end (mostly background)")
    parser.add_argument("--bg_per_subject", type=int, default=5,
                        help="Max pure-background slices to keep per subject "
                             "(default 5 — just enough for the model to learn "
                             "what empty slices look like)")
    parser.add_argument("--checkpoint_every", type=int, default=10,
                        help="Save metadata every N subjects")
    args = parser.parse_args()
    main(args)