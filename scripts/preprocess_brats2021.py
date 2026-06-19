"""
preprocess_brats2021.py  (RESUMABLE VERSION)
────────────────────────────────────────────────────────────────────────────────
Converts BraTS2021 3D NIfTI volumes → 2D axial .npy slices for pretraining.

RESUME SUPPORT:
  - Before processing each subject, checks if its slices already exist on disk
    (by checking for any file starting with "{subject_name}_slice_" in the
    output split folders).
  - If found, skips re-processing that subject entirely.
  - metadata.json is saved incrementally every `checkpoint_every` subjects,
    so even an unexpected shutdown loses at most a few subjects' worth of
    metadata (the .npy files themselves are never lost since they're written
    immediately as each subject completes).
  - Train/val split assignment is deterministic (fixed seed) so re-running
    always assigns the same subject to the same split — no leakage risk
    even across interrupted runs.

Run:
  python preprocess_brats2021.py \
    --root /path/to/brats2021 \
    --out  /path/to/output_slices \
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
        candidates = list(subject_dir.glob(f"{mod}.nii.gz")) or \
                     list(subject_dir.glob(f"{mod}.nii"))

        if not candidates:
            raise FileNotFoundError(
                f"Missing modality '{mod}' in {subject_dir}")

        try:
            vol = load_volume(candidates[0])
        except Exception as e:
            raise FileNotFoundError(f"Corrupted file {candidates[0]} ({e})")

        vol = normalise_volume(vol)
        volumes.append(vol)

    D = volumes[0].shape[axis]
    slices_out = []

    for z in range(skip, D - skip):
        if axis == 2:
            stack = np.stack([v[:, :, z] for v in volumes], axis=0)
        elif axis == 1:
            stack = np.stack([v[:, z, :] for v in volumes], axis=0)
        else:
            stack = np.stack([v[z, :, :] for v in volumes], axis=0)

        if brain_ratio(stack[0]) < min_brain:
            continue

        slices_out.append(stack)

    return slices_out


# ── resume helpers ────────────────────────────────────────────────────────────

def subject_already_done(out_dir: Path, split: str, subj_name: str) -> bool:
    """
    Check if this subject has already been processed by looking for
    any file matching "{subj_name}_slice_*.npy" in the split folder.
    """
    split_dir = out_dir / split
    if not split_dir.exists():
        return False
    pattern = f"{subj_name}_slice_*.npy"
    return any(split_dir.glob(pattern))


def load_existing_metadata(out_dir: Path, modalities, axis, skip, min_brain):
    """
    Load metadata.json if it exists (from a previous interrupted run),
    otherwise return a fresh metadata dict.
    """
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"✓ Found existing metadata.json — resuming")
        print(f"  Already recorded: {len(meta['train'])} train, "
              f"{len(meta['val'])} val slices")
        return meta
    return {"train": [], "val": [], "modalities": modalities,
            "axis": axis, "skip": skip, "min_brain": min_brain}


def rebuild_metadata_from_disk(out_dir: Path) -> dict:
    """
    Fallback: if metadata.json is missing/corrupted but .npy files exist
    on disk (e.g. process was killed before any metadata save), rebuild
    the file lists by scanning the output directories directly.
    """
    meta = {"train": [], "val": []}
    for split in ["train", "val"]:
        split_dir = out_dir / split
        if split_dir.exists():
            meta[split] = sorted([f.name for f in split_dir.glob("*.npy")])
    print(f"✓ Rebuilt metadata from disk — "
          f"{len(meta['train'])} train, {len(meta['val'])} val slices found")
    return meta


def save_metadata(out_dir: Path, metadata: dict):
    """Write metadata.json atomically (write to temp, then rename)."""
    tmp_path  = out_dir / "metadata.json.tmp"
    final_path = out_dir / "metadata.json"
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2)
    os.replace(tmp_path, final_path)   # atomic on most filesystems


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42)
    np.random.seed(42)

    root       = Path(args.root)
    out_dir    = Path(args.out)
    modalities = ["t1", "t1ce", "t2", "flair"]
    axis       = 2
    skip       = args.skip
    min_brain  = args.min_brain
    val_frac   = 0.1
    checkpoint_every = args.checkpoint_every

    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    # Find all subject directories
    subjects = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Found {len(subjects)} subjects")

    # Deterministic train/val split — same every run regardless of resume
    subjects_for_split = subjects.copy()
    random.shuffle(subjects_for_split)
    n_val   = max(1, int(len(subjects_for_split) * val_frac))
    val_set = set(s.name for s in subjects_for_split[:n_val])

    # ── Load or rebuild metadata ──
    metadata = load_existing_metadata(out_dir, modalities, axis, skip, min_brain)
    if not metadata["train"] and not metadata["val"]:
        # metadata.json missing but maybe .npy files exist from a crash
        disk_meta = rebuild_metadata_from_disk(out_dir)
        metadata["train"] = disk_meta["train"]
        metadata["val"]   = disk_meta["val"]

    total_train = len(metadata["train"])
    total_val   = len(metadata["val"])
    corrupted_count = 0
    skipped_already_done = 0
    processed_this_run = 0

    for i, subj in enumerate(tqdm(subjects, desc="Processing subjects")):
        split = "val" if subj.name in val_set else "train"

        # ── RESUME CHECK ──
        if subject_already_done(out_dir, split, subj.name):
            skipped_already_done += 1
            continue

        try:
            slices = extract_slices(subj, modalities, skip, min_brain, axis)
        except FileNotFoundError as e:
            corrupted_count += 1
            print(f"  [WARN] {e} — skipping")
            continue

        for idx, sl in enumerate(slices):
            fname = f"{subj.name}_slice_{idx:03d}.npy"
            np.save(out_dir / split / fname, sl.astype(np.float16))
            metadata[split].append(fname)

        if split == "train":
            total_train += len(slices)
        else:
            total_val += len(slices)

        processed_this_run += 1

        # ── Incremental metadata save ──
        if processed_this_run % checkpoint_every == 0:
            save_metadata(out_dir, metadata)

    # Final save
    save_metadata(out_dir, metadata)

    print(f"\n✓ Done")
    print(f"  Subjects skipped (already done) : {skipped_already_done}")
    print(f"  Subjects processed this run     : {processed_this_run}")
    print(f"  Subjects corrupted/missing      : {corrupted_count}")
    print(f"  Total train slices              : {total_train}")
    print(f"  Total val slices                : {total_val}")
    print(f"  Saved to                        : {out_dir}")


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
    parser.add_argument("--checkpoint_every", type=int, default=20,
                        help="Save metadata.json every N processed subjects")
    args = parser.parse_args()
    main(args)