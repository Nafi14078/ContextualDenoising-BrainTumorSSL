"""
evaluation/metrics.py
────────────────────────────────────────────────────────────────────────────────
All evaluation metrics for both stages:

Pretraining  : PSNR, SSIM
Fine-tuning  : Dice WT / TC / ET, HD95

BraTS sub-region mapping:
  WT (whole tumor)   = pred/mask > 0        (labels 1+2+3)
  TC (tumor core)    = pred/mask in {1,3}   (labels 1+3)
  ET (enhancing)     = pred/mask == 3       (label 3)
────────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity   as sk_ssim

# HD95 requires scipy
try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ── Pretraining metrics ───────────────────────────────────────────────────────

def compute_psnr(clean: np.ndarray,
                 denoised: np.ndarray,
                 data_range: float = 1.0) -> float:
    """
    PSNR between clean and denoised arrays.
    Both expected in [0, 1].
    Handles multi-channel (C, H, W) by averaging over channels.
    """
    if clean.ndim == 3:
        # (C, H, W) → average PSNR across channels
        psnrs = []
        for c in range(clean.shape[0]):
            psnrs.append(sk_psnr(clean[c], denoised[c],
                                 data_range=data_range))
        return float(np.mean(psnrs))
    return float(sk_psnr(clean, denoised, data_range=data_range))


def compute_ssim(clean: np.ndarray,
                 denoised: np.ndarray,
                 data_range: float = 1.0) -> float:
    """
    SSIM between clean and denoised arrays.
    Handles multi-channel (C, H, W) by averaging over channels.
    """
    if clean.ndim == 3:
        ssims = []
        for c in range(clean.shape[0]):
            ssims.append(sk_ssim(clean[c], denoised[c],
                                 data_range=data_range))
        return float(np.mean(ssims))
    return float(sk_ssim(clean, denoised, data_range=data_range))


# ── Segmentation metrics ──────────────────────────────────────────────────────

def _dice(pred_bin: np.ndarray, gt_bin: np.ndarray,
          smooth: float = 1e-5) -> float:
    """Binary Dice coefficient."""
    pred_bin = pred_bin.astype(bool)
    gt_bin   = gt_bin.astype(bool)
    intersection = (pred_bin & gt_bin).sum()
    denom        = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0    # both empty → perfect score
    return float((2.0 * intersection + smooth) / (denom + smooth))


def compute_dice_wt(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice for Whole Tumor (labels 1, 2, 3)."""
    return _dice(pred > 0, gt > 0)


def compute_dice_tc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice for Tumor Core (labels 1, 3)."""
    return _dice(np.isin(pred, [1, 3]), np.isin(gt, [1, 3]))


def compute_dice_et(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice for Enhancing Tumor (label 3)."""
    return _dice(pred == 3, gt == 3)


def compute_hd95(pred_bin: np.ndarray,
                 gt_bin:   np.ndarray) -> float:
    """
    95th percentile Hausdorff Distance.
    Returns 0.0 if both masks are empty (perfect),
    returns large value if one is empty and other is not.
    """
    if not SCIPY_AVAILABLE:
        return float("nan")

    pred_bin = pred_bin.astype(bool)
    gt_bin   = gt_bin.astype(bool)

    # Both empty
    if not pred_bin.any() and not gt_bin.any():
        return 0.0

    # One empty
    if not pred_bin.any() or not gt_bin.any():
        return 373.13   # diagonal of 240×240 image (worst case)

    # Distance transform from boundaries
    pred_border = pred_bin ^ _erode(pred_bin)
    gt_border   = gt_bin   ^ _erode(gt_bin)

    dist_pred_to_gt = distance_transform_edt(~gt_border)[pred_border]
    dist_gt_to_pred = distance_transform_edt(~pred_border)[gt_border]

    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Simple 3×3 binary erosion."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=1)


# ── Summary printer ───────────────────────────────────────────────────────────

def print_metrics(metrics: dict, stage: str = "val"):
    print(f"\n{'─'*40}")
    print(f"  {stage.upper()} METRICS")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")
    print(f"{'─'*40}\n")
