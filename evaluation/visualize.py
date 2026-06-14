"""
evaluation/visualize.py
────────────────────────────────────────────────────────────────────────────────
Visualization utilities for both stages.

  plot_denoising_comparison  : noisy | denoised | clean (per modality)
  plot_segmentation_overlay  : FLAIR slice + ground truth + prediction
  plot_training_curves       : loss / PSNR / Dice over epochs
────────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ── Label colours (BraTS convention) ─────────────────────────────────────────
LABEL_COLORS = {
    0: [0,   0,   0  ],   # background  — black
    1: [255, 0,   0  ],   # NCR/NET     — red
    2: [0,   255, 0  ],   # ED          — green
    3: [0,   0,   255],   # ET          — blue
}
LABEL_NAMES = {0: "BG", 1: "NCR/NET", 2: "ED", 3: "ET"}
MOD_NAMES   = ["T1", "T1ce", "T2", "FLAIR"]


def _to_numpy(t):
    """Convert tensor or array to numpy float in [0,1]."""
    if hasattr(t, "cpu"):
        t = t.detach().cpu().numpy()
    return np.clip(t.astype(np.float32), 0, 1)


def _label_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert integer label map (H,W) → RGB (H,W,3) uint8."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for lbl, color in LABEL_COLORS.items():
        rgb[mask == lbl] = color
    return rgb


# ── Denoising comparison ──────────────────────────────────────────────────────

def plot_denoising_comparison(noisy,
                               denoised,
                               clean=None,
                               save_path: str = None,
                               title: str = "Denoising Comparison"):
    """
    Args:
        noisy    : (4, H, W) or numpy
        denoised : (4, H, W) or numpy
        clean    : (4, H, W) or numpy  (optional — not always available)
        save_path: if given, saves figure instead of showing
    """
    noisy    = _to_numpy(noisy)
    denoised = _to_numpy(denoised)
    has_clean = clean is not None
    if has_clean:
        clean = _to_numpy(clean)

    n_cols = 3 if has_clean else 2
    n_rows = 4   # one per modality
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    col_titles = ["Noisy Input", "Denoised"]
    if has_clean:
        col_titles.append("Clean Reference")

    for row, mod in enumerate(MOD_NAMES):
        for col in range(n_cols):
            ax = axes[row, col]
            if col == 0:
                img = noisy[row]
            elif col == 1:
                img = denoised[row]
            else:
                img = clean[row]

            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(mod, fontsize=10, rotation=0,
                              labelpad=40, va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ── Segmentation overlay ──────────────────────────────────────────────────────

def plot_segmentation_overlay(image,
                               gt_mask,
                               pred_mask,
                               modality_idx: int = 3,
                               alpha: float = 0.45,
                               save_path: str = None,
                               title: str = "Segmentation"):
    """
    Args:
        image        : (4, H, W) — uses modality_idx (default 3 = FLAIR)
        gt_mask      : (H, W) long/int
        pred_mask    : (H, W) long/int
        modality_idx : which modality to use as background
        alpha        : overlay transparency
    """
    image     = _to_numpy(image)
    if hasattr(gt_mask, "cpu"):
        gt_mask   = gt_mask.cpu().numpy()
    if hasattr(pred_mask, "cpu"):
        pred_mask = pred_mask.cpu().numpy()

    bg   = image[modality_idx]                    # (H, W) in [0,1]
    bg3  = np.stack([bg, bg, bg], axis=-1)        # (H, W, 3)

    gt_rgb   = _label_to_rgb(gt_mask.astype(np.int32))
    pred_rgb = _label_to_rgb(pred_mask.astype(np.int32))

    def overlay(base, seg_rgb):
        seg_norm = seg_rgb / 255.0
        mask     = seg_rgb.sum(-1) > 0            # non-background pixels
        out      = base.copy()
        out[mask] = (1 - alpha) * base[mask] + alpha * seg_norm[mask]
        return out

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].imshow(bg, cmap="gray")
    axes[0].set_title(f"Input ({MOD_NAMES[modality_idx]})")
    axes[0].axis("off")

    axes[1].imshow(overlay(bg3, gt_rgb))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(overlay(bg3, pred_rgb))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # Legend
    patches = [mpatches.Patch(color=np.array(c)/255, label=LABEL_NAMES[l])
               for l, c in LABEL_COLORS.items() if l > 0]
    fig.legend(handles=patches, loc="lower center",
               ncol=3, fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(history: dict,
                         save_path: str = None,
                         title: str = "Training Curves"):
    """
    Args:
        history : dict with keys like "train_loss", "val_psnr",
                  "val_dice_wt" etc. Each value is a list over epochs.
    """
    keys = list(history.keys())
    n    = len(keys)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, key in enumerate(keys):
        axes[i].plot(history[key], linewidth=2)
        axes[i].set_title(key)
        axes[i].set_xlabel("Epoch")
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()
