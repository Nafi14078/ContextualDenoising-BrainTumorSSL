"""
finetuning/losses.py
────────────────────────────────────────────────────────────────────────────────
Segmentation losses for BraTS-PED fine-tuning.

  • DiceLoss        : soft Dice per class, averaged (now optionally WEIGHTED)
  • FocalLoss       : handles class imbalance (now optionally WEIGHTED per class)
  • CombinedLoss    : Dice + λ·Focal  (default λ=1)

BraTS label convention:
  0 = background
  1 = NCR/NET  (necrotic core)
  2 = ED       (peritumoral edema)
  3 = ET       (enhancing tumor)

Evaluation sub-regions (computed from raw labels):
  WT (whole tumor)      = labels {1, 2, 3}
  TC (tumor core)       = labels {1, 3}
  ET (enhancing tumor)  = label  {3}

CHANGE (this version): added optional per-class weighting to both losses.
ET is consistently the hardest, most fragile region (smallest, most
fine-grained, most affected by class imbalance during training). Both
DiceLoss and FocalLoss previously treated all foreground classes equally
when averaging — this let a model that's very good at NCR/ED (larger,
easier structures) mask mediocre ET performance in the aggregate training
signal. Passing higher weights for ET pushes the optimizer to actually
prioritize getting it right, rather than letting it be diluted by the
easier classes. Defaults to uniform weights (1.0 everywhere) if you don't
pass anything, so this is fully backward compatible.
────────────────────────────────────────────────────────────────────────────────
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss, optionally weighted per class, averaged over foreground
    classes (ignores background). Works on raw logits — applies softmax
    internally.
    """

    def __init__(self,
                 num_classes:   int   = 4,
                 smooth:        float = 1e-5,
                 ignore_bg:     bool  = True,
                 class_weights: Optional[Sequence[float]] = None):
        """
        class_weights: one weight per FOREGROUND class, in label order
          (e.g. [w_NCR, w_ED, w_ET] for the default ignore_bg=True case).
          Higher weight = that class's Dice loss counts for more in the
          average. None (default) = uniform weighting = original behavior.
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.ignore_bg   = ignore_bg

        n_fg = num_classes - (1 if ignore_bg else 0)
        if class_weights is None:
            class_weights = [1.0] * n_fg
        assert len(class_weights) == n_fg, (
            f"class_weights must have {n_fg} entries (one per foreground "
            f"class), got {len(class_weights)}")
        self.register_buffer(
            "class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)  raw logits
        targets : (B, H, W)     long labels {0..C-1}
        """
        probs = F.softmax(logits, dim=1)          # (B, C, H, W)

        # One-hot encode targets → (B, C, H, W)
        B, C, H, W = probs.shape
        targets_oh  = F.one_hot(targets, C).permute(0, 3, 1, 2).float()

        start_cls = 1 if self.ignore_bg else 0
        dice_per_class = []

        for c in range(start_cls, C):
            p   = probs[:, c]          # (B, H, W)
            t   = targets_oh[:, c]     # (B, H, W)
            intersection = (p * t).sum(dim=[1, 2])
            union        = p.sum(dim=[1, 2]) + t.sum(dim=[1, 2])
            dice         = (2. * intersection + self.smooth) / \
                           (union + self.smooth)
            dice_per_class.append(1. - dice.mean())

        dice_per_class = torch.stack(dice_per_class)             # (n_fg,)
        weights = self.class_weights.to(dice_per_class.device)
        return (dice_per_class * weights).sum() / weights.sum()


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class segmentation, optionally with an additional
    per-class weight on top of the standard focal (1-p_t)^gamma term.
    Reduces loss for easy (well-classified) pixels, focuses on hard ones.
    Essential for BraTS where tumor voxels << background voxels.
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 num_classes: int = 4,
                 class_weights: Optional[Sequence[float]] = None):
        """
        class_weights: one weight per class, INCLUDING background, in label
          order (e.g. [w_bg, w_NCR, w_ED, w_ET]). Applied per-pixel based on
          that pixel's true class, multiplicatively alongside `alpha`.
          None (default) = uniform weighting = original behavior.
        """
        super().__init__()
        self.gamma       = gamma
        self.alpha       = alpha
        self.num_classes = num_classes

        if class_weights is None:
            class_weights = [1.0] * num_classes
        assert len(class_weights) == num_classes, (
            f"class_weights must have {num_classes} entries (one per "
            f"class, including background), got {len(class_weights)}")
        self.register_buffer(
            "class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self,
                logits:  torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)
        targets : (B, H, W)
        """
        B, C, H, W = logits.shape

        # Cross-entropy per pixel
        ce_loss = F.cross_entropy(logits, targets,
                                  reduction="none")          # (B, H, W)

        # p_t = probability of the true class
        probs  = F.softmax(logits, dim=1)                    # (B, C, H, W)
        p_t    = probs.gather(1, targets.unsqueeze(1)) \
                      .squeeze(1)                            # (B, H, W)

        # Focal weight
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma

        # Per-pixel class weight, gathered from each pixel's true label
        weights = self.class_weights.to(logits.device)
        pixel_class_weight = weights[targets]                # (B, H, W)

        focal_loss = (pixel_class_weight * focal_weight * ce_loss).mean()
        return focal_loss


class CombinedSegLoss(nn.Module):
    """
    L = DiceLoss + λ · FocalLoss
    """

    def __init__(self,
                 num_classes:         int   = 4,
                 dice_weight:         float = 1.0,
                 focal_weight:        float = 1.0,
                 focal_gamma:         float = 2.0,
                 focal_alpha:         float = 0.25,
                 dice_class_weights:  Optional[Sequence[float]] = None,
                 focal_class_weights: Optional[Sequence[float]] = None):
        """
        dice_class_weights  : per-FOREGROUND-class weights for DiceLoss,
          e.g. [1.0, 1.0, 2.0] to weight ET 2x NCR/ED. None = uniform.
        focal_class_weights : per-class weights for FocalLoss, INCLUDING
          background, e.g. [1.0, 1.0, 1.0, 2.0]. None = uniform.
        """
        super().__init__()
        self.dice  = DiceLoss(num_classes, class_weights=dice_class_weights)
        self.focal = FocalLoss(focal_gamma, focal_alpha, num_classes,
                               class_weights=focal_class_weights)
        self.w_dice  = dice_weight
        self.w_focal = focal_weight

    def forward(self,
                logits:  torch.Tensor,
                targets: torch.Tensor) -> dict:
        """
        Returns dict with keys: total, dice, focal
        """
        d = self.dice(logits, targets)
        f = self.focal(logits, targets)
        total = self.w_dice * d + self.w_focal * f
        return {"total": total, "dice": d, "focal": f}