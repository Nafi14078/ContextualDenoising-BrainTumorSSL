"""
finetuning/losses.py
────────────────────────────────────────────────────────────────────────────────
Segmentation losses for BraTS-PED fine-tuning.

  • DiceLoss        : soft Dice per class, averaged
  • FocalLoss       : handles class imbalance (small tumors)
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
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss averaged over foreground classes (ignores background).
    Works on raw logits — applies softmax internally.
    """

    def __init__(self,
                 num_classes:   int   = 4,
                 smooth:        float = 1e-5,
                 ignore_bg:     bool  = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.ignore_bg   = ignore_bg

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

        return torch.stack(dice_per_class).mean()


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class segmentation.
    Reduces loss for easy (well-classified) pixels, focuses on hard ones.
    Essential for BraTS where tumor voxels << background voxels.
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 num_classes: int = 4):
        super().__init__()
        self.gamma       = gamma
        self.alpha       = alpha
        self.num_classes = num_classes

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

        focal_loss = (focal_weight * ce_loss).mean()
        return focal_loss


class CombinedSegLoss(nn.Module):
    """
    L = DiceLoss + λ · FocalLoss
    """

    def __init__(self,
                 num_classes:   int   = 4,
                 dice_weight:   float = 1.0,
                 focal_weight:  float = 1.0,
                 focal_gamma:   float = 2.0,
                 focal_alpha:   float = 0.25):
        super().__init__()
        self.dice  = DiceLoss(num_classes)
        self.focal = FocalLoss(focal_gamma, focal_alpha, num_classes)
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
