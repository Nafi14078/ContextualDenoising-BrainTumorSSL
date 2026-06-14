"""
pretraining/losses.py
────────────────────────────────────────────────────────────────────────────────
Two loss terms from the paper (Eq.10-12), adapted for 2D MRI slices.

L1  = pixel-wise L1 between denoised output and reference frame
L2  = slice consistency: warp adjacent denoised slices and compare

Total loss L = L1 + L2

Slice Consistency (replaces optical-flow warping from the paper):
  In the paper, they warp frame t+1 using optical flow F̂_{t→t+1}.
  For MRI slices, we replace this with a learned affine warping:
    - Estimate a lightweight affine grid from the difference between
      adjacent denoised slices
    - Warp slice t+1 onto the coordinate frame of slice t
    - Penalise the residual after warping

  This is simpler than RAFT but sufficient for anatomical slice-to-slice
  consistency where motion is small and smooth.
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── L1 Reconstruction Loss (Eq.10) ───────────────────────────────────────────

class L1ReconstructionLoss(nn.Module):
    """
    Pixel-wise L1 between denoised central slice and reference.
    Reference is updated recurrently (previous epoch's denoised output).
    On epoch 0, reference = original noisy central slice.
    """

    def forward(self,
                denoised:  torch.Tensor,
                reference: torch.Tensor) -> torch.Tensor:
        """
        denoised  : (B, C, H, W)
        reference : (B, C, H, W)
        """
        return F.l1_loss(denoised, reference)


# ── Slice Consistency Loss (Eq.11, MRI adaptation) ───────────────────────────

class SliceConsistencyLoss(nn.Module):
    """
    Encourages denoised slices to be consistent across the slice axis.

    For each adjacent pair (t, t+1) in the denoised window:
      1. Estimate a small affine perturbation between the two slices
         (modelled as a spatial transformer network grid)
      2. Warp slice t+1 toward slice t's coordinate frame
      3. Penalise L2 residual between warped(t+1) and t

    This is the MRI analogue of the optical-flow consistency loss L₂
    in Eq.11 of the paper.

    Implementation note: we use a simplified version where we directly
    compute the L2 distance after normalising each slice's intensity
    (soft consistency), without an explicit warp, since slice-to-slice
    deformation in MRI is small compared to inter-frame video motion.
    The full affine version is available as an optional upgrade.
    """

    def __init__(self, use_affine_warp: bool = False):
        """
        Args:
            use_affine_warp : if True, estimate + apply affine warp.
                              Default False (soft consistency only) —
                              recommended for Colab due to compute cost.
        """
        super().__init__()
        self.use_affine_warp = use_affine_warp

        if use_affine_warp:
            # Tiny CNN to predict 6-param affine matrix from slice pair
            self.affine_net = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1),  # 2 = concat of 2 slices (1 ch each)
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(16, 6)
            )
            # Initialise to identity transform
            self.affine_net[-1].weight.data.zero_()
            self.affine_net[-1].bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def _soft_consistency(self,
                          denoised_window: torch.Tensor) -> torch.Tensor:
        """
        Simple L2 consistency without warping.
        denoised_window : (B, N, C, H, W)
        """
        B, N, C, H, W = denoised_window.shape
        loss = torch.tensor(0.0, device=denoised_window.device)
        count = 0
        for t in range(N - 1):
            # Use only first modality (T1) for consistency signal
            s_t   = denoised_window[:, t,     0:1]   # (B, 1, H, W)
            s_tp1 = denoised_window[:, t + 1, 0:1]   # (B, 1, H, W)
            # Normalise before comparing
            s_t   = (s_t   - s_t.mean())   / (s_t.std()   + 1e-8)
            s_tp1 = (s_tp1 - s_tp1.mean()) / (s_tp1.std() + 1e-8)
            loss  = loss + F.mse_loss(s_t, s_tp1)
            count += 1
        return loss / max(count, 1)

    def _affine_consistency(self,
                            denoised_window: torch.Tensor) -> torch.Tensor:
        """
        Estimate affine warp between adjacent slices, apply it, then penalise.
        denoised_window : (B, N, C, H, W)
        """
        B, N, C, H, W = denoised_window.shape
        loss = torch.tensor(0.0, device=denoised_window.device)
        count = 0
        for t in range(N - 1):
            s_t   = denoised_window[:, t,     0:1]
            s_tp1 = denoised_window[:, t + 1, 0:1]
            pair  = torch.cat([s_t, s_tp1], dim=1)  # (B, 2, H, W)
            theta = self.affine_net(pair).view(B, 2, 3)
            grid  = F.affine_grid(theta, s_tp1.size(),
                                  align_corners=False)
            warped = F.grid_sample(s_tp1, grid,
                                   mode="bilinear",
                                   padding_mode="border",
                                   align_corners=False)
            loss   = loss + F.mse_loss(warped, s_t)
            count += 1
        return loss / max(count, 1)

    def forward(self, denoised_window: torch.Tensor) -> torch.Tensor:
        """
        denoised_window : (B, N, C, H, W)
                          — full denoised slice window
        """
        if self.use_affine_warp:
            return self._affine_consistency(denoised_window)
        return self._soft_consistency(denoised_window)


# ── Combined Loss ─────────────────────────────────────────────────────────────

class PretrainLoss(nn.Module):
    """
    L = L1 + λ · L2   (Eq.12 of the paper, λ=1)
    """

    def __init__(self, lambda_consistency: float = 1.0,
                 use_affine_warp: bool = False):
        super().__init__()
        self.l1_loss   = L1ReconstructionLoss()
        self.l2_loss   = SliceConsistencyLoss(use_affine_warp)
        self.lambda_c  = lambda_consistency

    def forward(self,
                denoised_central: torch.Tensor,
                reference:        torch.Tensor,
                denoised_window:  torch.Tensor) -> dict:
        """
        Args:
            denoised_central : (B, C, H, W)  — central slice output
            reference        : (B, C, H, W)  — recurrent reference
            denoised_window  : (B, N, C, H, W) — all denoised slices
                               (computed by running denoiser on each slice)

        Returns:
            dict with keys: total, l1, l2
        """
        l1 = self.l1_loss(denoised_central, reference)
        l2 = self.l2_loss(denoised_window)
        total = l1 + self.lambda_c * l2
        return {"total": total, "l1": l1, "l2": l2}
