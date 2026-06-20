"""
pretraining/sts_kernel.py
────────────────────────────────────────────────────────────────────────────────
Weighted SpatioTemporal Sampling kernels adapted for 2D MRI slices.

Paper → MRI mapping:
  Video frame        → MRI axial slice
  Temporal axis      → Slice-depth axis
  Optical flow (RAFT)→ Normalized Cross-Correlation (NCC) between adjacent slices
  Flow magnitude M̄  → Slice dissimilarity (1 - mean NCC)
  κ₀                 → Computed from dissimilarity via Eq.3

Two modules:
  WeightedTemporalKernel (𝒯) : slice-distance-based weighting
  WeightedSpatialKernel  (𝒮) : blind-spot pixel replacement (vectorized)
────────────────────────────────────────────────────────────────────────────────
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Slice Similarity (replaces RAFT optical flow) ────────────────────────────

class SliceSimilarityEstimator:
    """
    Computes normalized cross-correlation (NCC) between adjacent slices.
    Returns a scalar 'dissimilarity' analogous to mean optical flow magnitude.

    High dissimilarity → anatomy changes fast across slices (e.g. near lesion)
                       → κ₀ larger → rely more on nearby slices
    Low  dissimilarity → stable region → κ₀ smaller → look farther out
    """

    def __init__(self, window: int = 9):
        self.window = window

    @torch.no_grad()
    def compute_ncc(self,
                    s1: torch.Tensor,
                    s2: torch.Tensor) -> float:
        """
        NCC between two slices (B, C, H, W) or (C, H, W).
        Returns scalar in [-1, 1]; 1 = identical.
        """
        if s1.dim() == 3:
            s1 = s1.unsqueeze(0)
            s2 = s2.unsqueeze(0)

        # Use only first modality (T1) for similarity — stable anatomical signal
        s1 = s1[:, 0:1].float()
        s2 = s2[:, 0:1].float()

        mu1  = s1.mean(dim=[2, 3], keepdim=True)
        mu2  = s2.mean(dim=[2, 3], keepdim=True)
        std1 = s1.std(dim=[2, 3], keepdim=True)  + 1e-8
        std2 = s2.std(dim=[2, 3], keepdim=True)  + 1e-8

        s1n = (s1 - mu1) / std1
        s2n = (s2 - mu2) / std2

        w   = self.window
        ncc = F.avg_pool2d(s1n * s2n, w, stride=1, padding=w // 2)
        return ncc.mean().item()

    def compute_dissimilarity(self, window: torch.Tensor) -> float:
        """
        window: (N, C, H, W)
        Returns mean dissimilarity across consecutive slice pairs → M̄ analogue.
        """
        N   = window.shape[0]
        nccs = []
        for i in range(N - 1):
            ncc = self.compute_ncc(window[i], window[i + 1])
            nccs.append(ncc)
        mean_ncc = float(np.mean(nccs))
        # Convert similarity → dissimilarity (like flow magnitude)
        return max(0.0, 1.0 - mean_ncc)


# ── Weighted Temporal Kernel 𝒯 ───────────────────────────────────────────────

class WeightedTemporalKernel(nn.Module):
    """
    Implements Eq.1-3 from the STS-UVD paper, adapted for MRI slices.

    Assigns weights to each slice in the window such that:
      - The CENTRAL slice gets the MINIMUM weight κ (prevents noise identity
        mapping — the key trick from the paper)
      - FARTHER slices get HIGHER weights

    κ decays to 0 over training epochs, letting the network eventually
    ignore the central slice almost entirely during training.
    """

    def __init__(self, N: int = 7, L: float = 1.0, eta: float = 0.0003):
        """
        Args:
            N   : total slices in window (odd)
            L   : curvature of the weight function (paper uses L=1)
            eta : scaling for κ₀ computation (Eq.3)
        """
        super().__init__()
        assert N % 2 == 1
        self.N   = N
        self.C   = N // 2     # central slice index
        self.L   = L
        self.eta = eta

        # Similarity estimator (not a nn.Module, no params)
        self.similarity = SliceSimilarityEstimator()

    def compute_kappa0(self, dissimilarity: float) -> float:
        """
        Eq.3: κ₀ = 0.2 × (1 - e^{-η × M̄²})
        High dissimilarity → larger κ₀ → central slice still contributes early
        """
        return 0.2 * (1.0 - math.exp(-self.eta * (dissimilarity ** 2)))

    def compute_kappa(self, epoch: int, max_epoch: int,
                      kappa0: float) -> float:
        """
        Eq.2: κ = max(0, κ₀ × (1 - epoch/max_epoch))
        κ decays linearly to 0 — by final epoch, central slice weight → 0.
        """
        return max(0.0, kappa0 * (1.0 - epoch / max_epoch))

    def compute_weights(self, kappa: float) -> torch.Tensor:
        """
        Eq.1: ωₜ for each slice t in [0, N-1].
        Returns tensor of shape (N,).
        """
        weights = []
        C = self.C
        L = self.L
        for t in range(self.N):
            if t <= C:
                # Left side of window (before central)
                ratio = t / C if C > 0 else 0.0
                w     = kappa + (1.0 - kappa) * (1.0 - ratio ** L)
            else:
                # Right side of window (after central)
                ratio = (2 * C - t) / C if C > 0 else 0.0
                w     = kappa + (1.0 - kappa) * (1.0 - ratio ** L)
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

    def forward(self,
                features:    torch.Tensor,
                window:      torch.Tensor,
                epoch:       int,
                max_epoch:   int) -> torch.Tensor:
        """
        Args:
            features  : (B, N, C_feat, H, W) — after feature extraction
            window    : (B, N, C_mod, H, W)  — raw noisy slices (for NCC)
            epoch     : current training epoch
            max_epoch : total training epochs

        Returns:
            weighted_features: (B, N, C_feat, H, W)
        """
        B, N, C_feat, H, W = features.shape

        # Compute dissimilarity on the first batch item (cheap approximation)
        dissim   = self.similarity.compute_dissimilarity(window[0])
        kappa0   = self.compute_kappa0(dissim)
        kappa    = self.compute_kappa(epoch, max_epoch, kappa0)
        weights  = self.compute_weights(kappa)   # (N,)

        # Broadcast weights: (1, N, 1, 1, 1)
        w = weights.view(1, N, 1, 1, 1).to(features.device)
        return features * w


# ── Weighted Spatial Kernel 𝒮 ────────────────────────────────────────────────

class WeightedSpatialKernel(nn.Module):
    """
    Implements the blind-spot spatial sampling from the paper (Eq.6-7).

    For each slice in the window:
      1. Randomly select 10-20% of pixel positions {p}
      2. Replace each p with a neighbour q sampled from a 5×5 window,
         where probability favours pixels FARTHER from p (edge bias)

    This breaks the spatial self-correlation that would let the network
    learn a trivial identity mapping of noise.

    IMPORTANT: forward() is fully vectorized — no per-pixel Python loop,
    no .item() calls inside the loop, no GPU→CPU syncs. The original
    naive implementation looped over every selected pixel individually,
    which caused massive slowdowns (minutes per training step). This
    version processes all pixels across the whole batch in one shot
    using gather/scatter operations.
    """

    def __init__(self,
                 replace_ratio: float = 0.15,
                 window:        int   = 5,
                 alpha:         float = 3.0):
        """
        Args:
            replace_ratio : fraction of pixels to replace (0.10–0.20)
            window        : neighbourhood size (5×5)
            alpha         : edge-emphasis factor (Eq.7, paper uses 3)
        """
        super().__init__()
        self.replace_ratio = replace_ratio
        self.window        = window
        self.alpha         = alpha
        self.half          = window // 2

        # Pre-build probability table (constant, not learned)
        self._build_prob_table()

    def _build_prob_table(self):
        """
        Eq.7: P(q|p) ∝ e^{-α · d(p,q)} — wait, note the paper INVERTS this
        so that farther pixels have HIGHER probability. We use +α·d.
        """
        half    = self.half
        offsets = []
        dists   = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                if dy == 0 and dx == 0:
                    continue        # exclude self
                offsets.append((dy, dx))
                dists.append(abs(dy) + abs(dx))   # L1 distance

        dists_t = torch.tensor(dists, dtype=torch.float32)
        # Paper Eq.7: prefer FARTHER pixels → positive exponent of distance
        logits  = self.alpha * dists_t
        self.probs   = F.softmax(logits, dim=0)        # (K,) — K = window²-1
        self.offsets = offsets                          # list of (dy, dx)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized blind-spot pixel replacement.

        features : (B, N, C, H, W)
        Returns  : (B, N, C, H, W) with blind-spot pixel replacement
        """
        B, N, C, H, W = features.shape
        device = features.device
        BN     = B * N
        flat   = features.view(BN, C, H, W)

        n_pixels = max(1, int(H * W * self.replace_ratio))

        # Random pixel positions to replace — one independent set per (batch, slice)
        flat_idx = torch.stack([
            torch.randperm(H * W, device=device)[:n_pixels]
            for _ in range(BN)
        ], dim=0)                              # (BN, n_pixels)
        p_y = flat_idx // W
        p_x = flat_idx % W

        # Sample replacement offsets for ALL pixels at once
        probs  = self.probs.to(device)
        chosen = torch.multinomial(
            probs.unsqueeze(0).expand(BN * n_pixels, -1),
            num_samples=1
        ).squeeze(1).view(BN, n_pixels)         # (BN, n_pixels)

        offsets_tensor = torch.tensor(self.offsets, device=device)  # (K, 2)
        dy = offsets_tensor[chosen][..., 0]
        dx = offsets_tensor[chosen][..., 1]

        q_y = (p_y + dy).clamp(0, H - 1)
        q_x = (p_x + dx).clamp(0, W - 1)

        # Vectorized gather + scatter replacement
        flat_reshaped = flat.view(BN, C, H * W)
        src_idx = (q_y * W + q_x).unsqueeze(1).expand(-1, C, -1)
        gathered = torch.gather(flat_reshaped, 2, src_idx)

        out_flat = flat.clone().view(BN, C, H * W)
        dst_idx  = (p_y * W + p_x).unsqueeze(1).expand(-1, C, -1)
        out_flat.scatter_(2, dst_idx, gathered)

        return out_flat.view(B, N, C, H, W)


# ── Combined STS Module ───────────────────────────────────────────────────────

class STSModule(nn.Module):
    """
    Combines 𝒯 and 𝒮 into one forward pass.
    Called once per training iteration before the denoiser.
    """

    def __init__(self, N: int = 7, L: float = 1.0,
                 eta: float = 0.0003,
                 replace_ratio: float = 0.15,
                 window: int = 5, alpha: float = 3.0):
        super().__init__()
        self.T = WeightedTemporalKernel(N=N, L=L, eta=eta)
        self.S = WeightedSpatialKernel(replace_ratio=replace_ratio,
                                       window=window, alpha=alpha)

    def forward(self,
                features:  torch.Tensor,
                raw_window: torch.Tensor,
                epoch:     int,
                max_epoch: int) -> torch.Tensor:
        """
        features   : (B, N, C_feat, H, W)  after feature extraction G_φ
        raw_window : (B, N, C_mod, H, W)   original noisy slices (for NCC)
        Returns    : (B, N, C_feat, H, W)  spatiotemporally sampled
        """
        # Step 1: Temporal weighting (𝒯)
        x = self.T(features, raw_window, epoch, max_epoch)
        # Step 2: Spatial blind-spot replacement (𝒮)
        x = self.S(x)
        return x