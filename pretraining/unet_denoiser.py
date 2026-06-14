"""
pretraining/unet_denoiser.py
────────────────────────────────────────────────────────────────────────────────
2D U-Net denoiser — the backbone D_θ from the STS-UVD paper.

Architecture (closely following paper Section 3.2):
  • Feature extractor G_φ  : 3 group-conv layers, 21 channels each
  • Encoder                : 2 blocks, each = 5 × (3×3 conv + GroupNorm + GELU)
  • Decoder                : 3 blocks with skip connections
  • Output heads           : 1×1 convs → 4 channels (one per modality)

Input  : (B, N×C_mod, H, W)  — N slices concatenated along channel axis
          after STS sampling
Output : (B, C_mod, H, W)    — denoised central slice
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → GroupNorm → GELU. Default groups=8."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, groups: int = 8):
        super().__init__()
        pad      = kernel // 2
        gn_grps  = min(groups, out_ch)   # guard against small channel counts
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.GroupNorm(gn_grps, out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """5 ConvBlocks → MaxPool2d(2)."""
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 8):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBlock(in_ch,  out_ch, groups=gn_groups),
            ConvBlock(out_ch, out_ch, groups=gn_groups),
            ConvBlock(out_ch, out_ch, groups=gn_groups),
            ConvBlock(out_ch, out_ch, groups=gn_groups),
            ConvBlock(out_ch, out_ch, groups=gn_groups),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat   = self.convs(x)     # skip connection source
        pooled = self.pool(feat)
        return pooled, feat


class DecoderBlock(nn.Module):
    """Upsample → concat skip → 6 ConvBlocks."""
    def __init__(self, in_ch: int, skip_ch: int,
                 out_ch: int, gn_groups: int = 8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        merged  = (in_ch // 2) + skip_ch
        self.convs = nn.Sequential(
            ConvBlock(merged,  out_ch, groups=gn_groups),
            ConvBlock(out_ch,  out_ch, groups=gn_groups),
            ConvBlock(out_ch,  out_ch, groups=gn_groups),
            ConvBlock(out_ch,  out_ch, groups=gn_groups),
            ConvBlock(out_ch,  out_ch, groups=gn_groups),
            ConvBlock(out_ch,  out_ch, groups=gn_groups),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims don't match (can happen at odd sizes)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.convs(x)


# ── Feature extractor G_φ ─────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Shallow 3-layer group convolution network (paper Section 3).
    Takes one slice (B, C_mod, H, W) and outputs low-level features.
    The paper uses 21 channels and group convolutions.
    """
    def __init__(self, in_ch: int = 4, feat_ch: int = 21):
        super().__init__()
        gn = min(3, feat_ch)   # small group count for 21-channel tensors
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, padding=1, bias=False,
                      groups=1),
            nn.GroupNorm(gn, feat_ch),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False,
                      groups=3),
            nn.GroupNorm(gn, feat_ch),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False,
                      groups=3),
            nn.GroupNorm(gn, feat_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


# ── Full Denoiser ─────────────────────────────────────────────────────────────

class UNetDenoiser(nn.Module):
    """
    Full STS-UVD denoiser.

    Usage in training loop:
        1. For each slice in window: features[n] = G_phi(noisy[n])
        2. Apply STS module:        sampled = STS(features, noisy, epoch, max_ep)
        3. Flatten N×feat → denoiser input
        4. denoised_central = denoiser(sampled)

    The N sampled feature maps are concatenated along the channel axis
    before entering the U-Net.
    """

    def __init__(self,
                 N:           int = 7,
                 in_channels: int = 4,
                 feat_ch:     int = 21,
                 base_ch:     int = 48,
                 out_channels: int = 4,
                 gn_groups:   int = 8):
        """
        Args:
            N           : number of slices in window
            in_channels : modalities per slice (4)
            feat_ch     : feature extractor output channels (21)
            base_ch     : U-Net base channels (encoder block 1)
            out_channels: output modalities (4)
            gn_groups   : GroupNorm groups
        """
        super().__init__()
        self.N        = N
        self.feat_ch  = feat_ch
        self.out_ch   = out_channels

        # G_φ applied independently to each slice
        self.feature_extractor = FeatureExtractor(in_channels, feat_ch)

        # U-Net input = N stacked feature maps
        unet_in = N * feat_ch   # 7 × 21 = 147

        # ── Encoder ──
        self.enc1 = EncoderBlock(unet_in, base_ch,        gn_groups)
        self.enc2 = EncoderBlock(base_ch, base_ch * 2,    gn_groups)

        # ── Bottleneck ──
        self.bottleneck = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch * 4, groups=gn_groups),
            ConvBlock(base_ch * 4, base_ch * 4, groups=gn_groups),
            ConvBlock(base_ch * 4, base_ch * 4, groups=gn_groups),
        )

        # ── Decoder ──
        self.dec3 = DecoderBlock(base_ch * 4, base_ch * 2,
                                 base_ch * 2, gn_groups)
        self.dec2 = DecoderBlock(base_ch * 2, base_ch,
                                 base_ch,     gn_groups)
        self.dec1 = DecoderBlock(base_ch,     unet_in,
                                 base_ch // 2, gn_groups)

        # ── Output heads (1×1 convs, paper uses 3 layers: 348, 96, C) ──
        self.head = nn.Sequential(
            nn.Conv2d(base_ch // 2, 96,          1),
            nn.GELU(),
            nn.Conv2d(96,           out_channels, 1),
            nn.Sigmoid()    # output in [0, 1] — matches normalised input
        )

    def extract_all_features(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply G_φ to every slice in the window.

        Args:
            window : (B, N, C_mod, H, W)
        Returns:
            feats  : (B, N, feat_ch, H, W)
        """
        B, N, C, H, W = window.shape
        # Reshape to (B*N, C, H, W), extract, reshape back
        x    = window.view(B * N, C, H, W)
        feat = self.feature_extractor(x)      # (B*N, feat_ch, H, W)
        return feat.view(B, N, self.feat_ch, H, W)

    def forward(self, sampled_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sampled_features : (B, N, feat_ch, H, W)
                               — output of STS module
        Returns:
            denoised_central : (B, out_ch, H, W)
        """
        B, N, C_f, H, W = sampled_features.shape

        # Flatten N feature maps along channel axis
        x = sampled_features.view(B, N * C_f, H, W)

        # Encoder (save skip connections)
        x,  skip1 = self.enc1(x)
        x,  skip2 = self.enc2(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec3(x, skip2)
        x = self.dec2(x, skip1)

        # Final decoder expects a skip from the very first layer
        # We use the original flattened features as the skip
        skip0 = sampled_features.view(B, N * C_f, H, W)
        x = self.dec1(x, skip0)

        return self.head(x)

    def get_encoder_state_dict(self) -> dict:
        """
        Extract encoder weights for transfer to Swin UNETR.
        Includes: feature_extractor + enc1 + enc2 + bottleneck
        """
        encoder_keys = [
            "feature_extractor", "enc1", "enc2", "bottleneck"
        ]
        state = {}
        full_state = self.state_dict()
        for k, v in full_state.items():
            if any(k.startswith(prefix) for prefix in encoder_keys):
                state[k] = v
        return state
