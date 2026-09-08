"""
finetuning/swin_unetr_2d.py
────────────────────────────────────────────────────────────────────────────────
2D Segmentation U-Net for BraTS-PED tumor segmentation.

ARCHITECTURE CHANGE FROM ORIGINAL DESIGN:
  The original plan used Swin UNETR as the fine-tuning backbone. However,
  Swin UNETR is a Vision Transformer, and our pretraining produced a
  convolutional U-Net encoder (feature_extractor + enc1 + enc2 + bottleneck).
  These architectures are completely incompatible — weight transfer would
  match 0 out of 48 layers, making the "transfer learning" claim empty.

  This file uses a SEGMENTATION U-NET that shares the EXACT SAME encoder
  architecture as the pretraining U-Net (UNetDenoiser). This gives:
    • Direct, verified weight transfer of all 48 encoder layers
    • The encoder starts from pretrained denoising representations
    • Only the decoder + segmentation head are trained from scratch (Phase 1)
    • Phase 2 fine-tunes everything end-to-end

  This is architecturally cleaner and gives the strongest thesis argument:
  "our pretrained encoder, transferred directly, improves segmentation Dice."

  The thesis claim is the same — SSL pretraining on BraTS2021 → transfer to
  BraTS-PED segmentation — just with a backbone that actually supports it.

FIX (this version): the encoder only has TWO real downsampling stages.
  enc1 and enc2 each pool *after* computing their conv stack, so their
  skip connections (skip1, skip2) are captured BEFORE pooling:
    skip1 is captured at full resolution   (H,   W)
    skip2 is captured at half resolution   (H/2, W/2)
  So by the time the decoder reaches its last stage, the features are
  already back at full resolution (H, W) — the same resolution as the
  feature_extractor output (f0) they need to fuse with. The previous
  version still ran a stride-2 ConvTranspose2d there, upsampled to (2H,2W),
  then immediately had to bilinear-interpolate it back down to (H, W) to
  match f0's shape. That transpose conv was pure wasted compute whose
  output got thrown away by the interpolation — it never contributed a
  real decoding step. This version replaces that stage with SegFusionBlock,
  which does no upsampling and simply fuses two already-matching feature
  maps. Decoder stage count now correctly mirrors the true 2-level encoder.

TWO-PHASE TRAINING:
  Phase 1: freeze pretrained encoder, train decoder + seg head only
  Phase 2: unfreeze all, fine-tune end-to-end with differential LR
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ── Building blocks (same as pretraining/unet_denoiser.py) ───────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, groups=8):
        super().__init__()
        pad     = kernel // 2
        gn_grps = min(groups, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.GroupNorm(gn_grps, out_ch),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Identical to pretraining EncoderBlock — receives pretrained weights."""
    def __init__(self, in_ch, out_ch, gn_groups=8):
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
        feat   = self.convs(x)
        pooled = self.pool(feat)
        return pooled, feat


class SegDecoderBlock(nn.Module):
    """
    Decoder block that upsamples `x` by 2x, fuses with a skip connection at
    that resolution, then applies conv layers. Used where a real spatial
    resolution change is needed (dec2, dec1).
    """
    def __init__(self, in_ch, skip_ch, out_ch, gn_groups=8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        merged  = (in_ch // 2) + skip_ch
        self.convs = nn.Sequential(
            ConvBlock(merged,  out_ch, groups=gn_groups),
            ConvBlock(out_ch,  out_ch, groups=gn_groups),
            ConvBlock(out_ch,  out_ch, groups=gn_groups),
            ConvBlock(out_ch,  out_ch, groups=gn_groups),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.convs(x)


class SegFusionBlock(nn.Module):
    """
    Final decoder stage. Fuses two feature maps that are ALREADY at the
    same spatial resolution — no upsampling needed. Used for dec0, which
    fuses the full-resolution decoder output with the full-resolution
    feature_extractor output (f0). See the module-level FIX note for why
    this replaces a (now removed) redundant ConvTranspose2d step.
    """
    def __init__(self, in_ch, skip_ch, out_ch, gn_groups=8):
        super().__init__()
        merged = in_ch + skip_ch
        self.convs = nn.Sequential(
            ConvBlock(merged, out_ch, groups=gn_groups),
            ConvBlock(out_ch, out_ch, groups=gn_groups),
            ConvBlock(out_ch, out_ch, groups=gn_groups),
            ConvBlock(out_ch, out_ch, groups=gn_groups),
        )

    def forward(self, x, skip):
        # Defensive only: with the current encoder this is already a no-op,
        # since x and skip are both at full resolution by construction.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.convs(x)


class FeatureExtractor(nn.Module):
    """
    Identical to pretraining FeatureExtractor (G_φ).
    Must match exactly so pretrained weights load correctly.
    """
    def __init__(self, in_ch=4, feat_ch=21):
        super().__init__()
        gn = min(3, feat_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, padding=1, bias=False, groups=1),
            nn.GroupNorm(gn, feat_ch), nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False, groups=3),
            nn.GroupNorm(gn, feat_ch), nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False, groups=3),
            nn.GroupNorm(gn, feat_ch), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


# ── Main segmentation model ───────────────────────────────────────────────────

class SegUNet2D(nn.Module):
    """
    2D Segmentation U-Net whose encoder is initialised from pretrained
    denoising weights (feature_extractor + enc1 + enc2 + bottleneck).

    Input  : (B, 4, H, W)   — 4 MRI modalities
    Output : (B, num_classes, H, W)  — raw logits

    Encoder layers match pretraining UNetDenoiser exactly:
      feature_extractor → 21-channel low-level features (full res)
      enc1              → 48-channel spatial features (skip @ full res, pooled to stride 2)
      enc2              → 96-channel semantic features (skip @ stride 2, pooled to stride 4)
      bottleneck        → 192-channel deep features (stride 4)

    Decoder + head are NEW (random init), trained in Phase 1 with encoder frozen.
      dec2 : stride 4 -> stride 2, fuse skip2   (real upsample)
      dec1 : stride 2 -> stride 1, fuse skip1   (real upsample)
      dec0 : stride 1 -> stride 1, fuse f0      (fusion only, no upsample)
    """

    def __init__(self,
                 in_channels:  int = 4,
                 num_classes:  int = 4,
                 base_ch:      int = 48,
                 feat_ch:      int = 21,
                 gn_groups:    int = 8):
        super().__init__()
        self.feat_ch = feat_ch

        # ── Encoder (matches pretraining UNetDenoiser) ──
        self.feature_extractor = FeatureExtractor(in_channels, feat_ch)
        self.enc1 = EncoderBlock(feat_ch,      base_ch,     gn_groups)
        self.enc2 = EncoderBlock(base_ch,      base_ch * 2, gn_groups)

        # ── Bottleneck (matches pretraining) ──
        self.bottleneck = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch * 4, groups=gn_groups),
            ConvBlock(base_ch * 4, base_ch * 4, groups=gn_groups),
            ConvBlock(base_ch * 4, base_ch * 4, groups=gn_groups),
        )

        # ── Decoder (NEW — trained from scratch) ──
        self.dec2 = SegDecoderBlock(base_ch * 4, base_ch * 2,
                                    base_ch * 2, gn_groups)
        self.dec1 = SegDecoderBlock(base_ch * 2, base_ch,
                                    base_ch,     gn_groups)
        # dec0: fusion only (no upsample) — see FIX note above
        self.dec0 = SegFusionBlock(base_ch, feat_ch,
                                   base_ch // 2, gn_groups)

        # ── Segmentation head (NEW) ──
        self.seg_head = nn.Sequential(
            nn.Conv2d(base_ch // 2, base_ch // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_ch // 2, num_classes, 1)
        )

        # Track which layers are "pretrained" for freeze/unfreeze
        self._encoder_modules = [
            "feature_extractor", "enc1", "enc2", "bottleneck"
        ]

    def forward(self, x):
        # Feature extraction
        f0 = self.feature_extractor(x)     # (B, feat_ch, H, W)

        # Encoder
        x1, skip1 = self.enc1(f0)          # x1: (B, 48, H/2, W/2)  | skip1: (B, 48, H, W)
        x2, skip2 = self.enc2(x1)          # x2: (B, 96, H/4, W/4)  | skip2: (B, 96, H/2, W/2)

        # Bottleneck
        x3 = self.bottleneck(x2)           # (B, 192, H/4, W/4)

        # Decoder with skip connections
        x  = self.dec2(x3, skip2)          # (B, 96, H/2, W/2)
        x  = self.dec1(x,  skip1)          # (B, 48, H, W)
        x  = self.dec0(x,  f0)             # (B, 24, H, W)  — fusion only, no upsample

        return self.seg_head(x)            # (B, num_classes, H, W)

    # ── Pretrained weight loading ─────────────────────────────────────────────

    def load_pretrained_encoder(self, pretrained_path: str,
                                verbose: bool = True) -> int:
        """
        Load encoder weights from pretraining checkpoint.

        Handles the one known shape mismatch:
          enc1.convs.0.block.0.weight — pretrained (48, 147, 3, 3)
                                        vs model    (48, 21, 3, 3)

        Why this mismatch exists:
          During pretraining, enc1 receives N=7 stacked feature maps
          (7 × 21 = 147 input channels). During fine-tuning, enc1
          receives just the feature_extractor output (21 channels).

        Fix: reshape pretrained weight (48, 147, 3, 3) →
             (48, 7, 21, 3, 3), average over the 7 groups →
             (48, 21, 3, 3). This preserves the learned filter
             structure while adapting to the new input dimension.

        Note on scale: averaging (rather than summing) the 7 groups means
        this layer's output magnitude at init will generally be smaller
        than it was during pretraining (pretraining summed the contribution
        of 7 different input slices; fine-tuning applies the averaged
        kernel to a single slice). GroupNorm right after this layer absorbs
        most of that scale shift, but since Phase 1 freezes the encoder for
        `phase1_epochs`, this layer stays exactly as adapted for the whole
        of Phase 1. The printout below reports the weight-norm before/after
        adaptation so you can sanity-check it is not wildly off.

        Returns: number of matched layers (expect 48 after fix)
        """
        pretrained = torch.load(pretrained_path, map_location="cpu")
        model_dict = self.state_dict()

        matched  = {}
        adapted  = []
        skipped  = []
        misshape = []

        # Known mismatch key and its fix
        MISMATCH_KEY = "enc1.convs.0.block.0.weight"
        N_GROUPS     = 7   # pretraining used N=7 slice window

        for k, v in pretrained.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    # Direct match — load as-is
                    matched[k] = v

                elif (k == MISMATCH_KEY
                      and v.shape[1] == model_dict[k].shape[1] * N_GROUPS):
                    # Known mismatch: (48, 147, 3, 3) → (48, 21, 3, 3)
                    # Average across the N_GROUPS of feat_ch channels
                    out_ch, in_ch_total, kH, kW = v.shape
                    feat_ch = model_dict[k].shape[1]   # 21
                    pre_norm = v.norm().item()
                    adapted_weight = (
                        v.view(out_ch, N_GROUPS, feat_ch, kH, kW)
                         .mean(dim=1)                  # (48, 21, 3, 3)
                    )
                    post_norm = adapted_weight.norm().item()
                    matched[k] = adapted_weight
                    adapted.append(
                        f"{k}: {tuple(v.shape)} → averaged "
                        f"{N_GROUPS} groups → {tuple(adapted_weight.shape)} "
                        f"(||W|| {pre_norm:.3f} -> {post_norm:.3f})")

                else:
                    misshape.append(
                        f"{k}: pretrained {v.shape} "
                        f"vs model {model_dict[k].shape}")
            else:
                skipped.append(k)

        model_dict.update(matched)
        self.load_state_dict(model_dict, strict=False)

        if verbose:
            direct  = len(matched) - len(adapted)
            print(f"[Weight Transfer] Matched {len(matched)} / "
                  f"{len(pretrained)} pretrained layers")
            print(f"  Direct matches : {direct}")
            if adapted:
                print(f"  Adapted (group-averaged):")
                for a in adapted:
                    print(f"    {a}")
            if len(matched) == len(pretrained):
                print("  ✓ ALL 48 pretrained encoder layers loaded")
            if misshape:
                print(f"  Unexpected shape mismatches ({len(misshape)}):")
                for m in misshape:
                    print(f"    {m}")
            if skipped:
                print(f"  Skipped (not in model): {skipped[:3]}")

        return len(matched)

    # ── Two-phase training helpers ────────────────────────────────────────────

    def freeze_encoder(self):
        """
        Phase 1: freeze all pretrained encoder layers.
        Only decoder + seg_head gradients flow.
        Safe to call before wrapping in nn.DataParallel, or on the
        underlying .module after wrapping — DataParallel re-reads
        requires_grad from the source module on every forward call, so
        freezing/unfreezing between phases does not require re-wrapping.
        """
        frozen = 0
        for name, param in self.named_parameters():
            if any(name.startswith(m) for m in self._encoder_modules):
                param.requires_grad = False
                frozen += 1
            else:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"[Phase 1] Frozen {frozen} encoder params | "
              f"Trainable: {trainable:,} params (decoder + head)")

    def unfreeze_all(self):
        """Phase 2: unfreeze everything."""
        for param in self.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in self.parameters())
        print(f"[Phase 2] All {total:,} parameters unfrozen")

    def get_parameter_groups(self, base_lr: float) -> list:
        """
        Differential LR for Phase 2 / joint training:
          encoder (pretrained) → 10x lower LR to protect learned features
          decoder + head (new) → full base_lr

        (Previously this took a separate, unused `phase1_lr` argument that
        had no effect on the returned groups — removed for clarity.)
        """
        encoder_params = [p for n, p in self.named_parameters()
                          if any(n.startswith(m)
                                 for m in self._encoder_modules)]
        decoder_params = [p for n, p in self.named_parameters()
                          if not any(n.startswith(m)
                                     for m in self._encoder_modules)]
        return [
            {"params": encoder_params, "lr": base_lr * 0.1},
            {"params": decoder_params, "lr": base_lr},
        ]


# ── Build function ────────────────────────────────────────────────────────────

def build_model(cfg: dict, device) -> SegUNet2D:
    """Build segmentation model and load pretrained encoder weights."""

    model = SegUNet2D(
        in_channels = cfg["model"]["in_channels"],
        num_classes = cfg["model"]["out_channels"],
        base_ch     = cfg["model"].get("base_channels", 48),
        feat_ch     = cfg["model"].get("feat_ch", 21),
        gn_groups   = cfg["model"].get("group_norm_groups", 8),
    ).to(device)

    pretrained_path = cfg["model"].get("pretrained_encoder", None)
    if pretrained_path:
        if Path(pretrained_path).exists():
            matched = model.load_pretrained_encoder(pretrained_path)
            if matched == 0:
                print("[WARN] No layers matched — check encoder architecture "
                      "matches pretraining UNetDenoiser exactly.")
            elif matched == 48:
                print(f"  ✓ Perfect transfer: all 48 encoder layers loaded")
            else:
                print(f"  Partial transfer: {matched}/48 layers loaded")
        else:
            print(f"[WARN] Pretrained encoder not found: {pretrained_path}")
            print("       Training decoder from scratch (ablation baseline).")
    else:
        print("[INFO] No pretrained encoder specified — random init.")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"  Model params: {total_params:,} total | "
          f"{trainable_params:,} trainable")

    return model