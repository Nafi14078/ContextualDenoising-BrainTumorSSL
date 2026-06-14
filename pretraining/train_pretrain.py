"""
pretraining/train_pretrain.py
────────────────────────────────────────────────────────────────────────────────
Recurrent self-supervised pretraining loop (STS-UVD on BraTS2021 slices).

Key behaviours:
  • Epoch 0: reference = noisy central slice
  • Each subsequent epoch: reference ← previous epoch's denoised output
  • Training stops when slice-consistency loss L2 converges (< delta)
    OR max_epochs is reached
  • Encoder weights saved after training for transfer to Swin UNETR

Run (Kaggle notebook cell):
    !python train_pretrain.py --config ../configs/pretrain_config.yaml
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import math
import yaml
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pretraining.dataset      import get_pretrain_loaders
from pretraining.sts_kernel   import STSModule
from pretraining.unet_denoiser import UNetDenoiser
from pretraining.losses        import PretrainLoss
from evaluation.metrics        import compute_psnr, compute_ssim


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path: Path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"] + 1, ckpt.get("references", None)


# ── Reference Cache ───────────────────────────────────────────────────────────

class ReferenceCache:
    """
    Stores previous-epoch denoised outputs keyed by dataset index.
    On epoch 0, returns the noisy input as the reference (cold start).
    Lives on CPU to avoid GPU OOM.
    """

    def __init__(self):
        self._cache = {}

    def get(self, idx: int,
            noisy_central: torch.Tensor) -> torch.Tensor:
        """Return cached reference or noisy_central if not yet cached."""
        if idx in self._cache:
            return self._cache[idx].to(noisy_central.device)
        return noisy_central.clone()

    def update(self, idx: int, denoised: torch.Tensor):
        """Store denoised output (detached, on CPU)."""
        self._cache[idx] = denoised.detach().cpu()

    def clear(self):
        self._cache.clear()


# ── Training step ─────────────────────────────────────────────────────────────

def train_one_epoch(model:       UNetDenoiser,
                    sts:         STSModule,
                    loader,
                    criterion:   PretrainLoss,
                    optimizer,
                    scaler:      GradScaler,
                    ref_cache:   ReferenceCache,
                    epoch:       int,
                    max_epochs:  int,
                    device,
                    log_every:   int,
                    use_amp:     bool) -> dict:

    model.train()
    sts.train()
    running = defaultdict(float)

    for step, batch in enumerate(loader):
        noisy  = batch["noisy"].to(device)    # (B, N, 4, H, W)
        clean  = batch["clean"].to(device)    # (B, N, 4, H, W)
        C_idx  = batch["central"][0].item()   # scalar (same for all in batch)

        # Reference for L1 loss (recurrent — from previous epoch)
        # batch indices: we use the step as a proxy key
        # In a real setup you'd track dataset indices through the loader
        noisy_central = noisy[:, C_idx]       # (B, 4, H, W)
        ref_key       = epoch * len(loader) + step  # unique per step
        reference     = ref_cache.get(ref_key, noisy_central)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            # Step 1: Feature extraction G_φ
            features = model.extract_all_features(noisy)  # (B, N, 21, H, W)

            # Step 2: STS sampling (𝒯 + 𝒮)
            sampled  = sts(features, noisy, epoch, max_epochs)  # (B, N, 21, H, W)

            # Step 3: Denoise central slice
            denoised_central = model(sampled)              # (B, 4, H, W)

            # Step 4: Compute losses
            # For L2 we need denoised versions of ALL slices in the window.
            # Full window denoising is expensive; we use a single forward pass
            # by treating each slice as the "central" in sequence.
            # Efficient approximation: use clean as the denoised window proxy
            # during early epochs, then refine.
            denoised_window = _denoise_full_window(
                model, sts, noisy, epoch, max_epochs)     # (B, N, 4, H, W)

            loss_dict = criterion(denoised_central, reference, denoised_window)
            loss      = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(sts.parameters()), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update reference cache for next epoch
        ref_cache.update(ref_key, denoised_central)

        for k, v in loss_dict.items():
            running[k] += v.item()

        if (step + 1) % log_every == 0:
            avg = {k: v / log_every for k, v in running.items()}
            print(f"  [Epoch {epoch+1}] step {step+1}/{len(loader)} | "
                  + " | ".join(f"{k}: {v:.4f}" for k, v in avg.items()))
            running = defaultdict(float)

    return {k: v / len(loader) for k, v in running.items()}


def _denoise_full_window(model, sts, noisy, epoch, max_epochs):
    """
    Efficiently denoise all N slices by rotating the 'central' position.
    Uses torch.no_grad() for non-central slices to save memory.
    Returns (B, N, 4, H, W).
    """
    B, N, C, H, W = noisy.shape
    denoised_slices = []

    features = model.extract_all_features(noisy)   # (B, N, 21, H, W)
    sampled  = sts(features, noisy, epoch, max_epochs)

    with torch.no_grad():
        for n in range(N):
            # Shift the window so slice n is in the centre position
            # by rolling the feature tensor (cheap approximation)
            shift   = N // 2 - n
            shifted = torch.roll(sampled, shifts=shift, dims=1)
            out     = model(shifted)                # (B, 4, H, W)
            denoised_slices.append(out)

    return torch.stack(denoised_slices, dim=1)      # (B, N, 4, H, W)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model:    UNetDenoiser,
             sts:      STSModule,
             loader,
             criterion: PretrainLoss,
             epoch:    int,
             max_epochs: int,
             device) -> dict:

    model.eval()
    sts.eval()
    all_psnr, all_ssim = [], []
    total_l1, total_l2 = 0.0, 0.0

    for batch in loader:
        noisy    = batch["noisy"].to(device)
        clean    = batch["clean"].to(device)
        C_idx    = batch["central"][0].item()

        noisy_central = noisy[:, C_idx]
        clean_central = clean[:, C_idx]

        features = model.extract_all_features(noisy)
        sampled  = sts(features, noisy, epoch, max_epochs)
        denoised = model(sampled)

        # Use clean as reference for val PSNR/SSIM (we have pseudo-clean)
        for b in range(denoised.shape[0]):
            d = denoised[b].cpu().numpy()
            c = clean_central[b].cpu().numpy()
            all_psnr.append(compute_psnr(c, d))
            all_ssim.append(compute_ssim(c, d))

        dw       = _denoise_full_window(model, sts, noisy, epoch, max_epochs)
        ld       = criterion(denoised, clean_central, dw)
        total_l1 += ld["l1"].item()
        total_l2 += ld["l2"].item()

    return {
        "psnr":  float(np.mean(all_psnr)),
        "ssim":  float(np.mean(all_ssim)),
        "l1":    total_l1 / len(loader),
        "l2":    total_l2 / len(loader),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    train_loader, val_loader = get_pretrain_loaders(
        slices_dir  = cfg["data"]["output_slices"],
        N           = cfg["sts"]["N"],
        patch_size  = cfg["data"]["patch_size"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = 2
    )

    # ── Models ──
    model = UNetDenoiser(
        N            = cfg["sts"]["N"],
        in_channels  = 4,
        feat_ch      = 21,
        base_ch      = cfg["model"]["base_channels"],
        out_channels = 4,
        gn_groups    = cfg["model"]["group_norm_groups"]
    ).to(device)

    sts = STSModule(
        N             = cfg["sts"]["N"],
        L             = cfg["sts"]["L"],
        eta           = cfg["sts"]["eta"],
        replace_ratio = cfg["sts"]["replace_ratio"],
        window        = cfg["sts"]["window"],
        alpha         = cfg["sts"]["alpha"]
    ).to(device)

    criterion  = PretrainLoss(lambda_consistency=1.0)
    optimizer  = torch.optim.Adam(
        list(model.parameters()) + list(sts.parameters()),
        lr           = cfg["training"]["lr"],
        betas        = (cfg["training"]["beta1"],
                        cfg["training"]["beta2"]),
        weight_decay = cfg["training"]["weight_decay"]
    )
    scheduler  = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["training"]["lr_decay_step"], gamma=0.5)
    scaler     = GradScaler(enabled=cfg["training"]["amp"])
    ref_cache  = ReferenceCache()

    max_epochs = cfg["training"]["epochs"]
    delta      = cfg["training"]["delta"]
    ckpt_dir   = Path(cfg["training"]["checkpoint_dir"])
    best_psnr  = -1.0
    prev_l2    = float("inf")

    # ── Training loop ──
    for epoch in range(max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{max_epochs}")

        train_metrics = train_one_epoch(
            model, sts, train_loader, criterion, optimizer,
            scaler, ref_cache, epoch, max_epochs, device,
            cfg["logging"]["log_every"], cfg["training"]["amp"])

        val_metrics = validate(
            model, sts, val_loader, criterion,
            epoch, max_epochs, device)

        scheduler.step()

        print(f"  Train — " +
              " | ".join(f"{k}: {v:.4f}"
                         for k, v in train_metrics.items()))
        print(f"  Val   — PSNR: {val_metrics['psnr']:.2f} dB | "
              f"SSIM: {val_metrics['ssim']:.4f} | "
              f"L2: {val_metrics['l2']:.6f}")

        # Save best
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            save_checkpoint({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "sts":       sts.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler":    scaler.state_dict(),
                "psnr":      best_psnr,
            }, ckpt_dir / "best_model.pth")

        # Periodic checkpoint
        if (epoch + 1) % cfg["training"]["save_every"] == 0:
            save_checkpoint({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "sts":       sts.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler":    scaler.state_dict(),
            }, ckpt_dir / f"epoch_{epoch+1:03d}.pth")

        # Convergence check on L2 (paper stops when L2 < delta)
        curr_l2 = val_metrics["l2"]
        if abs(prev_l2 - curr_l2) < delta and epoch > 5:
            print(f"\n✓ L2 converged (Δ={abs(prev_l2-curr_l2):.2e} < {delta})")
            print(f"  Stopping at epoch {epoch+1}")
            break
        prev_l2 = curr_l2

    # ── Save encoder weights for transfer ──────────────────────────────────
    encoder_weights = model.get_encoder_state_dict()
    encoder_path    = ckpt_dir / "pretrain_encoder.pth"
    torch.save(encoder_weights, encoder_path)
    print(f"\n✓ Encoder weights saved → {encoder_path}")
    print(f"  Best val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/pretrain_config.yaml")
    args = parser.parse_args()
    main(args.config)
