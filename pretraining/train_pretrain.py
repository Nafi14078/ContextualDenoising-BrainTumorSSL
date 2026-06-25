"""
pretraining/train_pretrain.py
────────────────────────────────────────────────────────────────────────────────
Self-supervised pretraining loop (STS-UVD on BraTS2021 slices).

Key behaviours:
  • L1 reference is the noisy central slice every epoch (no cross-epoch
    recurrent caching — see note below for why this was removed)
  • Training stops when slice-consistency loss L2 converges (< delta)
    OR max_epochs is reached
  • Encoder weights saved after training for transfer to Swin UNETR
  • Resumable via --resume <checkpoint.pth>

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
from tqdm import tqdm

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


def load_checkpoint(path: Path, model, sts, optimizer, scaler, device):
    """
    Load a checkpoint and restore full training state.

    Returns:
        start_epoch : epoch to resume FROM (i.e. next epoch to run)
        best_psnr   : best validation PSNR seen so far (or -1.0 if unknown)
        scheduler_state : state dict for the LR scheduler, or None if the
                          checkpoint predates scheduler saving (older
                          checkpoints fall back to step-count fast-forward
                          in main())
    """
    print(f"\n[Resume] Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])

    if "sts" in ckpt:
        sts.load_state_dict(ckpt["sts"])
    else:
        print("  [WARN] Checkpoint has no 'sts' state — STS module "
              "kernels will use freshly initialised state.")

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        print("  [WARN] Checkpoint has no 'optimizer' state — "
              "optimizer momentum/state will restart fresh.")

    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt["epoch"] + 1
    best_psnr   = ckpt.get("psnr", -1.0)
    scheduler_state = ckpt.get("scheduler", None)

    print(f"  ✓ Resumed from epoch {ckpt['epoch'] + 1} "
          f"→ continuing at epoch {start_epoch + 1}")
    if best_psnr < 0:
        print(f"  [WARN] No 'psnr' recorded in this checkpoint — "
              f"best_psnr will reset and update on next validation.")
    else:
        print(f"  Best PSNR so far: {best_psnr:.2f} dB")
    print()

    return start_epoch, best_psnr, scheduler_state


# NOTE: The cross-epoch ReferenceCache mechanism described in earlier
# versions of this file has been REMOVED. It stored one tensor per
# unique training sample (47,115 of them at 576 KB each = ~26 GB),
# growing unboundedly through an epoch with no eviction — causing the
# RAM exhaustion seen on Kaggle. It also had a deeper correctness issue:
# since train uses random crop augmentation every epoch, a cached
# "previous epoch's denoised output" referred to a DIFFERENT spatial
# crop than the current epoch's crop, so the comparison wasn't even
# spatially aligned. The paper's recurrent reference design (Section 4)
# was built for a single short video of a few hundred/thousand frames,
# not a large multi-subject dataset with per-epoch random cropping —
# it doesn't transfer cleanly at this scale.
#
# Reference for the L1 loss is now simply the noisy central slice every
# epoch (matching what epoch 0 already did). The actual unsupervised
# denoising signal still comes from the STS blind-spot kernel (S) and
# temporal weighting (T), which are unaffected by this change.


# ── Training step ─────────────────────────────────────────────────────────────

def train_one_epoch(model:       UNetDenoiser,
                    sts:         STSModule,
                    loader,
                    criterion:   PretrainLoss,
                    optimizer,
                    scaler:      GradScaler,
                    epoch:       int,
                    max_epochs:  int,
                    device,
                    log_every:   int,
                    use_amp:     bool) -> dict:

    model.train()
    sts.train()
    running      = defaultdict(float)   # resets every log_every steps (for periodic logging)
    epoch_totals = defaultdict(float)   # NEVER resets — used for the true epoch-end average

    batch_size    = loader.batch_size
    total_samples = len(loader.dataset)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")

    for step, batch in enumerate(pbar):
        noisy   = batch["noisy"].to(device)    # (B, N, 4, H, W)
        clean   = batch["clean"].to(device)    # (B, N, 4, H, W)
        C_idx   = batch["central"][0].item()   # scalar (same for all in batch)

        # Reference for L1 loss — the noisy central slice itself.
        # (No cross-epoch recurrent caching — see note above class removal.)
        noisy_central = noisy[:, C_idx]       # (B, 4, H, W)
        reference     = noisy_central

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            # Step 1: Feature extraction G_φ
            features = model.extract_all_features(noisy)  # (B, N, 21, H, W)

            # Step 2: STS sampling (𝒯 + 𝒮) — computed ONCE, reused below
            sampled  = sts(features, noisy, epoch, max_epochs)  # (B, N, 21, H, W)

            # Step 3: Denoise central slice
            denoised_central = model(sampled)              # (B, 4, H, W)

            # Step 4: Reuse `sampled` for full-window denoising (no recompute)
            denoised_window = _denoise_full_window(
                model, sampled, sts.T.N)                  # (B, N, 4, H, W)

            loss_dict = criterion(denoised_central, reference, denoised_window)
            loss      = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(sts.parameters()), 1.0)
        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            running[k]      += v.item()
            epoch_totals[k] += v.item()

        # ── Live progress bar — updates every batch ──
        samples_done = (step + 1) * batch_size
        pct = samples_done / total_samples * 100
        pbar.set_postfix({
            "samples": f"{samples_done}/{total_samples}",
            "%":       f"{pct:.1f}",
            "loss":    f"{loss.item():.4f}"
        })

        if (step + 1) % log_every == 0:
            avg = {k: v / log_every for k, v in running.items()}
            tqdm.write(f"  [Epoch {epoch+1}] step {step+1}/{len(loader)} "
                      f"({pct:.1f}%) | "
                      + " | ".join(f"{k}: {v:.4f}" for k, v in avg.items()))
            running = defaultdict(float)

    return {k: v / len(loader) for k, v in epoch_totals.items()}


def _denoise_full_window(model, sampled, N):
    """
    Efficiently denoise all N slices by rotating the 'central' position.
    Reuses the already-computed `sampled` tensor (output of STS module) —
    avoids redundant feature extraction and STS sampling.

    sampled : (B, N, feat_ch, H, W) — output of sts(features, ...)
    Returns : (B, N, 4, H, W)
    """
    denoised_slices = []
    with torch.no_grad():
        for n in range(N):
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

        dw       = _denoise_full_window(model, sampled, sts.T.N)
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

def main(cfg_path: str, resume_path: str = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    train_loader, val_loader = get_pretrain_loaders(
        slices_dir          = cfg["data"]["output_slices"],
        N                   = cfg["sts"]["N"],
        patch_size          = cfg["data"]["patch_size"],
        batch_size          = cfg["training"]["batch_size"],
        num_workers         = 2,
        max_train_subjects  = cfg["data"].get("max_train_subjects", None),
        max_val_subjects    = cfg["data"].get("max_val_subjects", None)
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
        lr           = float(cfg["training"]["lr"]),
        betas        = (float(cfg["training"]["beta1"]),
                        float(cfg["training"]["beta2"])),
        weight_decay = float(cfg["training"]["weight_decay"])
    )
    scaler     = GradScaler(enabled=cfg["training"]["amp"])

    max_epochs = cfg["training"]["epochs"]
    delta      = cfg["training"]["delta"]
    ckpt_dir   = Path(cfg["training"]["checkpoint_dir"])

    # ── Resume state (defaults for a fresh run) ──
    start_epoch     = 0
    best_psnr       = -1.0
    scheduler_state = None

    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            start_epoch, best_psnr, scheduler_state = load_checkpoint(
                resume_path, model, sts, optimizer, scaler, device)
        else:
            print(f"[WARN] --resume path not found: {resume_path}")
            print("       Starting fresh from epoch 0 instead.")

    # ── Scheduler — created fresh, then restored or fast-forwarded ──
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["training"]["lr_decay_step"], gamma=0.5)

    if resume_path and start_epoch > 0:
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        else:
            # Older checkpoint without saved scheduler state — fast-forward
            # by replaying the same number of .step() calls. StepLR's
            # behaviour depends only on call count, so this is exact.
            print(f"  [Resume] No scheduler state found — fast-forwarding "
                  f"{start_epoch} scheduler steps to match.")
            for _ in range(start_epoch):
                scheduler.step()

    prev_l2 = float("inf")

    if start_epoch >= max_epochs:
        print(f"[WARN] Resumed epoch ({start_epoch}) >= max_epochs "
              f"({max_epochs}). Nothing to train — exiting.")
        return

    # ── Training loop ──
    for epoch in range(start_epoch, max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{max_epochs}")

        train_metrics = train_one_epoch(
            model, sts, train_loader, criterion, optimizer,
            scaler, epoch, max_epochs, device,
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
                "scheduler": scheduler.state_dict(),
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
                "scheduler": scheduler.state_dict(),
                "psnr":      best_psnr,
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
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint (.pth) to resume from. "
                             "Restores model, sts, optimizer, scaler, "
                             "scheduler, epoch, and best_psnr.")
    args = parser.parse_args()
    main(args.config, args.resume)