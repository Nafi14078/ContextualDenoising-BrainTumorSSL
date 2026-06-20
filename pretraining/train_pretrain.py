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

    NOTE on the recurrent reference cache: ReferenceCache lives entirely
    in memory and is NOT saved to disk (it would be enormous — one image
    per training sample). On resume, the cache starts cold again, so the
    first resumed epoch falls back to using the noisy central slice as
    the L1 reference instead of the previous epoch's denoised output.
    This is a known, accepted tradeoff — the recurrent refinement simply
    re-establishes itself within the first resumed epoch and is not a
    correctness bug.
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
    print(f"  [WARN] Reference cache restarts cold on resume — "
          f"recurrent refinement re-establishes within 1 epoch.\n")

    return start_epoch, best_psnr, scheduler_state


# ── Reference Cache ───────────────────────────────────────────────────────────

class ReferenceCache:
    """
    Stores previous-epoch denoised outputs keyed by STABLE dataset index
    (the (subject, central_slice) identity — see dataset.py's "index"
    field), not by batch/step position.

    Why this matters: the train DataLoader shuffles every epoch. A naive
    cache keyed by step number (epoch * len(loader) + step) would mix up
    completely different physical slices across epochs — e.g. "step 437"
    in epoch 1 is almost certainly a different slice than "step 437" in
    epoch 2. That breaks the recurrent reference mechanism described in
    the paper (Section 4): "update the reference frame using the output
    denoised frame from the previous epoch" only makes sense if you're
    talking about the SAME frame each time.

    Keying by dataset index fixes this — index 8471 always refers to the
    exact same (subject, slice) pair, every epoch, regardless of shuffle
    order, so the cache correctly tracks "this slice's own previous
    denoised output."

    On epoch 0 (or whenever a given index hasn't been seen yet), falls
    back to the noisy input — matching the paper's cold-start behaviour.
    Lives on CPU to avoid GPU OOM with large caches.
    """

    def __init__(self):
        self._cache = {}

    def get_batch(self,
                  indices:        torch.Tensor,
                  fallback_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices        : (B,) tensor/list of stable dataset indices
            fallback_batch  : (B, C, H, W) — used for any index not yet
                              cached (cold start)
        Returns:
            (B, C, H, W) — cached reference per-sample, or fallback
        """
        out = []
        for i, idx in enumerate(indices):
            idx = int(idx)
            if idx in self._cache:
                out.append(self._cache[idx].to(fallback_batch.device))
            else:
                out.append(fallback_batch[i].clone())
        return torch.stack(out, dim=0)

    def update_batch(self,
                     indices:  torch.Tensor,
                     denoised: torch.Tensor):
        """
        Args:
            indices  : (B,) tensor/list of stable dataset indices
            denoised : (B, C, H, W) — this epoch's denoised output,
                       becomes next epoch's reference for these exact
                       samples
        """
        for i, idx in enumerate(indices):
            idx = int(idx)
            self._cache[idx] = denoised[i].detach().cpu()

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

    batch_size    = loader.batch_size
    total_samples = len(loader.dataset)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")

    for step, batch in enumerate(pbar):
        noisy   = batch["noisy"].to(device)    # (B, N, 4, H, W)
        clean   = batch["clean"].to(device)    # (B, N, 4, H, W)
        C_idx   = batch["central"][0].item()   # scalar (same for all in batch)
        indices = batch["index"]               # (B,) — STABLE per-sample IDs

        # Reference for L1 loss — recurrent, keyed by stable sample identity
        # so it correctly tracks "this exact slice's previous-epoch output"
        # even though the DataLoader shuffles order every epoch.
        noisy_central = noisy[:, C_idx]       # (B, 4, H, W)
        reference     = ref_cache.get_batch(indices, noisy_central)

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

        # Update reference cache for next epoch — keyed by stable index
        ref_cache.update_batch(indices, denoised_central)

        for k, v in loss_dict.items():
            running[k] += v.item()

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

    return {k: v / len(loader) for k, v in running.items()}


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
    ref_cache  = ReferenceCache()

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