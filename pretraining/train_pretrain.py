"""
pretraining/train_pretrain.py
────────────────────────────────────────────────────────────────────────────────
Self-supervised pretraining loop (STS-UVD on BraTS2021 slices).

Key behaviours:
  • L1 reference is the noisy central slice every epoch (no cross-epoch
    recurrent caching — see note below for why)
  • Training stops when slice-consistency loss L2 converges (< delta)
    OR max_epochs is reached
  • Encoder weights saved after training for transfer to Swin UNETR
  • Resumable via --resume <checkpoint.pth>
  • Both training AND validation show a live tqdm progress bar (samples
    done / total, % complete, running metrics).
  • WeightedSpatialKernel's stochastic blind-spot replacement is disabled
    during validation (see sts_kernel.py) so val PSNR/SSIM aren't corrupted
    by a fresh random pixel-replacement draw every epoch.
  • Uses BOTH GPUs automatically when more than one is visible (e.g.
    Kaggle's T4 x2), via nn.DataParallel — see "Multi-GPU" note below.

Why there's no recurrent reference cache
─────────────────────────────────────────
An earlier version of this file implemented the paper's Section 4 design —
recurrently updating the L1 reference with each sample's own denoised
output from the previous epoch, via a disk-backed cache. It worked, but the
cache is a large (multi-GB) file, and having it live in checkpoint_dir
alongside the .pth checkpoints made downloading/zipping checkpoint_dir from
Kaggle slow and unreliable. By request, the cache was removed entirely:
checkpoint_dir now only ever contains .pth files, so it stays small and
downloads cleanly. L1 reference is the noisy central slice every epoch
(matching what the paper itself does at epoch 0) for the whole run.

Multi-GPU (nn.DataParallel)
─────────────────────────────
The original training step called three separate methods on the model
(`model.extract_all_features(...)`, then `sts(...)`, then `model(...)`)
instead of one `forward()` call. nn.DataParallel only parallelizes a
single forward() — calling separate methods on a wrapped model does NOT
get split across GPUs. To fix this, `STSUVDPipeline` below wraps the
entire per-batch computation (feature extraction -> STS sampling ->
central denoise -> full-window denoise) into one forward() call, so
nn.DataParallel can correctly scatter the batch dimension across all
visible GPUs and gather the results back.

Design notes:
  • `model` and `sts` stay as their own top-level objects in main() (NOT
    replaced by the pipeline) — this is what keeps checkpoint saving/
    loading, `get_encoder_state_dict()`, and --resume completely unchanged.
    The pipeline just holds references to them; wrapping it in
    DataParallel never touches model.state_dict() / sts.state_dict()
    (no "module." key prefix headaches).
  • GroupNorm (not BatchNorm) is used throughout the network, which is
    exactly what makes DataParallel safe here — GroupNorm's statistics are
    computed per-sample, so a small per-GPU batch shard doesn't distort
    normalization statistics the way BatchNorm would.
  • `pipeline.train()` / `pipeline.eval()` are used instead of separately
    toggling `model`/`sts` — this correctly propagates to both submodules
    whether or not DataParallel is wrapping them.
  • cfg["training"]["batch_size"] is the TOTAL batch size across all GPUs
    when DataParallel is active (each GPU gets batch_size // num_gpus).
    With the default batch_size=4 and 2 GPUs, that's only 2 samples/GPU —
    workable, but small enough that per-step overhead eats into the
    speedup. Consider raising batch_size (e.g. to 8) if VRAM allows, to
    get more benefit from the second GPU.
  • DataParallel gathers replicated outputs back to the primary GPU every
    step and re-broadcasts weights every forward call — expect realistic
    speedup in the ~1.3-1.6x range on 2 GPUs, not 2x. True
    DistributedDataParallel scales better but needs per-rank process
    launching, distributed samplers, and metric reduction across ranks —
    a bigger change than this file makes; ask if you want that instead.
  • Falls back to plain single-GPU (or CPU) execution automatically when
    only one device is visible — no special-casing needed elsewhere.

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


# ── Multi-GPU pipeline wrapper ──────────────────────────────────────────────

class STSUVDPipeline(nn.Module):
    """
    Combines feature extraction (G_φ) + STS sampling (𝒯, 𝒮) + denoising
    into a SINGLE forward() call, so nn.DataParallel can correctly split
    the batch dimension across multiple GPUs.

    Without this wrapper, the training step called three separate methods
    on the model (extract_all_features, then the STS module, then the
    model again) — nn.DataParallel only parallelizes one forward() call,
    so calling separate methods like that on a wrapped model runs
    entirely on a single GPU regardless of how many are wrapped.

    `model` and `sts` are stored by reference, not copied — this class is
    purely an orchestration wrapper. Checkpointing/resuming still saves
    and loads model.state_dict() / sts.state_dict() directly (see main()),
    completely unaffected by whether this pipeline is wrapped in
    DataParallel or not.
    """

    def __init__(self, model: UNetDenoiser, sts: STSModule):
        super().__init__()
        self.model = model
        self.sts   = sts

    def forward(self, noisy: torch.Tensor, epoch: int, max_epoch: int):
        """
        noisy : (B, N, 4, H, W) — this is the tensor nn.DataParallel
                scatters across GPUs along the batch dimension (dim 0)
                when this pipeline is wrapped. `epoch`/`max_epoch` are
                plain ints, so DataParallel broadcasts them unchanged to
                every replica rather than splitting them.

        Returns:
            denoised_central : (B, 4, H, W)
            denoised_window  : (B, N, 4, H, W)
            Both are automatically gathered back onto the primary GPU by
            DataParallel when wrapped (it recurses through tuple outputs).
        """
        features = self.model.extract_all_features(noisy)      # (B, N, 21, H, W)
        sampled  = self.sts(features, noisy, epoch, max_epoch) # (B, N, 21, H, W)
        denoised_central = self.model(sampled)                 # (B, 4, H, W)
        denoised_window  = _denoise_full_window(
            self.model, sampled, self.sts.T.N)                 # (B, N, 4, H, W)
        return denoised_central, denoised_window


# ── Training step ─────────────────────────────────────────────────────────────

def train_one_epoch(pipeline:    nn.Module,
                    loader,
                    criterion:   PretrainLoss,
                    optimizer,
                    scaler:      GradScaler,
                    epoch:       int,
                    max_epochs:  int,
                    device,
                    log_every:   int,
                    use_amp:     bool) -> dict:

    pipeline.train()   # propagates to model + sts whether or not DataParallel-wrapped
    running      = defaultdict(float)   # resets every log_every steps (for periodic logging)
    epoch_totals = defaultdict(float)   # NEVER resets — used for the true epoch-end average

    batch_size    = loader.batch_size
    total_samples = len(loader.dataset)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{max_epochs} [train]", unit="batch")

    for step, batch in enumerate(pbar):
        noisy   = batch["noisy"].to(device)    # (B, N, 4, H, W)
        clean   = batch["clean"].to(device)    # (B, N, 4, H, W)
        C_idx   = batch["central"][0].item()   # scalar (same for all in batch)

        # Reference for L1 loss — the noisy central slice itself.
        # (No cross-epoch recurrent caching — see module docstring.)
        noisy_central = noisy[:, C_idx]       # (B, 4, H, W)
        reference     = noisy_central

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            # Single forward() call — this is what makes multi-GPU
            # scattering via nn.DataParallel actually work (see
            # STSUVDPipeline docstring).
            denoised_central, denoised_window = pipeline(noisy, epoch, max_epochs)
            loss_dict = criterion(denoised_central, reference, denoised_window)
            loss      = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(pipeline.parameters()), 1.0)
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

    NOTE: called from inside STSUVDPipeline.forward(), i.e. once PER
    REPLICA when wrapped in nn.DataParallel — `model` here is that
    replica's local copy, `sampled` is already that replica's batch
    shard, so this runs correctly per-GPU with no changes needed.
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
def validate(pipeline:  nn.Module,
             loader,
             criterion: PretrainLoss,
             epoch:    int,
             max_epochs: int,
             device) -> dict:

    pipeline.eval()   # propagates to model + sts whether or not DataParallel-wrapped
    all_psnr, all_ssim = [], []
    total_l1, total_l2 = 0.0, 0.0

    batch_size    = loader.batch_size
    total_samples = len(loader.dataset)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{max_epochs} [val]  ", unit="batch")

    for step, batch in enumerate(pbar):
        noisy    = batch["noisy"].to(device)
        clean    = batch["clean"].to(device)
        C_idx    = batch["central"][0].item()

        clean_central = clean[:, C_idx]

        denoised, dw = pipeline(noisy, epoch, max_epochs)

        # Use clean as reference for val PSNR/SSIM (we have pseudo-clean)
        for b in range(denoised.shape[0]):
            d = denoised[b].cpu().numpy()
            c = clean_central[b].cpu().numpy()
            all_psnr.append(compute_psnr(c, d))
            all_ssim.append(compute_ssim(c, d))

        ld       = criterion(denoised, clean_central, dw)
        total_l1 += ld["l1"].item()
        total_l2 += ld["l2"].item()

        # ── Live progress bar — updates every batch ──
        samples_done   = min((step + 1) * batch_size, total_samples)
        pct             = samples_done / total_samples * 100
        running_psnr    = float(np.mean(all_psnr))
        running_ssim    = float(np.mean(all_ssim))
        pbar.set_postfix({
            "samples": f"{samples_done}/{total_samples}",
            "%":       f"{pct:.1f}",
            "psnr":    f"{running_psnr:.2f}",
            "ssim":    f"{running_ssim:.3f}",
        })

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
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus  = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs visible: {num_gpus}")

    # ── Data ──
    # num_workers bumped from 2 -> 4 when multi-GPU is active: two GPUs
    # consuming batches can outpace a 2-worker CPU dataloader, especially
    # since augmentation (crop/flip/rotate) happens per-sample in Python.
    train_loader, val_loader = get_pretrain_loaders(
        slices_dir          = cfg["data"]["output_slices"],
        N                   = cfg["sts"]["N"],
        patch_size          = cfg["data"]["patch_size"],
        batch_size          = cfg["training"]["batch_size"],
        num_workers         = 4 if num_gpus > 1 else 2,
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

    # ── Multi-GPU pipeline ──
    # `model` and `sts` remain the canonical objects used for checkpoint
    # save/load and get_encoder_state_dict() below — the pipeline just
    # wraps references to them for a single scatter-able forward() call.
    pipeline = STSUVDPipeline(model, sts).to(device)
    if num_gpus > 1:
        print(f"[Multi-GPU] Wrapping pipeline in nn.DataParallel across "
              f"{num_gpus} GPUs (device_ids=0..{num_gpus-1}).")
        print(f"[Multi-GPU] cfg.training.batch_size={cfg['training']['batch_size']} "
              f"is the TOTAL batch across all GPUs "
              f"(~{cfg['training']['batch_size'] // num_gpus} samples/GPU).")
        pipeline = nn.DataParallel(pipeline)
    else:
        print("[Multi-GPU] Only one GPU (or CPU) visible — running normally, "
              "no DataParallel wrapping.")

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
            pipeline, train_loader, criterion, optimizer,
            scaler, epoch, max_epochs, device,
            cfg["logging"]["log_every"], cfg["training"]["amp"])

        val_metrics = validate(
            pipeline, val_loader, criterion,
            epoch, max_epochs, device)

        scheduler.step()

        print(f"  Train — " +
              " | ".join(f"{k}: {v:.4f}"
                         for k, v in train_metrics.items()))
        print(f"  Val   — PSNR: {val_metrics['psnr']:.2f} dB | "
              f"SSIM: {val_metrics['ssim']:.4f} | "
              f"L2: {val_metrics['l2']:.6f}")

        # Save best — model/sts are the canonical (unwrapped) objects, so
        # this state_dict has ordinary keys regardless of DataParallel.
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
    # model is the canonical (unwrapped) object — unaffected by DataParallel.
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