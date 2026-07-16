"""
finetuning/train_finetune.py
────────────────────────────────────────────────────────────────────────────────
Two-phase fine-tuning loop for BraTS-PED 2D segmentation.

Phase 1 (epochs 1–30):
  Encoder frozen → only decoder + head trained
  LR = 0.0001

Phase 2 (epochs 31–100):
  All layers unfrozen
  Encoder LR = 10× lower than decoder (protects pretrained features)
  Decoder LR = 0.00005

Fixes vs original:
  - Resume support via --resume <checkpoint.pth>
  - tqdm progress bars for both train and val (never looks frozen)
  - Correct epoch-average (separate epoch_totals accumulator)
  - Correct dice_mean calculation (was dividing by 3 twice)
  - float() casts on all LR values from config
  - Scheduler state saved/restored in every checkpoint
  - Phase 2 correctly resumes without re-running Phase 1

Run:
    python train_finetune.py --config ../configs/finetune_config.yaml
    python train_finetune.py --config ../configs/finetune_config.yaml \
        --resume /kaggle/input/finetune-ckpt/best_model.pth
────────────────────────────────────────────────────────────────────────────────
"""

import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finetuning.dataset       import get_finetune_loaders
from finetuning.swin_unetr_2d import build_model
from finetuning.losses        import CombinedSegLoss
from evaluation.metrics       import (compute_dice_wt, compute_dice_tc,
                                      compute_dice_et, compute_hd95)


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Saved → {path}")


def load_checkpoint(path, model, optimizer, scaler, scheduler, device):
    """
    Restore full training state from a checkpoint.
    Returns: (start_epoch, start_phase, best_dice)
      - start_phase: 1 or 2 — which phase to resume from
      - start_epoch: which epoch within that phase to start at
    """
    print(f"\n[Resume] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        print("  [WARN] No optimizer state in checkpoint — "
              "optimizer restarts fresh")

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    elif scheduler is not None and "epoch" in ckpt:
        # Older checkpoint without scheduler state — fast-forward
        for _ in range(ckpt["epoch"] + 1):
            scheduler.step()
        print(f"  [WARN] No scheduler state — fast-forwarded "
              f"{ckpt['epoch']+1} steps")

    start_epoch = ckpt.get("epoch", 0) + 1
    start_phase = ckpt.get("phase", 1)
    best_dice   = ckpt.get("best_dice", -1.0)

    print(f"  ✓ Resumed from Phase {start_phase}, "
          f"epoch {ckpt.get('epoch', 0) + 1}")
    print(f"  Best Dice so far: {best_dice:.4f}\n")

    return start_epoch, start_phase, best_dice


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer,
                    scaler, device, log_every, use_amp,
                    epoch_label=""):
    model.train()
    running      = defaultdict(float)   # resets every log_every (for periodic logs)
    epoch_totals = defaultdict(float)   # never resets (for correct epoch average)

    batch_size    = loader.batch_size
    total_samples = len(loader.dataset)

    pbar = tqdm(loader, desc=f"Train {epoch_label}", unit="batch")

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)   # (B, 4, H, W)
        masks  = batch["mask"].to(device)    # (B, H, W)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits    = model(images)        # (B, num_classes, H, W)
            loss_dict = criterion(logits, masks)
            loss      = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            running[k]      += v.item()
            epoch_totals[k] += v.item()

        # Live progress bar
        samples_done = (step + 1) * batch_size
        pct          = samples_done / total_samples * 100
        pbar.set_postfix({
            "samples": f"{min(samples_done, total_samples)}/{total_samples}",
            "%":       f"{pct:.1f}",
            "loss":    f"{loss.item():.4f}"
        })

        if (step + 1) % log_every == 0:
            avg = {k: v / log_every for k, v in running.items()}
            tqdm.write(
                f"  {epoch_label} step {step+1}/{len(loader)} "
                f"({pct:.1f}%) | " +
                " | ".join(f"{k}: {v:.4f}" for k, v in avg.items()))
            running = defaultdict(float)

    return {k: v / len(loader) for k, v in epoch_totals.items()}


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, epoch_label=""):
    model.eval()
    dice_wt, dice_tc, dice_et = [], [], []
    hd95_wt, hd95_tc, hd95_et = [], [], []
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Val   {epoch_label}",
                unit="batch", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        logits    = model(images)
        loss_dict = criterion(logits, masks)
        total_loss += loss_dict["total"].item()

        preds = logits.argmax(dim=1)   # (B, H, W)

        for b in range(preds.shape[0]):
            p = preds[b].cpu().numpy()
            m = masks[b].cpu().numpy()

            dice_wt.append(compute_dice_wt(p, m))
            dice_tc.append(compute_dice_tc(p, m))
            dice_et.append(compute_dice_et(p, m))

            # HD95 is expensive — compute on every 10th sample
            if len(dice_wt) % 10 == 0:
                hd95_wt.append(compute_hd95(p > 0,           m > 0))
                hd95_tc.append(compute_hd95(np.isin(p,[1,3]),
                                             np.isin(m,[1,3])))
                hd95_et.append(compute_hd95(p == 3,          m == 3))

        # Update val bar with running dice
        if dice_wt:
            pbar.set_postfix({
                "WT": f"{np.mean(dice_wt):.3f}",
                "TC": f"{np.mean(dice_tc):.3f}",
                "ET": f"{np.mean(dice_et):.3f}",
            })

    # FIX: np.mean() already averages — don't divide by 3 again
    mean_dice = float(np.mean(
        [np.mean(dice_wt), np.mean(dice_tc), np.mean(dice_et)]))

    metrics = {
        "loss":      total_loss / len(loader),
        "dice_wt":   float(np.mean(dice_wt)),
        "dice_tc":   float(np.mean(dice_tc)),
        "dice_et":   float(np.mean(dice_et)),
        "dice_mean": mean_dice,
    }
    if hd95_wt:
        metrics["hd95_wt"] = float(np.mean(hd95_wt))
        metrics["hd95_tc"] = float(np.mean(hd95_tc))
        metrics["hd95_et"] = float(np.mean(hd95_et))

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg_path, resume_path=None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_finetune_loaders(
        slices_dir  = cfg["data"]["output_slices"],
        patch_size  = cfg["data"]["patch_size"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = 2
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(cfg, device)

    # ── Loss ───────────────────────────────────────────────────────────────────
    criterion = CombinedSegLoss(
        num_classes  = cfg["data"]["num_classes"],
        dice_weight  = float(cfg["loss"]["dice_weight"]),
        focal_weight = float(cfg["loss"]["focal_weight"]),
        focal_gamma  = float(cfg["loss"]["focal_gamma"]),
        focal_alpha  = float(cfg["loss"]["focal_alpha"]),
    )

    scaler   = GradScaler(enabled=cfg["training"]["amp"])
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])

    # ── Resume state defaults ─────────────────────────────────────────────────
    best_dice    = -1.0
    resume_phase = 1    # which phase to start from
    resume_epoch = 0    # which epoch within that phase to start from

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — freeze encoder, warm up decoder
    # ══════════════════════════════════════════════════════════════════════════
    p1_epochs = cfg["training"]["phase1_epochs"]
    p1_lr     = float(cfg["training"]["phase1_lr"])

    # Build Phase 1 optimizer + scheduler (needed even if resuming Phase 2,
    # so that load_checkpoint can restore scheduler state correctly)
    model.freeze_encoder()
    p1_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=p1_lr,
        weight_decay=float(cfg["training"]["weight_decay"])
    )
    p1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        p1_optimizer, T_max=p1_epochs)

    # Handle resume
    if resume_path and Path(resume_path).exists():
        resume_epoch, resume_phase, best_dice = load_checkpoint(
            resume_path, model, p1_optimizer, scaler, p1_scheduler, device)

    # Only run Phase 1 if we're resuming from Phase 1 or starting fresh
    if resume_phase == 1:
        print("\n" + "="*60)
        print("PHASE 1 — Decoder warmup (encoder frozen)")
        print("="*60)

        for epoch in range(resume_epoch, p1_epochs):
            label = f"[P1 {epoch+1}/{p1_epochs}]"
            print(f"\n{label}")

            train_metrics = train_one_epoch(
                model, train_loader, criterion, p1_optimizer,
                scaler, device, cfg["logging"]["log_every"],
                cfg["training"]["amp"], epoch_label=label)

            val_metrics = validate(
                model, val_loader, criterion, device, epoch_label=label)

            p1_scheduler.step()

            print(f"  Train — " +
                  " | ".join(f"{k}: {v:.4f}"
                             for k, v in train_metrics.items()))
            print(f"  Val   — WT: {val_metrics['dice_wt']:.4f} | "
                  f"TC: {val_metrics['dice_tc']:.4f} | "
                  f"ET: {val_metrics['dice_et']:.4f} | "
                  f"Mean: {val_metrics['dice_mean']:.4f}")

            if val_metrics["dice_mean"] > best_dice:
                best_dice = val_metrics["dice_mean"]
                save_checkpoint({
                    "epoch":     epoch,
                    "phase":     1,
                    "model":     model.state_dict(),
                    "optimizer": p1_optimizer.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "scheduler": p1_scheduler.state_dict(),
                    "best_dice": best_dice,
                    "dice_wt":   val_metrics["dice_wt"],
                    "dice_tc":   val_metrics["dice_tc"],
                    "dice_et":   val_metrics["dice_et"],
                }, ckpt_dir / "best_model.pth")
                print(f"  ★ New best mean Dice: {best_dice:.4f}")

            if (epoch + 1) % cfg["training"]["save_every"] == 0:
                save_checkpoint({
                    "epoch":     epoch,
                    "phase":     1,
                    "model":     model.state_dict(),
                    "optimizer": p1_optimizer.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "scheduler": p1_scheduler.state_dict(),
                    "best_dice": best_dice,
                }, ckpt_dir / f"phase1_epoch_{epoch+1:03d}.pth")

        # Reset resume_epoch for Phase 2 start
        resume_epoch = 0

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — unfreeze all, full fine-tuning with differential LR
    # ══════════════════════════════════════════════════════════════════════════
    p2_epochs = cfg["training"]["phase2_epochs"]
    p2_lr     = float(cfg["training"]["phase2_lr"])

    print("\n" + "="*60)
    print("PHASE 2 — Full fine-tuning (all layers unfrozen)")
    print("="*60)

    model.unfreeze_all()
    param_groups = model.get_parameter_groups(p1_lr, p2_lr)
    p2_optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg["training"]["weight_decay"]))
    p2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        p2_optimizer, T_max=p2_epochs)

    # If resuming from Phase 2, reload optimizer/scheduler for Phase 2
    if resume_phase == 2 and resume_path and Path(resume_path).exists():
        _, _, best_dice = load_checkpoint(
            resume_path, model, p2_optimizer, scaler, p2_scheduler, device)

    for epoch in range(resume_epoch, p2_epochs):
        label = f"[P2 {epoch+1}/{p2_epochs}]"
        print(f"\n{label}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, p2_optimizer,
            scaler, device, cfg["logging"]["log_every"],
            cfg["training"]["amp"], epoch_label=label)

        val_metrics = validate(
            model, val_loader, criterion, device, epoch_label=label)

        p2_scheduler.step()

        print(f"  Train — " +
              " | ".join(f"{k}: {v:.4f}"
                         for k, v in train_metrics.items()))
        print(f"  Val   — WT: {val_metrics['dice_wt']:.4f} | "
              f"TC: {val_metrics['dice_tc']:.4f} | "
              f"ET: {val_metrics['dice_et']:.4f} | "
              f"Mean: {val_metrics['dice_mean']:.4f}")
        if "hd95_wt" in val_metrics:
            print(f"  HD95  — WT: {val_metrics['hd95_wt']:.2f} | "
                  f"TC: {val_metrics['hd95_tc']:.2f} | "
                  f"ET: {val_metrics['hd95_et']:.2f}")

        if val_metrics["dice_mean"] > best_dice:
            best_dice = val_metrics["dice_mean"]
            save_checkpoint({
                "epoch":     epoch,
                "phase":     2,
                "model":     model.state_dict(),
                "optimizer": p2_optimizer.state_dict(),
                "scaler":    scaler.state_dict(),
                "scheduler": p2_scheduler.state_dict(),
                "best_dice": best_dice,
                "dice_wt":   val_metrics["dice_wt"],
                "dice_tc":   val_metrics["dice_tc"],
                "dice_et":   val_metrics["dice_et"],
            }, ckpt_dir / "best_model.pth")
            print(f"  ★ New best mean Dice: {best_dice:.4f}")

        if (epoch + 1) % cfg["training"]["save_every"] == 0:
            save_checkpoint({
                "epoch":     epoch,
                "phase":     2,
                "model":     model.state_dict(),
                "optimizer": p2_optimizer.state_dict(),
                "scaler":    scaler.state_dict(),
                "scheduler": p2_scheduler.state_dict(),
                "best_dice": best_dice,
            }, ckpt_dir / f"phase2_epoch_{epoch+1:03d}.pth")

    print(f"\n✓ Training complete. Best mean Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="../configs/finetune_config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from. "
                             "Automatically detects which phase to resume.")
    args = parser.parse_args()
    main(args.config, args.resume)