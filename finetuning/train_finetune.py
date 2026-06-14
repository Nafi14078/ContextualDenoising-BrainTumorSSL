"""
finetuning/train_finetune.py
────────────────────────────────────────────────────────────────────────────────
Two-phase fine-tuning loop for BraTS-PED 2D segmentation.

Phase 1 (epochs 1–30):
  Encoder frozen → only decoder + head trained
  LR = 1e-4

Phase 2 (epochs 31–100):
  All layers unfrozen
  Encoder LR = 5e-6  (10× lower than decoder to protect pretrained features)
  Decoder LR = 5e-5

Run:
    python train_finetune.py --config ../configs/finetune_config.yaml
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


# ── One epoch ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer,
                    scaler, device, log_every, use_amp):
    model.train()
    running = defaultdict(float)

    for step, batch in enumerate(loader):
        images = batch["image"].to(device)   # (B, 4, H, W)
        masks  = batch["mask"].to(device)    # (B, H, W)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits    = model(images)        # (B, 4, H, W)
            loss_dict = criterion(logits, masks)
            loss      = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            running[k] += v.item()

        if (step + 1) % log_every == 0:
            avg = {k: v / log_every for k, v in running.items()}
            print(f"  step {step+1}/{len(loader)} | " +
                  " | ".join(f"{k}: {v:.4f}" for k, v in avg.items()))
            running = defaultdict(float)

    return {k: v / len(loader) for k, v in running.items()}


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    dice_wt, dice_tc, dice_et = [], [], []
    hd95_wt, hd95_tc, hd95_et = [], [], []
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        logits = model(images)
        loss_dict = criterion(logits, masks)
        total_loss += loss_dict["total"].item()

        preds = logits.argmax(dim=1)   # (B, H, W)

        for b in range(preds.shape[0]):
            p = preds[b].cpu().numpy()
            m = masks[b].cpu().numpy()

            dice_wt.append(compute_dice_wt(p, m))
            dice_tc.append(compute_dice_tc(p, m))
            dice_et.append(compute_dice_et(p, m))

            # HD95 is expensive — compute on subset for speed
            if len(dice_wt) % 10 == 0:
                hd95_wt.append(compute_hd95(p > 0,      m > 0))
                hd95_tc.append(compute_hd95(
                    np.isin(p, [1, 3]), np.isin(m, [1, 3])))
                hd95_et.append(compute_hd95(p == 3,     m == 3))

    metrics = {
        "loss":    total_loss / len(loader),
        "dice_wt": float(np.mean(dice_wt)),
        "dice_tc": float(np.mean(dice_tc)),
        "dice_et": float(np.mean(dice_et)),
        "dice_mean": float(np.mean(dice_wt + dice_tc + dice_et)) / 3,
    }
    if hd95_wt:
        metrics["hd95_wt"] = float(np.mean(hd95_wt))
        metrics["hd95_tc"] = float(np.mean(hd95_tc))
        metrics["hd95_et"] = float(np.mean(hd95_et))

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    train_loader, val_loader = get_finetune_loaders(
        slices_dir  = cfg["data"]["output_slices"],
        patch_size  = cfg["data"]["patch_size"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = 2
    )

    # ── Model ──
    model = build_model(cfg, device)

    # ── Loss ──
    criterion = CombinedSegLoss(
        num_classes  = cfg["data"]["num_classes"],
        dice_weight  = cfg["loss"]["dice_weight"],
        focal_weight = cfg["loss"]["focal_weight"],
        focal_gamma  = cfg["loss"]["focal_gamma"],
        focal_alpha  = cfg["loss"]["focal_alpha"],
    )

    scaler   = GradScaler(enabled=cfg["training"]["amp"])
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    best_dice = -1.0

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — freeze encoder, warm up decoder
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PHASE 1 — Decoder warmup (encoder frozen)")
    print("="*60)

    model.freeze_encoder()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["phase1_lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["phase1_epochs"])

    for epoch in range(cfg["training"]["phase1_epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['training']['phase1_epochs']} [Phase 1]")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, cfg["logging"]["log_every"],
            cfg["training"]["amp"])

        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss: {train_metrics['total']:.4f}")
        print(f"  Val   — WT: {val_metrics['dice_wt']:.4f} | "
              f"TC: {val_metrics['dice_tc']:.4f} | "
              f"ET: {val_metrics['dice_et']:.4f}")

        mean_dice = (val_metrics['dice_wt'] +
                     val_metrics['dice_tc'] +
                     val_metrics['dice_et']) / 3.0

        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint({
                "epoch": epoch, "phase": 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "dice_wt": val_metrics["dice_wt"],
                "dice_tc": val_metrics["dice_tc"],
                "dice_et": val_metrics["dice_et"],
            }, ckpt_dir / "best_model.pth")

        if (epoch + 1) % cfg["training"]["save_every"] == 0:
            save_checkpoint({
                "epoch": epoch, "phase": 1,
                "model": model.state_dict(),
            }, ckpt_dir / f"phase1_epoch_{epoch+1:03d}.pth")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — unfreeze all, full fine-tuning with differential LR
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PHASE 2 — Full fine-tuning (all layers unfrozen)")
    print("="*60)

    model.unfreeze_all()
    param_groups = model.get_parameter_groups(
        cfg["training"]["phase1_lr"],
        cfg["training"]["phase2_lr"])
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["phase2_epochs"])

    for epoch in range(cfg["training"]["phase2_epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['training']['phase2_epochs']} [Phase 2]")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, cfg["logging"]["log_every"],
            cfg["training"]["amp"])

        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss: {train_metrics['total']:.4f}")
        print(f"  Val   — WT: {val_metrics['dice_wt']:.4f} | "
              f"TC: {val_metrics['dice_tc']:.4f} | "
              f"ET: {val_metrics['dice_et']:.4f}")
        if "hd95_wt" in val_metrics:
            print(f"  HD95  — WT: {val_metrics['hd95_wt']:.2f} | "
                  f"TC: {val_metrics['hd95_tc']:.2f} | "
                  f"ET: {val_metrics['hd95_et']:.2f}")

        mean_dice = (val_metrics['dice_wt'] +
                     val_metrics['dice_tc'] +
                     val_metrics['dice_et']) / 3.0

        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint({
                "epoch": epoch, "phase": 2,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "dice_wt": val_metrics["dice_wt"],
                "dice_tc": val_metrics["dice_tc"],
                "dice_et": val_metrics["dice_et"],
            }, ckpt_dir / "best_model.pth")
            print(f"  ★ New best mean Dice: {best_dice:.4f}")

        if (epoch + 1) % cfg["training"]["save_every"] == 0:
            save_checkpoint({
                "epoch": epoch, "phase": 2,
                "model": model.state_dict(),
            }, ckpt_dir / f"phase2_epoch_{epoch+1:03d}.pth")

    print(f"\n✓ Training complete. Best mean Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="../configs/finetune_config.yaml")
    args = parser.parse_args()
    main(args.config)
