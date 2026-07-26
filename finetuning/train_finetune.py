"""
finetuning/train_finetune.py
────────────────────────────────────────────────────────────────────────────────
Fine-tuning loop for BraTS-PED 2D segmentation.

Two training modes (controlled by --joint flag):

  DEFAULT (two-phase):
    Phase 1 (epochs 1-30): encoder frozen, decoder warms up
    Phase 2 (epochs 31-100): all layers unfrozen, differential LR

  JOINT (--joint flag):
    Single phase: all layers trained together from epoch 1
    Encoder LR = 0.1 x decoder LR (no hard freeze)
    Recommended for pretrained encoder

Best model saved on single val Dice (standard research practice).
TTA used in validation for better accuracy.
Early stopping when val Dice does not improve for patience epochs.

Run:
    python train_finetune.py --config ../configs/finetune_config.yaml
    python train_finetune.py --config ../configs/finetune_config.yaml --joint
    python train_finetune.py --config ../configs/finetune_config.yaml --joint --resume /path/ckpt.pth
"""

import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
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
    print(f"  Saved -> {path}")


def load_checkpoint(path, model, optimizer, scaler, scheduler, device):
    """
    Restore full training state. Handles optimizer group mismatch gracefully.
    Returns: (start_epoch, start_phase, best_dice)
    """
    print(f"\n[Resume] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        ckpt_groups  = len(ckpt["optimizer"]["param_groups"])
        model_groups = len(optimizer.param_groups)
        if ckpt_groups == model_groups:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print(f"  [WARN] Optimizer group mismatch "
                  f"({ckpt_groups} vs {model_groups}) -- restarts fresh")
    else:
        print("  [WARN] No optimizer state -- restarts fresh")

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    elif scheduler is not None and "epoch" in ckpt:
        for _ in range(ckpt["epoch"] + 1):
            scheduler.step()
        print(f"  [WARN] No scheduler state -- fast-forwarded {ckpt['epoch']+1} steps")

    start_epoch = ckpt.get("epoch", 0) + 1
    start_phase = ckpt.get("phase", 1)
    best_dice   = ckpt.get("best_dice", -1.0)

    print(f"  Resumed from Phase {start_phase}, epoch {start_epoch}")
    print(f"  Best Dice so far: {best_dice:.4f}\n")
    return start_epoch, start_phase, best_dice


# ── Test-Time Augmentation ────────────────────────────────────────────────────

@torch.no_grad()
def predict_tta(model, images, device):
    """Average predictions across 4 rotations + horizontal flip."""
    model.eval()
    preds = []
    for k in range(4):
        aug   = torch.rot90(images, k=k, dims=[2, 3])
        logit = model(aug.to(device))
        logit = torch.rot90(logit, k=-k, dims=[2, 3])
        preds.append(F.softmax(logit, dim=1))
    aug   = torch.flip(images, dims=[3])
    logit = model(aug.to(device))
    logit = torch.flip(logit, dims=[3])
    preds.append(F.softmax(logit, dim=1))
    return torch.stack(preds).mean(0)


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer,
                    scaler, device, log_every, use_amp, epoch_label=""):
    model.train()
    running      = defaultdict(float)
    epoch_totals = defaultdict(float)
    batch_size    = loader.batch_size
    total_samples = len(loader.dataset)
    pbar = tqdm(loader, desc=f"Train {epoch_label}", unit="batch")

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits    = model(images)
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

        samples_done = (step + 1) * batch_size
        pct          = samples_done / total_samples * 100
        pbar.set_postfix({
            "samples": f"{min(samples_done, total_samples)}/{total_samples}",
            "%": f"{pct:.1f}", "loss": f"{loss.item():.4f}"})

        if (step + 1) % log_every == 0:
            avg = {k: v / log_every for k, v in running.items()}
            tqdm.write(
                f"  {epoch_label} step {step+1}/{len(loader)} ({pct:.1f}%) | " +
                " | ".join(f"{k}: {v:.4f}" for k, v in avg.items()))
            running = defaultdict(float)

    return {k: v / len(loader) for k, v in epoch_totals.items()}


# ── Validation with TTA ───────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device,
             use_tta=True, epoch_label=""):
    model.eval()
    dice_wt, dice_tc, dice_et = [], [], []
    hd95_wt, hd95_tc, hd95_et = [], [], []
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Val   {epoch_label}", unit="batch", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        logits    = model(images)
        loss_dict = criterion(logits, masks)
        total_loss += loss_dict["total"].item()

        if use_tta:
            probs = predict_tta(model, images, device)
            preds = probs.argmax(dim=1)
        else:
            preds = logits.argmax(dim=1)

        for b in range(preds.shape[0]):
            p = preds[b].cpu().numpy()
            m = masks[b].cpu().numpy()
            dice_wt.append(compute_dice_wt(p, m))
            dice_tc.append(compute_dice_tc(p, m))
            dice_et.append(compute_dice_et(p, m))
            if len(dice_wt) % 10 == 0:
                hd95_wt.append(compute_hd95(p > 0,            m > 0))
                hd95_tc.append(compute_hd95(np.isin(p,[1,3]), np.isin(m,[1,3])))
                hd95_et.append(compute_hd95(p == 3,           m == 3))

        if dice_wt:
            pbar.set_postfix({
                "WT": f"{np.mean(dice_wt):.3f}",
                "TC": f"{np.mean(dice_tc):.3f}",
                "ET": f"{np.mean(dice_et):.3f}"})

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


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_best(model, optimizer, scaler, scheduler,
              epoch, phase, best_dice, val_metrics, ckpt_dir):
    save_checkpoint({
        "epoch":     epoch,
        "phase":     phase,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_dice": best_dice,       # single val Dice — standard practice
        "dice_wt":   val_metrics["dice_wt"],
        "dice_tc":   val_metrics["dice_tc"],
        "dice_et":   val_metrics["dice_et"],
    }, ckpt_dir / "best_model.pth")


def save_periodic(model, optimizer, scaler, scheduler,
                  epoch, phase, best_dice, ckpt_dir, prefix):
    save_checkpoint({
        "epoch":     epoch,
        "phase":     phase,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_dice": best_dice,
    }, ckpt_dir / f"{prefix}_epoch_{epoch+1:03d}.pth")


# ── Joint training (single-phase, no freeze) ──────────────────────────────────

def run_joint_training(model, cfg, train_loader, val_loader,
                       criterion, scaler, device, resume_path=None):
    """
    All layers trained together from epoch 1.
    Encoder gets 10x lower LR than decoder — differential LR replaces hard freeze.
    Best model saved on single val Dice (standard research practice).
    """
    total_epochs = (cfg["training"]["phase1_epochs"] +
                    cfg["training"]["phase2_epochs"])
    p1_lr        = float(cfg["training"]["phase1_lr"])
    p2_lr        = float(cfg["training"]["phase2_lr"])
    ckpt_dir     = Path(cfg["training"]["checkpoint_dir"])
    log_every    = cfg["logging"]["log_every"]
    use_amp      = cfg["training"]["amp"]
    save_every   = cfg["training"]["save_every"]
    patience     = cfg["training"].get("early_stop_patience", 15)

    model.unfreeze_all()
    param_groups = model.get_parameter_groups(p1_lr, p2_lr)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["training"].get("cosine_T0", 20),
        T_mult=cfg["training"].get("cosine_Tmult", 2),
        eta_min=1e-7)

    best_dice   = -1.0
    start_epoch = 0
    no_improve  = 0

    if resume_path and Path(resume_path).exists():
        start_epoch, _, best_dice = load_checkpoint(
            resume_path, model, optimizer, scaler, scheduler, device)

    print("\n" + "="*60)
    print("JOINT TRAINING -- all layers, differential LR")
    print(f"  Encoder LR : {p2_lr*0.1:.2e} | Decoder LR: {p2_lr:.2e}")
    print(f"  Total epochs: {total_epochs} | Early stop patience: {patience}")
    print(f"  Best model: saved on single val mean Dice (standard)")
    print("="*60)

    for epoch in range(start_epoch, total_epochs):
        label = f"[J {epoch+1}/{total_epochs}]"
        print(f"\n{label}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, log_every, use_amp, epoch_label=label)

        val_metrics = validate(
            model, val_loader, criterion, device,
            use_tta=True, epoch_label=label)

        scheduler.step()

        print(f"  Train -- " +
              " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))
        print(f"  Val   -- WT: {val_metrics['dice_wt']:.4f} | "
              f"TC: {val_metrics['dice_tc']:.4f} | "
              f"ET: {val_metrics['dice_et']:.4f} | "
              f"Mean: {val_metrics['dice_mean']:.4f}")
        if "hd95_wt" in val_metrics:
            print(f"  HD95  -- WT: {val_metrics['hd95_wt']:.2f} | "
                  f"TC: {val_metrics['hd95_tc']:.2f} | "
                  f"ET: {val_metrics['hd95_et']:.2f}")
        print(f"  LR (decoder): {optimizer.param_groups[1]['lr']:.2e}")

        # Save best on single val Dice — standard research approach
        curr_dice = val_metrics["dice_mean"]
        if curr_dice > best_dice:
            best_dice  = curr_dice
            no_improve = 0
            save_best(model, optimizer, scaler, scheduler,
                      epoch, 2, best_dice, val_metrics, ckpt_dir)
            print(f"  New best Dice: {best_dice:.4f}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{patience} epochs "
                  f"(best: {best_dice:.4f})")

        if (epoch + 1) % save_every == 0:
            save_periodic(model, optimizer, scaler, scheduler,
                          epoch, 2, best_dice, ckpt_dir, "joint")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    print(f"\nJoint training complete. Best val Dice: {best_dice:.4f}")
    return best_dice


# ── Two-phase training ────────────────────────────────────────────────────────

def run_two_phase_training(model, cfg, train_loader, val_loader,
                           criterion, scaler, device, resume_path=None):
    """
    Original two-phase training.
    Phase 1: encoder frozen, decoder warms up.
    Phase 2: all layers unfrozen, differential LR.
    Best model saved on single val Dice — standard research practice.
    """
    p1_epochs  = cfg["training"]["phase1_epochs"]
    p1_lr      = float(cfg["training"]["phase1_lr"])
    p2_epochs  = cfg["training"]["phase2_epochs"]
    p2_lr      = float(cfg["training"]["phase2_lr"])
    ckpt_dir   = Path(cfg["training"]["checkpoint_dir"])
    log_every  = cfg["logging"]["log_every"]
    use_amp    = cfg["training"]["amp"]
    save_every = cfg["training"]["save_every"]
    patience   = cfg["training"].get("early_stop_patience", 15)

    best_dice    = -1.0
    resume_phase = 1
    resume_epoch = 0

    # ── Phase 1 setup ──────────────────────────────────────────────────────
    model.freeze_encoder()
    p1_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=p1_lr, weight_decay=float(cfg["training"]["weight_decay"]))
    p1_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        p1_optimizer, T_0=10, T_mult=1, eta_min=1e-7)

    if resume_path and Path(resume_path).exists():
        resume_epoch, resume_phase, best_dice = load_checkpoint(
            resume_path, model, p1_optimizer, scaler, p1_scheduler, device)

    # ── Phase 1 ────────────────────────────────────────────────────────────
    if resume_phase == 1:
        print("\n" + "="*60)
        print("PHASE 1 -- Decoder warmup (encoder frozen)")
        print(f"  Best model: saved on single val mean Dice (standard)")
        print("="*60)
        no_improve = 0

        for epoch in range(resume_epoch, p1_epochs):
            label = f"[P1 {epoch+1}/{p1_epochs}]"
            print(f"\n{label}")

            train_metrics = train_one_epoch(
                model, train_loader, criterion, p1_optimizer,
                scaler, device, log_every, use_amp, epoch_label=label)
            val_metrics = validate(
                model, val_loader, criterion, device,
                use_tta=True, epoch_label=label)
            p1_scheduler.step()

            print(f"  Train -- " +
                  " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))
            print(f"  Val   -- WT: {val_metrics['dice_wt']:.4f} | "
                  f"TC: {val_metrics['dice_tc']:.4f} | "
                  f"ET: {val_metrics['dice_et']:.4f} | "
                  f"Mean: {val_metrics['dice_mean']:.4f}")

            curr_dice = val_metrics["dice_mean"]
            if curr_dice > best_dice:
                best_dice  = curr_dice
                no_improve = 0
                save_best(model, p1_optimizer, scaler, p1_scheduler,
                          epoch, 1, best_dice, val_metrics, ckpt_dir)
                print(f"  New best Dice: {best_dice:.4f}")
            else:
                no_improve += 1
                print(f"  No improvement for {no_improve}/{patience} epochs")

            if (epoch + 1) % save_every == 0:
                save_periodic(model, p1_optimizer, scaler, p1_scheduler,
                              epoch, 1, best_dice, ckpt_dir, "phase1")

        resume_epoch = 0

    # ── Phase 2 ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2 -- Full fine-tuning (all layers unfrozen)")
    print(f"  Best model: saved on single val mean Dice (standard)")
    print("="*60)

    model.unfreeze_all()
    param_groups = model.get_parameter_groups(p1_lr, p2_lr)
    p2_optimizer = torch.optim.AdamW(
        param_groups, weight_decay=float(cfg["training"]["weight_decay"]))
    p2_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        p2_optimizer,
        T_0=cfg["training"].get("cosine_T0", 20),
        T_mult=cfg["training"].get("cosine_Tmult", 2),
        eta_min=1e-7)

    if resume_phase == 2 and resume_path and Path(resume_path).exists():
        _, _, best_dice = load_checkpoint(
            resume_path, model, p2_optimizer, scaler, p2_scheduler, device)

    no_improve = 0

    for epoch in range(resume_epoch, p2_epochs):
        label = f"[P2 {epoch+1}/{p2_epochs}]"
        print(f"\n{label}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, p2_optimizer,
            scaler, device, log_every, use_amp, epoch_label=label)
        val_metrics = validate(
            model, val_loader, criterion, device,
            use_tta=True, epoch_label=label)
        p2_scheduler.step()

        print(f"  Train -- " +
              " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))
        print(f"  Val   -- WT: {val_metrics['dice_wt']:.4f} | "
              f"TC: {val_metrics['dice_tc']:.4f} | "
              f"ET: {val_metrics['dice_et']:.4f} | "
              f"Mean: {val_metrics['dice_mean']:.4f}")
        if "hd95_wt" in val_metrics:
            print(f"  HD95  -- WT: {val_metrics['hd95_wt']:.2f} | "
                  f"TC: {val_metrics['hd95_tc']:.2f} | "
                  f"ET: {val_metrics['hd95_et']:.2f}")
        print(f"  LR (decoder): {p2_optimizer.param_groups[1]['lr']:.2e}")

        curr_dice = val_metrics["dice_mean"]
        if curr_dice > best_dice:
            best_dice  = curr_dice
            no_improve = 0
            save_best(model, p2_optimizer, scaler, p2_scheduler,
                      epoch, 2, best_dice, val_metrics, ckpt_dir)
            print(f"  New best Dice: {best_dice:.4f}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{patience} epochs "
                  f"(best: {best_dice:.4f})")

        if (epoch + 1) % save_every == 0:
            save_periodic(model, p2_optimizer, scaler, p2_scheduler,
                          epoch, 2, best_dice, ckpt_dir, "phase2")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")
    return best_dice


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg_path, resume_path=None, joint=False):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode  : {'JOINT (single-phase, no freeze)' if joint else 'TWO-PHASE'}")

    train_loader, val_loader = get_finetune_loaders(
        slices_dir  = cfg["data"]["output_slices"],
        patch_size  = cfg["data"]["patch_size"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = 2)

    model = build_model(cfg, device)

    criterion = CombinedSegLoss(
        num_classes  = cfg["data"]["num_classes"],
        dice_weight  = float(cfg["loss"]["dice_weight"]),
        focal_weight = float(cfg["loss"]["focal_weight"]),
        focal_gamma  = float(cfg["loss"]["focal_gamma"]),
        focal_alpha  = float(cfg["loss"]["focal_alpha"]))

    scaler = GradScaler(enabled=cfg["training"]["amp"])

    if joint:
        run_joint_training(model, cfg, train_loader, val_loader,
                           criterion, scaler, device, resume_path)
    else:
        run_two_phase_training(model, cfg, train_loader, val_loader,
                               criterion, scaler, device, resume_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/finetune_config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--joint", action="store_true",
                        help="Joint single-phase training (no encoder freeze).")
    args = parser.parse_args()
    main(args.config, args.resume, args.joint)