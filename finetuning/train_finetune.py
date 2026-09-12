"""
finetuning/train_finetune.py
────────────────────────────────────────────────────────────────────────────────
Fine-tuning loop for BraTS-PED 2D segmentation.

SINGLE-PHASE FULL FINE-TUNING (no encoder freezing, no phase split):
  Every parameter — pretrained encoder included — is trainable from epoch 1
  with a single learning rate and a single optimizer, the standard way most
  fine-tuning is done. (Earlier versions of this script had a two-phase
  freeze-then-unfreeze scheme and a separate encoder/decoder LR split —
  both have been removed here at the user's request in favor of plain,
  uniform full fine-tuning.)

Best model saved on single val Dice (standard research practice).
TTA used in validation for better accuracy.
Early stopping when val Dice does not improve for `patience` epochs.

Learning rate / epoch count come from the config, with fallbacks so this
still runs against an existing two-phase-style config without edits:
  epochs : cfg["training"]["epochs"]
           -> falls back to phase1_epochs + phase2_epochs if "epochs" absent
  lr     : cfg["training"]["lr"]
           -> falls back to phase2_lr if "lr" absent
For a clean config going forward, just set:
  training:
    epochs: 100
    lr: 0.00005
and you can delete phase1_epochs / phase2_epochs / phase1_lr / phase2_lr.

────────────────────────────────────────────────────────────────────────────────
MULTI-GPU (both Kaggle T4s)
────────────────────────────────────────────────────────────────────────────────
This script automatically uses every visible GPU via nn.DataParallel — no
flags needed, it just detects torch.cuda.device_count() > 1.

Why plain nn.DataParallel makes GPU-0 memory blow up (and the fix used here):
  By default, DataParallel scatters the input batch across GPUs, runs the
  model forward on each shard, then GATHERS every shard's full output
  tensor back onto GPU 0 before you can compute the loss there. For
  segmentation that means the full (B, C, H, W) logits tensor gets
  duplicated on GPU 0 alongside its own shard, gradients, and optimizer
  state — so GPU 0 can end up needing meaningfully more memory than GPU 1
  and be the one that OOMs first, even though "both GPUs are only half
  full" by naive accounting.

  Fix: `ModelWithLoss` below wraps the model AND the loss function
  together, so the loss is computed independently on each GPU, on that
  GPU's own shard of logits. Only three small scalar tensors (total, dice,
  focal loss) get gathered back to GPU 0 instead of the full logits tensor.
  Gradient reduction during backward still happens automatically through
  autograd exactly as it always does with DataParallel — this only changes
  what gets *gathered*, not how gradients flow.

  Validation does not train, so this concern doesn't apply there — a
  separate `infer_model` (DataParallel-wrapped raw model, no loss) is used
  during validation/TTA so both GPUs help speed up eval too.

Effective batch size: DataParallel splits `training.batch_size` across your
GPUs (e.g. batch_size=8 with 2 GPUs -> 4 samples per GPU per step). If you
want each GPU to still see ~8 samples per step, raise `training.batch_size`
in the config to ~16 now that you have 2 GPUs.

Run:
    python train_finetune.py --config ../configs/finetune_config.yaml
    python train_finetune.py --config ../configs/finetune_config.yaml --resume /path/ckpt.pth
────────────────────────────────────────────────────────────────────────────────
"""

import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finetuning.dataset       import get_finetune_loaders
from finetuning.swin_unetr_2d import build_model
from finetuning.losses        import CombinedSegLoss
from evaluation.metrics       import (compute_dice_wt, compute_dice_tc,
                                      compute_dice_et, compute_hd95)


# ── Multi-GPU wrapper ─────────────────────────────────────────────────────────

class ModelWithLoss(nn.Module):
    """
    Wraps model + loss so that when replicated across GPUs by
    nn.DataParallel, each GPU computes ITS OWN loss locally from its own
    shard of logits. Only small scalar tensors get gathered back to the
    primary GPU, instead of the full (B, C, H, W) logits tensor. This keeps
    GPU-0 memory usage in line with the other GPUs instead of it silently
    ballooning. See the module docstring above for the full explanation.
    """
    def __init__(self, model: nn.Module, criterion: nn.Module):
        super().__init__()
        self.model     = model
        self.criterion = criterion

    def forward(self, images, masks):
        logits    = self.model(images)
        loss_dict = self.criterion(logits, masks)
        # unsqueeze(0): DataParallel concatenates per-GPU outputs along dim 0,
        # so each GPU must return at least a 1-D tensor, not a bare scalar.
        total = loss_dict["total"].unsqueeze(0)
        dice  = loss_dict["dice"].unsqueeze(0)
        focal = loss_dict["focal"].unsqueeze(0)
        return total, dice, focal


def gpu_mem_string() -> str:
    """Short 'g0=1.2G|g1=1.1G' string for the tqdm postfix."""
    if not torch.cuda.is_available():
        return "cpu"
    parts = []
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        parts.append(f"g{i}={used:.1f}G")
    return "|".join(parts)


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
    `model` must be the raw (unwrapped) model — never the DataParallel wrapper
    — so state_dict keys stay plain and portable across GPU-count changes.
    Returns: (start_epoch, best_dice)
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
    best_dice   = ckpt.get("best_dice", -1.0)

    print(f"  Resumed at epoch {start_epoch}")
    print(f"  Best Dice so far: {best_dice:.4f}\n")
    return start_epoch, best_dice


# ── Test-Time Augmentation ────────────────────────────────────────────────────

@torch.no_grad()
def predict_tta(infer_model, images, device):
    """
    Average predictions across 4 rotations + horizontal flip.
    `infer_model` is the (possibly DataParallel-wrapped) raw model, used
    for inference only — no loss involved, so both GPUs help here too.
    """
    infer_model.eval()
    preds = []
    for k in range(4):
        aug   = torch.rot90(images, k=k, dims=[2, 3])
        logit = infer_model(aug.to(device))
        logit = torch.rot90(logit, k=-k, dims=[2, 3])
        preds.append(F.softmax(logit, dim=1))
    aug   = torch.flip(images, dims=[3])
    logit = infer_model(aug.to(device))
    logit = torch.flip(logit, dims=[3])
    preds.append(F.softmax(logit, dim=1))
    return torch.stack(preds).mean(0)


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(train_model, raw_model, loader, optimizer,
                    scaler, device, log_every, use_amp, epoch_label=""):
    """
    `train_model` is ModelWithLoss, optionally DataParallel-wrapped — used
    for the forward+backward pass so both GPUs share the training work.
    `raw_model` is the underlying SegUNet2D — used for gradient clipping,
    since clip_grad_norm_ needs the actual parameter tensors, not the
    wrapper's.
    """
    train_model.train()
    running      = defaultdict(float)
    epoch_totals = defaultdict(float)
    batch_size    = loader.batch_size
    total_samples = len(loader.dataset)
    pbar = tqdm(loader, desc=f"Train {epoch_label}", unit="batch",
                dynamic_ncols=True, leave=True)

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            total, dice, focal = train_model(images, masks)
            loss = total.mean()   # mean over GPUs (each already meaned over its shard)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_dict = {
            "total": loss.item(),
            "dice":  dice.mean().item(),
            "focal": focal.mean().item(),
        }
        for k, v in loss_dict.items():
            running[k]      += v
            epoch_totals[k] += v

        samples_done = min((step + 1) * batch_size, total_samples)
        pct          = samples_done / total_samples * 100
        pbar.set_postfix({
            "samples": f"{samples_done}/{total_samples}",
            "%": f"{pct:.1f}",
            "loss": f"{loss.item():.4f}",
            "gpu": gpu_mem_string(),
        })

        if (step + 1) % log_every == 0:
            avg = {k: v / log_every for k, v in running.items()}
            tqdm.write(
                f"  {epoch_label} step {step+1}/{len(loader)} ({pct:.1f}%) | " +
                " | ".join(f"{k}: {v:.4f}" for k, v in avg.items()) +
                f" | gpu: {gpu_mem_string()}")
            running = defaultdict(float)

    return {k: v / len(loader) for k, v in epoch_totals.items()}


# ── Validation with TTA ───────────────────────────────────────────────────────

@torch.no_grad()
def validate(infer_model, loader, criterion, device,
             use_tta=True, epoch_label=""):
    """
    `infer_model` is the raw model (optionally DataParallel-wrapped, no
    loss attached) — used both for the plain forward pass loss and for TTA.
    """
    infer_model.eval()
    dice_wt, dice_tc, dice_et = [], [], []
    hd95_wt, hd95_tc, hd95_et = [], [], []
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Val   {epoch_label}", unit="batch",
                dynamic_ncols=True, leave=True)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)

        logits    = infer_model(images)
        loss_dict = criterion(logits, masks)
        total_loss += loss_dict["total"].item()

        if use_tta:
            probs = predict_tta(infer_model, images, device)
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
                "ET": f"{np.mean(dice_et):.3f}",
                "gpu": gpu_mem_string(),
            })

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

def save_best(raw_model, optimizer, scaler, scheduler,
              epoch, best_dice, val_metrics, ckpt_dir):
    save_checkpoint({
        "epoch":     epoch,
        "model":     raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_dice": best_dice,       # single val Dice — standard practice
        "dice_wt":   val_metrics["dice_wt"],
        "dice_tc":   val_metrics["dice_tc"],
        "dice_et":   val_metrics["dice_et"],
    }, ckpt_dir / "best_model.pth")


def save_periodic(raw_model, optimizer, scaler, scheduler,
                  epoch, best_dice, ckpt_dir):
    save_checkpoint({
        "epoch":     epoch,
        "model":     raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_dice": best_dice,
    }, ckpt_dir / f"epoch_{epoch+1:03d}.pth")


# ── Training ───────────────────────────────────────────────────────────────────

def run_training(train_model, infer_model, raw_model, cfg,
                 train_loader, val_loader, criterion, scaler,
                 device, resume_path=None):
    """
    Single-phase full fine-tuning: every parameter trains from epoch 1 with
    one optimizer and one learning rate. No freezing, no phase split.
    Best model saved on single val Dice (standard research practice).
    """
    total_epochs = cfg["training"].get(
        "epochs",
        cfg["training"].get("phase1_epochs", 0) + cfg["training"].get("phase2_epochs", 100))
    lr = float(cfg["training"].get("lr", cfg["training"].get("phase2_lr", 1e-4)))

    ckpt_dir   = Path(cfg["training"]["checkpoint_dir"])
    log_every  = cfg["logging"]["log_every"]
    use_amp    = cfg["training"]["amp"]
    save_every = cfg["training"]["save_every"]
    patience   = cfg["training"].get("early_stop_patience", 15)

    # Every parameter is trainable — explicit call for clarity, though a
    # freshly built model already has requires_grad=True everywhere.
    raw_model.unfreeze_all()

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=lr, weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["training"].get("cosine_T0", 20),
        T_mult=cfg["training"].get("cosine_Tmult", 2),
        eta_min=1e-7)

    best_dice   = -1.0
    start_epoch = 0
    no_improve  = 0

    if resume_path and Path(resume_path).exists():
        start_epoch, best_dice = load_checkpoint(
            resume_path, raw_model, optimizer, scaler, scheduler, device)

    print("\n" + "="*60)
    print("FULL FINE-TUNING -- all layers trainable from epoch 1")
    print(f"  LR: {lr:.2e} | Total epochs: {total_epochs} | "
          f"Early stop patience: {patience}")
    print(f"  Best model: saved on single val mean Dice (standard)")
    print("="*60)

    for epoch in range(start_epoch, total_epochs):
        label = f"[{epoch+1}/{total_epochs}]"
        print(f"\n{label}")

        train_metrics = train_one_epoch(
            train_model, raw_model, train_loader, optimizer,
            scaler, device, log_every, use_amp, epoch_label=label)

        val_metrics = validate(
            infer_model, val_loader, criterion, device,
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
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best on single val Dice — standard research approach
        curr_dice = val_metrics["dice_mean"]
        if curr_dice > best_dice:
            best_dice  = curr_dice
            no_improve = 0
            save_best(raw_model, optimizer, scaler, scheduler,
                      epoch, best_dice, val_metrics, ckpt_dir)
            print(f"  New best Dice: {best_dice:.4f}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{patience} epochs "
                  f"(best: {best_dice:.4f})")

        if (epoch + 1) % save_every == 0:
            save_periodic(raw_model, optimizer, scaler, scheduler,
                          epoch, best_dice, ckpt_dir)

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")
    return best_dice


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg_path, resume_path=None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device} | GPUs visible: {n_gpus}")
    print("Mode  : FULL FINE-TUNING (single phase, no encoder freezing)")

    train_loader, val_loader = get_finetune_loaders(
        slices_dir  = cfg["data"]["output_slices"],
        patch_size  = cfg["data"]["patch_size"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["training"].get("num_workers", 4),
        oversample_et = cfg["data"].get("oversample_et", False),
        et_oversample_factor = float(cfg["data"].get("et_oversample_factor", 5.0)))

    if n_gpus > 1:
        per_gpu = cfg["training"]["batch_size"] // n_gpus
        print(f"[Multi-GPU] batch_size={cfg['training']['batch_size']} "
              f"split across {n_gpus} GPUs (~{per_gpu}/GPU). "
              f"Raise training.batch_size in the config if GPUs look under-used.")

    # raw_model: the actual SegUNet2D. All custom methods (unfreeze_all,
    # load_pretrained_encoder, state_dict/load_state_dict for checkpoints)
    # are ALWAYS called on this object directly, never on a DataParallel
    # wrapper.
    raw_model = build_model(cfg, device)

    criterion = CombinedSegLoss(
        num_classes  = cfg["data"]["num_classes"],
        dice_weight  = float(cfg["loss"]["dice_weight"]),
        focal_weight = float(cfg["loss"]["focal_weight"]),
        focal_gamma  = float(cfg["loss"]["focal_gamma"]),
        focal_alpha  = float(cfg["loss"]["focal_alpha"]),
        dice_class_weights  = cfg["loss"].get("dice_class_weights", None),
        focal_class_weights = cfg["loss"].get("focal_class_weights", None))

    # train_model: used ONLY for the forward+backward training pass.
    # Wraps model+loss together so DataParallel gathers small scalars
    # instead of full logits (keeps GPU-0 memory from ballooning).
    train_model = ModelWithLoss(raw_model, criterion).to(device)

    # infer_model: used ONLY for validation/TTA forward passes (no
    # backward, so gathering logits back to GPU 0 here is cheap/transient).
    infer_model = raw_model

    if n_gpus > 1:
        device_ids  = list(range(n_gpus))
        train_model = nn.DataParallel(train_model, device_ids=device_ids)
        infer_model = nn.DataParallel(raw_model,   device_ids=device_ids)
        print(f"[Multi-GPU] Wrapped for training + inference on GPUs {device_ids}")

    scaler = GradScaler(enabled=cfg["training"]["amp"])

    run_training(train_model, infer_model, raw_model, cfg,
                train_loader, val_loader, criterion, scaler,
                device, resume_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/finetune_config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from.")
    args = parser.parse_args()
    main(args.config, args.resume)