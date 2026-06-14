# Thesis Project — STS-UVD Pretraining + BraTS-PED Segmentation

## Overview
Two-stage pipeline:
1. **Pretraining** — Self-supervised denoising on BraTS2021 (STS-UVD adapted for 2D MRI slices)
2. **Fine-tuning** — Transfer pretrained encoder to Swin UNETR for BraTS-PED tumor segmentation

**Goals:**
- Pretraining  : PSNR > 38 dB, SSIM > 0.95 on BraTS2021 slices
- Fine-tuning  : SOTA Dice WT/TC/ET on BraTS-PED pediatric tumor segmentation

---

## Project Structure

```
thesis_project/
├── configs/
│   ├── pretrain_config.yaml      ← all pretraining hyperparameters
│   └── finetune_config.yaml      ← all fine-tuning hyperparameters
├── scripts/
│   ├── preprocess_brats2021.py   ← 3D NIfTI → 2D .npy slices (pretraining)
│   └── preprocess_bratsped.py    ← 3D NIfTI → 2D .npy slices + masks (finetune)
├── pretraining/
│   ├── dataset.py                ← slice window dataset loader
│   ├── sts_kernel.py             ← Weighted Temporal (T) + Spatial (S) kernels
│   ├── unet_denoiser.py          ← U-Net denoiser backbone (G_phi + D_theta)
│   ├── losses.py                 ← L1 + slice consistency loss
│   └── train_pretrain.py         ← recurrent training loop
├── finetuning/
│   ├── dataset.py                ← BraTS-PED slice dataset with augmentation
│   ├── swin_unetr_2d.py          ← 2D Swin UNETR + weight injection
│   ├── losses.py                 ← Dice + Focal loss
│   └── train_finetune.py         ← two-phase training loop
└── evaluation/
    ├── metrics.py                ← PSNR, SSIM, Dice WT/TC/ET, HD95
    └── visualize.py              ← denoising plots + segmentation overlays
```

---

## Setup

### 1. Install dependencies (run once on Kaggle)
```bash
pip install monai nibabel scikit-image scipy pyyaml tqdm matplotlib
```

### 2. Preprocess BraTS2021 (run once, save output as Kaggle Dataset)
```bash
python scripts/preprocess_brats2021.py \
  --root /kaggle/input/brats2021/BraTS2021_Training_Data \
  --out  /kaggle/working/brats2021_slices \
  --skip 10 \
  --min_brain 0.05
```
Then upload `/kaggle/working/brats2021_slices/` as a new Kaggle Dataset called `brats2021-preprocessed`.

### 3. Preprocess BraTS-PED (run once, save output as Kaggle Dataset)
```bash
python scripts/preprocess_bratsped.py \
  --root /kaggle/input/bratsped/BraTS-PED_Training_Data \
  --out  /kaggle/working/bratsped_slices \
  --skip 10 \
  --min_tumor 0.01
```
Then upload as `bratsped-preprocessed`.

---

## Stage 1 — Pretraining

### Update config
Edit `configs/pretrain_config.yaml`:
```yaml
data:
  brats2021_root:  "/kaggle/input/brats2021/BraTS2021_Training_Data"
  output_slices:   "/kaggle/input/brats2021-preprocessed/slices"
training:
  checkpoint_dir:  "/kaggle/working/checkpoints/pretrain"
```

### Run
```bash
cd thesis_project
python pretraining/train_pretrain.py --config configs/pretrain_config.yaml
```

### Output
- Best model: `/kaggle/working/checkpoints/pretrain/best_model.pth`
- Encoder weights for transfer: `/kaggle/working/checkpoints/pretrain/pretrain_encoder.pth`

Upload `pretrain_encoder.pth` as a Kaggle Dataset called `pretrained-weights`.

---

## Stage 2 — Fine-tuning

### Update config
Edit `configs/finetune_config.yaml`:
```yaml
data:
  bratsped_root:      "/kaggle/input/bratsped/BraTS-PED_Training_Data"
  output_slices:      "/kaggle/input/bratsped-preprocessed/slices"
model:
  pretrained_encoder: "/kaggle/input/pretrained-weights/pretrain_encoder.pth"
training:
  checkpoint_dir:     "/kaggle/working/checkpoints/finetune"
```

### Run
```bash
python finetuning/train_finetune.py --config configs/finetune_config.yaml
```

---

## Training Strategy (Kaggle Free Tier)

| Stage | Sessions needed | Tip |
|---|---|---|
| Preprocessing | 1 session each | Upload outputs as Kaggle Dataset immediately |
| Pretraining | 3–5 sessions | Save checkpoint every 5 epochs, resume from last |
| Fine-tuning | 4–6 sessions | Phase 1 and Phase 2 can be separate sessions |

**Always save to `/kaggle/working/` and upload as dataset — runtime resets lose `/tmp/`.**

---

## Expected Results

| Metric | Target | SOTA (for reference) |
|---|---|---|
| PSNR (pretraining) | > 38 dB | ~41 dB (STS-UVD paper) |
| SSIM (pretraining) | > 0.95 | ~0.984 (STS-UVD paper) |
| Dice WT | > 0.88 | ~0.90 (BraTS-PED leaderboard) |
| Dice TC | > 0.82 | ~0.85 |
| Dice ET | > 0.78 | ~0.82 |
| HD95 WT | < 10 mm | ~7 mm |

---

## Thesis Ablation Experiments

Run these 4 experiments to support your thesis claim:

| Experiment | Command change |
|---|---|
| **Ours (full)** | pretrained_encoder = pretrain_encoder.pth |
| **Scratch baseline** | Set pretrained_encoder to null in config |
| **No temporal kernel** | Set sts.N = 1 in pretrain_config.yaml |
| **No spatial kernel** | Set sts.replace_ratio = 0.0 |

---

## Citation

If you use this work in your thesis, cite the STS-UVD paper:
```
Aiyetigbo et al., "Generalizable Unsupervised Microscopy Video Denoising
via Weighted SpatioTemporal Sampling", CVPR Workshop 2024.
```
