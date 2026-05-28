import os
import torch
import numpy as np
from tqdm import tqdm

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from torch.utils.data import DataLoader

from src.models.ssl_model import (
    SSLModel
)

from src.data.dataloader import (
    build_ssl_dataloader
)

from src.utils.config_loader import (
    load_config
)


def denormalize(image):

    image = image.astype(
        np.float32
    )

    image_min = image.min()

    image_max = image.max()

    return (
        image - image_min
    ) / (
        image_max
        - image_min
        + 1e-8
    )


@torch.no_grad()
def evaluate():

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    print(
        f"Using device: "
        f"{device}"
    )

    config = load_config(
        "configs/pretrain.yaml"
    )

    _, val_loader = (
        build_ssl_dataloader(
            config
        )
    )

    model = SSLModel()

    weights = torch.load(

        "pretrained_weights/"
        "ssl_encoder.pth",

        map_location=device
    )

    model.load_state_dict(
        weights
    )

    model.to(device)

    model.eval()

    psnr_scores = []
    ssim_scores = []

    print(
        "Evaluating..."
    )

    for x, y in tqdm(
        val_loader
    ):

        x = x.to(device)

        y = y.to(device)

        pred = model(x)

        pred = (
            pred.cpu()
            .numpy()
        )

        y = (
            y.cpu()
            .numpy()
        )

        batch_size = (
            pred.shape[0]
        )

        for b in range(
            batch_size
        ):

            for c in range(
                pred.shape[1]
            ):

                gt = y[b, c]

                recon = pred[b, c]

                gt = denormalize(gt)

                recon = denormalize(
                    recon
                )

                # Brain/tumor region
                mask = gt > 0

                if (
                    mask.sum()
                    == 0
                ):
                    continue

                gt_masked = (
                    gt[mask]
                )

                recon_masked = (
                    recon[mask]
                )

                psnr = (
                    peak_signal_noise_ratio(
                        gt_masked,
                        recon_masked,
                        data_range=1.0
                    )
                )

                # SSIM needs 2D
                ssim = (
                    structural_similarity(
                        gt,
                        recon,
                        data_range=1.0
                    )
                )

                psnr_scores.append(
                    psnr
                )

                ssim_scores.append(
                    ssim
                )

    avg_psnr = np.mean(
        psnr_scores
    )

    avg_ssim = np.mean(
        ssim_scores
    )

    print("\nResults")
    print("-" * 30)

    print(
        f"Average PSNR: "
        f"{avg_psnr:.4f}"
    )

    print(
        f"Average SSIM: "
        f"{avg_ssim:.4f}"
    )


if __name__ == "__main__":

    evaluate()