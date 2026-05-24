import os
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.seed import set_seed
from src.utils.config_loader import load_config

from src.data.dataloader import (
    build_ssl_dataloader
)

from src.models.ssl_model import (
    SSLModel
)

from src.training.ssl_trainer import (
    SSLTrainer
)


def main():

    # --------------------------
    # Load config
    # --------------------------

    config = load_config(
        "configs/pretrain.yaml"
    )

    # --------------------------
    # Set seed
    # --------------------------

    set_seed(
        config["seed"]
    )

    # --------------------------
    # Device
    # --------------------------

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # --------------------------
    # Create folders
    # --------------------------

    os.makedirs(
        "outputs/checkpoints",
        exist_ok=True
    )

    os.makedirs(
        "outputs/logs",
        exist_ok=True
    )

    os.makedirs(
        "outputs/visualizations",
        exist_ok=True
    )

    os.makedirs(
        "pretrained_weights",
        exist_ok=True
    )

    # --------------------------
    # Dataloader
    # --------------------------

    print(
        "Building dataloaders..."
    )

    train_loader, val_loader = (
        build_ssl_dataloader(
            config
        )
    )

    print(
        f"Train batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    # --------------------------
    # Model
    # --------------------------

    model = SSLModel()

    model = model.to(device)

    print(
        "Model loaded."
    )

    # --------------------------
    # Optimizer
    # --------------------------

    optimizer = AdamW(

        model.parameters(),

        lr=config[
            "learning_rate"
        ],

        weight_decay=config[
            "weight_decay"
        ]
    )

    # --------------------------
    # Scheduler
    # --------------------------

    scheduler = (
        CosineAnnealingLR(

            optimizer,

            T_max=config[
                "epochs"
            ]
        )
    )

    # --------------------------
    # Trainer
    # --------------------------

    trainer = SSLTrainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        optimizer=optimizer,

        scheduler=scheduler,

        config=config,

        device=device
    )

    # --------------------------
    # Train
    # --------------------------

    print(
        "Starting SSL training..."
    )

    trainer.train()

    print(
        "Training complete."
    )


if __name__ == "__main__":

    main()