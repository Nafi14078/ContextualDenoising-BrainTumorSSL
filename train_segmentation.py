import os
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.seed import (
    set_seed
)

from src.utils.config_loader import (
    load_config
)

from src.data.ped_dataloader import (
    build_ped_dataloader
)

from src.models.segmentation_model import (
    SegmentationModel,
    load_pretrained_encoder
)

from src.training.segmentation_trainer import (
    SegmentationTrainer
)


def main():

    # ==========================
    # Load Config
    # ==========================

    config = load_config(
        "configs/finetune.yaml"
    )

    # ==========================
    # Seed
    # ==========================

    set_seed(
        config["seed"]
    )

    # ==========================
    # Device
    # ==========================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ==========================
    # Create Folders
    # ==========================

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

    # ==========================
    # Dataloader
    # ==========================

    print(
        "Building PED dataloaders..."
    )

    train_loader, val_loader = (

        build_ped_dataloader(
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

    # ==========================
    # Model
    # ==========================

    model = (
        SegmentationModel()
    )

    # ==========================
    # Load SSL Encoder
    # ==========================

    model = (
        load_pretrained_encoder(

            model,

            config[
                "pretrained_weights"
            ]
        )
    )

    model = model.to(
        device
    )

    print(
        "Segmentation model ready."
    )

    # ==========================
    # Optimizer
    # ==========================

    optimizer = AdamW(

        model.parameters(),

        lr=config[
            "learning_rate"
        ],

        weight_decay=config[
            "weight_decay"
        ]
    )

    # ==========================
    # Scheduler
    # ==========================

    scheduler = (
        CosineAnnealingLR(

            optimizer,

            T_max=config[
                "epochs"
            ]
        )
    )

    # ==========================
    # Trainer
    # ==========================

    trainer = (
        SegmentationTrainer(

            model=model,

            train_loader=
            train_loader,

            val_loader=
            val_loader,

            optimizer=
            optimizer,

            scheduler=
            scheduler,

            config=config,

            device=device
        )
    )

    # ==========================
    # Start Training
    # ==========================

    print(
        "Starting "
        "BraTS-PED fine-tuning..."
    )

    trainer.train()

    print(
        "Fine-tuning complete."
    )


if __name__ == "__main__":

    main()