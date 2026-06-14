from src.data.ped_dataloader import (
    build_ped_dataloader
)

import yaml

with open(
    "configs/finetune.yaml",
    "r"
) as f:

    config = yaml.safe_load(f)

train_loader, val_loader = (
    build_ped_dataloader(
        config
    )
)

print(
    len(train_loader.dataset)
)

print(
    len(val_loader.dataset)
)