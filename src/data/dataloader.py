import os

from sklearn.model_selection import (
    train_test_split
)

from torch.utils.data import (
    DataLoader
)

from src.data.brats_dataset import (
    BraTSDataset
)

from src.data.ssl_dataset import (
    SSLDataset
)


def get_patient_dirs(path):

    patient_dirs = []

    for patient in os.listdir(path):

        patient_path = (
            os.path.join(
                path,
                patient
            )
        )

        if os.path.isdir(
            patient_path
        ):

            patient_dirs.append(
                patient_path
            )

    return sorted(
        patient_dirs
    )


def build_ssl_dataloader(config):

    patient_dirs = (
        get_patient_dirs(
            config[
                "dataset_path"
            ]
        )
    )

    train_dirs, val_dirs = (
        train_test_split(

            patient_dirs,

            train_size=config[
                "train_split"
            ],

            random_state=config[
                "seed"
            ]
        )
    )

    train_base = (
        BraTSDataset(

            train_dirs,

            config[
                "modalities"
            ]
        )
    )

    val_base = (
        BraTSDataset(

            val_dirs,

            config[
                "modalities"
            ]
        )
    )

    train_dataset = (
        SSLDataset(

            train_base,

            num_slices=config[
                "num_input_slices"
            ],

            image_size=config[
                "image_size"
            ]
        )
    )

    val_dataset = (
        SSLDataset(

            val_base,

            num_slices=config[
                "num_input_slices"
            ],

            image_size=config[
                "image_size"
            ]
        )
    )

    train_loader = (
        DataLoader(

            train_dataset,

            batch_size=config[
                "batch_size"
            ],

            shuffle=True,

            num_workers=config[
                "num_workers"
            ],

            pin_memory=True,

            persistent_workers=(
                config[
                    "num_workers"
                ] > 0
            ),

            prefetch_factor=2,

            drop_last=True
        )
    )

    val_loader = (
        DataLoader(

            val_dataset,

            batch_size=config[
                "batch_size"
            ],

            shuffle=False,

            num_workers=config[
                "num_workers"
            ],

            pin_memory=True,

            persistent_workers=(
                config[
                    "num_workers"
                ] > 0
            ),

            prefetch_factor=2
        )
    )

    return (
        train_loader,
        val_loader
    )