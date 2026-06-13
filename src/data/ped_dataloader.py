import os

from sklearn.model_selection import (
    train_test_split
)

from torch.utils.data import (
    DataLoader
)

from src.data.brats_ped_dataset import (
    BraTSPEDDataset
)


def get_patient_dirs(path):

    patient_dirs = []

    for patient in os.listdir(path):

        patient_path = os.path.join(
            path,
            patient
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


def build_ped_dataloader(config):

    patient_dirs = get_patient_dirs(

        config[
            "dataset_path"
        ]
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

    train_dataset = (

        BraTSPEDDataset(

            patient_dirs=train_dirs,

            modalities=config[
                "modalities"
            ],

            num_slices=config[
                "num_input_slices"
            ],

            image_size=config[
                "image_size"
            ]
        )
    )

    val_dataset = (

        BraTSPEDDataset(

            patient_dirs=val_dirs,

            modalities=config[
                "modalities"
            ],

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