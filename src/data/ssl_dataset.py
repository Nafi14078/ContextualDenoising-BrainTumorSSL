import random
import torch
import numpy as np
from torch.utils.data import Dataset


class SSLDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        num_slices=5,
        image_size=128
    ):

        self.base_dataset = base_dataset

        self.num_slices = num_slices

        self.half = num_slices // 2

        self.image_size = image_size

    def __len__(self):

        return len(self.base_dataset)

    def center_crop(self, image):

        h, w = image.shape

        crop_h = self.image_size
        crop_w = self.image_size

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        return image[
            top:top+crop_h,
            left:left+crop_w
        ]

    def __getitem__(self, idx):

        volume = self.base_dataset[idx]

        depth = volume.shape[1]

        center_slice = random.randint(
            self.half,
            depth - self.half - 1
        )

        stacked_slices = []

        for slice_idx in range(
            center_slice - self.half,
            center_slice + self.half + 1
        ):

            slice_data = volume[
                :,
                slice_idx
            ]

            cropped = []

            for modality_slice in slice_data:

                cropped.append(
                    self.center_crop(
                        modality_slice
                    )
                )

            cropped = np.stack(cropped)

            stacked_slices.append(
                cropped
            )

        stacked_slices = np.concatenate(
            stacked_slices,
            axis=0
        )

        target = volume[
            :,
            center_slice
        ]

        cropped_target = []

        for m in target:

            cropped_target.append(
                self.center_crop(m)
            )

        target = np.stack(
            cropped_target
        )

        return (
            torch.tensor(
                stacked_slices,
                dtype=torch.float32
            ),
            torch.tensor(
                target,
                dtype=torch.float32
            )
        )