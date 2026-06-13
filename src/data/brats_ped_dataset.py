import os
import random

import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset


class BraTSPEDDataset(Dataset):

    def __init__(
        self,
        patient_dirs,
        modalities,
        num_slices=5,
        image_size=128
    ):

        self.patient_dirs = patient_dirs

        self.modalities = modalities

        self.num_slices = num_slices

        self.image_size = image_size

        self.half = (
            num_slices // 2
        )

    def __len__(self):

        return len(
            self.patient_dirs
        )

    def load_nifti(
        self,
        path
    ):

        image = nib.load(
            path
        )

        return image.get_fdata()

    def normalize(
        self,
        image
    ):

        image = image.astype(
            np.float32
        )

        non_zero = image > 0

        if np.sum(non_zero) > 0:

            mean = image[
                non_zero
            ].mean()

            std = image[
                non_zero
            ].std()

            image[
                non_zero
            ] = (

                image[
                    non_zero
                ] - mean

            ) / (

                std + 1e-8
            )

        return image

    def center_crop(
        self,
        image
    ):

        h, w = image.shape

        crop_h = (
            self.image_size
        )

        crop_w = (
            self.image_size
        )

        top = (
            h - crop_h
        ) // 2

        left = (
            w - crop_w
        ) // 2

        return image[
            top:top + crop_h,
            left:left + crop_w
        ]

    def __getitem__(
        self,
        idx
    ):

        patient_dir = (
            self.patient_dirs[idx]
        )

        patient_id = os.path.basename(
            patient_dir
        )

        # --------------------------
        # Load modalities
        # --------------------------

        modalities_data = []

        for modality in self.modalities:

            path = os.path.join(

                patient_dir,

                f"{patient_id}-{modality}.nii.gz"
            )

            image = self.load_nifti(
                path
            )

            image = self.normalize(
                image
            )

            modalities_data.append(
                image
            )

        volume = np.stack(
            modalities_data,
            axis=0
        )

        # --------------------------
        # Load segmentation
        # --------------------------

        seg_path = os.path.join(

            patient_dir,

            f"{patient_id}-seg.nii.gz"
        )

        segmentation = (
            self.load_nifti(
                seg_path
            )
        )

        # --------------------------
        # Random center slice
        # --------------------------

        depth = volume.shape[1]

        center_slice = random.randint(

            self.half,

            depth -
            self.half -
            1
        )

        # --------------------------
        # Build 20-channel input
        # --------------------------

        stacked_slices = []

        for slice_idx in range(

            center_slice -
            self.half,

            center_slice +
            self.half +
            1
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

            cropped = np.stack(
                cropped
            )

            stacked_slices.append(
                cropped
            )

        stacked_slices = np.concatenate(

            stacked_slices,

            axis=0
        )

        # --------------------------
        # Target mask
        # --------------------------

        target_mask = segmentation[
            :,
            :,
            center_slice
        ]

        target_mask = (
            self.center_crop(
                target_mask
            )
        )

        target_mask = (
            target_mask.astype(
                np.int64
            )
        )

        return (

            torch.tensor(

                stacked_slices,

                dtype=torch.float32
            ),

            torch.tensor(

                target_mask,

                dtype=torch.long
            )
        )