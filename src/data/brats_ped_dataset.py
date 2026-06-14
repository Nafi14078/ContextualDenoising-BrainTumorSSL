import os

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

        # --------------------------------
        # Build slice-level index
        # --------------------------------

        self.samples = []

        print(
            "Building slice index..."
        )

        for patient_dir in self.patient_dirs:

            patient_id = os.path.basename(
                patient_dir
            )

            example_modality = (
                self.get_file_path(
                    patient_dir,
                    f"{patient_id}-{self.modalities[0]}"
                )
            )

            volume = self.load_nifti(
                example_modality
            )

            depth = volume.shape[2]

            for center_slice in range(

                self.half,

                depth - self.half
            ):

                self.samples.append(

                    (
                        patient_dir,
                        center_slice
                    )
                )

        print(
            f"Total samples: "
            f"{len(self.samples)}"
        )

    def __len__(self):

        return len(
            self.samples
        )

    # --------------------------------
    # Supports BOTH .nii and .nii.gz
    # --------------------------------

    def get_file_path(
        self,
        patient_dir,
        filename
    ):

        nii_path = os.path.join(
            patient_dir,
            filename + ".nii"
        )

        nii_gz_path = os.path.join(
            patient_dir,
            filename + ".nii.gz"
        )

        if os.path.exists(
            nii_path
        ):
            return nii_path

        if os.path.exists(
            nii_gz_path
        ):
            return nii_gz_path

        raise FileNotFoundError(
            f"Cannot find file: "
            f"{filename}"
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

        patient_dir, center_slice = (

            self.samples[idx]
        )

        patient_id = os.path.basename(
            patient_dir
        )

        # --------------------------------
        # Load modalities
        # --------------------------------

        modalities_data = []

        for modality in self.modalities:

            path = self.get_file_path(

                patient_dir,

                f"{patient_id}-{modality}"
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

        # volume shape:
        # [4, H, W, D]

        # --------------------------------
        # Load segmentation
        # --------------------------------

        seg_path = self.get_file_path(

            patient_dir,

            f"{patient_id}-seg"
        )

        segmentation = (
            self.load_nifti(
                seg_path
            )
        )

        # --------------------------------
        # Build 20-channel input
        # --------------------------------

        stacked_slices = []

        for slice_idx in range(

            center_slice - self.half,

            center_slice + self.half + 1
        ):

            slice_data = volume[
                :,
                :,
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

        # --------------------------------
        # Target mask
        # --------------------------------

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