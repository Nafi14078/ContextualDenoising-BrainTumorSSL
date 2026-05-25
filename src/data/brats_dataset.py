import os
import nibabel as nib
import numpy as np

from torch.utils.data import Dataset


class BraTSDataset(Dataset):

    def __init__(
        self,
        patient_dirs,
        modalities
    ):

        self.modalities = modalities

        print(
            "Checking dataset integrity..."
        )

        self.patient_dirs = (
            self.filter_valid_patients(
                patient_dirs
            )
        )

        print(
            f"Valid patients: "
            f"{len(self.patient_dirs)}"
        )

    def __len__(self):

        return len(
            self.patient_dirs
        )

    def load_nifti(
        self,
        path
    ):

        image = nib.load(path)

        return image.get_fdata()

    def normalize(
        self,
        image
    ):

        image = image.astype(
            np.float32
        )

        non_zero = image > 0

        if np.sum(
            non_zero
        ) > 0:

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

    def find_valid_path(
        self,
        patient_dir,
        modality
    ):

        possible_paths = [

            os.path.join(
                patient_dir,
                f"{modality}.nii.gz"
            ),

            os.path.join(
                patient_dir,
                f"{modality}.nii"
            )
        ]

        for p in possible_paths:

            if (

                os.path.exists(
                    p
                )

                and

                os.path.getsize(
                    p
                ) > 0
            ):

                return p

        return None

    def filter_valid_patients(
        self,
        patient_dirs
    ):

        valid_patients = []

        for patient_dir in patient_dirs:

            patient_ok = True

            for modality in (
                self.modalities
            ):

                path = (
                    self.find_valid_path(
                        patient_dir,
                        modality
                    )
                )

                if path is None:

                    patient_ok = False
                    break

            if patient_ok:

                valid_patients.append(
                    patient_dir
                )

        return valid_patients

    def __getitem__(
        self,
        idx
    ):

        patient_dir = (
            self.patient_dirs[idx]
        )

        modalities_data = []

        for modality in (
            self.modalities
        ):

            path = (
                self.find_valid_path(
                    patient_dir,
                    modality
                )
            )

            if path is None:

                raise FileNotFoundError(

                    f"{modality} "
                    f"not found in "
                    f"{patient_dir}"
                )

            image = (
                self.load_nifti(
                    path
                )
            )

            image = (
                self.normalize(
                    image
                )
            )

            modalities_data.append(
                image
            )

        modalities_data = (
            np.stack(
                modalities_data,
                axis=0
            )
        )

        return (
            modalities_data.astype(
                np.float32
            )
        )