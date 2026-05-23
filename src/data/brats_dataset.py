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

        self.patient_dirs = patient_dirs
        self.modalities = modalities

    def __len__(self):

        return len(self.patient_dirs)

    def load_nifti(self, path):

        image = nib.load(path)

        return image.get_fdata()

    def normalize(self, image):

        image = image.astype(np.float32)

        non_zero = image > 0

        if np.sum(non_zero) > 0:

            mean = image[non_zero].mean()

            std = image[non_zero].std()

            image[non_zero] = (
                image[non_zero] - mean
            ) / (std + 1e-8)

        return image

    def __getitem__(self, idx):

        patient_dir = self.patient_dirs[idx]

        modalities_data = []

        for modality in self.modalities:

            path = os.path.join(
                patient_dir,
                f"{modality}.nii.gz"
            )

            image = self.load_nifti(path)

            image = self.normalize(image)

            modalities_data.append(image)

        modalities_data = np.stack(
            modalities_data,
            axis=0
        )

        return modalities_data.astype(
            np.float32
        )