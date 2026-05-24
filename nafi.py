from src.data.brats_dataset import BraTSDataset
import os

patient_dirs = [
    os.path.join(
        "datasets/brats2021",
        "BraTS2021_00000"
    )
]

dataset = BraTSDataset(
    patient_dirs,
    ["t1", "t1ce", "t2", "flair"]
)

sample = dataset[0]

print(sample.shape)
print(sample.dtype)