import os
import shutil

SOURCE_DIR = "temp_extract"
SAVE_DIR = "datasets/brats2021"

os.makedirs(SAVE_DIR, exist_ok=True)

count = 0

for root, dirs, files in os.walk(SOURCE_DIR):

    nii_files = [
        f for f in files
        if f.endswith(".nii.gz")
    ]

    if len(nii_files) == 0:
        continue

    patient_name = os.path.basename(root)

    patient_save_dir = os.path.join(
        SAVE_DIR,
        patient_name
    )

    os.makedirs(
        patient_save_dir,
        exist_ok=True
    )

    for file_name in nii_files:

        source_path = os.path.join(
            root,
            file_name
        )

        lower_name = file_name.lower()

        if "t1ce" in lower_name:
            new_name = "t1ce.nii.gz"

        elif "_t1" in lower_name:
            new_name = "t1.nii.gz"

        elif "t2" in lower_name:
            new_name = "t2.nii.gz"

        elif "flair" in lower_name:
            new_name = "flair.nii.gz"

        elif "seg" in lower_name:
            new_name = "seg.nii.gz"

        else:
            continue

        destination_path = os.path.join(
            patient_save_dir,
            new_name
        )

        shutil.copy2(
            source_path,
            destination_path
        )

    count += 1

print(f"Done. {count} patients processed.")