import yaml
import torch
import numpy as np

from tqdm import tqdm

from src.data.ped_dataloader import (
    build_ped_dataloader
)

from src.models.segmentation_model import (
    SegmentationModel
)


# ==================================================
# BraTS Regions
# ==================================================

def get_regions(mask):

    wt = (
        mask > 0
    )

    tc = np.logical_or(
        mask == 1,
        mask == 3
    )

    et = (
        mask == 3
    )

    return (
        wt,
        tc,
        et
    )


# ==================================================
# Global Statistics
# ==================================================

def compute_stats(
    pred,
    target
):

    tp = np.logical_and(
        pred,
        target
    ).sum()

    fp = np.logical_and(
        pred,
        np.logical_not(target)
    ).sum()

    fn = np.logical_and(
        np.logical_not(pred),
        target
    ).sum()

    return (
        tp,
        fp,
        fn
    )


# ==================================================
# Evaluation
# ==================================================

def evaluate():

    with open(
        "configs/finetune.yaml",
        "r"
    ) as f:

        config = yaml.safe_load(
            f
        )

    device = torch.device(

        config["device"]

        if torch.cuda.is_available()

        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    print(
        "Building validation dataloader..."
    )

    _, val_loader = (
        build_ped_dataloader(
            config
        )
    )

    model = (
        SegmentationModel()
        .to(device)
    )

    print(
        "Loading best model..."
    )

    checkpoint = torch.load(

        config[
            "best_model_path"
        ],

        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    metrics = {

        "WT": {
            "tp": 0,
            "fp": 0,
            "fn": 0
        },

        "TC": {
            "tp": 0,
            "fp": 0,
            "fn": 0
        },

        "ET": {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    }

    print(
        "Evaluating..."
    )

    with torch.no_grad():

        for images, masks in tqdm(
            val_loader
        ):

            images = images.to(
                device
            )

            logits = model(
                images
            )

            preds = torch.argmax(
                logits,
                dim=1
            )

            preds = (
                preds.cpu()
                .numpy()
            )

            masks = (
                masks.cpu()
                .numpy()
            )

            for pred, target in zip(
                preds,
                masks
            ):

                pred_wt, pred_tc, pred_et = (
                    get_regions(
                        pred
                    )
                )

                gt_wt, gt_tc, gt_et = (
                    get_regions(
                        target
                    )
                )

                regions = {

                    "WT": (
                        pred_wt,
                        gt_wt
                    ),

                    "TC": (
                        pred_tc,
                        gt_tc
                    ),

                    "ET": (
                        pred_et,
                        gt_et
                    )
                }

                for name in regions:

                    p, g = (
                        regions[
                            name
                        ]
                    )

                    tp, fp, fn = (
                        compute_stats(
                            p,
                            g
                        )
                    )

                    metrics[
                        name
                    ]["tp"] += tp

                    metrics[
                        name
                    ]["fp"] += fp

                    metrics[
                        name
                    ]["fn"] += fn

    print("\n")

    print("=" * 60)
    print("BraTS-PED Segmentation Results")
    print("=" * 60)

    for region in [
        "WT",
        "TC",
        "ET"
    ]:

        tp = metrics[
            region
        ]["tp"]

        fp = metrics[
            region
        ]["fp"]

        fn = metrics[
            region
        ]["fn"]

        dice = (

            2 * tp

        ) / (

            2 * tp
            + fp
            + fn
            + 1e-8
        )

        precision = (

            tp

        ) / (

            tp
            + fp
            + 1e-8
        )

        recall = (

            tp

        ) / (

            tp
            + fn
            + 1e-8
        )

        f1 = (

            2
            * precision
            * recall

        ) / (

            precision
            + recall
            + 1e-8
        )

        print(
            f"\n{region}"
        )

        print(
            f"Dice      : {dice:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )


if __name__ == "__main__":

    evaluate()