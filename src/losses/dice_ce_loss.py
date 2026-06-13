import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):

    def __init__(
        self,
        smooth=1e-5,
        dice_weight=0.5,
        ce_weight=0.5
    ):

        super().__init__()

        self.smooth = smooth

        self.dice_weight = (
            dice_weight
        )

        self.ce_weight = (
            ce_weight
        )

        self.ce_loss = (
            nn.CrossEntropyLoss()
        )

    def dice_loss(
        self,
        logits,
        targets
    ):

        num_classes = (
            logits.shape[1]
        )

        probs = F.softmax(
            logits,
            dim=1
        )

        targets_one_hot = (
            F.one_hot(
                targets,
                num_classes
            )
            .permute(
                0, 3, 1, 2
            )
            .float()
        )

        dims = (0, 2, 3)

        intersection = (
            probs *
            targets_one_hot
        ).sum(
            dims
        )

        denominator = (
            probs +
            targets_one_hot
        ).sum(
            dims
        )

        dice = (

            2 *
            intersection +
            self.smooth

        ) / (

            denominator +
            self.smooth
        )

        dice_loss = (
            1 -
            dice.mean()
        )

        return dice_loss

    def forward(
        self,
        logits,
        targets
    ):

        dice = self.dice_loss(
            logits,
            targets
        )

        ce = self.ce_loss(
            logits,
            targets
        )

        total = (

            self.dice_weight *
            dice +

            self.ce_weight *
            ce
        )

        return total