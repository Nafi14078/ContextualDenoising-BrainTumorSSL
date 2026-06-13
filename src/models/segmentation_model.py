import torch
import torch.nn as nn

from src.models.resunet_encoder import (
    ConvBlock
)


class SegmentationModel(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------
        # Encoder
        # --------------------------

        self.encoder1 = ConvBlock(
            20,
            64
        )

        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = ConvBlock(
            64,
            128
        )

        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(
            128,
            256
        )

        # --------------------------
        # Decoder
        # --------------------------

        self.up1 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = ConvBlock(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = ConvBlock(
            128,
            64
        )

        # --------------------------
        # Segmentation Head
        # --------------------------

        self.segmentation_head = (
            nn.Conv2d(
                64,
                4,      # classes: 0,1,2,3
                kernel_size=1
            )
        )

    def forward(self, x):

        # --------------------------
        # Encoder
        # --------------------------

        e1 = self.encoder1(
            x
        )

        e2 = self.encoder2(
            self.pool1(e1)
        )

        b = self.bottleneck(
            self.pool2(e2)
        )

        # --------------------------
        # Decoder
        # --------------------------

        d1 = self.up1(
            b
        )

        d1 = torch.cat(
            [d1, e2],
            dim=1
        )

        d1 = self.decoder1(
            d1
        )

        d2 = self.up2(
            d1
        )

        d2 = torch.cat(
            [d2, e1],
            dim=1
        )

        d2 = self.decoder2(
            d2
        )

        # --------------------------
        # Segmentation Output
        # --------------------------

        return self.segmentation_head(
            d2
        )


# ==================================================
# Load SSL Pretrained Weights
# ==================================================

def load_pretrained_encoder(
    model,
    weight_path
):

    print(
        f"Loading SSL weights from "
        f"{weight_path}"
    )

    pretrained = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True
    )

    model_dict = model.state_dict()

    matched_layers = {}

    skipped_layers = []

    for key, value in pretrained.items():

        if (
            key in model_dict
            and
            model_dict[key].shape
            == value.shape
        ):

            matched_layers[key] = value

        else:

            skipped_layers.append(
                key
            )

    model_dict.update(
        matched_layers
    )

    model.load_state_dict(
        model_dict
    )

    print(
        f"Loaded "
        f"{len(matched_layers)} "
        f"layers from SSL model."
    )

    print(
        f"Skipped "
        f"{len(skipped_layers)} "
        f"layers."
    )

    return model