import torch.nn as nn
import torch

from src.models.resunet_encoder import (
    ConvBlock
)


class SSLModel(nn.Module):

    def __init__(self):

        super().__init__()

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

        self.up1 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.decoder1 = ConvBlock(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.decoder2 = ConvBlock(
            128,
            64
        )

        self.final = nn.Conv2d(
            64,
            4,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool1(e1)
        )

        b = self.bottleneck(
            self.pool2(e2)
        )

        d1 = self.up1(b)

        d1 = torch.cat(
            [d1, e2],
            dim=1
        )

        d1 = self.decoder1(d1)

        d2 = self.up2(d1)

        d2 = torch.cat(
            [d2, e1],
            dim=1
        )

        d2 = self.decoder2(d2)

        return self.final(d2)