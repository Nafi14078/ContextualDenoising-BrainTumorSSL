import torch
import random

from src.ssl.weighted_sampling import (
    get_slice_weights
)


class WeightedDenoising:

    def __init__(
        self,
        num_slices=5,
        noise_std=0.1,
        mask_ratio=0.15
    ):

        self.num_slices = num_slices

        self.noise_std = noise_std

        self.mask_ratio = mask_ratio

        self.weights = get_slice_weights(
            num_slices
        )

    def add_noise(self, x):

        noise = torch.randn_like(x)

        return x + (
            noise * self.noise_std
        )

    def random_mask(self, x):

        mask = (
            torch.rand_like(x)
            > self.mask_ratio
        )

        return x * mask

    def weighted_slice_scaling(self, x):

        b, c, h, w = x.shape

        num_modalities = 4

        x = x.view(
            b,
            self.num_slices,
            num_modalities,
            h,
            w
        )

        weights = self.weights.to(
            x.device
        )

        weights = weights.view(
            1,
            self.num_slices,
            1,
            1,
            1
        )

        x = x * weights

        x = x.view(
            b,
            c,
            h,
            w
        )

        return x

    def __call__(self, x):

        x = self.weighted_slice_scaling(
            x
        )

        if random.random() > 0.5:

            x = self.add_noise(x)

        if random.random() > 0.5:

            x = self.random_mask(x)

        return x