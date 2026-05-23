import os
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.cuda.amp import (
    autocast,
    GradScaler
)

from src.losses.ssl_loss import (
    ssl_loss
)

from src.ssl.corruption import (
    WeightedDenoising
)


class SSLTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,
        device
    ):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config

        self.device = device

        self.scaler = GradScaler()

        self.corruption = (
            WeightedDenoising(
                num_slices=config[
                    "num_input_slices"
                ],
                noise_std=config[
                    "noise_std"
                ],
                mask_ratio=config[
                    "mask_ratio"
                ]
            )
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):

        self.model.train()

        running_loss = 0

        for x, y in tqdm(
            self.train_loader
        ):

            x = x.to(self.device)

            y = y.to(self.device)

            x = self.corruption(x)

            self.optimizer.zero_grad()

            with autocast():

                pred = self.model(x)

                loss = ssl_loss(
                    pred,
                    y
                )

            self.scaler.scale(
                loss
            ).backward()

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            running_loss += (
                loss.item()
            )

        return (
            running_loss
            /
            len(self.train_loader)
        )

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0

        for x, y in self.val_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            x = self.corruption(x)

            with autocast():

                pred = self.model(x)

                loss = ssl_loss(
                    pred,
                    y
                )

            running_loss += (
                loss.item()
            )

        return (
            running_loss
            /
            len(self.val_loader)
        )

    def plot_curve(self):

        plt.figure(figsize=(8,5))

        plt.plot(
            self.train_losses,
            label="Train Loss"
        )

        plt.plot(
            self.val_losses,
            label="Validation Loss"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend()

        plt.grid(True)

        save_path = os.path.join(
            "outputs",
            "visualizations",
            "ssl_loss_curve.png"
        )

        plt.savefig(
            save_path
        )

        plt.close()

    def train(self):

        best_loss = 999999

        for epoch in range(
            self.config["epochs"]
        ):

            train_loss = (
                self.train_epoch()
            )

            val_loss = (
                self.validate()
            )

            self.scheduler.step()

            self.train_losses.append(
                train_loss
            )

            self.val_losses.append(
                val_loss
            )

            print(
                f"Epoch "
                f"{epoch+1} | "
                f"Train "
                f"{train_loss:.4f} | "
                f"Val "
                f"{val_loss:.4f}"
            )

            if val_loss < best_loss:

                best_loss = val_loss

                torch.save(
                    self.model.state_dict(),
                    "pretrained_weights/ssl_encoder.pth"
                )

        self.plot_curve()