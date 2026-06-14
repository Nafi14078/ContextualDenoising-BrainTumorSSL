import os
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.losses.dice_ce_loss import (
    DiceCELoss
)


class SegmentationTrainer:

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

        self.loss_fn = DiceCELoss()

        self.scaler = (
            torch.amp.GradScaler(
                "cuda"
            )
        )

        self.train_losses = []
        self.val_losses = []

        self.checkpoint_path = (
            "outputs/checkpoints/"
            "segmentation_checkpoint.pth"
        )

        self.best_model_path = (
            "outputs/checkpoints/"
            "best_segmentation_model.pth"
        )

        self.start_epoch = 0

    # =====================================
    # Dice Metric
    # =====================================

    def compute_dice(
        self,
        prediction,
        target
    ):

        prediction = torch.argmax(
            prediction,
            dim=1
        )

        dices = []

        for cls in [1, 2, 3]:

            pred_cls = (
                prediction == cls
            ).float()

            target_cls = (
                target == cls
            ).float()

            intersection = (
                pred_cls *
                target_cls
            ).sum()

            union = (
                pred_cls.sum()
                +
                target_cls.sum()
            )

            dice = (
                (
                    2 * intersection
                    + 1e-5
                )
                /
                (
                    union
                    + 1e-5
                )
            )

            dices.append(
                dice.item()
            )

        return sum(dices) / len(dices)

    # =====================================
    # Save Checkpoint
    # =====================================

    def save_checkpoint(
        self,
        epoch,
        best_dice
    ):

        checkpoint = {

            "epoch":
            epoch,

            "model_state_dict":
            self.model.state_dict(),

            "optimizer_state_dict":
            self.optimizer.state_dict(),

            "scheduler_state_dict":
            self.scheduler.state_dict(),

            "scaler_state_dict":
            self.scaler.state_dict(),

            "best_dice":
            best_dice,

            "train_losses":
            self.train_losses,

            "val_losses":
            self.val_losses
        }

        torch.save(
            checkpoint,
            self.checkpoint_path
        )

    # =====================================
    # Resume Training
    # =====================================

    def load_checkpoint(self):

        if os.path.exists(
            self.checkpoint_path
        ):

            print(
                "Resuming training..."
            )

            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=
                self.device
            )

            self.model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

            self.scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

            self.scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

            self.start_epoch = (
                checkpoint[
                    "epoch"
                ] + 1
            )

            self.train_losses = (
                checkpoint[
                    "train_losses"
                ]
            )

            self.val_losses = (
                checkpoint[
                    "val_losses"
                ]
            )

            best_dice = (
                checkpoint[
                    "best_dice"
                ]
            )

            print(
                f"Resuming from "
                f"epoch "
                f"{self.start_epoch}"
            )

            return best_dice

        return 0.0

    # =====================================
    # Train One Epoch
    # =====================================

    def train_epoch(self):

        self.model.train()

        running_loss = 0

        loop = tqdm(
            self.train_loader
        )

        for images, masks in loop:

            images = images.to(
                self.device,
                non_blocking=True
            )

            masks = masks.to(
                self.device,
                non_blocking=True
            )

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda"
            ):

                outputs = (
                    self.model(
                        images
                    )
                )

                loss = (
                    self.loss_fn(
                        outputs,
                        masks
                    )
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

            loop.set_postfix(
                loss=
                loss.item()
            )

        return (
            running_loss
            /
            len(
                self.train_loader
            )
        )

    # =====================================
    # Validation
    # =====================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0
        running_dice = 0

        for images, masks in self.val_loader:

            images = images.to(
                self.device,
                non_blocking=True
            )

            masks = masks.to(
                self.device,
                non_blocking=True
            )

            with torch.amp.autocast(
                "cuda"
            ):

                outputs = (
                    self.model(
                        images
                    )
                )

                loss = (
                    self.loss_fn(
                        outputs,
                        masks
                    )
                )

            running_loss += (
                loss.item()
            )

            running_dice += (
                self.compute_dice(
                    outputs,
                    masks
                )
            )

        avg_loss = (
            running_loss
            /
            len(
                self.val_loader
            )
        )

        avg_dice = (
            running_dice
            /
            len(
                self.val_loader
            )
        )

        return (
            avg_loss,
            avg_dice
        )

    # =====================================
    # Plot Curves
    # =====================================

    def plot_curve(self):

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            self.train_losses,
            label="Train Loss"
        )

        plt.plot(
            self.val_losses,
            label="Validation Loss"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        plt.legend()

        plt.grid(True)

        plt.savefig(

            "outputs/visualizations/"
            "segmentation_loss_curve.png"
        )

        plt.close()

    # =====================================
    # Main Training Loop
    # =====================================

    def train(self):

        best_dice = (
            self.load_checkpoint()
        )

        for epoch in range(

            self.start_epoch,

            self.config[
                "epochs"
            ]
        ):

            print(
                f"\nEpoch "
                f"{epoch+1}/"
                f"{self.config['epochs']}"
            )

            train_loss = (
                self.train_epoch()
            )

            val_loss, val_dice = (
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
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Val Loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"Val Dice: "
                f"{val_dice:.4f}"
            )

            if val_dice > best_dice:

                best_dice = (
                    val_dice
                )

                torch.save(

                    self.model.state_dict(),

                    self.best_model_path
                )

                print(
                    "Best model saved."
                )

            self.save_checkpoint(
                epoch,
                best_dice
            )

            self.plot_curve()

        print(
            "Training finished."
        )