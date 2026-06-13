import torch

from src.models.segmentation_model import (
    SegmentationModel,
    load_pretrained_encoder
)

model = SegmentationModel()

model = load_pretrained_encoder(
    model,
    "pretrained_weights/ssl_encoder.pth"
)

print("Success")