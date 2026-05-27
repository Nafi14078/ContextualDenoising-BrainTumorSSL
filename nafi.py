import torch

checkpoint = torch.load(
    "ssl_checkpoint.pth",
    map_location="cpu"
)

print(checkpoint["epoch"])