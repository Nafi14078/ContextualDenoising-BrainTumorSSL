import torch.nn.functional as F


def ssl_loss(pred, target):

    l1 = F.l1_loss(
        pred,
        target
    )

    mse = F.mse_loss(
        pred,
        target
    )

    loss = (
        0.7 * l1
        +
        0.3 * mse
    )

    return loss