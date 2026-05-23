import torch


def get_slice_weights(num_slices=5):

    center = num_slices // 2

    distances = []

    for i in range(num_slices):

        distance = abs(i - center)

        distances.append(distance)

    distances = torch.tensor(
        distances,
        dtype=torch.float32
    )

    weights = 1 / (distances + 1)

    weights = weights / weights.sum()

    return weights