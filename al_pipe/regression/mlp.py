"""MLP for prediction task."""

import torch


class MLP(torch.nn.Module):
    """MLP layer for classifications."""

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):  # noqa: D107
        # TODO: note this sizes variable
        super().__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1]) if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]  # noqa: E741
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()

    def forward(self, x):  # noqa: D102
        return self.network(x)
