"""Classification head: Global Average Pooling + FC layer."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head with GAP, optional dropout, and linear projection.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of output classes.
        dropout: Dropout probability before the linear layer.
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
