"""
MobileNetV3 backbone for FlashCls.

Variants: Small (lightweight mobile), Large (accuracy-optimized mobile).
Uses torchvision pretrained ImageNet weights.
Returns feature tensor after last conv layer (before classifier).
"""

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
)

logger = logging.getLogger(__name__)


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small backbone.

    Optimized for mobile/edge with ~2.5M params.
    Output channels: 576.

    Args:
        pretrained: Load ImageNet-pretrained weights from torchvision.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)

        self.features = net.features
        self._out_channels = 576

        if pretrained:
            logger.info("Loaded pretrained MobileNetV3-Small backbone")

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class MobileNetV3Large(nn.Module):
    """MobileNetV3-Large backbone.

    Higher accuracy than Small with ~5.4M params.
    Output channels: 960.

    Args:
        pretrained: Load ImageNet-pretrained weights from torchvision.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)

        self.features = net.features
        self._out_channels = 960

        if pretrained:
            logger.info("Loaded pretrained MobileNetV3-Large backbone")

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
