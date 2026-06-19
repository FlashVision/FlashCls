"""
ResNet backbone for FlashCls.

Variants: ResNet-18, ResNet-34, ResNet-50.
Uses torchvision pretrained ImageNet weights.
Returns feature tensor after layer4 (avgpool + fc stripped).
"""

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

logger = logging.getLogger(__name__)

_VARIANT_MAP = {
    "resnet18": (models.resnet18, ResNet18_Weights.DEFAULT, 512),
    "resnet34": (models.resnet34, ResNet34_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, ResNet50_Weights.DEFAULT, 2048),
}


class ResNet(nn.Module):
    """ResNet backbone for classification.

    Classic residual network. Strips avgpool + fc, returns spatial features.

    Args:
        variant: Model variant ("resnet18", "resnet34", "resnet50").
        pretrained: Load ImageNet-pretrained weights from torchvision.
    """

    def __init__(self, variant: str = "resnet18", pretrained: bool = True):
        super().__init__()
        variant = variant.lower()
        if variant not in _VARIANT_MAP:
            raise ValueError(
                f"Unknown ResNet variant '{variant}'. "
                f"Choose from: {list(_VARIANT_MAP.keys())}"
            )

        factory_fn, weights, out_ch = _VARIANT_MAP[variant]
        net = factory_fn(weights=weights if pretrained else None)

        self._out_channels = out_ch

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        if pretrained:
            logger.info("Loaded pretrained %s backbone", variant)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
