"""
EfficientNet backbone for FlashCls.

Variants: B0, B1, B2, B3, B4.
Uses torchvision pretrained ImageNet weights.
Returns feature tensor after final conv (classifier stripped).
"""

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
)

logger = logging.getLogger(__name__)

_VARIANT_MAP = {
    "b0": (models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
    "b1": (models.efficientnet_b1, EfficientNet_B1_Weights.DEFAULT, 1280),
    "b2": (models.efficientnet_b2, EfficientNet_B2_Weights.DEFAULT, 1408),
    "b3": (models.efficientnet_b3, EfficientNet_B3_Weights.DEFAULT, 1536),
    "b4": (models.efficientnet_b4, EfficientNet_B4_Weights.DEFAULT, 1792),
}


class EfficientNet(nn.Module):
    """EfficientNet backbone for classification.

    Compound-scaled CNN optimized for accuracy/efficiency trade-off.

    Args:
        variant: Model variant ("b0", "b1", "b2", "b3", "b4").
        pretrained: Load ImageNet-pretrained weights from torchvision.
    """

    def __init__(self, variant: str = "b0", pretrained: bool = True):
        super().__init__()
        variant = variant.lower()
        if variant not in _VARIANT_MAP:
            raise ValueError(
                f"Unknown EfficientNet variant '{variant}'. "
                f"Choose from: {list(_VARIANT_MAP.keys())}"
            )

        factory_fn, weights, out_ch = _VARIANT_MAP[variant]
        net = factory_fn(weights=weights if pretrained else None)

        self.features = net.features
        self._out_channels = out_ch

        if pretrained:
            logger.info("Loaded pretrained EfficientNet-%s backbone", variant.upper())

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
