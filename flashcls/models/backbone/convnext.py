"""
ConvNeXt backbone for FlashCls.

Variants: Tiny, Small.
Modern pure-ConvNet architecture (Liu et al., 2022) competitive with ViTs.
Uses torchvision pretrained ImageNet weights.
Returns feature tensor after final stage (classifier stripped).
"""

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
)

logger = logging.getLogger(__name__)

_VARIANT_MAP = {
    "convnext_tiny": (models.convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, 768),
    "convnext_small": (models.convnext_small, ConvNeXt_Small_Weights.DEFAULT, 768),
}


class ConvNeXt(nn.Module):
    """ConvNeXt backbone for classification.

    A pure ConvNet that matches or exceeds Vision Transformer accuracy.
    Uses depthwise convolutions, LayerNorm, and GELU activations.

    Args:
        variant: Model variant ("convnext_tiny", "convnext_small").
        pretrained: Load ImageNet-pretrained weights from torchvision.
    """

    def __init__(self, variant: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()
        variant = variant.lower()
        if variant not in _VARIANT_MAP:
            raise ValueError(
                f"Unknown ConvNeXt variant '{variant}'. "
                f"Choose from: {list(_VARIANT_MAP.keys())}"
            )

        factory_fn, weights, out_ch = _VARIANT_MAP[variant]
        net = factory_fn(weights=weights if pretrained else None)

        self.features = net.features
        self._out_channels = out_ch

        if pretrained:
            logger.info("Loaded pretrained %s backbone", variant)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
