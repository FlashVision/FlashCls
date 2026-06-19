"""
ShuffleNetV2 backbone for FlashCls.

Variants: 0.5x, 1.0x, 1.5x, 2.0x
Returns feature tensor after final stage (before classifier).
"""

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
    ShuffleNet_V2_X1_5_Weights,
    ShuffleNet_V2_X2_0_Weights,
)

logger = logging.getLogger(__name__)

_VARIANT_MAP = {
    "0.5x": (models.shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights.DEFAULT, 1024),
    "1.0x": (models.shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights.DEFAULT, 1024),
    "1.5x": (models.shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights.DEFAULT, 1024),
    "2.0x": (models.shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights.DEFAULT, 2048),
}

_OUT_CHANNELS = {
    "0.5x": 192,
    "1.0x": 464,
    "1.5x": 704,
    "2.0x": 976,
}


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 backbone for classification.

    Args:
        model_size: Variant string ("0.5x", "1.0x", "1.5x", "2.0x").
        pretrained: Load ImageNet-pretrained weights from torchvision.
    """

    def __init__(self, model_size: str = "1.0x", pretrained: bool = True):
        super().__init__()
        if model_size not in _VARIANT_MAP:
            raise ValueError(
                f"Unknown ShuffleNetV2 size '{model_size}'. "
                f"Choose from: {list(_VARIANT_MAP.keys())}"
            )

        factory_fn, weights, final_conv_ch = _VARIANT_MAP[model_size]
        net = factory_fn(weights=weights if pretrained else None)

        self._out_channels = _OUT_CHANNELS[model_size]

        self.conv1 = net.conv1
        self.maxpool = net.maxpool
        self.stage2 = net.stage2
        self.stage3 = net.stage3
        self.stage4 = net.stage4

        if pretrained:
            logger.info("Loaded pretrained ShuffleNetV2-%s backbone", model_size)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
