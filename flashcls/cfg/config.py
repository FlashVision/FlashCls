"""
FlashCls configuration module.

Provides model configuration via MODEL_SIZE_MAP and the get_config() factory.
"""

import logging
from typing import Any, Dict, Optional

from flashcls.models.backbone import (
    ShuffleNetV2,
    MobileNetV3Small,
    MobileNetV3Large,
    EfficientNet,
    ResNet,
    ConvNeXt,
    VisionTransformer,
)

logger = logging.getLogger(__name__)

MODEL_SIZE_MAP: Dict[str, Dict[str, Any]] = {
    # ShuffleNetV2 family (ultra-lightweight, ~1-7M params)
    "shuffle-0.5x": {"backbone": "shufflenet", "backbone_size": "0.5x", "out_channels": 192},
    "shuffle-1.0x": {"backbone": "shufflenet", "backbone_size": "1.0x", "out_channels": 464},
    "shuffle-1.5x": {"backbone": "shufflenet", "backbone_size": "1.5x", "out_channels": 704},
    # MobileNetV3 family (mobile-optimized)
    "mobilenet-s": {"backbone": "mobilenetv3_small", "out_channels": 576},
    "mobilenet-l": {"backbone": "mobilenetv3_large", "out_channels": 960},
    # EfficientNet family (accuracy-optimized)
    "efficientnet-b0": {"backbone": "efficientnet_b0", "out_channels": 1280},
    "efficientnet-b1": {"backbone": "efficientnet_b1", "out_channels": 1280},
    "efficientnet-b2": {"backbone": "efficientnet_b2", "out_channels": 1408},
    "efficientnet-b3": {"backbone": "efficientnet_b3", "out_channels": 1536},
    # ResNet family (classic)
    "resnet18": {"backbone": "resnet18", "out_channels": 512},
    "resnet34": {"backbone": "resnet34", "out_channels": 512},
    "resnet50": {"backbone": "resnet50", "out_channels": 2048},
    # ConvNeXt family (modern CNN, 2022 SOTA)
    "convnext-t": {"backbone": "convnext_tiny", "out_channels": 768},
    "convnext-s": {"backbone": "convnext_small", "out_channels": 768},
    # Vision Transformer family
    "vit-t": {"backbone": "vit_tiny", "out_channels": 192},
    "vit-s": {"backbone": "vit_small", "out_channels": 384},
}

_BACKBONE_REGISTRY = {
    "shufflenet": ShuffleNetV2,
    "mobilenetv3_small": MobileNetV3Small,
    "mobilenetv3_large": MobileNetV3Large,
    "efficientnet_b0": EfficientNet,
    "efficientnet_b1": EfficientNet,
    "efficientnet_b2": EfficientNet,
    "efficientnet_b3": EfficientNet,
    "resnet18": ResNet,
    "resnet34": ResNet,
    "resnet50": ResNet,
    "convnext_tiny": ConvNeXt,
    "convnext_small": ConvNeXt,
    "vit_tiny": VisionTransformer,
    "vit_small": VisionTransformer,
}

_BACKBONE_KWARGS = {
    "shufflenet": lambda cfg, pretrained: {"model_size": cfg.get("backbone_size", "1.0x"), "pretrained": pretrained},
    "mobilenetv3_small": lambda cfg, pretrained: {"pretrained": pretrained},
    "mobilenetv3_large": lambda cfg, pretrained: {"pretrained": pretrained},
    "efficientnet_b0": lambda cfg, pretrained: {"variant": "b0", "pretrained": pretrained},
    "efficientnet_b1": lambda cfg, pretrained: {"variant": "b1", "pretrained": pretrained},
    "efficientnet_b2": lambda cfg, pretrained: {"variant": "b2", "pretrained": pretrained},
    "efficientnet_b3": lambda cfg, pretrained: {"variant": "b3", "pretrained": pretrained},
    "resnet18": lambda cfg, pretrained: {"variant": "resnet18", "pretrained": pretrained},
    "resnet34": lambda cfg, pretrained: {"variant": "resnet34", "pretrained": pretrained},
    "resnet50": lambda cfg, pretrained: {"variant": "resnet50", "pretrained": pretrained},
    "convnext_tiny": lambda cfg, pretrained: {"variant": "convnext_tiny", "pretrained": pretrained},
    "convnext_small": lambda cfg, pretrained: {"variant": "convnext_small", "pretrained": pretrained},
    "vit_tiny": lambda cfg, pretrained: {"variant": "vit_tiny", "pretrained": pretrained},
    "vit_small": lambda cfg, pretrained: {"variant": "vit_small", "pretrained": pretrained},
}


def get_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration by name.

    Args:
        model_name: Key from MODEL_SIZE_MAP (e.g. "shuffle-1.0x", "resnet50").

    Returns:
        Configuration dict with backbone type and output channels.

    Raises:
        ValueError: If model_name is not recognized.
    """
    model_name = model_name.lower()
    if model_name not in MODEL_SIZE_MAP:
        available = ", ".join(sorted(MODEL_SIZE_MAP.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    return MODEL_SIZE_MAP[model_name].copy()


def build_backbone(model_name: str, pretrained: bool = True):
    """Instantiate a backbone network from a model name.

    Args:
        model_name: Key from MODEL_SIZE_MAP (e.g. "shuffle-1.0x", "resnet50").
        pretrained: Whether to load pretrained ImageNet weights.

    Returns:
        Instantiated backbone module.
    """
    cfg = get_config(model_name)
    backbone_key = cfg["backbone"]

    if backbone_key not in _BACKBONE_REGISTRY:
        raise ValueError(f"No backbone registered for key '{backbone_key}'")

    cls = _BACKBONE_REGISTRY[backbone_key]
    kwargs_fn = _BACKBONE_KWARGS[backbone_key]
    kwargs = kwargs_fn(cfg, pretrained)

    backbone = cls(**kwargs)
    logger.info(
        "Built backbone '%s' (out_channels=%d, pretrained=%s)",
        model_name, cfg["out_channels"], pretrained,
    )
    return backbone


def list_models() -> list:
    """Return sorted list of all available model names."""
    return sorted(MODEL_SIZE_MAP.keys())
