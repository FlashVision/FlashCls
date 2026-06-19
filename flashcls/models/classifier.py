"""FlashClassifier — lightweight classification model (backbone + GAP + head)."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashcls.models.backbone.shufflenet import ShuffleNetV2

logger = logging.getLogger(__name__)

MODEL_SIZE_MAP = {
    "0.5x": {"channels": [24, 48, 96, 192, 1024], "last_conv": 1024},
    "1.0x": {"channels": [24, 116, 232, 464, 1024], "last_conv": 1024},
    "1.5x": {"channels": [24, 176, 352, 704, 1024], "last_conv": 1024},
    "2.0x": {"channels": [24, 244, 488, 976, 2048], "last_conv": 2048},
}

BACKBONE_SIZE_ALIAS = {
    "m-0.5x": "0.5x",
    "m": "1.0x",
    "m-1.5x": "1.5x",
}


class FlashClassifier(nn.Module):
    """Ultra-lightweight image classifier.

    Architecture::

        Input → ShuffleNetV2 backbone → Global Average Pooling → Dropout → Linear → Logits

    Args:
        num_classes: Number of output classes.
        input_size: Spatial input resolution (int or (H, W)).
        backbone_size: ShuffleNetV2 width multiplier (``"0.5x"``, ``"1.0x"``, ``"1.5x"``).
        dropout: Dropout probability before the final linear layer.
        pretrained: Load ImageNet-pretrained backbone weights.
        class_names: Human-readable class names for prediction output.
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_size: Union[int, Tuple[int, int]] = (224, 224),
        backbone_size: str = "1.0x",
        dropout: float = 0.2,
        pretrained: bool = True,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        real_size = BACKBONE_SIZE_ALIAS.get(backbone_size, backbone_size)

        self.num_classes = num_classes
        self.input_size = input_size
        self.backbone_size = real_size
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        self.backbone = ShuffleNetV2(model_size=real_size, pretrained=pretrained)
        feature_channels = self.backbone.out_channels

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feature_channels, num_classes)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "FlashClassifier: backbone=%s, classes=%d, params=%s",
            real_size, num_classes, f"{total_params:,}",
        )

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images ``[B, 3, H, W]``.
            targets: Optional ground-truth class indices ``[B]``.

        Returns:
            Dict with ``"logits"`` and optionally ``"loss"``.
        """
        features = self.backbone(x)
        pooled = self.gap(features).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(logits, targets)
        return out

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, top_k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """Classify a batch and return sorted ``(class_name, probability)`` lists.

        Args:
            x: Input images ``[B, 3, H, W]``.
            top_k: Number of top predictions to return per image.

        Returns:
            A list of length ``B``, each element a list of ``(name, prob)`` tuples.
        """
        self.eval()
        features = self.backbone(x)
        pooled = self.gap(features).flatten(1)
        logits = self.fc(pooled)
        probs = F.softmax(logits, dim=1)

        batch_results = []
        for i in range(probs.size(0)):
            topk_probs, topk_idx = probs[i].topk(min(top_k, self.num_classes))
            preds = []
            for prob, idx in zip(topk_probs.cpu().tolist(), topk_idx.cpu().tolist()):
                name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
                preds.append((name, prob))
            batch_results.append(preds)
        return batch_results

    def get_model_info(self) -> Dict:
        """Return a summary dict of model parameters and architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone_size": self.backbone_size,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "total_params": total_params,
            "trainable_params": trainable,
            "size_mb": total_params * 4 / (1024 * 1024),
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled feature vector ``[B, C]`` (before the classifier head)."""
        features = self.backbone(x)
        return self.gap(features).flatten(1)


def build_model(config) -> FlashClassifier:
    """Build a FlashClassifier from a configuration object or namespace.

    The config must have attributes: ``num_classes``, ``input_size``,
    ``backbone_size``, ``dropout``, ``pretrained``.
    Optionally: ``class_names``.
    """
    return FlashClassifier(
        num_classes=getattr(config, "num_classes", 10),
        input_size=getattr(config, "input_size", 224),
        backbone_size=getattr(config, "backbone_size", "1.0x"),
        dropout=getattr(config, "dropout", 0.2),
        pretrained=getattr(config, "pretrained", True),
        class_names=getattr(config, "class_names", None),
    )
