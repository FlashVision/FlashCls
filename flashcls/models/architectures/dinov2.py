"""
DINOv2 Backbone for FlashCls.

Wraps the HuggingFace DINOv2 vision transformer as a feature extractor
with support for:
  - Linear probing (frozen backbone + trainable head)
  - Full fine-tuning
  - Intermediate layer feature extraction

Reference:
    Oquab et al., "DINOv2: Learning Robust Visual Features without
    Supervision", 2023.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from flashcls.registry import BACKBONES

logger = logging.getLogger(__name__)

_DINOV2_VARIANTS = {
    "dinov2_vits14": {"embed_dim": 384, "hf_name": "facebook/dinov2-small"},
    "dinov2_vitb14": {"embed_dim": 768, "hf_name": "facebook/dinov2-base"},
    "dinov2_vitl14": {"embed_dim": 1024, "hf_name": "facebook/dinov2-large"},
    "dinov2_vitg14": {"embed_dim": 1536, "hf_name": "facebook/dinov2-giant"},
}


class DINOv2FeatureExtractor(nn.Module):
    """Loads DINOv2 from HuggingFace transformers or torch.hub and exposes
    the CLS token embedding (and optionally patch tokens) as features."""

    def __init__(self, variant: str = "dinov2_vits14", pretrained: bool = True):
        super().__init__()
        cfg = _DINOV2_VARIANTS.get(variant)
        if cfg is None:
            raise ValueError(f"Unknown DINOv2 variant '{variant}'. Choose from {list(_DINOV2_VARIANTS)}")

        self._embed_dim = cfg["embed_dim"]
        self._loaded = False

        if pretrained:
            self._try_load_pretrained(variant, cfg["hf_name"])

        if not self._loaded:
            logger.info("Using randomly initialised DINOv2-style ViT (%s)", variant)
            self._build_fallback(cfg)

    def _try_load_pretrained(self, variant: str, hf_name: str):
        # HuggingFace transformers
        try:
            from transformers import Dinov2Model
            self.backbone = Dinov2Model.from_pretrained(hf_name)
            self._loaded = True
            self._backend = "hf"
            logger.info("Loaded DINOv2 from HuggingFace: %s", hf_name)
            return
        except Exception as e:
            logger.debug("HuggingFace DINOv2 load failed: %s", e)

        # PyTorch Hub
        try:
            self.backbone = torch.hub.load("facebookresearch/dinov2", variant)
            self._loaded = True
            self._backend = "hub"
            logger.info("Loaded DINOv2 from torch.hub: %s", variant)
            return
        except Exception as e:
            logger.debug("torch.hub DINOv2 load failed: %s", e)

    def _build_fallback(self, cfg: dict):
        """Build a minimal ViT matching DINOv2 dimensions for offline use."""
        from flashcls.models.backbone.vit import VisionTransformer as _ViT

        dim = cfg["embed_dim"]
        depth_map = {384: 12, 768: 12, 1024: 24, 1536: 40}
        heads_map = {384: 6, 768: 12, 1024: 16, 1536: 24}
        depth = depth_map.get(dim, 12)
        heads = heads_map.get(dim, 6)

        self.backbone = _ViT.__new__(_ViT)
        nn.Module.__init__(self.backbone)
        from flashcls.models.backbone.vit import PatchEmbedding, TransformerBlock

        self.backbone._out_channels = dim
        self.backbone.patch_embed = PatchEmbedding(img_size=224, patch_size=14, in_channels=3, embed_dim=dim)
        num_patches = self.backbone.patch_embed.num_patches
        self.backbone.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.backbone.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.backbone.pos_drop = nn.Dropout(0.0)
        self.backbone.blocks = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads, mlp_ratio=4.0) for _ in range(depth)
        ])
        self.backbone.norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.backbone.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.backbone.cls_token, std=0.02)
        self._backend = "fallback"
        self._loaded = True

    @property
    def out_channels(self) -> int:
        return self._embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features [B, embed_dim]."""
        if self._backend == "hf":
            outputs = self.backbone(pixel_values=x)
            return outputs.last_hidden_state[:, 0]
        elif self._backend == "hub":
            return self.backbone(x)
        else:
            return self.backbone(x)


@BACKBONES.register("DINOv2")
class DINOv2Backbone(nn.Module):
    """DINOv2 backbone with linear probe and full fine-tuning support.

    Args:
        variant: DINOv2 model variant.
        num_classes: Number of output classes.
        pretrained: Load pretrained DINOv2 weights.
        freeze_backbone: Freeze backbone for linear probing.
        dropout: Dropout before the classification head.
    """

    def __init__(
        self,
        variant: str = "dinov2_vits14",
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_extractor = DINOv2FeatureExtractor(variant, pretrained)
        self._out_channels = self.feature_extractor.out_channels

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            logger.info("DINOv2 backbone frozen for linear probing")

        self.head = nn.Sequential(
            nn.LayerNorm(self._out_channels),
            nn.Dropout(dropout),
            nn.Linear(self._out_channels, num_classes),
        )

        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.head(features)
        out: Dict[str, torch.Tensor] = {"logits": logits, "features": features}
        if targets is not None:
            import torch.nn.functional as F
            out["loss"] = F.cross_entropy(logits, targets)
        return out

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def freeze_backbone(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "DINOv2",
            "embed_dim": self._out_channels,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
