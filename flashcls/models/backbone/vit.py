"""
Vision Transformer (ViT) backbone for FlashCls.

Variants:
  - ViT-Tiny: Custom lightweight (patch_size=16, dim=192, depth=12, heads=3)
  - ViT-Small: patch_size=16, dim=384, depth=12, heads=6

ViT-Small uses torchvision pretrained weights (vit_b_16 is too large, so we
use the standard ViT architecture with custom configs for tiny/small).

Returns CLS token embedding as feature vector.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings via strided convolution."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional dropout."""

    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block: MHSA + FFN with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


_VARIANT_CONFIGS = {
    "vit_tiny": {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "vit_small": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
}


class VisionTransformer(nn.Module):
    """Vision Transformer backbone for classification.

    Splits image into non-overlapping patches, processes with transformer
    encoder, and returns the CLS token embedding as features.

    Args:
        variant: Model variant ("vit_tiny", "vit_small").
        img_size: Input image resolution (assumed square).
        pretrained: Load pretrained weights if available.
        drop_rate: Dropout rate for embeddings and MLP.
        attn_drop_rate: Dropout rate for attention weights.
    """

    def __init__(self, variant: str = "vit_tiny", img_size: int = 224,
                 pretrained: bool = True, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0):
        super().__init__()
        variant = variant.lower()
        if variant not in _VARIANT_CONFIGS:
            raise ValueError(
                f"Unknown ViT variant '{variant}'. "
                f"Choose from: {list(_VARIANT_CONFIGS.keys())}"
            )

        cfg = _VARIANT_CONFIGS[variant]
        embed_dim = cfg["embed_dim"]
        depth = cfg["depth"]
        num_heads = cfg["num_heads"]
        mlp_ratio = cfg["mlp_ratio"]

        self._out_channels = embed_dim
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=16, in_channels=3, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

        if pretrained:
            self._load_pretrained(variant)

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_pretrained(self, variant: str):
        """Attempt to load pretrained weights from torchvision ViT models."""
        try:
            if variant == "vit_small":
                from torchvision.models import vit_b_16, ViT_B_16_Weights
                ref_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
                self._transfer_from_vit_b16(ref_model)
                logger.info("Loaded partial pretrained weights for ViT-Small from ViT-B/16")
            else:
                logger.info(
                    "No pretrained weights available for %s, using random init", variant
                )
        except Exception as e:
            logger.warning("Could not load pretrained ViT weights: %s", e)

    def _transfer_from_vit_b16(self, ref_model: nn.Module):
        """Transfer compatible weights from ViT-B/16 (dim=768) to ViT-Small (dim=384).

        Only transfers patch embedding (proj layer) by truncating channels.
        """
        try:
            ref_patch_weight = ref_model.conv_proj.weight.data
            my_dim = self._out_channels
            if ref_patch_weight.shape[0] >= my_dim:
                self.patch_embed.proj.weight.data.copy_(ref_patch_weight[:my_dim])
                if ref_model.conv_proj.bias is not None and self.patch_embed.proj.bias is not None:
                    self.patch_embed.proj.bias.data.copy_(ref_model.conv_proj.bias.data[:my_dim])
        except Exception:
            pass

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0]
