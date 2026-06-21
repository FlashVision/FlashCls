"""
Self-Supervised Learning (SSL) for FlashCls.

Implements DINO-style self-distillation and MAE-style masked image modelling
for pretraining visual backbones without labels.

References:
  - Caron et al., "Emerging Properties in Self-Supervised Vision
    Transformers" (DINO), ICCV 2021.
  - He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE),
    CVPR 2022.
"""

import copy
import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiCropWrapper(nn.Module):
    """Wrapper that processes global and local crops through a backbone,
    concatenates outputs, and applies a projection head."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """Process a list of multi-crop augmented views.

        Args:
            x_list: List of tensors, each [B, 3, H_i, W_i].

        Returns:
            Concatenated projected features [B * num_crops, proj_dim].
        """
        outputs = []
        for x in x_list:
            if hasattr(self.backbone, "extract_features"):
                feat = self.backbone.extract_features(x)
            elif hasattr(self.backbone, "forward"):
                out = self.backbone(x)
                feat = out if isinstance(out, torch.Tensor) else out.get("features", out.get("logits"))
            else:
                feat = self.backbone(x)

            if feat.dim() > 2:
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)

            outputs.append(self.head(feat))

        return torch.cat(outputs, dim=0)


class DINOHead(nn.Module):
    """DINO projection head: MLP + L2-norm + prototypes."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, bottleneck_dim: int = 256, out_dim: int = 65536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class DINOLoss(nn.Module):
    """DINO self-distillation loss with centering and sharpening.

    The teacher output is centered (to prevent collapse) and sharpened
    (low temperature), while the student uses a higher temperature.

    Args:
        out_dim: Output dimension (number of prototypes).
        num_crops: Total number of crops (global + local).
        num_global: Number of global crops (teacher only sees these).
        teacher_temp: Teacher sharpening temperature.
        student_temp: Student temperature.
        center_momentum: EMA momentum for the center vector.
    """

    def __init__(
        self,
        out_dim: int = 65536,
        num_crops: int = 8,
        num_global: int = 2,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.num_crops = num_crops
        self.num_global = num_global
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DINO cross-entropy loss.

        Args:
            student_output: [B * num_crops, out_dim] student logits.
            teacher_output: [B * num_global, out_dim] teacher logits.
        """
        student_out = student_output / self.student_temp
        student_chunks = student_out.chunk(self.num_crops)

        teacher_out = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1
        ).detach()
        teacher_chunks = teacher_out.chunk(self.num_global)

        total_loss = 0
        n_terms = 0
        for t_idx, tch in enumerate(teacher_chunks):
            for s_idx, stu in enumerate(student_chunks):
                if s_idx == t_idx:
                    continue
                loss = -torch.sum(tch * F.log_softmax(stu, dim=-1), dim=-1).mean()
                total_loss += loss
                n_terms += 1

        total_loss /= max(n_terms, 1)
        self._update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_output: torch.Tensor):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MAEDecoder(nn.Module):
    """Lightweight MAE decoder for masked image modelling.

    Reconstructs masked patches from the encoder's visible-patch
    representations plus learnable mask tokens.

    Args:
        encoder_dim: Encoder output dimension.
        decoder_dim: Decoder hidden dimension.
        decoder_depth: Number of decoder transformer layers.
        decoder_heads: Number of decoder attention heads.
        patch_size: Patch size used by the encoder.
        num_patches: Total number of patches.
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        decoder_dim: int = 192,
        decoder_depth: int = 4,
        decoder_heads: int = None,
        patch_size: int = 16,
        num_patches: int = 196,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.pixel_dim = patch_size * patch_size * 3

        if decoder_heads is None:
            for h in [6, 4, 3, 2, 1]:
                if decoder_dim % h == 0:
                    decoder_heads = h
                    break

        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.pixel_dim)

    def forward(
        self,
        encoded: torch.Tensor,
        visible_indices: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Decode masked patches.

        Args:
            encoded: [B, N_vis, encoder_dim] encoded visible patches.
            visible_indices: [B, N_vis] indices of visible patches.
            mask_indices: [B, N_mask] indices of masked patches.

        Returns:
            [B, N_mask, patch_size^2 * 3] predicted pixel values.
        """
        B = encoded.shape[0]
        x = self.decoder_embed(encoded)

        mask_tokens = self.mask_token.expand(B, mask_indices.shape[1], -1)

        # Combine visible and mask tokens and restore positional order
        N_total = visible_indices.shape[1] + mask_indices.shape[1]
        full = torch.zeros(B, N_total, x.shape[-1], device=x.device, dtype=x.dtype)
        full.scatter_(1, visible_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]), x)
        full.scatter_(1, mask_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]), mask_tokens)

        # Add CLS token position and apply positional encoding
        full = full + self.decoder_pos_embed[:, 1:N_total + 1]

        full = self.decoder_blocks(full)
        full = self.decoder_norm(full)

        # Predict only masked positions
        pred = self.decoder_pred(
            torch.gather(full, 1, mask_indices.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
        )
        return pred


class SSLTrainer:
    """Self-supervised learning trainer supporting DINO and MAE methods.

    Args:
        backbone: The backbone model to pretrain.
        method: 'dino' or 'mae'.
        feature_dim: Backbone output feature dimension.
        device: Target device.
        dino_out_dim: DINO projection output dimension.
        dino_teacher_temp: DINO teacher temperature.
        dino_student_temp: DINO student temperature.
        dino_ema_decay: EMA momentum for the teacher network.
        mae_mask_ratio: Fraction of patches to mask for MAE.
        mae_decoder_dim: MAE decoder hidden dimension.
        mae_decoder_depth: MAE decoder depth.
    """

    def __init__(
        self,
        backbone: nn.Module,
        method: str = "dino",
        feature_dim: int = 384,
        device: str = "cuda",
        dino_out_dim: int = 65536,
        dino_teacher_temp: float = 0.04,
        dino_student_temp: float = 0.1,
        dino_ema_decay: float = 0.996,
        mae_mask_ratio: float = 0.75,
        mae_decoder_dim: int = 192,
        mae_decoder_depth: int = 4,
    ):
        self.method = method.lower()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone = backbone.to(self.device)

        if self.method == "dino":
            self._setup_dino(feature_dim, dino_out_dim, dino_teacher_temp, dino_student_temp, dino_ema_decay)
        elif self.method == "mae":
            self._setup_mae(feature_dim, mae_mask_ratio, mae_decoder_dim, mae_decoder_depth)
        else:
            raise ValueError(f"Unknown SSL method '{method}'. Choose 'dino' or 'mae'.")

        logger.info("SSLTrainer initialised: method=%s, feature_dim=%d", method, feature_dim)

    def _setup_dino(self, feature_dim, out_dim, teacher_temp, student_temp, ema_decay):
        head = DINOHead(feature_dim, out_dim=out_dim)
        self.student = MultiCropWrapper(self.backbone, head).to(self.device)
        self.teacher = copy.deepcopy(self.student).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.dino_loss = DINOLoss(
            out_dim=out_dim, teacher_temp=teacher_temp, student_temp=student_temp,
        ).to(self.device)
        self.ema_decay = ema_decay

    def _setup_mae(self, feature_dim, mask_ratio, decoder_dim, decoder_depth):
        self.mask_ratio = mask_ratio
        patch_size = 16
        img_size = 224
        num_patches = (img_size // patch_size) ** 2

        self.mae_decoder = MAEDecoder(
            encoder_dim=feature_dim, decoder_dim=decoder_dim,
            decoder_depth=decoder_depth, patch_size=patch_size,
            num_patches=num_patches,
        ).to(self.device)

    def train_step_dino(self, crops: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single DINO training step.

        Args:
            crops: List of augmented crop tensors. First 2 are global crops
                (used by the teacher), the rest are local crops.

        Returns:
            Dict with 'loss'.
        """
        crops = [c.to(self.device) for c in crops]
        global_crops = crops[:2]

        student_out = self.student(crops)
        with torch.no_grad():
            teacher_out = self.teacher(global_crops)

        loss = self.dino_loss(student_out, teacher_out)
        return {"loss": loss}

    @torch.no_grad()
    def update_teacher(self):
        """EMA update of teacher from student weights."""
        m = self.ema_decay
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)

    def train_step_mae(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single MAE training step.

        Args:
            images: [B, 3, H, W] input images.

        Returns:
            Dict with 'loss' (mean squared error on masked patches).
        """
        images = images.to(self.device)
        B, C, H, W = images.shape
        patch_size = self.mae_decoder.patch_size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w

        num_mask = int(num_patches * self.mask_ratio)
        num_vis = num_patches - num_mask

        # Random masking
        noise = torch.rand(B, num_patches, device=self.device)
        ids_shuffle = noise.argsort(dim=1)
        visible_indices = ids_shuffle[:, :num_vis]
        mask_indices = ids_shuffle[:, num_vis:]

        # Patchify the image for reconstruction targets
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, num_patches_h * num_patches_w, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(B, num_patches, -1)

        # Encode only visible patches (simplified: run full encoder, gather visible)
        if hasattr(self.backbone, "extract_features"):
            features = self.backbone.extract_features(images)
        else:
            out = self.backbone(images)
            features = out if isinstance(out, torch.Tensor) else out.get("features", out.get("logits"))

        if features.dim() == 2:
            features = features.unsqueeze(1).expand(-1, num_vis, -1)
        elif features.shape[1] > num_vis:
            features = features[:, :num_vis]

        pred = self.mae_decoder(features, visible_indices, mask_indices)

        target = torch.gather(patches, 1, mask_indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        loss = F.mse_loss(pred, target)

        return {"loss": loss}

    def train_step(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Dispatch to the appropriate method's training step."""
        if self.method == "dino":
            return self.train_step_dino(*args, **kwargs)
        return self.train_step_mae(*args, **kwargs)
