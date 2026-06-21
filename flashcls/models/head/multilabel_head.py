"""
Multi-Label Classification Head.

Supports:
  - Sigmoid output for independent per-class predictions
  - Asymmetric Loss for handling class imbalance in multi-label settings
  - Per-class learnable or fixed thresholds for inference

Reference:
    Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification",
    ICCV 2021.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashcls.registry import HEADS

logger = logging.getLogger(__name__)


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Applies different focusing parameters (gamma) for positive and negative
    samples, combined with probability shifting (clipping) for negatives
    to reduce the contribution of easy negatives.

    Args:
        gamma_pos: Focusing parameter for positive samples.
        gamma_neg: Focusing parameter for negative samples.
        clip: Probability margin for negative shifting.
        eps: Epsilon for numerical stability.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw logits (before sigmoid).
            targets: [B, C] binary multi-label targets.

        Returns:
            Scalar loss.
        """
        probs = torch.sigmoid(logits)

        # Probability shifting for negatives
        probs_neg = probs
        if self.clip > 0:
            probs_neg = (probs - self.clip).clamp(min=0)

        # Positive and negative log-likelihoods
        log_pos = torch.log(probs.clamp(min=self.eps))
        log_neg = torch.log((1 - probs_neg).clamp(min=self.eps))

        # Asymmetric focusing
        if self.gamma_pos > 0:
            pos_weight = (1 - probs) ** self.gamma_pos
            log_pos = log_pos * pos_weight

        if self.gamma_neg > 0:
            neg_weight = probs_neg ** self.gamma_neg
            log_neg = log_neg * neg_weight

        loss = -(targets * log_pos + (1 - targets) * log_neg)
        return loss.mean()


class AsymmetricLossOptimized(nn.Module):
    """Numerically stable version of Asymmetric Loss using logsigmoid.

    More stable for mixed-precision training.

    Args:
        gamma_pos: Focusing parameter for positive samples.
        gamma_neg: Focusing parameter for negative samples.
        clip: Probability margin for negative shifting.
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits (numerically stable)
        bce_pos = F.logsigmoid(logits)
        bce_neg = F.logsigmoid(-logits)

        probs = torch.sigmoid(logits).detach()

        # Focusing weights
        if self.gamma_pos > 0:
            pos_weight = (1 - probs) ** self.gamma_pos
        else:
            pos_weight = 1.0

        if self.gamma_neg > 0:
            neg_probs = probs
            if self.clip > 0:
                neg_probs = (probs - self.clip).clamp(min=0)
            neg_weight = neg_probs ** self.gamma_neg
        else:
            neg_weight = 1.0

        loss = -(targets * pos_weight * bce_pos + (1 - targets) * neg_weight * bce_neg)
        return loss.mean()


@HEADS.register("MultiLabelHead")
class MultiLabelHead(nn.Module):
    """Multi-label classification head with sigmoid output.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of label classes.
        dropout: Dropout probability before the linear layer.
        loss_type: Loss function type ('bce', 'asl', 'asl_optimized').
        gamma_pos: ASL positive focusing parameter.
        gamma_neg: ASL negative focusing parameter.
        asl_clip: ASL probability clipping margin.
        init_thresholds: Initial per-class thresholds for inference.
        learnable_thresholds: Whether thresholds are learnable parameters.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.1,
        loss_type: str = "asl",
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        init_thresholds: float = 0.5,
        learnable_thresholds: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

        if loss_type == "asl":
            self.criterion = AsymmetricLoss(gamma_pos, gamma_neg, asl_clip)
        elif loss_type == "asl_optimized":
            self.criterion = AsymmetricLossOptimized(gamma_pos, gamma_neg, asl_clip)
        else:
            self.criterion = None

        if learnable_thresholds:
            self.thresholds = nn.Parameter(
                torch.full((num_classes,), init_thresholds)
            )
        else:
            self.register_buffer(
                "thresholds",
                torch.full((num_classes,), init_thresholds),
            )

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Feature tensor [B, C, H, W] or [B, C] (pre-pooled).
            targets: Optional [B, num_classes] multi-hot targets.

        Returns:
            Dict with 'logits' and optionally 'loss'.
        """
        if x.dim() == 4:
            x = self.gap(x).flatten(1)
        x = self.drop(x)
        logits = self.fc(x)

        out: Dict[str, torch.Tensor] = {"logits": logits}

        if targets is not None:
            if self.criterion is not None:
                out["loss"] = self.criterion(logits, targets.float())
            else:
                out["loss"] = F.binary_cross_entropy_with_logits(logits, targets.float())

        return out

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        class_names: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """Predict multi-label classifications.

        Args:
            x: Feature tensor [B, C, H, W] or [B, C].
            class_names: Optional human-readable class names.

        Returns:
            List of lists of (class_name, probability) for active labels.
        """
        out = self.forward(x)
        probs = torch.sigmoid(out["logits"])
        active = probs > self.thresholds.unsqueeze(0)

        results = []
        for b in range(probs.shape[0]):
            preds = []
            for c in active[b].nonzero(as_tuple=False).squeeze(-1):
                c = c.item()
                name = class_names[c] if class_names and c < len(class_names) else str(c)
                preds.append((name, probs[b, c].item()))
            preds.sort(key=lambda x: x[1], reverse=True)
            results.append(preds)
        return results

    def set_thresholds(self, thresholds: torch.Tensor):
        """Update per-class thresholds."""
        if isinstance(self.thresholds, nn.Parameter):
            self.thresholds.data.copy_(thresholds)
        else:
            self.thresholds.copy_(thresholds)

    def optimal_thresholds(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """Find optimal per-class thresholds via grid search on F1 score.

        Args:
            logits: [N, C] model logits.
            targets: [N, C] ground-truth multi-hot labels.

        Returns:
            [C] optimal threshold per class.
        """
        probs = torch.sigmoid(logits)
        best_thresholds = torch.full((self.num_classes,), 0.5, device=logits.device)

        for c in range(self.num_classes):
            best_f1 = 0.0
            for t_val in torch.linspace(0.1, 0.9, num_steps):
                preds = (probs[:, c] > t_val).float()
                tp = (preds * targets[:, c]).sum()
                fp = (preds * (1 - targets[:, c])).sum()
                fn = ((1 - preds) * targets[:, c]).sum()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds[c] = t_val

        return best_thresholds
