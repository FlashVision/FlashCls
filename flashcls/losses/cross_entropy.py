"""Cross-entropy loss variants for classification.

- LabelSmoothingCrossEntropy: standard label smoothing
- SoftTargetCrossEntropy: for mixup/cutmix soft targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing.

    Args:
        smoothing: Label smoothing factor in ``[0, 1)``.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.size(-1)

        # One-hot smooth targets
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.sum(dim=-1) / num_classes

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy loss for soft targets (from mixup/cutmix).

    Expects targets as probability distributions ``[B, C]``.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1)
        return loss.mean()
