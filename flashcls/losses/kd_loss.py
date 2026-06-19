"""Knowledge Distillation loss for classification.

KL divergence between teacher and student softened logits
(Hinton et al., 2015).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationKDLoss(nn.Module):
    """KL-divergence logit distillation for classification.

    Matches the teacher's soft probability distribution using
    temperature-scaled KL divergence.

    Args:
        temperature: Softmax temperature for KL divergence.
        alpha: Weight for KD loss (vs hard-label loss).
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined KD + hard-label loss.

        Args:
            student_logits: ``[B, C]`` student output logits.
            teacher_logits: ``[B, C]`` teacher output logits (detached).
            targets: ``[B]`` integer class labels.

        Returns:
            Scalar loss.
        """
        T = self.temperature

        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

        hard_loss = F.cross_entropy(student_logits, targets)

        return self.alpha * kd_loss + (1.0 - self.alpha) * hard_loss
