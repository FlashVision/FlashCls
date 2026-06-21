"""
Knowledge Distillation for FlashCls.

Teacher-student framework supporting:
  - Logit distillation via KL divergence (Hinton et al., 2015)
  - Feature matching between teacher and student intermediate representations
  - Hybrid loss combining task loss, KD loss, and feature matching loss

Reference:
    Hinton et al., "Distilling the Knowledge in a Neural Network", 2015.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """Combined knowledge distillation loss.

    Supports logit distillation (KL divergence on softened probabilities),
    feature matching (L2 loss between projected feature maps), and a
    standard task loss, all weighted by configurable coefficients.

    Args:
        temperature: Softening temperature for KL divergence.
        alpha: Weight of the KD loss relative to the task loss.
            Final loss = alpha * kd_loss + (1 - alpha) * task_loss + beta * feat_loss.
        beta: Weight of feature matching loss.
        feature_dims: If provided, (student_dim, teacher_dim) for the
            linear projection used in feature matching.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.0,
        feature_dims: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

        self.projector = None
        if feature_dims is not None:
            student_dim, teacher_dim = feature_dims
            self.projector = nn.Sequential(
                nn.Linear(student_dim, teacher_dim),
                nn.ReLU(inplace=True),
                nn.Linear(teacher_dim, teacher_dim),
            )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined distillation loss.

        Args:
            student_logits: [B, C] student model logits.
            teacher_logits: [B, C] teacher model logits (detached).
            targets: [B] ground-truth class indices.
            student_features: Optional [B, D_s] student features.
            teacher_features: Optional [B, D_t] teacher features.

        Returns:
            Dict with 'loss', 'kd_loss', 'task_loss', and optionally 'feat_loss'.
        """
        T = self.temperature

        # KL divergence on soft targets
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        # Standard task loss
        task_loss = F.cross_entropy(student_logits, targets)

        loss = self.alpha * kd_loss + (1 - self.alpha) * task_loss

        result = {
            "kd_loss": kd_loss.detach(),
            "task_loss": task_loss.detach(),
        }

        # Feature matching loss
        if (
            self.beta > 0
            and student_features is not None
            and teacher_features is not None
        ):
            if self.projector is not None:
                student_features = self.projector(student_features)

            # L2 normalise then compute cosine distance
            s_norm = F.normalize(student_features, dim=-1)
            t_norm = F.normalize(teacher_features.detach(), dim=-1)
            feat_loss = (1 - (s_norm * t_norm).sum(dim=-1)).mean()

            loss = loss + self.beta * feat_loss
            result["feat_loss"] = feat_loss.detach()

        result["loss"] = loss
        return result


class DistillationTrainer:
    """High-level trainer wrapper for knowledge distillation.

    Manages teacher model loading, freezing, and the training loop that
    uses both teacher and student models.

    Args:
        student: Student model (nn.Module with forward returning dict with 'logits').
        teacher: Teacher model (same interface, larger capacity).
        temperature: KD temperature.
        alpha: KD loss weight.
        beta: Feature matching loss weight.
        feature_dims: Optional (student_dim, teacher_dim) for feature projection.
        device: Target device.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.0,
        feature_dims: Optional[Tuple[int, int]] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.criterion = DistillationLoss(
            temperature=temperature, alpha=alpha, beta=beta, feature_dims=feature_dims,
        ).to(self.device)

        logger.info(
            "DistillationTrainer initialised: T=%.1f, alpha=%.2f, beta=%.2f",
            temperature, alpha, beta,
        )

    def train_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Single training step returning loss dict.

        The caller is responsible for optimizer.zero_grad / loss.backward / step.
        """
        images = images.to(self.device)
        targets = targets.to(self.device)

        student_out = self.student(images)
        with torch.no_grad():
            teacher_out = self.teacher(images)

        student_logits = student_out["logits"]
        teacher_logits = teacher_out["logits"]

        student_features = student_out.get("features")
        teacher_features = teacher_out.get("features")

        return self.criterion(
            student_logits, teacher_logits, targets,
            student_features, teacher_features,
        )

    @classmethod
    def from_checkpoint(
        cls,
        student: nn.Module,
        teacher_path: str,
        teacher_builder_fn,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.0,
        feature_dims: Optional[Tuple[int, int]] = None,
        device: str = "cuda",
    ) -> "DistillationTrainer":
        """Create a DistillationTrainer by loading the teacher from a checkpoint.

        Args:
            student: The student model.
            teacher_path: Path to teacher checkpoint (.pth).
            teacher_builder_fn: Callable that returns a fresh teacher nn.Module.
            temperature: KD temperature.
            alpha: KD loss weight.
            beta: Feature matching weight.
            feature_dims: Feature dimensions for projection.
            device: Target device.
        """
        teacher = teacher_builder_fn()
        ckpt = torch.load(teacher_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        teacher.load_state_dict(sd, strict=False)
        logger.info("Teacher loaded from %s", teacher_path)

        return cls(
            student=student, teacher=teacher,
            temperature=temperature, alpha=alpha, beta=beta,
            feature_dims=feature_dims, device=device,
        )
