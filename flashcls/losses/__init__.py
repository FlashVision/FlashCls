from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .kd_loss import ClassificationKDLoss

__all__ = [
    "LabelSmoothingCrossEntropy",
    "SoftTargetCrossEntropy",
    "ClassificationKDLoss",
]
