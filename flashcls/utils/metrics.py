"""Classification evaluation metrics."""

from typing import Dict, Tuple

import torch
import numpy as np


def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[float, ...]:
    """Compute top-k accuracy for the given logits and targets.

    Args:
        logits: ``[B, C]`` unnormalized class scores.
        targets: ``[B]`` ground-truth class indices.
        topk: Tuple of k values.

    Returns:
        Tuple of accuracy percentages for each k.
    """
    maxk = min(max(topk), logits.size(1))
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        k = min(k, logits.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append((correct_k / batch_size * 100.0).item())
    return tuple(result)


def per_class_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[int, float]:
    """Compute per-class accuracy.

    Returns:
        Dict mapping class_id to accuracy percentage.
    """
    preds = logits.argmax(dim=1)
    result = {}
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() == 0:
            continue
        correct = (preds[mask] == c).float().sum()
        result[c] = (correct / mask.sum() * 100.0).item()
    return result


def confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix.

    Returns:
        ``[C, C]`` confusion matrix where ``[i, j]`` is the count of
        samples with true label ``i`` predicted as label ``j``.
    """
    preds = logits.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm
