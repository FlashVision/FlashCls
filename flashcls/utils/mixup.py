"""Mixup and CutMix augmentation for classification training."""

import numpy as np
import torch


class Mixup:
    """Apply Mixup and/or CutMix to a batch of images and labels.

    When both ``mixup_alpha`` and ``cutmix_alpha`` are > 0, one of the two
    is randomly chosen per batch with equal probability.

    Args:
        mixup_alpha: Beta distribution parameter for Mixup.
        cutmix_alpha: Beta distribution parameter for CutMix.
        num_classes: Number of classes (for one-hot targets).
        prob: Probability of applying any augmentation.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        num_classes: int = 10,
        prob: float = 1.0,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes
        self.prob = prob

    def __call__(self, images: torch.Tensor, targets: torch.Tensor):
        """Apply Mixup or CutMix to a batch.

        Args:
            images: ``[B, C, H, W]`` batch of images.
            targets: ``[B]`` integer class labels.

        Returns:
            Tuple of (mixed_images, soft_targets) where soft_targets is
            ``[B, num_classes]``.
        """
        if np.random.random() > self.prob:
            return images, self._one_hot(targets)

        use_cutmix = False
        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            use_cutmix = np.random.random() > 0.5
        elif self.cutmix_alpha > 0:
            use_cutmix = True

        if use_cutmix:
            return self._cutmix(images, targets)
        return self._mixup(images, targets)

    def _mixup(self, images, targets):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        mixed = lam * images + (1 - lam) * images[index]
        targets_one_hot = self._one_hot(targets)
        mixed_targets = lam * targets_one_hot + (1 - lam) * targets_one_hot[index]
        return mixed, mixed_targets

    def _cutmix(self, images, targets):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        _, _, h, w = images.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        cy = np.random.randint(h)
        cx = np.random.randint(w)

        y1 = max(0, cy - cut_h // 2)
        y2 = min(h, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(w, cx + cut_w // 2)

        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
        targets_one_hot = self._one_hot(targets)
        mixed_targets = lam_actual * targets_one_hot + (1 - lam_actual) * targets_one_hot[index]
        return mixed, mixed_targets

    def _one_hot(self, targets):
        return torch.zeros(targets.size(0), self.num_classes,
                           device=targets.device).scatter_(1, targets.unsqueeze(1), 1.0)
