"""
DataLoader creation for classification datasets.
"""

import logging
from typing import List, Optional, Tuple

from torch.utils.data import DataLoader

from .dataset import ClassificationDataset
from .transforms import TrainTransform, ValTransform

logger = logging.getLogger(__name__)


def create_dataloader(
    data_dir: str,
    batch_size: int = 64,
    input_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    is_train: bool = True,
    class_names: Optional[List[str]] = None,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    auto_augment: Optional[str] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for classification.

    Args:
        data_dir: Path to ImageFolder-format directory.
        batch_size: Batch size.
        input_size: Input image size (H, W).
        num_workers: Number of data loading workers.
        is_train: If True, apply training augmentations and shuffle.
        class_names: Optional explicit class ordering.
        mean: Normalization mean.
        std: Normalization std.
        auto_augment: AutoAugment policy name.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        DataLoader instance.
    """
    if is_train:
        transform = TrainTransform(
            input_size=input_size,
            mean=mean,
            std=std,
            auto_augment=auto_augment,
        )
    else:
        transform = ValTransform(
            input_size=input_size,
            mean=mean,
            std=std,
        )

    dataset = ClassificationDataset(
        root_dir=data_dir,
        transform=transform,
        class_names=class_names,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
    )

    logger.info(
        "DataLoader: %d samples, batch=%d, workers=%d, shuffle=%s",
        len(dataset), batch_size, num_workers, is_train,
    )

    return loader
