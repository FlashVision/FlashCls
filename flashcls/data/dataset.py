"""Classification dataset — ImageFolder: class subfolders with images."""

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class ClassificationDataset(Dataset):
    """ImageFolder-style classification dataset.

    Expects a directory with one sub-folder per class::

        root_dir/
            cat/
                img001.jpg
                img002.jpg
            dog/
                img003.jpg
                ...

    Args:
        root_dir: Path to the root directory.
        transform: Callable transform applied to each PIL image.
        class_names: Optional explicit class ordering. If not provided,
            classes are sorted alphabetically from sub-folder names.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform

        if class_names:
            self.class_names = class_names
        else:
            self.class_names = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self.samples: List[Tuple[str, int]] = []
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = self.class_to_idx[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((os.path.join(class_dir, fname), label))

        logger.info(
            "Loaded %d images across %d classes from %s",
            len(self.samples), self.num_classes, root_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
