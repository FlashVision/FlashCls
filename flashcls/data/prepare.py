"""
Dataset preparation utilities.

Verify dataset structure, convert formats, and split datasets.
"""

import os
import shutil
import random
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def verify_dataset(data_dir: str) -> bool:
    """Verify that a directory has ImageFolder structure.

    Args:
        data_dir: Path to dataset directory.

    Returns:
        True if valid ImageFolder structure found.
    """
    if not os.path.isdir(data_dir):
        logger.error("Directory does not exist: %s", data_dir)
        return False

    subdirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    if not subdirs:
        logger.error("No class subdirectories found in %s", data_dir)
        return False

    total_images = 0
    for subdir in subdirs:
        class_dir = os.path.join(data_dir, subdir)
        images = [
            f for f in os.listdir(class_dir)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ]
        total_images += len(images)
        if not images:
            logger.warning("Class directory '%s' has no images", subdir)

    if total_images == 0:
        logger.error("No images found in %s", data_dir)
        return False

    logger.info(
        "Dataset verified: %d classes, %d images in %s",
        len(subdirs), total_images, data_dir,
    )
    return True


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    copy: bool = True,
) -> Tuple[str, str]:
    """Split an ImageFolder dataset into train/val splits.

    Args:
        source_dir: Path to source ImageFolder directory.
        output_dir: Path to output directory (will create train/ and val/ inside).
        train_ratio: Fraction of data for training.
        seed: Random seed for reproducibility.
        copy: If True, copy files. If False, move them.

    Returns:
        Tuple of (train_dir, val_dir) paths.
    """
    random.seed(seed)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    classes = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])

    for class_name in classes:
        src_class_dir = os.path.join(source_dir, class_name)
        images = sorted([
            f for f in os.listdir(src_class_dir)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ])

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create output directories
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        transfer_fn = shutil.copy2 if copy else shutil.move

        for img in train_images:
            transfer_fn(os.path.join(src_class_dir, img), os.path.join(train_class_dir, img))
        for img in val_images:
            transfer_fn(os.path.join(src_class_dir, img), os.path.join(val_class_dir, img))

    logger.info(
        "Dataset split: %d classes, train=%s, val=%s",
        len(classes), train_dir, val_dir,
    )
    return train_dir, val_dir


def convert_flat_to_imagefolder(
    source_dir: str,
    output_dir: str,
    labels_file: str,
    delimiter: str = ",",
) -> str:
    """Convert a flat directory with a labels file to ImageFolder format.

    Args:
        source_dir: Directory containing all images.
        output_dir: Output ImageFolder directory.
        labels_file: Path to CSV/TSV with columns: filename, class_name.
        delimiter: Delimiter for the labels file.

    Returns:
        Path to output directory.
    """
    with open(labels_file, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("filename"):
            continue
        parts = line.split(delimiter)
        if len(parts) < 2:
            continue
        filename, class_name = parts[0].strip(), parts[1].strip()

        src_path = os.path.join(source_dir, filename)
        if not os.path.isfile(src_path):
            continue

        dst_dir = os.path.join(output_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_path, os.path.join(dst_dir, filename))

    logger.info("Converted flat dataset to ImageFolder at %s", output_dir)
    return output_dir
