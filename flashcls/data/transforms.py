"""
Image transforms for training, validation, and inference.
"""

from typing import List, Tuple

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrainTransform:
    """Training transform with data augmentation.

    Applies: RandomResizedCrop, HorizontalFlip, ColorJitter, Normalize.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        mean: List[float] = None,
        std: List[float] = None,
        color_jitter: Tuple[float, ...] = (0.4, 0.4, 0.4, 0.1),
        hflip_prob: float = 0.5,
        auto_augment: str = None,
    ):
        mean = mean or IMAGENET_MEAN
        std = std or IMAGENET_STD

        augments = [
            transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
        ]

        if hflip_prob > 0:
            augments.append(transforms.RandomHorizontalFlip(p=hflip_prob))

        if auto_augment:
            augments.append(transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.IMAGENET
            ))
        elif color_jitter:
            augments.append(transforms.ColorJitter(*color_jitter))

        augments.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.transform = transforms.Compose(augments)

    def __call__(self, img):
        return self.transform(img)


class ValTransform:
    """Validation transform: Resize + CenterCrop + Normalize."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        mean: List[float] = None,
        std: List[float] = None,
        resize_ratio: float = 256 / 224,
    ):
        mean = mean or IMAGENET_MEAN
        std = std or IMAGENET_STD
        resize_size = int(input_size[0] * resize_ratio)

        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transform(img)


class InferenceTransform:
    """Inference transform: same as validation but returns PIL-compatible output."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        mean: List[float] = None,
        std: List[float] = None,
    ):
        mean = mean or IMAGENET_MEAN
        std = std or IMAGENET_STD
        resize_size = int(input_size[0] * 256 / 224)

        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transform(img)
