"""Logging utilities."""

import os
import sys
import logging
from datetime import datetime


def setup_logger(
    name: str = "FlashCls",
    save_dir: str = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    logger.handlers = []

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(save_dir, f"train_{ts}.log"))
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class AverageMeter:
    """Compute and store running average."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"
