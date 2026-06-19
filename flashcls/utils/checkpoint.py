"""Checkpoint utilities for saving and loading classification models."""

import os
from typing import Dict

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    metrics: Dict = None,
    scheduler=None,
    config: Dict = None,
    ema=None,
) -> str:
    """Save training checkpoint with model config embedded."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics or {},
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()

    if config is not None:
        checkpoint["config"] = config
    elif hasattr(model, "num_classes"):
        checkpoint["config"] = {
            "num_classes": getattr(model, "num_classes", 10),
            "input_size": getattr(model, "input_size", (224, 224)),
            "backbone_size": getattr(model, "backbone_size", "1.0x"),
            "class_names": getattr(model, "class_names", []),
        }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")
    return save_path


def save_inference_weights(
    model: torch.nn.Module,
    save_path: str,
    config: Dict = None,
    half: bool = False,
) -> str:
    """Save inference-only weights, optionally as FP16."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    state_dict = model.state_dict()
    if half:
        state_dict = {
            k: v.half() if v.dtype == torch.float32 else v
            for k, v in state_dict.items()
        }

    checkpoint = {"model_state_dict": state_dict, "half": half}

    if config is not None:
        checkpoint["config"] = config
    elif hasattr(model, "num_classes"):
        checkpoint["config"] = {
            "num_classes": getattr(model, "num_classes", 10),
            "input_size": getattr(model, "input_size", (224, 224)),
            "backbone_size": getattr(model, "backbone_size", "1.0x"),
            "class_names": getattr(model, "class_names", []),
        }

    torch.save(checkpoint, save_path)
    size_mb = os.path.getsize(save_path) / 1e6
    precision = "FP16" if half else "FP32"
    print(f"Inference weights saved: {save_path} ({size_mb:.2f} MB, {precision})")
    return save_path


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device="cuda"):
    """Load training checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "state_dict" in checkpoint:
        sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    print(f"Model loaded from: {checkpoint_path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, KeyError):
            print("  Optimizer state skipped (architecture mismatch).")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (ValueError, KeyError):
            print("  Scheduler state skipped.")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "metrics": checkpoint.get("metrics", {}),
    }
