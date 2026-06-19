"""Visualization utilities for classification."""

import colorsys
from typing import Dict, List, Tuple

import cv2
import numpy as np


def make_color_palette(n: int) -> Dict[int, Tuple[int, int, int]]:
    """Generate *n* visually distinct BGR colors using HSV spacing."""
    palette = {}
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.90)
        palette[i] = (int(b * 255), int(g * 255), int(r * 255))
    return palette


def draw_classification(
    image: np.ndarray,
    class_name: str,
    probability: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 1.0,
) -> np.ndarray:
    """Draw classification label on an image.

    Args:
        image: BGR image.
        class_name: Predicted class name.
        probability: Confidence probability.
        position: Text position ``(x, y)``.
        color: BGR color for the text background.
        font_scale: Font size scale.

    Returns:
        Annotated image copy.
    """
    output = image.copy()
    text = f"{class_name}: {probability:.2f}"
    thickness = max(1, int(font_scale * 1.5))

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    pad = 6

    cv2.rectangle(output, (x, y - th - pad), (x + tw + pad * 2, y + pad), color, -1)
    cv2.putText(output, text, (x + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
):
    """Plot and save a confusion matrix.

    Args:
        cm: ``[C, C]`` confusion matrix.
        class_names: List of class names.
        save_path: Path to save the image.
        title: Plot title.
        figsize: Figure size in inches.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for confusion matrix plots: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")
