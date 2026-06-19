"""FlashCls Predictor — classify images."""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

from flashcls.cfg import get_config
from flashcls.models.classifier import FlashClassifier
from flashcls.data.transforms import InferenceTransform

logger = logging.getLogger(__name__)


class Predictor:
    """High-level inference wrapper for FlashCls.

    Example::

        from flashcls import Predictor

        pred = Predictor(model_path="workspace/model_best_inference.pth")
        results = pred.classify("test.jpg")
        for cls_name, prob in results:
            print(f"{cls_name}: {prob:.4f}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        input_size: int = 224,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if model_path is None:
            raise ValueError("model_path is required")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        cfg = get_config()
        backbone_size = cfg.model.backbone_size
        num_classes = cfg.model.num_classes
        inp_size = cfg.model.input_size
        class_names = list(cfg.class_names) if cfg.class_names else None

        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
            num_classes = ckpt_cfg.get("num_classes", num_classes)
            inp_size = ckpt_cfg.get("input_size", inp_size)
            if "class_names" in ckpt_cfg and ckpt_cfg["class_names"]:
                class_names = ckpt_cfg["class_names"]

        if isinstance(inp_size, int):
            inp_size = (inp_size, inp_size)
        self.input_size = inp_size
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        self.model = FlashClassifier(
            num_classes=num_classes,
            input_size=inp_size,
            backbone_size=backbone_size,
            pretrained=False,
            class_names=self.class_names,
        )

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model = self.model.to(self.device).eval()
        self.transform = InferenceTransform(input_size=self.input_size)

    @torch.no_grad()
    def classify(self, image) -> List[Tuple[str, float]]:
        """Classify a single image.

        Args:
            image: PIL Image or path to image file.

        Returns:
            Sorted list of ``(class_name, probability)`` tuples.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        results = self.model.predict(tensor, top_k=len(self.class_names))
        return results[0]

    def predict(self, source, output_dir: Optional[str] = None) -> List:
        """Classify image(s) from a path or directory.

        Args:
            source: Path to image or directory.
            output_dir: If set, saves annotated output here.

        Returns:
            Classification results.
        """
        source = str(source)
        if os.path.isdir(source):
            return self._predict_directory(source, output_dir)
        return self.classify(source)

    def _predict_directory(self, dir_path: str, output_dir: Optional[str] = None) -> List:
        all_results = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for path in sorted(Path(dir_path).glob(ext)):
                result = self.classify(str(path))
                top_cls, top_prob = result[0]
                logger.info("%s → %s (%.4f)", path.name, top_cls, top_prob)
                all_results.append((str(path), result))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "predictions.csv"), "w") as f:
                f.write("filename,predicted_class,probability\n")
                for path, preds in all_results:
                    cls, prob = preds[0]
                    f.write(f"{Path(path).name},{cls},{prob:.6f}\n")
            logger.info("Predictions saved to %s/predictions.csv", output_dir)

        return all_results
