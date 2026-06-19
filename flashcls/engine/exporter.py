"""FlashCls Exporter — export models to ONNX."""

import os
import logging
from typing import Optional, Tuple

import torch

from flashcls.cfg import get_config
from flashcls.models.classifier import FlashClassifier

logger = logging.getLogger(__name__)


class Exporter:
    """Export a FlashClassifier model to ONNX format.

    Example::

        from flashcls import Exporter

        exporter = Exporter(model_path="workspace/model_best_inference.pth")
        exporter.export_onnx("model.onnx")
    """

    def __init__(
        self,
        model_path: str,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        self.model_path = model_path

        cfg = get_config()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        backbone_size = cfg.model.backbone_size
        num_classes = cfg.model.num_classes
        inp_size = cfg.model.input_size

        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
            num_classes = ckpt_cfg.get("num_classes", num_classes)
            inp_size = ckpt_cfg.get("input_size", inp_size)

        if input_size is not None:
            inp_size = input_size
        if isinstance(inp_size, int):
            inp_size = (inp_size, inp_size)

        self.input_size = inp_size
        self.num_classes = num_classes

        self.model = FlashClassifier(
            num_classes=num_classes,
            input_size=inp_size,
            backbone_size=backbone_size,
            pretrained=False,
        )

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model loaded: %s parameters", f"{total_params:,}")

    def export(self, output="model.onnx", simplify=True, **kwargs) -> str:
        return self.export_onnx(output_path=output, simplify=simplify, **kwargs)

    def export_onnx(
        self,
        output_path: str = "model.onnx",
        opset_version: int = 11,
        simplify: bool = True,
        dynamic_batch: bool = True,
    ) -> str:
        """Export model to ONNX format."""
        inp_h, inp_w = self.input_size
        dummy_input = torch.randn(1, 3, inp_h, inp_w)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

        class _Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model(x)["logits"]

        wrapper = _Wrapper(self.model)

        torch.onnx.export(
            wrapper, dummy_input, output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=True,
        )
        logger.info("ONNX exported: %s", output_path)

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                onnx_model = onnx.load(output_path)
                simplified, _ = onnx_simplify(onnx_model)
                onnx.save(simplified, output_path)
                logger.info("ONNX model simplified successfully")
            except ImportError:
                logger.warning("onnxsim not installed, skipping simplification")

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info("Output: %s (%.2f MB)", output_path, file_size)
        return output_path
