"""FlashCls Validator — compute Top-1/Top-5 accuracy on a validation set."""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from flashcls.cfg import get_config
from flashcls.models.classifier import FlashClassifier
from flashcls.data import create_dataloader
from flashcls.utils.metrics import top_k_accuracy, per_class_accuracy

logger = logging.getLogger(__name__)


class Validator:
    """Validate a FlashClassifier model on an ImageFolder dataset.

    Example::

        from flashcls import Validator

        val = Validator(model_path="workspace/checkpoint_best.pth", val_dir="data/val")
        results = val.validate()
        print(f"Top-1: {results['top1']:.2f}%")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        device: str = "cuda",
        batch_size: int = 64,
        workers: int = 4,
        input_size: int = 224,
        val_dir: Optional[str] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.workers = workers
        self.input_size = (input_size, input_size)

        cfg = get_config()
        self.val_dir = val_dir or cfg.data.val_dir

        if model is not None:
            self.model = model.to(self.device)
            self.class_names = list(cfg.class_names) if cfg.class_names else []
        elif model_path is not None:
            self.model, self.class_names = self._load_model(model_path, cfg)
        else:
            raise ValueError("Either model_path or model must be provided")

    def _load_model(self, model_path, cfg):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        backbone_size = cfg.model.backbone_size
        num_classes = cfg.model.num_classes
        class_names = list(cfg.class_names) if cfg.class_names else []

        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
            num_classes = ckpt_cfg.get("num_classes", num_classes)
            if "class_names" in ckpt_cfg and ckpt_cfg["class_names"]:
                class_names = ckpt_cfg["class_names"]

        model = FlashClassifier(
            num_classes=num_classes,
            input_size=self.input_size,
            backbone_size=backbone_size,
            pretrained=False,
            class_names=class_names,
        )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(self.device).eval()
        return model, class_names

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return accuracy metrics.

        Returns:
            Dict with keys: ``top1``, ``top5``, ``val_loss``, ``per_class_acc``.
        """
        self.model.eval()

        val_loader = create_dataloader(
            data_dir=self.val_dir,
            batch_size=self.batch_size,
            input_size=self.input_size,
            num_workers=self.workers,
            is_train=False,
            class_names=self.class_names if self.class_names else None,
        )

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_samples = 0

        for images, targets in val_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            output = self.model(images)
            logits = output["logits"]
            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

            all_preds.append(logits.cpu())
            all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        top1, top5 = top_k_accuracy(all_preds, all_targets, topk=(1, 5))
        avg_loss = total_loss / max(n_samples, 1)

        num_classes = self.model.num_classes
        per_cls_acc = per_class_accuracy(all_preds, all_targets, num_classes)

        result = {
            "top1": top1,
            "top5": top5,
            "val_loss": avg_loss,
            "per_class_acc": per_cls_acc,
        }

        logger.info("Validation: Top-1 = %.2f%%, Top-5 = %.2f%%, Loss = %.4f",
                     top1, top5, avg_loss)

        if self.class_names and per_cls_acc:
            for cid, acc in sorted(per_cls_acc.items()):
                name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
                logger.info("  %s: %.2f%%", name, acc)

        return result
