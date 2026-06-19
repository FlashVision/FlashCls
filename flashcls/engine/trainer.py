"""FlashCls Trainer — wraps the full training loop into a reusable class."""

import os
import copy
import math
import logging
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from flashcls.cfg import get_config
from flashcls.models.classifier import FlashClassifier
from flashcls.models.lora import (
    apply_lora, apply_qlora, merge_lora_weights, get_lora_state_dict,
)
from flashcls.data import create_dataloader, verify_dataset
from flashcls.losses import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from flashcls.utils import (
    save_checkpoint, load_checkpoint, save_inference_weights, setup_logger, AverageMeter,
)
from flashcls.utils.metrics import top_k_accuracy
from flashcls.utils.mixup import Mixup
from flashcls.utils.torchtune_optim import (
    apply_activation_checkpointing,
    ActivationOffloadHook,
    create_optimizer,
    compile_model as torchtune_compile,
)

logger = logging.getLogger(__name__)


class ModelEMA:
    """Exponential Moving Average of model weights with adaptive decay warmup."""

    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup: int = 2000):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.target_decay = decay
        self.warmup = warmup
        self.num_updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @property
    def decay(self):
        return min(self.target_decay,
                   (1 + self.num_updates) / (self.warmup + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = self.decay
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self):
        return {
            "ema_state": self.ema.state_dict(),
            "target_decay": self.target_decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict):
        self.ema.load_state_dict(state["ema_state"], strict=False)
        self.target_decay = state.get("target_decay", self.target_decay)
        self.warmup = state.get("warmup", self.warmup)
        self.num_updates = state.get("num_updates", 0)


MODEL_SIZE_MAP = {
    "m": {"backbone": "1.0x"},
    "m-1.5x": {"backbone": "1.5x"},
    "m-0.5x": {"backbone": "0.5x"},
}


class Trainer:
    """High-level trainer for FlashCls.

    Example::

        from flashcls import Trainer

        trainer = Trainer(
            epochs=100,
            batch_size=64,
            model_size="m",
            train_dir="data/train",
            val_dir="data/val",
            lora=True,
        )
        trainer.train()
    """

    def __init__(
        self,
        # Basic training
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        workers: int = 4,
        save_dir: str = "workspace/classification",
        resume: Optional[str] = None,
        device: str = "cuda",
        warmup_epochs: int = 5,
        patience: int = 30,
        # Model
        model_size: str = "m",
        input_size: int = 224,
        num_classes: Optional[int] = None,
        dropout: float = 0.2,
        finetune: Optional[str] = None,
        # Data
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        # Augmentation
        label_smoothing: float = 0.1,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        auto_augment: Optional[str] = None,
        # Performance
        amp: bool = False,
        grad_accum: int = 1,
        # torchtune optimizations
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        optimizer_in_bwd: bool = False,
        use_8bit_optimizer: bool = False,
        compile: bool = False,
        # LoRA
        lora: bool = False,
        lora_variant: str = "standard",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: Optional[List[str]] = None,
        qlora: bool = False,
        qlora_dtype: str = "int8",
        # Knowledge Distillation
        kd: bool = False,
        kd_teacher_path: Optional[str] = None,
        kd_teacher_size: str = "m-1.5x",
        kd_temperature: float = 4.0,
        kd_alpha: float = 0.7,
        # Config override
        config: Any = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.workers = workers
        self.save_dir = save_dir
        self.resume = resume
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.model_size = model_size
        self.input_size = (input_size, input_size)
        self.num_classes = num_classes
        self.dropout = dropout
        self.finetune = finetune
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.class_names = class_names
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.auto_augment = auto_augment
        self.amp = amp
        self.grad_accum = max(1, grad_accum)
        self.activation_checkpointing = activation_checkpointing
        self.activation_offloading = activation_offloading
        self.optimizer_in_bwd = optimizer_in_bwd
        self.use_8bit_optimizer = use_8bit_optimizer
        self.compile = compile
        self.lora = lora
        self.lora_variant = lora_variant
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = lora_targets or ["backbone"]
        self.qlora = qlora
        self.qlora_dtype = qlora_dtype
        self.kd = kd
        self.kd_teacher_path = kd_teacher_path
        self.kd_teacher_size = kd_teacher_size
        self.kd_temperature = kd_temperature
        self.kd_alpha = kd_alpha

        self._config = config or get_config()
        self._model_cfg = MODEL_SIZE_MAP[self.model_size]

        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            if device not in ("cpu", ""):
                logger.warning("CUDA unavailable; falling back to CPU.")

        os.makedirs(self.save_dir, exist_ok=True)
        self._logger = setup_logger("FlashCls", self.save_dir)

    def train(self) -> Dict[str, float]:
        """Run the full training loop. Returns dict with best_top1 and best_loss."""
        cfg = self._config

        if self.train_dir:
            cfg.data.train_dir = self.train_dir
        if self.val_dir:
            cfg.data.val_dir = self.val_dir

        # Verify dataset
        if not verify_dataset(cfg.data.train_dir):
            raise FileNotFoundError(f"Training dataset not found at {cfg.data.train_dir}")
        if not verify_dataset(cfg.data.val_dir):
            raise FileNotFoundError(f"Validation dataset not found at {cfg.data.val_dir}")

        # Discover classes
        if self.class_names:
            class_names = self.class_names
        else:
            class_names = sorted([
                d for d in os.listdir(cfg.data.train_dir)
                if os.path.isdir(os.path.join(cfg.data.train_dir, d))
            ])

        num_classes = self.num_classes or len(class_names)

        self._logger.info("=" * 60)
        self._logger.info("FlashCls Training")
        self._logger.info("=" * 60)
        self._logger.info(f"Device: {self.device}")
        self._logger.info(f"Model: {self.model_size}, Input: {self.input_size}")
        self._logger.info(f"Epochs: {self.epochs}, Batch: {self.batch_size}, LR: {self.lr}")
        self._logger.info(f"Classes ({num_classes}): {class_names[:10]}{'...' if len(class_names) > 10 else ''}")

        # Data loaders
        train_loader = create_dataloader(
            data_dir=cfg.data.train_dir,
            batch_size=self.batch_size,
            input_size=self.input_size,
            num_workers=self.workers,
            is_train=True,
            class_names=class_names,
            auto_augment=self.auto_augment,
        )
        val_loader = create_dataloader(
            data_dir=cfg.data.val_dir,
            batch_size=self.batch_size,
            input_size=self.input_size,
            num_workers=self.workers,
            is_train=False,
            class_names=class_names,
        )

        # Build model
        model = FlashClassifier(
            num_classes=num_classes,
            input_size=self.input_size,
            backbone_size=self._model_cfg["backbone"],
            dropout=self.dropout,
            pretrained=True,
            class_names=class_names,
        ).to(self.device)

        # Apply LoRA / QLoRA
        model = self._apply_lora(model)

        # Fine-tune from checkpoint
        if self.finetune and not self.resume:
            ckpt = torch.load(self.finetune, map_location=self.device, weights_only=False)
            src_sd = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(src_sd, strict=False)
            self._logger.info(f"Fine-tune weights loaded from: {self.finetune}")

        # Log model info
        info = model.get_model_info()
        self._logger.info(f"Params: {info['total_params']:,} ({info['params_mb']:.2f} MB)")

        # Loss function
        mixup_fn = None
        if self.mixup_alpha > 0 or self.cutmix_alpha > 0:
            mixup_fn = Mixup(
                mixup_alpha=self.mixup_alpha,
                cutmix_alpha=self.cutmix_alpha,
                num_classes=num_classes,
            )
            criterion = SoftTargetCrossEntropy()
        else:
            criterion = LabelSmoothingCrossEntropy(smoothing=self.label_smoothing)

        # KD teacher
        teacher_model = None
        if self.kd and self.kd_teacher_path:
            teacher_model = self._load_teacher(num_classes, class_names)

        # AMP
        scaler = None
        if self.amp and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            self._logger.info("AMP enabled")

        # torchtune optimizations
        if self.activation_checkpointing:
            apply_activation_checkpointing(model)
        offload_hook = None
        if self.activation_offloading:
            offload_hook = ActivationOffloadHook()
            offload_hook.register(model)
        if self.compile:
            model = torchtune_compile(model)

        # Optimizer
        optimizer = create_optimizer(
            model, lr=self.lr, weight_decay=0.05,
            use_8bit=self.use_8bit_optimizer, optimizer_in_bwd=self.optimizer_in_bwd,
        )

        # LR schedule (warmup + cosine)
        eta_min = 1e-5
        eta_min_factor = eta_min / self.lr

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_factor + (1.0 - eta_min_factor) * cosine

        scheduler = None
        if not self.optimizer_in_bwd:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # EMA
        ema = ModelEMA(model, decay=0.9998, warmup=2000)

        # Resume
        start_epoch = 0
        best_loss = float("inf")
        best_top1 = 0.0

        if self.resume:
            ckpt = load_checkpoint(model, self.resume, optimizer, scheduler, self.device)
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt.get("loss", float("inf"))
            raw_ckpt = torch.load(self.resume, map_location=self.device, weights_only=False)
            if raw_ckpt and "ema_state_dict" in raw_ckpt:
                ema.load_state_dict(raw_ckpt["ema_state_dict"])
            else:
                ema = ModelEMA(model, decay=0.9998, warmup=2000)
            self._logger.info(f"Resumed from epoch {start_epoch}")

        model_config = {
            "num_classes": num_classes,
            "input_size": self.input_size,
            "backbone_size": self._model_cfg["backbone"],
            "dropout": self.dropout,
            "class_names": class_names,
        }

        # Training loop
        self._logger.info("\nStarting training...")
        epochs_without_improvement = 0

        for epoch in range(start_epoch, self.epochs):
            if self.optimizer_in_bwd:
                lr_factor = lr_lambda(epoch)
                current_lr = self.lr * lr_factor
                optimizer.set_lr(current_lr)
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            self._logger.info(f"\nEpoch {epoch + 1}/{self.epochs} (lr={current_lr:.6f})")

            train_loss = self._train_one_epoch(
                model, train_loader, optimizer, criterion, ema, scaler,
                mixup_fn, teacher_model,
            )

            # Validate
            if (epoch + 1) % cfg.train.val_interval == 0:
                val_loss, top1, top5 = self._validate(ema.ema, val_loader)

                if val_loss < best_loss:
                    best_loss = val_loss

                if top1 > best_top1:
                    best_top1 = top1
                    epochs_without_improvement = 0
                    save_checkpoint(
                        model, optimizer, epoch, val_loss,
                        os.path.join(self.save_dir, "checkpoint_best.pth"),
                        scheduler=scheduler, config=model_config,
                    )
                    save_inference_weights(
                        ema.ema,
                        os.path.join(self.save_dir, "model_best_inference.pth"),
                        config=model_config,
                    )
                    self._logger.info(f"  Best model saved (Top-1: {best_top1:.2f}%)")
                else:
                    epochs_without_improvement += cfg.train.val_interval

                self._logger.info(
                    f"  Val Loss: {val_loss:.4f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%"
                )

                if self.patience > 0 and epochs_without_improvement >= self.patience:
                    self._logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Save latest
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(self.save_dir, "checkpoint_last.pth"),
                scheduler=scheduler, config=model_config, ema=ema,
            )

            if scheduler is not None:
                scheduler.step()

        # Final save
        if self.lora or self.qlora:
            lora_path = os.path.join(self.save_dir, "lora_adapters.pth")
            torch.save(get_lora_state_dict(ema.ema), lora_path)
            merge_lora_weights(ema.ema)

        save_inference_weights(
            ema.ema,
            os.path.join(self.save_dir, "model_final_inference.pth"),
            config=model_config,
        )

        if offload_hook is not None:
            offload_hook.remove()

        self._logger.info("=" * 60)
        self._logger.info("Training Complete!")
        self._logger.info(f"Best Top-1: {best_top1:.2f}%  |  Best Loss: {best_loss:.4f}")
        self._logger.info("=" * 60)

        return {"best_top1": best_top1, "best_loss": best_loss}

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        if self.qlora:
            model = apply_qlora(
                model, rank=self.lora_rank, alpha=self.lora_alpha,
                dropout=self.lora_dropout, target_modules=self.lora_targets,
                quant_dtype=self.qlora_dtype, variant=self.lora_variant,
            )
            self._logger.info(f"QLoRA applied (rank={self.lora_rank})")
        elif self.lora:
            model = apply_lora(
                model, rank=self.lora_rank, alpha=self.lora_alpha,
                dropout=self.lora_dropout, target_modules=self.lora_targets,
                variant=self.lora_variant,
            )
            self._logger.info(f"LoRA applied (rank={self.lora_rank})")
        return model

    def _load_teacher(self, num_classes, class_names):
        """Load teacher model for knowledge distillation."""
        teacher_cfg = MODEL_SIZE_MAP.get(self.kd_teacher_size, MODEL_SIZE_MAP["m-1.5x"])
        teacher = FlashClassifier(
            num_classes=num_classes,
            input_size=self.input_size,
            backbone_size=teacher_cfg["backbone"],
            pretrained=False,
            class_names=class_names,
        ).to(self.device)

        ckpt = torch.load(self.kd_teacher_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        teacher.load_state_dict(sd, strict=False)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        self._logger.info(f"KD teacher loaded: {self.kd_teacher_size}")
        return teacher

    def _train_one_epoch(self, model, dataloader, optimizer, criterion, ema, scaler,
                         mixup_fn, teacher_model):
        model.train()
        use_amp = scaler is not None
        loss_meter = AverageMeter("Loss")

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            if mixup_fn is not None:
                images, targets = mixup_fn(images, targets)

            with torch.amp.autocast(self.device.type, enabled=use_amp):
                output = model(images)
                logits = output["logits"]
                loss = criterion(logits, targets)

                # Knowledge distillation
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_out = teacher_model(images)
                    teacher_logits = teacher_out["logits"]
                    T = self.kd_temperature
                    kd_loss = nn.functional.kl_div(
                        nn.functional.log_softmax(logits / T, dim=1),
                        nn.functional.softmax(teacher_logits / T, dim=1),
                        reduction="batchmean",
                    ) * (T * T)
                    loss = self.kd_alpha * kd_loss + (1 - self.kd_alpha) * loss

                loss = loss / self.grad_accum

            if torch.isnan(loss):
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update(model)

            loss_meter.update(loss.item() * self.grad_accum)

            if (batch_idx + 1) % 20 == 0:
                self._logger.info(
                    f"  [{batch_idx+1}/{len(dataloader)}] Loss: {loss_meter.avg:.4f}"
                )

        return loss_meter.avg

    @torch.no_grad()
    def _validate(self, model, dataloader):
        model.eval()
        loss_meter = AverageMeter("Loss")
        top1_meter = AverageMeter("Top1")
        top5_meter = AverageMeter("Top5")
        criterion = nn.CrossEntropyLoss()

        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            output = model(images)
            logits = output["logits"]
            loss = criterion(logits, targets)

            top1, top5 = top_k_accuracy(logits, targets, topk=(1, 5))

            loss_meter.update(loss.item(), images.size(0))
            top1_meter.update(top1, images.size(0))
            top5_meter.update(top5, images.size(0))

        return loss_meter.avg, top1_meter.avg, top5_meter.avg
