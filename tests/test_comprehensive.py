"""Comprehensive test suite for FlashCls."""

import subprocess
import sys

import pytest
import torch
import torch.nn as nn

B, C, H, W = 2, 3, 64, 64
NUM_CLASSES = 5


@pytest.fixture
def dummy_input():
    return torch.randn(B, C, H, W)


# ===================================================================
# 1. MODEL ARCHITECTURES
# ===================================================================


class TestFlashClassifier:
    def test_forward(self, dummy_input):
        from flashcls.models.classifier import FlashClassifier

        model = FlashClassifier(num_classes=NUM_CLASSES, input_size=H, backbone_size="0.5x", pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert "logits" in out
        assert out["logits"].shape == (B, NUM_CLASSES)

    def test_forward_with_loss(self, dummy_input):
        from flashcls.models.classifier import FlashClassifier

        model = FlashClassifier(num_classes=NUM_CLASSES, input_size=H, backbone_size="0.5x", pretrained=False)
        targets = torch.randint(0, NUM_CLASSES, (B,))
        out = model(dummy_input, targets=targets)
        assert "loss" in out
        assert torch.isfinite(out["loss"])

    def test_gradient_flow(self):
        from flashcls.models.classifier import FlashClassifier

        model = FlashClassifier(num_classes=3, input_size=64, backbone_size="0.5x", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        out = model(x)
        out["logits"].sum().backward()
        assert x.grad is not None

    def test_predict(self, dummy_input):
        from flashcls.models.classifier import FlashClassifier

        model = FlashClassifier(num_classes=NUM_CLASSES, input_size=H, backbone_size="0.5x", pretrained=False)
        results = model.predict(dummy_input, top_k=3)
        assert len(results) == B
        assert len(results[0]) == 3

    def test_model_info(self):
        from flashcls.models.classifier import FlashClassifier

        model = FlashClassifier(num_classes=5, input_size=64, backbone_size="0.5x", pretrained=False)
        info = model.get_model_info()
        assert info["total_params"] > 0
        assert info["num_classes"] == 5

    def test_extract_features(self, dummy_input):
        from flashcls.models.classifier import FlashClassifier

        model = FlashClassifier(num_classes=NUM_CLASSES, input_size=H, backbone_size="0.5x", pretrained=False)
        model.eval()
        with torch.no_grad():
            feats = model.extract_features(dummy_input)
        assert feats.dim() == 2
        assert feats.shape[0] == B

    def test_backbone_sizes(self):
        from flashcls.models.classifier import FlashClassifier

        for size in ["0.5x", "1.0x"]:
            model = FlashClassifier(num_classes=3, input_size=64, backbone_size=size, pretrained=False)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, 64, 64))
            assert out["logits"].shape == (1, 3)


class TestDINOv2:
    def test_forward(self):
        from flashcls.models.architectures.dinov2 import DINOv2Backbone

        model = DINOv2Backbone(variant="dinov2_vits14", num_classes=5, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert "logits" in out
        assert out["logits"].shape == (1, 5)

    def test_freeze_unfreeze(self):
        from flashcls.models.architectures.dinov2 import DINOv2Backbone

        model = DINOv2Backbone(variant="dinov2_vits14", num_classes=5, pretrained=False)
        model.freeze_backbone()
        frozen = sum(1 for p in model.feature_extractor.parameters() if not p.requires_grad)
        assert frozen > 0
        model.unfreeze_backbone()
        unfrozen = sum(1 for p in model.feature_extractor.parameters() if p.requires_grad)
        assert unfrozen > 0

    def test_model_info(self):
        from flashcls.models.architectures.dinov2 import DINOv2Backbone

        model = DINOv2Backbone(variant="dinov2_vits14", num_classes=5, pretrained=False)
        info = model.get_model_info()
        assert info["name"] == "DINOv2"


class TestClassificationHead:
    def test_forward(self):
        from flashcls.models.head import ClassificationHead

        head = ClassificationHead(in_channels=192, num_classes=10)
        x = torch.randn(2, 192, 4, 4)
        out = head(x)
        assert out.shape == (2, 10)


class TestMultiLabelHead:
    def test_forward(self):
        from flashcls.models.head import MultiLabelHead

        head = MultiLabelHead(in_channels=64, num_classes=5)
        x = torch.randn(2, 64, 4, 4)
        out = head(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 5)

    def test_with_loss(self):
        from flashcls.models.head import MultiLabelHead

        head = MultiLabelHead(in_channels=64, num_classes=5, loss_type="asl")
        x = torch.randn(2, 64, 4, 4)
        targets = torch.zeros(2, 5)
        targets[0, 0] = 1
        targets[1, 2] = 1
        out = head(x, targets=targets)
        assert "loss" in out
        assert torch.isfinite(out["loss"])

    def test_predict(self):
        from flashcls.models.head import MultiLabelHead

        head = MultiLabelHead(in_channels=64, num_classes=5)
        x = torch.randn(2, 64, 4, 4)
        results = head.predict(x, class_names=["a", "b", "c", "d", "e"])
        assert isinstance(results, list)
        assert len(results) == 2


class TestAsymmetricLoss:
    def test_forward(self):
        from flashcls.models.head import AsymmetricLoss

        loss_fn = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[0, 0] = 1.0
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)


# ===================================================================
# 2. LOSSES
# ===================================================================


class TestLosses:
    def test_label_smoothing_ce(self):
        from flashcls.losses import LabelSmoothingCrossEntropy

        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_soft_target_ce(self):
        from flashcls.losses import SoftTargetCrossEntropy

        loss_fn = SoftTargetCrossEntropy()
        logits = torch.randn(4, 10)
        targets = torch.softmax(torch.randn(4, 10), dim=-1)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_kd_loss(self):
        from flashcls.losses import ClassificationKDLoss

        loss_fn = ClassificationKDLoss(temperature=4.0)
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(student, teacher, targets)
        assert torch.isfinite(loss)

    def test_loss_gradient(self):
        from flashcls.losses import LabelSmoothingCrossEntropy

        loss_fn = LabelSmoothingCrossEntropy()
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.randint(0, 5, (4,))
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


# ===================================================================
# 3. REGISTRY
# ===================================================================


class TestRegistry:
    def test_backbone_registry(self):
        from flashcls.registry import BACKBONES

        assert "DINOv2" in BACKBONES

    def test_head_registry(self):
        from flashcls.registry import HEADS

        assert "MultiLabelHead" in HEADS

    def test_registry_build(self):
        from flashcls.registry import Registry

        reg = Registry("test")

        @reg.register("Foo")
        class Foo:
            pass

        assert "Foo" in reg
        assert len(reg) == 1


# ===================================================================
# 4. CLI
# ===================================================================


class TestCLI:
    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "flashcls.cli", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "FlashCls" in result.stdout

    def test_no_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "flashcls.cli"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0


# ===================================================================
# 5. ENGINE
# ===================================================================


class TestEngine:
    def test_imports(self):
        from flashcls.engine.trainer import Trainer
        from flashcls.engine.validator import Validator

        assert Trainer is not None
        assert Validator is not None

    def test_callbacks(self):
        from flashcls.engine.callbacks import CallbackList, EarlyStopping

        cb_list = CallbackList()
        assert cb_list is not None
        assert EarlyStopping is not None


# ===================================================================
# 6. DATA
# ===================================================================


class TestData:
    def test_transforms_import(self):
        from flashcls.data.transforms import TrainTransform, ValTransform

        train_t = TrainTransform(input_size=(64, 64))
        val_t = ValTransform(input_size=(64, 64))
        assert train_t is not None
        assert val_t is not None


# ===================================================================
# 7. UTILS
# ===================================================================


class TestUtils:
    def test_metrics(self):
        from flashcls.utils.metrics import top_k_accuracy

        logits = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0], [1.0, 0.5, 2.0], [2.0, 0.5, 1.0]])
        targets = torch.tensor([0, 1, 2, 1])
        top1, top2 = top_k_accuracy(logits, targets, topk=(1, 2))
        assert 0 <= top1 <= 100
        assert 0 <= top2 <= 100

    def test_mixup(self):
        from flashcls.utils.mixup import Mixup

        mixer = Mixup(mixup_alpha=1.0, num_classes=5)
        assert mixer is not None

    def test_logger(self):
        from flashcls.utils.logger import setup_logger

        log = setup_logger("test")
        assert log is not None


# ===================================================================
# 8. SOLUTIONS
# ===================================================================


class TestSolutions:
    def test_image_tagger_import(self):
        from flashcls.solutions import ImageTagger

        assert ImageTagger is not None

    def test_quality_inspector_import(self):
        from flashcls.solutions import QualityInspector

        assert QualityInspector is not None


# ===================================================================
# 9. TRAINING — KD, SSL
# ===================================================================


class TestTraining:
    def test_distillation_import(self):
        from flashcls.training.distillation import DistillationTrainer

        assert DistillationTrainer is not None

    def test_ssl_import(self):
        from flashcls.training.ssl import SSLTrainer

        assert SSLTrainer is not None


# ===================================================================
# 10. LoRA
# ===================================================================


class TestLoRA:
    def test_apply_lora(self):
        from flashcls.models.classifier import FlashClassifier
        from flashcls.models.lora import apply_lora

        model = FlashClassifier(num_classes=5, input_size=64, backbone_size="0.5x", pretrained=False)
        model = apply_lora(model, rank=4, alpha=8.0)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable > 0

    def test_merge_lora(self):
        from flashcls.models.lora import LoRALinear, merge_lora_weights

        model = nn.Sequential(LoRALinear(16, 16, rank=4))
        merged = merge_lora_weights(model)
        assert merged is not None


# ===================================================================
# 11. CONFIG
# ===================================================================


class TestConfig:
    def test_get_config(self):
        from flashcls.cfg import get_config

        cfg = get_config("shuffle-1.0x")
        assert cfg["backbone"] == "shufflenet"

    def test_list_models(self):
        from flashcls.cfg import list_models

        models = list_models()
        assert len(models) > 5

    def test_build_backbone(self):
        from flashcls.cfg import build_backbone

        bb = build_backbone("shuffle-0.5x", pretrained=False)
        x = torch.randn(1, 3, 64, 64)
        out = bb(x)
        assert out.shape[0] == 1


# ===================================================================
# 12. EDGE CASES
# ===================================================================


class TestEdgeCases:
    def test_wrong_channels(self):
        from flashcls.models.classifier import FlashClassifier

        model = FlashClassifier(num_classes=3, input_size=64, backbone_size="0.5x", pretrained=False)
        with pytest.raises(RuntimeError):
            model(torch.randn(1, 1, 64, 64))

    def test_single_class(self):
        from flashcls.models.classifier import FlashClassifier

        model = FlashClassifier(num_classes=1, input_size=64, backbone_size="0.5x", pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 64, 64))
        assert out["logits"].shape == (1, 1)


# ===================================================================
# 13. INTEGRATION
# ===================================================================


class TestIntegration:
    def test_full_pipeline(self):
        from flashcls.models.classifier import FlashClassifier

        model = FlashClassifier(num_classes=3, input_size=64, backbone_size="0.5x", pretrained=False)
        model.train()
        x = torch.randn(2, 3, 64, 64)
        targets = torch.randint(0, 3, (2,))

        out = model(x, targets=targets)
        loss = out["loss"]
        assert torch.isfinite(loss)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        preds = model.predict(x, top_k=2)
        assert len(preds) == 2

    def test_all_backbones_basic(self):
        from flashcls.models.backbone import MobileNetV3Small, ResNet, ShuffleNetV2, VisionTransformer

        x = torch.randn(1, 3, 64, 64)

        bb = ShuffleNetV2(model_size="0.5x", pretrained=False)
        out = bb(x)
        assert out.shape[0] == 1

        bb2 = MobileNetV3Small(pretrained=False)
        out2 = bb2(x)
        assert out2.shape[0] == 1

        bb3 = ResNet(variant="resnet18", pretrained=False)
        out3 = bb3(x)
        assert out3.shape[0] == 1

        bb4 = VisionTransformer(variant="vit_tiny", pretrained=False)
        out4 = bb4(torch.randn(1, 3, 224, 224))
        assert out4.shape[0] == 1


# ===================================================================
# 14. ANALYTICS
# ===================================================================


class TestAnalytics:
    def test_benchmark_import(self):
        from flashcls.analytics import Benchmark

        assert Benchmark is not None

    def test_profiler_import(self):
        from flashcls.analytics import Profiler

        assert Profiler is not None
