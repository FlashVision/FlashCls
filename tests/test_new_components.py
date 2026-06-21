"""Tests for new FlashCls components: DINOv2, distillation, SSL, multi-label head."""

import torch
import torch.nn as nn


class TestDINOv2Backbone:
    def test_forward(self):
        from flashcls.models.architectures.dinov2 import DINOv2Backbone

        model = DINOv2Backbone(
            variant="dinov2_vits14",
            num_classes=10,
            pretrained=False,
            freeze_backbone=False,
            dropout=0.0,
        )
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 10)
        assert "features" in out

    def test_freeze_backbone(self):
        from flashcls.models.architectures.dinov2 import DINOv2Backbone

        model = DINOv2Backbone(
            variant="dinov2_vits14",
            num_classes=5,
            pretrained=False,
            freeze_backbone=True,
        )
        backbone_params = sum(p.requires_grad for p in model.feature_extractor.parameters())
        assert backbone_params == 0

    def test_unfreeze(self):
        from flashcls.models.architectures.dinov2 import DINOv2Backbone

        model = DINOv2Backbone(
            variant="dinov2_vits14",
            num_classes=5,
            pretrained=False,
            freeze_backbone=True,
        )
        model.unfreeze_backbone()
        trainable = sum(p.requires_grad for p in model.feature_extractor.parameters())
        total = sum(1 for _ in model.feature_extractor.parameters())
        assert trainable == total

    def test_with_targets(self):
        from flashcls.models.architectures.dinov2 import DINOv2Backbone

        model = DINOv2Backbone(
            variant="dinov2_vits14",
            num_classes=5,
            pretrained=False,
        )
        x = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([1, 3])
        out = model(x, targets)
        assert "loss" in out
        assert out["loss"].requires_grad

    def test_model_info(self):
        from flashcls.models.architectures.dinov2 import DINOv2Backbone

        model = DINOv2Backbone(variant="dinov2_vits14", num_classes=10, pretrained=False)
        info = model.get_model_info()
        assert info["name"] == "DINOv2"
        assert info["total_params"] > 0

    def test_registry(self):
        from flashcls.registry import BACKBONES

        assert "DINOv2" in BACKBONES


class TestDistillation:
    def test_distillation_loss(self):
        from flashcls.training.distillation import DistillationLoss

        criterion = DistillationLoss(temperature=4.0, alpha=0.7, beta=0.0)
        student_logits = torch.randn(4, 10, requires_grad=True)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        result = criterion(student_logits, teacher_logits, targets)
        assert "loss" in result
        assert "kd_loss" in result
        assert "task_loss" in result
        assert result["loss"].requires_grad

    def test_distillation_loss_with_features(self):
        from flashcls.training.distillation import DistillationLoss

        criterion = DistillationLoss(
            temperature=4.0,
            alpha=0.7,
            beta=0.5,
            feature_dims=(128, 256),
        )
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        student_features = torch.randn(4, 128)
        teacher_features = torch.randn(4, 256)
        result = criterion(
            student_logits,
            teacher_logits,
            targets,
            student_features,
            teacher_features,
        )
        assert "feat_loss" in result
        assert result["loss"].requires_grad

    def test_distillation_trainer(self):
        from flashcls.training.distillation import DistillationTrainer

        class SimpleModel(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc = nn.Linear(3 * 32 * 32, dim)
                self.head = nn.Linear(dim, 5)

            def forward(self, x, targets=None):
                feat = self.fc(x.flatten(1))
                logits = self.head(feat)
                out = {"logits": logits, "features": feat}
                return out

        student = SimpleModel(64)
        teacher = SimpleModel(128)

        trainer = DistillationTrainer(
            student,
            teacher,
            temperature=4.0,
            alpha=0.7,
            device="cpu",
        )
        images = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, 5, (2,))
        result = trainer.train_step(images, targets)
        assert "loss" in result


class TestSSL:
    def test_dino_loss(self):
        from flashcls.training.ssl import DINOLoss

        criterion = DINOLoss(out_dim=128, num_crops=4, num_global=2)
        student_out = torch.randn(8, 128)
        teacher_out = torch.randn(4, 128)
        loss = criterion(student_out, teacher_out)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_mae_decoder(self):
        from flashcls.training.ssl import MAEDecoder

        decoder = MAEDecoder(
            encoder_dim=64,
            decoder_dim=32,
            decoder_depth=1,
            decoder_heads=2,
            patch_size=16,
            num_patches=196,
        )
        encoded = torch.randn(2, 50, 64)
        vis_idx = torch.arange(50).unsqueeze(0).expand(2, -1)
        mask_idx = torch.arange(50, 196).unsqueeze(0).expand(2, -1)
        pred = decoder(encoded, vis_idx, mask_idx)
        assert pred.shape == (2, 146, 16 * 16 * 3)

    def test_ssl_trainer_mae(self):
        from flashcls.training.ssl import SSLTrainer

        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 224 * 224, 64)

            def extract_features(self, x):
                return self.fc(x.flatten(1))

        backbone = DummyBackbone()
        trainer = SSLTrainer(
            backbone,
            method="mae",
            feature_dim=64,
            device="cpu",
            mae_decoder_dim=48,
            mae_decoder_depth=1,
        )
        images = torch.randn(2, 3, 224, 224)
        result = trainer.train_step_mae(images)
        assert "loss" in result


class TestMultiLabelHead:
    def test_forward(self):
        from flashcls.models.head.multilabel_head import MultiLabelHead

        head = MultiLabelHead(in_channels=64, num_classes=10, loss_type="asl")
        x = torch.randn(2, 64, 7, 7)
        targets = torch.zeros(2, 10)
        targets[0, [1, 3]] = 1.0
        targets[1, [0, 5, 7]] = 1.0
        out = head(x, targets)
        assert "logits" in out
        assert out["logits"].shape == (2, 10)
        assert "loss" in out
        assert out["loss"].requires_grad

    def test_forward_pre_pooled(self):
        from flashcls.models.head.multilabel_head import MultiLabelHead

        head = MultiLabelHead(in_channels=64, num_classes=5, loss_type="bce")
        x = torch.randn(2, 64)
        out = head(x)
        assert out["logits"].shape == (2, 5)

    def test_predict(self):
        from flashcls.models.head.multilabel_head import MultiLabelHead

        head = MultiLabelHead(in_channels=32, num_classes=3)
        head.eval()
        x = torch.randn(1, 32, 7, 7)
        preds = head.predict(x, class_names=["cat", "dog", "bird"])
        assert len(preds) == 1
        for name, prob in preds[0]:
            assert isinstance(name, str)
            assert 0 <= prob <= 1

    def test_asymmetric_loss(self):
        from flashcls.models.head.multilabel_head import AsymmetricLoss

        loss_fn = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[0, [1, 3]] = 1.0
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_set_thresholds(self):
        from flashcls.models.head.multilabel_head import MultiLabelHead

        head = MultiLabelHead(in_channels=32, num_classes=5, init_thresholds=0.5)
        new_thr = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7])
        head.set_thresholds(new_thr)
        assert torch.allclose(head.thresholds, new_thr)

    def test_registry(self):
        from flashcls.registry import HEADS

        assert "MultiLabelHead" in HEADS
