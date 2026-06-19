"""Tests for FlashClassifier model."""

import pytest
import torch

from flashcls.models.classifier import FlashClassifier, build_model, BACKBONE_CHANNELS
from flashcls.cfg import get_config


class TestFlashClassifier:
    """Test FlashClassifier model variants."""

    @pytest.mark.parametrize("backbone_size,expected_channels", [
        ("0.5x", 192),
        ("1.0x", 464),
        ("1.5x", 704),
    ])
    def test_backbone_channels(self, backbone_size, expected_channels):
        assert BACKBONE_CHANNELS[backbone_size][-1] == expected_channels

    @pytest.mark.parametrize("backbone_size", ["0.5x", "1.0x", "1.5x"])
    def test_forward_pass(self, backbone_size):
        model = FlashClassifier(
            num_classes=10,
            input_size=(224, 224),
            backbone_size=backbone_size,
            pretrained=False,
        )
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert "logits" in output
        assert "probs" in output
        assert output["logits"].shape == (2, 10)
        assert output["probs"].shape == (2, 10)
        # Probs should sum to 1
        assert torch.allclose(output["probs"].sum(dim=1), torch.ones(2), atol=1e-5)

    def test_predict(self):
        model = FlashClassifier(
            num_classes=5,
            input_size=(224, 224),
            backbone_size="0.5x",
            pretrained=False,
            class_names=["cat", "dog", "bird", "fish", "horse"],
        )
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        results = model.predict(x)
        assert len(results) == 1
        assert len(results[0]) == 5
        # Sorted by probability
        probs = [p for _, p in results[0]]
        assert probs == sorted(probs, reverse=True)

    def test_get_model_info(self):
        model = FlashClassifier(
            num_classes=10,
            backbone_size="1.0x",
            pretrained=False,
        )
        info = model.get_model_info()
        assert info["name"] == "FlashClassifier"
        assert info["num_classes"] == 10
        assert info["total_params"] > 0
        assert info["params_mb"] > 0

    def test_build_model_from_config(self):
        cfg = get_config(model_size="m", input_size=224, num_classes=20)
        model = build_model(cfg)
        assert model.num_classes == 20
        assert model.input_size == (224, 224)
        assert model.backbone_size == "1.0x"

    def test_get_features(self):
        model = FlashClassifier(
            num_classes=10,
            backbone_size="0.5x",
            pretrained=False,
        )
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = model.get_features(x)
        assert features.shape == (2, 192)

    def test_different_input_sizes(self):
        for size in [128, 224, 256, 320]:
            model = FlashClassifier(
                num_classes=5,
                input_size=(size, size),
                backbone_size="0.5x",
                pretrained=False,
            )
            model.eval()
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                output = model(x)
            assert output["logits"].shape == (1, 5)
