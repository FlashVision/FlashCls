"""FlashCls — Ultra-lightweight image classification."""

__version__ = "1.0.0"

from flashcls.models.classifier import FlashClassifier
from flashcls.models.lora import apply_lora, apply_qlora, merge_lora_weights
from flashcls.engine.trainer import Trainer
from flashcls.engine.predictor import Predictor
from flashcls.engine.validator import Validator
from flashcls.engine.exporter import Exporter
from flashcls.cfg import get_config
from flashcls.analytics import Benchmark

__all__ = [
    "FlashClassifier", "Trainer", "Validator", "Predictor", "Exporter",
    "apply_lora", "apply_qlora", "merge_lora_weights", "get_config",
    "Benchmark",
    "__version__",
]
