# FlashCls

[![CI](https://github.com/FlashVision/FlashCls/actions/workflows/ci.yml/badge.svg)](https://github.com/FlashVision/FlashCls/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/flashcls)](https://pypi.org/project/flashcls/)

**Ultra-lightweight real-time image classification** built on ShuffleNetV2 backbone. Part of the [FlashVision](https://github.com/FlashVision) family.

## What is FlashCls?

FlashCls is a production-ready image classification framework designed for edge deployment. It combines a lightweight ShuffleNetV2 backbone with Global Average Pooling and a simple linear head to achieve excellent accuracy with minimal parameters.

**Key Features:**
- **Tiny models** — 0.35M to 2.5M parameters
- **Real-time** — >1000 FPS on GPU, >100 FPS on mobile
- **LoRA fine-tuning** — 6 variants for parameter-efficient adaptation
- **Knowledge Distillation** — train small models from large teachers
- **ImageFolder format** — just organize images into class folders
- **ONNX export** — deploy anywhere

## Models

| Model | Backbone | Params | FP16 Size | Top-1 (ImageNet) |
|-------|----------|--------|-----------|------------------|
| FlashCls-m-0.5x | ShuffleNetV2 0.5x | ~0.35M | ~0.7 MB | ~60.3% |
| FlashCls-m | ShuffleNetV2 1.0x | ~1.3M | ~2.5 MB | ~69.4% |
| FlashCls-m-1.5x | ShuffleNetV2 1.5x | ~2.5M | ~4.8 MB | ~72.6% |

## Installation

```bash
pip install flashcls

# Or from source
git clone https://github.com/FlashVision/FlashCls.git
cd FlashCls
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
from flashcls import FlashClassifier, Trainer, Predictor

# Train on your dataset (ImageFolder format)
trainer = Trainer(
    train_dir="data/train",
    val_dir="data/val",
    model_size="m",
    epochs=50,
    batch_size=64,
)
trainer.train()

# Inference
predictor = Predictor(model_path="workspace/classification/model_best_inference.pth")
results = predictor.classify("test_image.jpg")
for class_name, probability in results[:5]:
    print(f"{class_name}: {probability:.2%}")
```

### CLI

```bash
# Verify installation
flashcls check

# Train
flashcls train --train-dir data/train --val-dir data/val --model-size m --epochs 100

# Predict
flashcls predict --model best.pth --source image.jpg

# Validate
flashcls val --model best.pth --val-dir data/val

# Export to ONNX
flashcls export --model best.pth --output model.onnx --simplify
```

## Dataset Format

FlashCls uses ImageFolder format:

```
data/
├── train/
│   ├── cat/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   ├── dog/
│   │   ├── img003.jpg
│   │   └── img004.jpg
│   └── bird/
│       └── img005.jpg
└── val/
    ├── cat/
    ├── dog/
    └── bird/
```

## LoRA Fine-Tuning

Fine-tune with minimal parameters using LoRA:

```python
trainer = Trainer(
    train_dir="data/train",
    val_dir="data/val",
    model_size="m",
    lora=True,
    lora_rank=8,
    epochs=30,
)
trainer.train()
```

Available variants: `standard`, `dora`, `lora_plus`, `adalora`, `ortho`, `lora_fa`

## Knowledge Distillation

Train a small student from a large teacher:

```python
trainer = Trainer(
    train_dir="data/train",
    val_dir="data/val",
    model_size="m-0.5x",
    kd=True,
    kd_teacher_path="teacher_model.pth",
    kd_teacher_size="m-1.5x",
    kd_temperature=4.0,
)
trainer.train()
```

## Solutions

### Image Tagger
```python
from flashcls import Predictor
from flashcls.solutions import ImageTagger

predictor = Predictor(model_path="model.pth")
tagger = ImageTagger(predictor)
results = tagger.tag_directory("images/", sort_into_folders=True)
```

### Quality Inspector
```python
from flashcls.solutions import QualityInspector

inspector = QualityInspector(predictor, good_classes=["good"])
result = inspector.inspect("part_image.jpg")
print(f"Verdict: {result['verdict']}")
```

## Benchmarks

```python
from flashcls.analytics import Benchmark

bench = Benchmark()
results = bench.compare_all()
```

## Architecture

```
Input Image [B, 3, 224, 224]
    │
    ▼
┌─────────────────────┐
│  ShuffleNetV2        │  Lightweight backbone
│  (0.5x / 1.0x / 1.5x) │
└─────────────────────┘
    │
    ▼ [B, C, 7, 7]
┌─────────────────────┐
│  Global Avg Pool     │
└─────────────────────┘
    │
    ▼ [B, C]
┌─────────────────────┐
│  Dropout + Linear    │  Classification head
└─────────────────────┘
    │
    ▼ [B, num_classes]
  Logits / Softmax
```

## License

MIT License. See [LICENSE](LICENSE) for details.
