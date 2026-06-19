# Changelog

All notable changes to FlashCls will be documented in this file.

## [1.0.0] — 2026-06-19

### Added
- **Package structure** — `pip install` from GitHub or PyPI
- **CLI** — `flashcls train`, `predict`, `val`, `export`, `check`, `settings`, `version`
- **Python API** — `Trainer`, `Predictor`, `Exporter`, `Validator`
- **Models** — FlashCls-m-0.5x (~0.35M), FlashCls-m (~1.3M), FlashCls-m-1.5x (~2.5M)
- **LoRA fine-tuning** — 6 variants (standard, dora, lora_plus, adalora, ortho, lora_fa)
- **QLoRA** — INT8/NF4 quantized base weights + LoRA
- **Knowledge Distillation** — teacher-student training with configurable temperature
- **Solutions** — ImageTagger, QualityInspector
- **Analytics** — Benchmark, Profiler
- **ONNX export** — with simplification support
- **Mixed precision** — AMP (FP16) training
- **Mixup / CutMix** — advanced data augmentation
- **CI/CD** — GitHub Actions (lint + test on Python 3.9-3.12, auto-publish to PyPI)
- **Examples** — 5 runnable example scripts

### Architecture
- ShuffleNetV2 backbone (0.5x, 1.0x, 1.5x)
- Global Average Pooling + Dropout + Linear classification head
- Label smoothing cross-entropy loss
