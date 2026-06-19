# Training

## Basic Training

```python
from flashcls import Trainer

trainer = Trainer(
    train_dir="data/train",
    val_dir="data/val",
    model_size="m",
    epochs=100,
    batch_size=64,
    label_smoothing=0.1,
    mixup_alpha=0.2,
    amp=True,
)
trainer.train()
```

## Knowledge Distillation

```python
trainer = Trainer(
    model_size="m-0.5x",
    kd=True,
    kd_teacher_path="teacher.pth",
    kd_teacher_size="m-1.5x",
    kd_temperature=4.0,
)
```

## YAML Config

```bash
flashcls train --config configs/flashcls_m_custom.yaml
```
