# LoRA Fine-Tuning

## Basic LoRA

```python
from flashcls import Trainer

trainer = Trainer(
    train_dir="data/train",
    val_dir="data/val",
    lora=True,
    lora_rank=8,
    epochs=30,
)
trainer.train()
```

## Variants

| Variant | Description |
|---------|-------------|
| `standard` | Classic LoRA |
| `dora` | Weight-decomposed |
| `lora_plus` | Asymmetric LR |
| `adalora` | Adaptive rank |
| `ortho` | Orthogonal regularization |
| `lora_fa` | Freeze A matrix |

## QLoRA

```python
trainer = Trainer(qlora=True, qlora_dtype="int8")
```
