# Quick Start

## Train

```python
from flashcls import Trainer

trainer = Trainer(train_dir="data/train", val_dir="data/val", model_size="m", epochs=100)
trainer.train()
```

## Predict

```python
from flashcls import Predictor

predictor = Predictor(model_path="workspace/classification/model_best_inference.pth")
results = predictor.classify("test.jpg")
for cls, prob in results[:5]:
    print(f"{cls}: {prob:.2%}")
```

## Export

```bash
flashcls export --model best.pth --output model.onnx --simplify
```
