# Models

## Variants

| Model | Backbone | Params | FP16 Size |
|-------|----------|--------|-----------|
| FlashCls-m-0.5x | ShuffleNetV2 0.5x | ~0.35M | ~0.7 MB |
| FlashCls-m | ShuffleNetV2 1.0x | ~1.3M | ~2.5 MB |
| FlashCls-m-1.5x | ShuffleNetV2 1.5x | ~2.5M | ~4.8 MB |

## Architecture

ShuffleNetV2 backbone → Global Average Pooling → Dropout → Linear classification head.
