"""Example: Fine-tune with LoRA."""

import argparse
from flashcls import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--variant", default="standard", choices=["standard", "dora", "lora_plus", "adalora", "ortho", "lora_fa"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    trainer = Trainer(
        train_dir=args.train_dir, val_dir=args.val_dir,
        model_size="m", epochs=args.epochs, lora=True,
        lora_variant=args.variant, lora_rank=args.rank, amp=True,
    )
    results = trainer.train()
    print(f"LoRA ({args.variant}) Best Top-1: {results['best_top1']:.2f}%")


if __name__ == "__main__":
    main()
