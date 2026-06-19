"""Example: Train on a custom ImageFolder dataset."""

import argparse
from flashcls import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--model-size", default="m", choices=["m-0.5x", "m", "m-1.5x"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    trainer = Trainer(
        train_dir=args.train_dir, val_dir=args.val_dir,
        model_size=args.model_size, epochs=args.epochs,
        batch_size=args.batch_size, label_smoothing=0.1, amp=True,
    )
    results = trainer.train()
    print(f"Best Top-1: {results['best_top1']:.2f}%")


if __name__ == "__main__":
    main()
