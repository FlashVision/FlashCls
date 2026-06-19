"""Example: Classify a single image."""

import argparse
from flashcls import Predictor


def main():
    parser = argparse.ArgumentParser(description="Classify an image")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    predictor = Predictor(model_path=args.model, device=args.device, top_k=args.top_k)
    results = predictor.classify(args.image)

    print(f"\nResults for: {args.image}")
    for i, (cls, prob) in enumerate(results, 1):
        print(f"  {i}. {cls:<20} {prob:>6.2%}")


if __name__ == "__main__":
    main()
