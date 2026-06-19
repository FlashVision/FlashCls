"""Example: Export to ONNX."""

import argparse
from flashcls import Exporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="model.onnx")
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()

    exporter = Exporter(model_path=args.model)
    path = exporter.export_onnx(output_path=args.output, simplify=args.simplify)
    print(f"Exported: {path}")


if __name__ == "__main__":
    main()
