"""FlashCls CLI — command-line interface for training, validation, prediction, and export."""

import argparse
import sys


def _colored(text, color):
    colors = {"green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m", "red": "\033[91m", "bold": "\033[1m"}
    return f"{colors.get(color, '')}{text}\033[0m"


def _print_banner():
    print(_colored("FlashCls", "bold") + f" v{_get_version()}")
    print(_colored("Ultra-lightweight image classification", "blue"))
    print()


def _get_version():
    from flashcls import __version__
    return __version__


def cmd_version(args):
    _print_banner()


def cmd_settings(args):
    import torch
    import platform
    import numpy as np

    _print_banner()
    print(_colored("System", "bold"))
    print(f"  Python:      {platform.python_version()}")
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Machine:     {platform.machine()}")
    print()
    print(_colored("Dependencies", "bold"))
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  CUDA:        {torch.version.cuda or 'Not available'}")
    print(f"  cuDNN:       {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print()
    print(_colored("Hardware", "bold"))
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM:        {mem:.1f} GB")
    else:
        print("  GPU:         None (CPU only)")
    print(f"  CPU cores:   {__import__('os').cpu_count()}")


def cmd_check(args):
    _print_banner()
    errors = []

    print(_colored("Checking installation...", "bold"))
    print()

    try:
        import flashcls  # noqa: F401
        print(f"  {_colored('✓', 'green')} flashcls package")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} flashcls package: {e}")
        errors.append(str(e))

    try:
        from flashcls.engine import Trainer, Predictor, Exporter, Validator  # noqa: F401
        print(f"  {_colored('✓', 'green')} engine (Trainer, Predictor, Exporter, Validator)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} engine: {e}")
        errors.append(str(e))

    try:
        from flashcls.solutions import ImageTagger, QualityInspector  # noqa: F401
        print(f"  {_colored('✓', 'green')} solutions (ImageTagger, QualityInspector)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} solutions: {e}")
        errors.append(str(e))

    try:
        from flashcls.analytics import Benchmark, Profiler  # noqa: F401
        print(f"  {_colored('✓', 'green')} analytics (Benchmark, Profiler)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} analytics: {e}")
        errors.append(str(e))

    try:
        import torch
        from flashcls.models.classifier import FlashClassifier
        model = FlashClassifier(num_classes=10, input_size=224,
                                backbone_size="1.0x", pretrained=False)
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 3, 224, 224))
        print(f"  {_colored('✓', 'green')} model forward pass (FlashClassifier-m, 224px)")
    except Exception as e:
        print(f"  {_colored('✗', 'red')} model forward pass: {e}")
        errors.append(str(e))

    import torch
    if torch.cuda.is_available():
        print(f"  {_colored('✓', 'green')} CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  {_colored('⚠', 'yellow')} No CUDA GPU (training will be slow)")

    print()
    if errors:
        print(_colored(f"✗ {len(errors)} check(s) failed", "red"))
        sys.exit(1)
    else:
        print(_colored("✓ All checks passed! FlashCls is ready.", "green"))


def cmd_train(args):
    from flashcls.engine.trainer import Trainer

    if args.config:
        from flashcls.cfg import load_yaml_config
        cfg = load_yaml_config(args.config)
        print(f"{_colored('Config:', 'bold')} {args.config}")
        trainer = Trainer(config=cfg, device=args.device)
    else:
        if not args.train_dir or not args.val_dir:
            print(_colored("Error:", "red") + " --train-dir and --val-dir are required (or use --config)")
            sys.exit(1)
        kwargs = {
            "model_size": args.model_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "save_dir": args.save_dir,
        }
        if args.lora:
            kwargs["lora"] = True
        if args.qlora:
            kwargs["qlora"] = True
        if args.lr:
            kwargs["learning_rate"] = args.lr
        if args.workers is not None:
            kwargs["workers"] = args.workers
        if args.label_smoothing is not None:
            kwargs["label_smoothing"] = args.label_smoothing
        trainer = Trainer(**kwargs)

    trainer.train()


def cmd_predict(args):
    from flashcls.engine.predictor import Predictor

    predictor = Predictor(model_path=args.model, device=args.device)
    results = predictor.predict(args.source, output_dir=args.output)

    if isinstance(results, list) and results and isinstance(results[0], tuple):
        print(f"\n{_colored('Classification results:', 'green')}")
        for cls_name, prob in results[:10]:
            print(f"  {cls_name}: {prob:.4f}")


def cmd_val(args):
    from flashcls.engine.validator import Validator
    validator = Validator(model_path=args.model, val_dir=args.val_dir, device=args.device)
    validator.validate()


def cmd_export(args):
    from flashcls.engine.exporter import Exporter
    exporter = Exporter(model_path=args.model)
    path = exporter.export(output=args.output, simplify=args.simplify)
    print(f"\n{_colored('✓', 'green')} Exported: {path}")


def main():
    parser = argparse.ArgumentParser(
        prog="flashcls",
        description="FlashCls: Ultra-lightweight image classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashcls check                                Verify installation
  flashcls train --train-dir data/train --val-dir data/val
  flashcls predict --model best.pth --source photo.jpg
  flashcls export --model best.pth --output model.onnx --simplify

Documentation: https://github.com/FlashVision/FlashCls
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("version", help="Show version info")
    subparsers.add_parser("settings", help="Show system settings (Python, PyTorch, CUDA, GPU)")
    subparsers.add_parser("check", help="Verify installation and run health check")

    train_p = subparsers.add_parser("train", help="Train a FlashClassifier model")
    train_p.add_argument("--config", default=None, help="Path to YAML config")
    train_p.add_argument("--model-size", default="m", choices=["m-0.5x", "m", "m-1.5x"])
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=None)
    train_p.add_argument("--label-smoothing", type=float, default=None)
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--train-dir", default=None, help="Path to training images (ImageFolder)")
    train_p.add_argument("--val-dir", default=None, help="Path to validation images (ImageFolder)")
    train_p.add_argument("--save-dir", default="workspace/train", help="Output directory")
    train_p.add_argument("--workers", type=int, default=None)
    train_p.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    train_p.add_argument("--qlora", action="store_true", help="Enable QLoRA fine-tuning")

    pred_p = subparsers.add_parser("predict", help="Classify image(s)")
    pred_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    pred_p.add_argument("--source", required=True, help="Image path or directory")
    pred_p.add_argument("--device", default="cuda")
    pred_p.add_argument("--output", default=None, help="Output directory for annotated results")

    val_p = subparsers.add_parser("val", help="Validate model on dataset")
    val_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    val_p.add_argument("--val-dir", required=True, help="Path to validation images")
    val_p.add_argument("--device", default="cuda")

    exp_p = subparsers.add_parser("export", help="Export model to ONNX format")
    exp_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    exp_p.add_argument("--output", default="model.onnx", help="Output path")
    exp_p.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")

    args = parser.parse_args()

    if args.command is None:
        _print_banner()
        parser.print_help()
        sys.exit(0)

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "train": cmd_train,
        "predict": cmd_predict,
        "val": cmd_val,
        "export": cmd_export,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
