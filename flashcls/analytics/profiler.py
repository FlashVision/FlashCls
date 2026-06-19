"""
Model profiler — layer-wise latency and memory analysis.
"""

import time
import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Profiler:
    """Profile FlashClassifier models layer-by-layer.

    Example::

        from flashcls.analytics import Profiler

        profiler = Profiler()
        report = profiler.profile(model)
        for layer in report:
            print(f"{layer['name']}: {layer['time_ms']:.3f} ms, {layer['params']:,} params")
    """

    def __init__(self, device: str = "cuda", num_iters: int = 100):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_iters = num_iters

    def profile(self, model: nn.Module, input_size: int = 224) -> List[Dict]:
        """Profile model and return per-module timing.

        Args:
            model: Model to profile.
            input_size: Input image size.

        Returns:
            List of dicts with name, time_ms, params, output_shape.
        """
        model = model.to(self.device).eval()
        dummy = torch.randn(1, 3, input_size, input_size, device=self.device)

        timings = {}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                if name not in timings:
                    timings[name] = {"times": [], "params": 0, "output_shape": None}
                timings[name]["times"].append(time.perf_counter())
                timings[name]["params"] = sum(p.numel() for p in module.parameters())
                if isinstance(out, torch.Tensor):
                    timings[name]["output_shape"] = list(out.shape)
                elif isinstance(out, (list, tuple)) and len(out) > 0:
                    timings[name]["output_shape"] = list(out[-1].shape) if isinstance(out[-1], torch.Tensor) else None
            return hook

        for name, module in model.named_children():
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

        with torch.no_grad():
            for _ in range(self.num_iters):
                model(dummy)

        for h in hooks:
            h.remove()

        report = []
        for name, data in timings.items():
            times = data["times"]
            if len(times) > 1:
                avg_time = (times[-1] - times[0]) / max(len(times) - 1, 1) * 1000
            else:
                avg_time = 0.0
            report.append({
                "name": name,
                "time_ms": avg_time,
                "params": data["params"],
                "output_shape": data["output_shape"],
            })

        return report

    def summary(self, model: nn.Module, input_size: int = 224) -> str:
        """Get a formatted profiling summary."""
        report = self.profile(model, input_size)
        lines = [f"{'Module':<20} {'Time (ms)':<12} {'Params':<12} {'Output Shape'}"]
        lines.append("-" * 70)
        for layer in report:
            shape_str = str(layer["output_shape"]) if layer["output_shape"] else "N/A"
            lines.append(
                f"{layer['name']:<20} {layer['time_ms']:<12.3f} {layer['params']:<12,} {shape_str}"
            )
        return "\n".join(lines)
