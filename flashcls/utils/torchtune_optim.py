"""
Torchtune-inspired memory & performance optimizations for FlashCls.

Techniques ported from https://github.com/meta-pytorch/torchtune:
  - Activation checkpointing (gradient checkpointing)
  - Activation offloading to CPU
  - Fused optimizer step into backward pass
  - 8-bit AdamW via bitsandbytes
  - torch.compile wrapper
"""

import logging

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = logging.getLogger(__name__)


def apply_activation_checkpointing(model, target_modules=None):
    """Wrap submodules with gradient checkpointing to trade compute for memory."""
    if target_modules is None:
        target_modules = ["backbone"]

    wrapped = []
    for name in target_modules:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        original_forward = mod.forward

        def _make_ckpt_forward(orig_fn):
            def ckpt_forward(*args, **kwargs):
                if torch.is_grad_enabled():
                    return torch_checkpoint(orig_fn, *args, use_reentrant=False, **kwargs)
                return orig_fn(*args, **kwargs)
            return ckpt_forward

        mod.forward = _make_ckpt_forward(original_forward)
        wrapped.append(name)

    if wrapped:
        logger.info("Activation checkpointing enabled for: %s", ", ".join(wrapped))
    return model


class ActivationOffloadHook:
    """Offload activations to CPU during forward, reload on backward."""

    def __init__(self):
        self._handles = []

    def register(self, model, target_modules=None):
        if target_modules is None:
            target_modules = ["backbone"]
        for name in target_modules:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            for submod in mod.modules():
                h = submod.register_forward_hook(self._offload_hook)
                self._handles.append(h)
        logger.info("Activation offloading registered on %d sub-modules", len(self._handles))
        return self

    @staticmethod
    def _offload_hook(module, input, output):
        if not isinstance(output, torch.Tensor) or not output.requires_grad:
            return output
        cpu_output = output.detach().cpu()
        device = output.device

        class _Reload(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, cpu_copy):
                ctx.device = device
                ctx.cpu_copy = cpu_copy
                return x
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None

        return _Reload.apply(output, cpu_output)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def create_optimizer(model, lr, weight_decay=0.05, use_8bit=False,
                     optimizer_in_bwd=False, betas=(0.9, 0.999)):
    if optimizer_in_bwd:
        return _OptimizerInBwdWrapper(model, lr, weight_decay, use_8bit, betas)

    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr,
                                       weight_decay=weight_decay, betas=betas)
            logger.info("Using bitsandbytes 8-bit AdamW")
            return opt
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to standard AdamW")

    return torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=weight_decay, betas=betas)


class _OptimizerInBwdWrapper:
    def __init__(self, model, lr, weight_decay, use_8bit, betas):
        self.param_groups = [{"lr": lr}]
        self._per_param_optims = {}
        self._hooks = []

        opt_cls = torch.optim.AdamW
        if use_8bit:
            try:
                import bitsandbytes as bnb
                opt_cls = bnb.optim.AdamW8bit
            except ImportError:
                pass

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            opt = opt_cls([p], lr=lr, weight_decay=weight_decay, betas=betas)
            self._per_param_optims[name] = opt

            def _make_hook(optimizer):
                def hook(grad):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                return hook
            h = p.register_post_accumulate_grad_hook(_make_hook(opt))
            self._hooks.append(h)

    def step(self):
        pass

    def zero_grad(self, set_to_none=True):
        pass

    def state_dict(self):
        return {n: o.state_dict() for n, o in self._per_param_optims.items()}

    def load_state_dict(self, sd):
        for n, o in self._per_param_optims.items():
            if n in sd:
                o.load_state_dict(sd[n])

    def set_lr(self, lr):
        self.param_groups[0]["lr"] = lr
        for o in self._per_param_optims.values():
            for pg in o.param_groups:
                pg["lr"] = lr


def compile_model(model, backend="inductor"):
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available (requires PyTorch >= 2.0)")
        return model
    try:
        compiled = torch.compile(model, backend=backend)
        logger.info("torch.compile enabled (backend=%s)", backend)
        return compiled
    except Exception as e:
        logger.warning("torch.compile failed: %s. Using eager mode.", e)
        return model


def get_memory_stats(device):
    if device.type != "cuda":
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def log_memory_stats(device, prefix=""):
    stats = get_memory_stats(device)
    if stats:
        logger.info(
            "%sGPU Memory: %.1f MB allocated, %.1f MB reserved, %.1f MB peak",
            f"[{prefix}] " if prefix else "",
            stats["allocated_mb"], stats["reserved_mb"], stats["max_allocated_mb"],
        )
