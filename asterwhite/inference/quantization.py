"""
AsterWhite :: inference.quantization

Dynamic INT8 quantization (torch.ao.quantization.quantize_dynamic) for
CPU inference -- what configs/edge_inference.yaml's inference.quantize
flag turns on. This is post-training dynamic quantization (weights only,
computed once at load time; activations quantized on the fly per-op) --
it needs no calibration dataset, which is why it's the right default for
"quantize whatever checkpoint you hand it," unlike static/QAT
quantization, which need one.

CUDA/ROCm/MPS/XPU: dynamic quantization is a CPU-backend feature in
stock PyTorch; quantize_for_inference raises rather than silently
skipping if asked to quantize a non-CPU model, so a misconfigured
--device + --quantize combination fails loudly instead of quietly
serving unquantized weights under a "quantized" label.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.error_tracing import AsterWhiteError, get_logger

logger = get_logger("asterwhite.inference.quantization")


def quantize_for_inference(model: nn.Module) -> nn.Module:
    """Returns a NEW module with every nn.Linear dynamically quantized to
    INT8 -- the base model passed in is left untouched. Must be called
    with the model already on CPU (see module docstring)."""
    devices = {p.device.type for p in model.parameters()}
    if devices != {"cpu"}:
        raise AsterWhiteError(f"quantize_for_inference requires a CPU-resident model (found device(s) "
                               f"{devices}) -- torch's dynamic quantization backend is CPU-only. Move the "
                               f"model to CPU first (model.to('cpu')), or don't set inference.quantize for "
                               f"a GPU deployment.")
    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    orig_params = sum(p.numel() for p in model.parameters())
    logger.info(f"dynamic INT8 quantization applied to all nn.Linear layers "
                f"(pre-quantization param count: {orig_params:,})")
    return quantized
