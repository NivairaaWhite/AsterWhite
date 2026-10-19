"""
AsterWhite :: training.lora

Low-Rank Adaptation (Hu et al. 2021). Freezes the base weight of a target
`nn.Linear` and learns a rank-r delta `(alpha/r) * B @ A` instead of the
full matrix -- the spec's "Parameter-efficient fine-tuning such as LoRA or
adapters" platform requirement, and standard practice for fine-tuning a
modern LLM without the memory cost of full-parameter training.

Usage:
    model = AsterWhiteCore(cfg.backbone)
    apply_lora(model, target_substrings=("q_proj", "v_proj"), r=8, alpha=16)
    mark_only_lora_as_trainable(model)
    # ... train normally; only LoRA A/B matrices accumulate gradients ...
    save_lora_adapter(model, "adapters/my_style")
    # later, on a fresh base model:
    apply_lora(fresh_model, target_substrings=("q_proj", "v_proj"), r=8, alpha=16)
    load_lora_adapter(fresh_model, "adapters/my_style")
    # or, to deploy without the extra matmul at inference time:
    merge_lora_weights(fresh_model)
"""
from __future__ import annotations

import os
from typing import Dict, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from ..utils.error_tracing import AsterWhiteError, get_logger

logger = get_logger("asterwhite.training.lora")


class LoRALinear(nn.Module):
    """Drop-in replacement for an existing nn.Linear: same forward shape,
    frozen base weight, plus a trainable low-rank delta."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert r > 0
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.scaling = alpha / r

        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)  # B starts at zero -> delta starts at exactly zero
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def apply_lora(model: nn.Module, target_substrings: Sequence[str] = ("q_proj", "v_proj"),
               r: int = 8, alpha: int = 16, dropout: float = 0.0) -> int:
    """Walks `model`, replacing every nn.Linear submodule whose dotted name
    contains one of `target_substrings` with a LoRALinear wrapper, in
    place. Returns the count of layers wrapped (raises if zero -- a silent
    no-op here would mean "training" that touches nothing)."""
    replaced = 0
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name for t in target_substrings):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
    if replaced == 0:
        raise AsterWhiteError(f"apply_lora matched zero nn.Linear layers for target_substrings="
                               f"{target_substrings} -- check the names against model.named_modules().")
    logger.info(f"LoRA applied to {replaced} layer(s): r={r} alpha={alpha} target_substrings={target_substrings}")
    return replaced


def mark_only_lora_as_trainable(model: nn.Module, train_bias: bool = False) -> None:
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_bias = train_bias and name.endswith(".bias")
        param.requires_grad_(is_lora or is_bias)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA-only training: {n_trainable:,}/{n_total:,} parameters trainable "
                f"({100 * n_trainable / max(1, n_total):.2f}%)")


def _lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().cpu().contiguous() for name, p in model.named_parameters()
            if "lora_A" in name or "lora_B" in name}


def save_lora_adapter(model: nn.Module, directory: str, metadata: Dict[str, str] = None) -> str:
    os.makedirs(directory, exist_ok=True)
    tensors = _lora_state_dict(model)
    if not tensors:
        raise AsterWhiteError("no LoRA parameters found on this model -- call apply_lora() first.")
    path = os.path.join(directory, "adapter.safetensors")
    save_file(tensors, path, metadata=metadata or {})
    logger.info(f"LoRA adapter saved -> {path} ({len(tensors)} tensors)")
    return path


def load_lora_adapter(model: nn.Module, directory: str, strict: bool = True) -> int:
    path = os.path.join(directory, "adapter.safetensors")
    if not os.path.exists(path):
        raise AsterWhiteError(f"no adapter.safetensors in {directory}")
    tensors = load_file(path)
    own = dict(model.named_parameters())
    loaded = 0
    for name, tensor in tensors.items():
        if name not in own:
            if strict:
                raise AsterWhiteError(f"adapter tensor {name!r} has no matching parameter on this model -- "
                                       f"did you call apply_lora() with the same target_substrings/r first?")
            continue
        own[name].data.copy_(tensor)
        loaded += 1
    logger.info(f"LoRA adapter loaded <- {path} ({loaded} tensors)")
    return loaded


def merge_lora_weights(model: nn.Module) -> int:
    """Folds every LoRALinear's delta into a plain nn.Linear, in place --
    for deploying a fine-tuned adapter without paying the extra matmul (and
    extra file) at inference time. Irreversible on this model instance."""
    merged = 0
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                plain = nn.Linear(child.in_features, child.out_features, bias=child.base.bias is not None)
                plain.weight.data.copy_(child.merged_weight())
                if child.base.bias is not None:
                    plain.bias.data.copy_(child.base.bias.data)
                setattr(parent, child_name, plain)
                merged += 1
    logger.info(f"merged {merged} LoRA layer(s) back into plain nn.Linear for deployment")
    return merged
