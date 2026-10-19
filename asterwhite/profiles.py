"""
AsterWhite :: profiles

Loads AsterWhiteConfig from configs/*.yaml instead of editing Python
source (spec 4.2: "Never depend on editing Python source to change normal
training settings"). Each profile YAML may set `extends: <other_profile>`
(defaulting to `base`) and only needs to list the keys it overrides --
deep-merged onto its parent before construction.

    cfg = load_profile("nvidia_single_gpu", overrides={"training": {"max_steps": 500}})

`overrides` (typically built from argparse/CLI flags) is merged in LAST,
so a command-line flag always wins over the profile file. validate_config
runs before returning -- a bad profile or override fails here, before any
model is allocated.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

from .config import AsterWhiteConfig, AsterWhiteConfigError, validate_config

_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

REQUIRED_PROFILES = [
    "base", "smoke_cpu", "tiny_cpu", "laptop_low_memory", "apple_mps", "intel_xpu",
    "nvidia_single_gpu", "amd_rocm_single_gpu", "multi_gpu_ddp", "multi_gpu_fsdp",
    "multi_node", "edge_inference", "production_inference", "long_context_extension",
    "pretrained_small",
]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(name: str) -> Dict[str, Any]:
    path = os.path.join(_CONFIGS_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise AsterWhiteConfigError(f"no such profile {name!r} (looked for {path}). "
                                     f"Known profiles: {REQUIRED_PROFILES}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise AsterWhiteConfigError(f"profile {name!r} did not parse to a mapping (got {type(data).__name__})")
    return data


def _resolve_chain(name: str, _seen: Optional[set] = None) -> Dict[str, Any]:
    _seen = _seen or set()
    if name in _seen:
        raise AsterWhiteConfigError(f"circular 'extends' chain detected at profile {name!r}")
    _seen.add(name)
    data = _load_yaml(name)
    parent_name = data.pop("extends", None)
    if parent_name is None and name != "base":
        parent_name = "base"
    if parent_name is not None:
        parent = _resolve_chain(parent_name, _seen)
        return _deep_merge(parent, data)
    return data


def load_profile(name: str, overrides: Optional[Dict[str, Any]] = None) -> AsterWhiteConfig:
    merged = _resolve_chain(name)
    if overrides:
        merged = _deep_merge(merged, overrides)
    # Profile YAML may carry deployment metadata (device/notes) that isn't
    # part of AsterWhiteConfig itself -- keep it out of the dataclass, but
    # don't silently drop it either; callers that care can read it back
    # via load_profile_raw().
    merged_config_only = {k: v for k, v in merged.items() if k in ("name", "backbone", "narrative", "training", "grpo", "inference")}
    cfg = AsterWhiteConfig.from_dict(merged_config_only)
    validate_config(cfg)
    return cfg


def load_profile_raw(name: str) -> Dict[str, Any]:
    """Full merged profile dict, including non-AsterWhiteConfig deployment
    metadata (e.g. `device`, `notes`, `world_size_hint`) -- useful for CLI
    tooling that wants to print/inspect a profile without constructing a
    model config."""
    return _resolve_chain(name)


def list_profiles() -> list:
    if not os.path.isdir(_CONFIGS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(_CONFIGS_DIR) if f.endswith(".yaml"))
