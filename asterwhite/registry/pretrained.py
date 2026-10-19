"""
AsterWhite :: registry.pretrained

Locating and loading SELF-CONTAINED pretrained model bundles.

A bundle is an ordinary checkpoint directory (registry.checkpoint's atomic,
checksummed safetensors + JSON format -- `latest.json`, `model.json`,
`model.safetensors`) that ALSO carries a `tokenizer/` subdirectory:

    pretrained/asterwhite-small/
        latest.json  model.json  model.safetensors    <- weights + the exact config they were trained with
        tokenizer/tokenizer.json  tokenizer_meta.json <- the vocabulary those weights belong to
        MODEL_CARD.md

Everything a loader needs travels inside that one directory, so
`train.py generate --load pretrained/asterwhite-small` needs no --profile
and no --tokenizer-dir: the architecture is read back from the checkpoint's
stored config (a profile can never silently disagree with the weights it is
being loaded into -- the strict state-dict load would fail with a wall of
shape mismatches), and the tokenizer is found next to the weights.

`train.py export` produces a bundle from a training checkpoint.

The bundled default is looked up under `pretrained/` at the repository root,
or under $ASTERWHITE_PRETRAINED_DIR if set (an empty value means "no
bundles", which is how the test suite exercises the no-model path).
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional

from ..config import AsterWhiteConfig, BackboneConfig, validate_config
from ..utils.error_tracing import AsterWhiteError, get_logger
from .checkpoint import resolve_checkpoint_name

logger = get_logger("asterwhite.registry.pretrained")

DEFAULT_BUNDLE_NAME = "asterwhite-small"
PRETRAINED_DIR_ENV = "ASTERWHITE_PRETRAINED_DIR"
TOKENIZER_SUBDIR = "tokenizer"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pretrained_root() -> str:
    """Directory searched for bundles. $ASTERWHITE_PRETRAINED_DIR wins when
    set (even if empty -> no bundles); otherwise <repo>/pretrained."""
    env = os.environ.get(PRETRAINED_DIR_ENV)
    if env is not None:
        return env
    return os.path.join(_REPO_ROOT, "pretrained")


def is_checkpoint_dir(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "latest.json"))


def default_bundle_dir() -> Optional[str]:
    """Path of the bundled default model, or None if it is not present
    (e.g. a source checkout without the weights, or an empty override)."""
    root = pretrained_root()
    if not root:
        return None
    path = os.path.join(root, DEFAULT_BUNDLE_NAME)
    return path if is_checkpoint_dir(path) else None


def bundled_tokenizer_dir(checkpoint_dir: Optional[str]) -> Optional[str]:
    """`<checkpoint_dir>/tokenizer` if it holds a tokenizer, else None."""
    if not checkpoint_dir:
        return None
    candidate = os.path.join(checkpoint_dir, TOKENIZER_SUBDIR)
    return candidate if os.path.isfile(os.path.join(candidate, "tokenizer.json")) else None


def read_checkpoint_metadata(checkpoint_dir: str, name: Optional[str] = None, which: str = "latest") -> Dict[str, Any]:
    """Reads ONLY the JSON sidecar (cheap -- no tensors are touched and no
    checksum is computed; load_checkpoint does the integrity check when the
    weights are actually loaded)."""
    resolved = resolve_checkpoint_name(checkpoint_dir, name, which)
    meta_path = os.path.join(checkpoint_dir, f"{resolved}.json")
    if not os.path.isfile(meta_path):
        raise AsterWhiteError(f"checkpoint {resolved!r} has no metadata file at {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def stored_config(checkpoint_dir: str, name: Optional[str] = None, which: str = "latest") -> Optional[AsterWhiteConfig]:
    """The AsterWhiteConfig a checkpoint was saved with (Trainer.save_checkpoint
    stores it), or None for a checkpoint written without one."""
    meta = read_checkpoint_metadata(checkpoint_dir, name, which)
    raw = meta.get("config")
    if not raw:
        return None
    try:
        cfg = AsterWhiteConfig.from_dict(raw)
        validate_config(cfg)
    except Exception as exc:  # noqa: BLE001 -- re-raised with the checkpoint named
        raise AsterWhiteError(f"the config stored inside checkpoint {checkpoint_dir!r} is invalid: "
                               f"{type(exc).__name__}: {exc}", cause=exc) from exc
    return cfg


def effective_context_length(backbone: BackboneConfig) -> int:
    """Longest sequence this backbone was set up to handle: max_seq_len, or
    that times the RoPE scaling factor when context extension is configured."""
    rs = backbone.rope_scaling
    if rs.strategy != "none":
        base = rs.original_max_position_embeddings or backbone.max_seq_len
        return int(base * rs.factor)
    return backbone.max_seq_len


def adopt_checkpoint_config(cfg: Optional[AsterWhiteConfig], checkpoint_dir: str, *,
                            keep_profile_settings: bool, name: Optional[str] = None,
                            which: str = "latest") -> AsterWhiteConfig:
    """Returns the config to use for running the weights in `checkpoint_dir`.

    The BACKBONE (architecture) always comes from the checkpoint -- it is
    part of the weights' identity, and any other value cannot load.

    keep_profile_settings=True  (the user passed --profile / a profile is
        configured): narrative/training/grpo/inference come from `cfg`
        (the profile) -- the profile can still tune sampling defaults,
        concurrency, timeouts.
    keep_profile_settings=False (no profile requested): EVERYTHING comes
        from the checkpoint, so a bundle's own tuned inference defaults
        apply.

    `inference.max_total_tokens` is capped at the backbone's effective
    context length: running a model past the length it was trained for
    degrades output, and a profile written for a bigger model (e.g.
    production_inference's 8192) must not silently push a small one there.

    A checkpoint with no stored config returns `cfg` unchanged (the old
    behavior); with no `cfg` either that is an error."""
    ckpt_cfg = stored_config(checkpoint_dir, name, which)
    if ckpt_cfg is None:
        if cfg is None:
            raise AsterWhiteError(f"checkpoint {checkpoint_dir!r} has no stored config and none was supplied -- "
                                   f"pass --profile matching the architecture it was trained with.")
        return cfg

    if cfg is None or not keep_profile_settings:
        merged = ckpt_cfg
    else:
        merged = copy.deepcopy(cfg)
        merged.backbone = ckpt_cfg.backbone
        if cfg.backbone != ckpt_cfg.backbone:
            logger.info(f"architecture taken from the checkpoint's stored config, not the profile "
                        f"(profile asked for d_model={cfg.backbone.d_model}, n_layers={cfg.backbone.n_layers}; "
                        f"checkpoint has d_model={ckpt_cfg.backbone.d_model}, n_layers={ckpt_cfg.backbone.n_layers})")

    context = effective_context_length(merged.backbone)
    if merged.inference.max_total_tokens > context:
        logger.warning(f"inference.max_total_tokens={merged.inference.max_total_tokens} exceeds this model's "
                        f"context length ({context}) -- capping it there")
        merged.inference.max_total_tokens = context
        merged.inference.default_max_new_tokens = min(merged.inference.default_max_new_tokens, context)
    validate_config(merged)
    return merged
