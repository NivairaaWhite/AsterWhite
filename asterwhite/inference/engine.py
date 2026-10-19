"""
AsterWhite :: inference.engine

The one place a checkpoint + tokenizer actually get loaded and turned into
runnable generation -- asterwhite.serving's HTTP handlers all go through
this, so there is exactly one code path to get right (auth/schema/
streaming-format differences live in the HTTP layer; loading, device
placement, quantization, and token-by-token decode all live here).

Thread-safety: AsterWhiteCore.generate doesn't mutate any shared module
state beyond reading parameters (this repo's norm layers are LayerNorm/
RMSNorm-style, not running-stats BatchNorm), so concurrent calls against
the same model instance are almost certainly fine -- "almost certainly"
is not a guarantee this module is willing to bet request correctness on,
so a lock serializes actual generate() calls regardless. Concurrency
across REQUESTS still happens (asterwhite.serving runs generation in a
worker thread per request via asyncio.to_thread) -- this lock only
prevents two of those threads from calling generate() on the shared model
at the literal same instant.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from ..config import AsterWhiteConfig
from ..core.device import DeviceInfo, DeviceManager
from ..core.model import AsterWhiteCore, sample_token
from ..registry.checkpoint import load_checkpoint
from ..registry.pretrained import adopt_checkpoint_config, bundled_tokenizer_dir
from ..tokenization import AsterWhiteTokenizer
from ..utils.error_tracing import AsterWhiteError, get_logger
from .quantization import quantize_for_inference

logger = get_logger("asterwhite.inference.engine")

UNTRAINED_MODEL_VERSION = "untrained-random-init"


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" (hit EOS) | "length" (hit max_new_tokens/budget)
    latency_ms: float


class InferenceEngine:
    """checkpoint_dir: trained weights. When it holds a stored config (every
    checkpoint Trainer/`train.py export` wrote does) the ARCHITECTURE is taken
    from it, so `cfg` (a serving profile) only has to supply serving settings
    -- sampling defaults, concurrency, timeouts. tokenizer_dir defaults to
    `<checkpoint_dir>/tokenizer` when that exists (a self-contained bundle)."""

    def __init__(self, cfg: AsterWhiteConfig, checkpoint_dir: Optional[str] = None,
                 tokenizer_dir: Optional[str] = None, device_override: Optional[str] = None):
        if checkpoint_dir:
            cfg = adopt_checkpoint_config(cfg, checkpoint_dir, keep_profile_settings=True)
            if tokenizer_dir is None:
                tokenizer_dir = bundled_tokenizer_dir(checkpoint_dir)
        self.cfg = cfg
        self.device_info: DeviceInfo = DeviceManager.resolve(device_override)
        self._lock = threading.Lock()

        self.tokenizer: Optional[AsterWhiteTokenizer] = (
            AsterWhiteTokenizer.load(tokenizer_dir) if tokenizer_dir else None
        )
        tokenizer_hash = self.tokenizer.compute_hash() if self.tokenizer else None
        if self.tokenizer is not None and self.tokenizer.vocab_size != cfg.backbone.vocab_size:
            logger.warning(f"tokenizer vocab_size={self.tokenizer.vocab_size} overrides "
                            f"config backbone.vocab_size={cfg.backbone.vocab_size}")
            cfg.backbone.vocab_size = self.tokenizer.vocab_size

        self.model = AsterWhiteCore(cfg.backbone)
        if checkpoint_dir:
            tensors, metadata = load_checkpoint(checkpoint_dir, expected_tokenizer_hash=tokenizer_hash)
            model_sd = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
            self.model.load_state_dict(model_sd, strict=True)
            self.model_version = f"step_{metadata.get('step', 'unknown')}"
            logger.info(f"InferenceEngine loaded trained weights from {checkpoint_dir} "
                        f"(model_version={self.model_version})")
        else:
            self.model_version = UNTRAINED_MODEL_VERSION
            logger.warning("InferenceEngine constructed with no checkpoint_dir -- serving RANDOM, "
                            "UNTRAINED weights. Every response will note this via model_version.")

        self.model.to(self.device_info.device)
        self.model.eval()

        if cfg.inference.quantize:
            if self.device_info.device.type != "cpu":
                raise AsterWhiteError(f"inference.quantize=true requires a CPU device (resolved device "
                                       f"was {self.device_info.device}) -- dynamic quantization is a "
                                       f"CPU-only backend in stock PyTorch.")
            self.model = quantize_for_inference(self.model)
            self.model_version += "-int8"

    def _encode(self, text: str):
        # A PROMPT to be continued: [BOS] text, with NO trailing EOS (the
        # default encode() would append one and tell the model the document
        # is already over).
        if self.tokenizer is not None:
            return self.tokenizer.encode_prompt(text)
        return [b % self.cfg.backbone.vocab_size for b in text.encode("utf-8")]

    def _decode(self, ids) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")

    def _resolved_sampling(self, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
        i = self.cfg.inference
        return (
            max_new_tokens if max_new_tokens is not None else i.default_max_new_tokens,
            temperature if temperature is not None else i.default_temperature,
            top_p if top_p is not None else i.default_top_p,
            (top_k if top_k is not None else i.default_top_k) or None,
            repetition_penalty if repetition_penalty is not None else i.default_repetition_penalty,
        )

    def generate_stream(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None,
                         top_p: Optional[float] = None, top_k: Optional[int] = None,
                         repetition_penalty: Optional[float] = None) -> Iterator[str]:
        """Yields one decoded piece per newly generated token, using the
        model's own incremental KV/SSM cache (see core.model.AsterWhiteCore.
        forward's cache param) so this is genuine token-by-token decode,
        not a chunked-up non-streaming call.

        Deliberately does NOT call AsterWhiteCore.generate() in a
        max_new_tokens=1 loop -- that would reset the repetition-penalty
        context to just the single most-recently-fed token on every
        iteration (verified by instrumenting core.model.sample_token
        during development: the context length was 1 for every step after
        the first). Instead this inlines the same forward+sample_token
        logic generate() uses, but keeps `full_ids` growing across the
        ENTIRE stream so repetition_penalty sees everything generated so
        far, exactly like a single non-streaming generate() call would."""
        max_new_tokens, temperature, top_p, top_k, repetition_penalty = self._resolved_sampling(
            max_new_tokens, temperature, top_p, top_k, repetition_penalty)

        prompt_ids = torch.tensor([self._encode(prompt)], device=self.device_info.device)
        if prompt_ids.shape[1] >= self.cfg.inference.max_total_tokens:
            raise AsterWhiteError(f"prompt is {prompt_ids.shape[1]} tokens, at or beyond "
                                   f"inference.max_total_tokens={self.cfg.inference.max_total_tokens}")
        max_new_tokens = min(max_new_tokens, self.cfg.inference.max_total_tokens - prompt_ids.shape[1])

        eos_id = self.tokenizer.eos_id if self.tokenizer is not None else None
        with self._lock:
            cache = self.model.new_cache()
            full_ids = prompt_ids  # grows every iteration; the repetition-penalty context
            logits = self.model.forward(full_ids, cache=cache)
            next_logits = logits[:, -1, :]
            for _ in range(max_new_tokens):
                next_id = sample_token(next_logits, temperature, top_p, top_k, repetition_penalty, full_ids)
                full_ids = torch.cat([full_ids, next_id], dim=1)
                piece = self._decode(next_id[0].tolist())
                yield piece
                if eos_id is not None and next_id.item() == eos_id:
                    return
                logits = self.model.forward(next_id, cache=cache)
                next_logits = logits[:, -1, :]

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None,
                 top_p: Optional[float] = None, top_k: Optional[int] = None,
                 repetition_penalty: Optional[float] = None) -> GenerationResult:
        t0 = time.time()
        resolved_max_new_tokens = self._resolved_sampling(max_new_tokens, temperature, top_p, top_k, repetition_penalty)[0]
        pieces = list(self.generate_stream(prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty))
        text = "".join(pieces)
        prompt_tokens = len(self._encode(prompt))
        completion_tokens = len(pieces)
        finish_reason = "length" if completion_tokens >= resolved_max_new_tokens else "stop"
        return GenerationResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                 finish_reason=finish_reason, latency_ms=(time.time() - t0) * 1000)
