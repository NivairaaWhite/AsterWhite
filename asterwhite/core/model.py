"""AsterWhite :: core.model -- AsterWhiteCore, the hybrid attention/SSM
decoder backbone shared by every subsystem: generation_orchestrator's prose
synthesis, agent_simulation's character dialogue, and alignment/grpo's
policy are all this same class, conditioned differently -- one set of
weights, many roles."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ..config import BackboneConfig, MODEL_NAME
from .layers import HybridBlock, RMSNorm


def _expand_pattern(pattern: str, n_layers: int) -> List[str]:
    return [pattern[i % len(pattern)] for i in range(n_layers)]


class AsterWhiteCore(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.name = MODEL_NAME
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        pattern = _expand_pattern(cfg.layer_pattern, cfg.n_layers)
        self.block_kinds = pattern
        self.blocks = nn.ModuleList([
            HybridBlock(kind, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff,
                        cfg.max_seq_len, cfg.rope_theta, cfg.ssm_state_dim,
                        cfg.ssm_conv_kernel, cfg.norm_eps, cfg.dropout,
                        rope_scaling=cfg.rope_scaling, qk_norm=cfg.qk_norm,
                        moe_cfg=cfg.moe, layer_index=i)
            for i, kind in enumerate(pattern)
        ])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Multi-token prediction (Gloeckle et al. 2024, parallel-head
        # variant): mtp_heads[i] predicts offset (i+2) from the SAME final
        # hidden state lm_head predicts offset +1 from -- zero extra trunk
        # forward passes. Untied, independently-initialized linear heads
        # (kept simple/robust over weight-tying them to tok_emb too).
        # Empty ModuleList (not None) when disabled, so `len(self.mtp_heads)`
        # and `for h in self.mtp_heads` work unconditionally everywhere
        # (training.trainer.Trainer, asterwhite.inference.speculative)
        # without an extra `is not None` check at every call site.
        self.mtp_heads = nn.ModuleList([
            nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) for _ in range(cfg.mtp_n_future_tokens)
        ])
        # Populated as a side effect of forward() whenever mtp_heads is
        # non-empty -- see forward()'s docstring for why this is a stashed
        # attribute rather than a second return value.
        self._last_mtp_logits: Optional[List[torch.Tensor]] = None

        # Real activation checkpointing (torch.utils.checkpoint) is applied
        # in forward_hidden below when this is True -- defaults off (exact
        # original behavior) for anyone constructing/using this class
        # without a Trainer. training.trainer.Trainer sets this to
        # TrainingConfig.activation_checkpointing right after construction;
        # it is deliberately a plain attribute, not a BackboneConfig field,
        # since it's a training-time compute/memory tradeoff rather than a
        # property of the architecture itself (the same weights mean the
        # same thing whether or not this is on).
        self.activation_checkpointing: bool = False

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def new_cache(self) -> List[dict]:
        """One (initially empty) state dict per block -- attention blocks
        fill it with k/v/offset, SSM blocks fill it with h/conv_buf. Persist
        this list (e.g. via torch.save) alongside the generated token ids to
        pause a run and resume it later from the exact same token."""
        return [dict() for _ in self.blocks]

    def forward_hidden(self, input_ids: torch.Tensor, cache: Optional[List[dict]] = None,
                        hidden_prefix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The trunk, through the final norm, BEFORE lm_head -- what
        forward() itself uses, and what training.narrative_objectives.
        pooled_hidden_states / narrative.revision's pooled-hidden pattern
        / asterwhite.inference.speculative's MTP drafting all want direct
        access to (a plain hidden-state tensor, not logits, and without
        each of those call sites re-deriving the same tok_emb+blocks+norm_f
        loop this method already is). Same (B, T, d_model) shape contract
        `forward`'s old inline version had; hidden_prefix positions are
        consumed as context and trimmed from the OUTPUT here (mathematically
        identical to trimming logits afterward, since lm_head/any other head
        applied downstream is a per-position linear map)."""
        x = self.tok_emb(input_ids)
        prefix_len = 0
        if hidden_prefix is not None:
            prefix_len = hidden_prefix.shape[1]
            x = torch.cat([hidden_prefix, x], dim=1)
        # Activation checkpointing recomputes each block's activations
        # during backward instead of storing them from the forward pass --
        # trading ~30% more compute time for memory that stops scaling with
        # n_layers. Deliberately restricted to cache is None (i.e. training,
        # never incremental generation): checkpoint() re-runs the wrapped
        # function a SECOND time during backward, and a HybridBlock given a
        # real kv_cache/ssm_state dict MUTATES it in place (appends k/v,
        # advances the SSM recurrent state) -- running that twice would
        # silently double-mutate the cache. Training never passes a cache
        # (always a single forward over the whole sequence), so this
        # restriction costs nothing in the one situation this is meant for.
        use_checkpoint = (self.activation_checkpointing and self.training
                           and torch.is_grad_enabled() and cache is None)
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache is not None else None
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, block_cache, use_reentrant=False)
            else:
                x = block(x, cache=block_cache)
        x = self.norm_f(x)
        return x[:, prefix_len:, :]

    def forward(self, input_ids: torch.Tensor, cache: Optional[List[dict]] = None,
                hidden_prefix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        input_ids: (B, T) int64 token ids.
        hidden_prefix: optional (B, P, d_model) conditioning vectors -- plan
            beats / world-state / character embeddings -- prepended before
            the block stack. This is how generation_orchestrator,
            agent_simulation, and universe_tracker steer this one shared
            backbone without separate weights per subsystem.
        cache: from new_cache(); pass the SAME list back in on every
            subsequent call to stream token-by-token.
        Returns logits (B, T, vocab_size) over the input_ids positions only
        (hidden_prefix positions are consumed as context, not predicted).

        Side effects (both opt-in via config, both no-ops -- and zero
        extra compute -- when disabled):
          - self._last_mtp_logits: set to a list of (B, T, vocab_size)
            tensors (one per BackboneConfig.mtp_n_future_tokens head) when
            mtp_heads is non-empty, else None. Computed INSIDE this same
            call (not via a second forward pass through self) specifically
            so a DDP/FSDP-wrapped caller (which only instruments the
            single self(...) call itself, not arbitrary submodule calls
            made afterward) still synchronizes mtp_heads' gradients
            correctly -- see training.trainer.Trainer._forward_loss.
          - every MoEFeedForward MLP in the block stack (if any) sets its
            own .last_aux_loss on every call -- see collect_moe_aux_loss().
        """
        hidden = self.forward_hidden(input_ids, cache=cache, hidden_prefix=hidden_prefix)
        logits = self.lm_head(hidden)
        if len(self.mtp_heads) > 0:
            self._last_mtp_logits = [head(hidden) for head in self.mtp_heads]
        else:
            self._last_mtp_logits = None
        return logits

    def collect_moe_aux_loss(self) -> torch.Tensor:
        """Sums every MoEFeedForward layer's last_aux_loss (see
        core.moe.MoEFeedForward) from the MOST RECENT forward() call.
        Returns a 0-valued tensor (not None, not a bare float -- so
        callers can always do loss + coef * this without a branch) when
        the model has no MoE layers or none have run yet."""
        total: Optional[torch.Tensor] = None
        for block in self.blocks:
            aux = getattr(block.mlp, "last_aux_loss", None)
            if aux is not None:
                total = aux if total is None else total + aux
        if total is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return total

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_p: float = 0.9, top_k: Optional[int] = None, repetition_penalty: float = 1.0,
                 hidden_prefix: Optional[torch.Tensor] = None,
                 eos_token_id: Optional[int] = None,
                 cache: Optional[List[dict]] = None,
                 max_total_tokens: Optional[int] = None) -> Tuple[torch.Tensor, List[dict]]:
        """Token-by-token incremental decode. Pass in a `cache` saved from a
        previous (possibly checkpointed-to-disk) call, together with
        whatever token comes next, to resume generation from the exact
        point it left off -- the manual-intervention / resume-from-token
        hook generation_orchestrator exposes to the author. Returns
        (generated_ids, cache) so the caller can persist `cache` for the
        next pause.

        top_k, repetition_penalty: standard modern sampling controls
        (repetition_penalty follows the CTRL/Keskar-et-al. convention:
        positive logits for already-seen tokens are divided by the
        penalty, negative logits multiplied -- both push toward less
        repetition regardless of a logit's sign).
        max_total_tokens: if set, raises before generating past this many
        total tokens (prompt + generated) -- the serving layer's context
        budget enforcement point; None (default) preserves the previous
        unbounded behavior for library/CLI callers.

        For a (usually) faster alternative when this model has
        BackboneConfig.mtp_n_future_tokens > 0, see
        asterwhite.inference.speculative.speculative_generate -- same
        cache/resume contract, self-speculative drafting via mtp_heads."""
        self.eval()
        if max_total_tokens is not None and input_ids.shape[1] + max_new_tokens > max_total_tokens:
            raise ValueError(f"requested prompt({input_ids.shape[1]}) + max_new_tokens({max_new_tokens}) "
                              f"= {input_ids.shape[1] + max_new_tokens} exceeds max_total_tokens="
                              f"{max_total_tokens}")
        if cache is None:
            cache = self.new_cache()
        logits = self.forward(input_ids, cache=cache, hidden_prefix=hidden_prefix)
        out = input_ids
        next_logits = logits[:, -1, :]
        for _ in range(max_new_tokens):
            next_id = _sample(next_logits, temperature, top_p, top_k, repetition_penalty, out)
            out = torch.cat([out, next_id], dim=1)
            if eos_token_id is not None and torch.all(next_id.squeeze(-1) == eos_token_id):
                break
            logits = self.forward(next_id, cache=cache)
            next_logits = logits[:, -1, :]
        return out, cache


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, generated_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(1, generated_ids, score)


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: Optional[int],
                  repetition_penalty: float, generated_ids: torch.Tensor) -> torch.Tensor:
    """Public entry point for the same sampling logic AsterWhiteCore.generate
    uses internally -- exposed so a caller that needs token-by-token control
    (asterwhite.inference.engine.InferenceEngine.generate_stream, which
    must maintain its OWN growing repetition-penalty context across
    streamed calls rather than resetting it every call) can reuse this
    exactly instead of re-implementing or subtly diverging from it."""
    logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)
    logits = logits / max(temperature, 1e-5)
    if top_k is not None and 0 < top_k < logits.shape[-1]:
        kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, 1)
    return torch.gather(sorted_idx, -1, choice)


_sample = sample_token  # backward-compatible alias for the original private name
