"""
AsterWhite :: inference.speculative

Self-speculative decoding using the backbone's own multi-token-prediction
heads (core.model.AsterWhiteCore.mtp_heads -- the Gloeckle et al. 2024
parallel-head variant, see BackboneConfig.mtp_n_future_tokens) as a
zero-extra-trunk-forward-pass drafter: one ordinary forward call already
produces a hidden state that lm_head projects to offset +1 and each
mtp_heads[i] projects to offset (i+2) -- so K = 1 + len(mtp_heads) draft
tokens are available immediately after the model produces its most recent
real token, with no second "draft model" and no extra pass through the
(expensive) block stack.

Verification follows the standard speculative-decoding rejection-sampling
algorithm (Leviathan et al. 2023 / Chen et al. 2023): the K drafted tokens
are checked in ONE forward pass through the FULL model (not the cheap MTP
heads), and the longest verified-correct prefix is kept; the first
disagreement is corrected by resampling from the residual distribution
(p - q, renormalized) rather than simply substituting the target's argmax
-- so accepted output matches the TARGET model's own distribution, never
merely the (possibly weaker) drafter's guesses. When every drafted token
is accepted, one further "bonus" token is sampled for free from the
verifier's own distribution at the position right after the draft.

Honesty about scope: acceptance rate (and therefore the actual speedup
over ordinary sequential decoding) depends entirely on how well-trained
the MTP heads are. With freshly-initialized (random) heads, expect a low
acceptance rate and performance close to -- not faster than -- sequential
decoding, since a rejected draft costs a wasted verification pass. The
algorithm's OUTPUT is correct regardless (see test_speculative.py's
exact-match-with-sequential-decoding test under a "perfect drafter"): a
badly-trained drafter costs speed, never correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..core.model import AsterWhiteCore
from ..utils.error_tracing import AsterWhiteError, get_logger

logger = get_logger("asterwhite.inference.speculative")


def clone_cache(cache: List[dict]) -> List[dict]:
    """Deep-clones a cache list (each block's attention k/v/offset or SSM
    h/conv_buf dict) so mutating the clone can never touch the original --
    the snapshot-before-drafting, discard-and-redo-if-rejected strategy in
    speculative_generate below depends on this being a true copy, not a
    shared reference to the same tensors."""
    return [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in block_cache.items()} for block_cache in cache]


@dataclass
class SpeculativeStats:
    """Running totals across an entire speculative_generate call --
    accepted / proposed is the empirical acceptance rate, the single
    number that determines whether this was actually faster than
    sequential decoding for this drafter/prompt/model combination."""
    proposed: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


def _draft_tokens(model: AsterWhiteCore, last_hidden: torch.Tensor, temperature: float, greedy: bool,
                   generator: Optional[torch.Generator]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """last_hidden: (B, d_model), the final hidden state at the most
    recently confirmed real position. Returns (draft_ids: (B, K) long,
    draft_logits: list of K tensors each (B, vocab)) using lm_head for
    offset +1 and mtp_heads for offsets +2..+1+len(mtp_heads) -- zero
    extra trunk forward passes, see module docstring."""
    heads = [model.lm_head] + list(model.mtp_heads)
    draft_ids, draft_logits = [], []
    for head in heads:
        logits = head(last_hidden)  # (B, vocab)
        draft_logits.append(logits)
        if greedy:
            draft_ids.append(logits.argmax(dim=-1, keepdim=True))
        else:
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
            draft_ids.append(torch.multinomial(probs, 1, generator=generator))
    return torch.cat(draft_ids, dim=1), draft_logits  # (B, K), list[K] of (B, vocab)


def _verify_one_position(draft_token: torch.Tensor, draft_logits: torch.Tensor, target_logits: torch.Tensor,
                          temperature: float, greedy: bool,
                          generator: Optional[torch.Generator]) -> Tuple[bool, Optional[torch.Tensor]]:
    """One step of the Leviathan/Chen rejection-sampling test. Returns
    (accepted, correction_token_or_None) for a single batch-1 position.
    Only called with batch size 1 -- see speculative_generate's docstring
    for why batching is out of scope for this reference implementation."""
    if greedy:
        accepted = bool((draft_token == target_logits.argmax(dim=-1)).all().item())
        if accepted:
            return True, None
        return False, target_logits.argmax(dim=-1, keepdim=True)

    p = F.softmax(target_logits / max(temperature, 1e-5), dim=-1)
    q = F.softmax(draft_logits / max(temperature, 1e-5), dim=-1)
    p_x = p.gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
    q_x = q.gather(-1, draft_token.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
    accept_prob = torch.clamp(p_x / q_x, max=1.0)
    u = torch.rand((), generator=generator) if generator is not None else torch.rand(())
    if bool((u < accept_prob).item()):
        return True, None
    residual = torch.clamp(p - q, min=0.0)
    residual_sum = residual.sum(dim=-1, keepdim=True)
    if float(residual_sum.item()) <= 1e-8:
        return False, target_logits.argmax(dim=-1, keepdim=True)
    residual = residual / residual_sum
    return False, torch.multinomial(residual, 1, generator=generator)


def _sample_bonus(logits: torch.Tensor, temperature: float, greedy: bool,
                   generator: Optional[torch.Generator]) -> torch.Tensor:
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
    return torch.multinomial(probs, 1, generator=generator)


@torch.no_grad()
def speculative_generate(model: AsterWhiteCore, input_ids: torch.Tensor, max_new_tokens: int,
                          temperature: float = 1.0, greedy: bool = False,
                          eos_token_id: Optional[int] = None, cache: Optional[List[dict]] = None,
                          generator: Optional[torch.Generator] = None,
                          ) -> Tuple[torch.Tensor, List[dict], SpeculativeStats]:
    """Drop-in (batch-size-1 only, see below) alternative to
    AsterWhiteCore.generate for models with BackboneConfig.mtp_n_future_tokens
    > 0 -- same cache/resume contract (pass a previously-returned `cache`
    back in to continue), same output distribution (see this module's
    docstring), usually fewer trunk forward passes for the same number of
    output tokens.

    Batch size 1 only: the per-position accept/reject decision can accept
    a DIFFERENT number of tokens for different batch elements, which would
    need ragged/padded cache handling to batch correctly -- out of scope
    for this reference implementation (a per-request decoding loop, as
    inference.engine.InferenceEngine already uses, calls this once per
    request rather than batching requests together). Raises AsterWhiteError
    on a batch dimension other than 1 rather than silently only handling
    the first row.

    Not currently wired into InferenceEngine/the HTTP serving layer by
    default -- call it directly for now (e.g. from a custom script, or
    swapped in for AsterWhiteCore.generate inside inference.engine for
    your own deployment) once you've trained MTP heads worth drafting
    with; see this module's docstring for why an untrained model won't
    actually be faster."""
    if input_ids.shape[0] != 1:
        raise AsterWhiteError(f"speculative_generate only supports batch size 1 (got {input_ids.shape[0]}) -- "
                               f"see this function's docstring for why; call it once per request instead of "
                               f"batching requests together.")
    if len(model.mtp_heads) == 0:
        raise AsterWhiteError("speculative_generate requires backbone.mtp_n_future_tokens > 0 (this model has "
                               "no mtp_heads) -- use AsterWhiteCore.generate for ordinary sequential decoding.")
    model.eval()
    if cache is None:
        cache = model.new_cache()
    stats = SpeculativeStats()

    hidden = model.forward_hidden(input_ids, cache=cache)
    last_hidden = hidden[:, -1, :]
    out = input_ids
    generated = 0

    while generated < max_new_tokens:
        draft_ids, draft_logits = _draft_tokens(model, last_hidden, temperature, greedy, generator)
        k_max = min(draft_ids.shape[1], max_new_tokens - generated)
        draft_ids = draft_ids[:, :k_max]

        pre_draft_cache = clone_cache(cache)
        verify_cache = clone_cache(cache)
        verify_hidden = model.forward_hidden(draft_ids, cache=verify_cache)  # (1, K, d_model)
        verify_logits = model.lm_head(verify_hidden)  # (1, K, vocab) -- target distribution at each drafted slot

        n_accept = 0
        correction_id: Optional[torch.Tensor] = None
        # target_logits[i] is what draft_ids[:, i] must be checked against.
        # draft_ids[:, 0] was produced by lm_head(last_hidden) -- exactly
        # the model's real, unconditional next-token distribution given the
        # confirmed context so far -- so it is its OWN target and always
        # trivially accepted (no trunk forward pass needed to confirm the
        # very first drafted token). draft_ids[:, i] for i >= 1 was drafted
        # from that SAME stale hidden state (the parallel-MTP-head
        # approximation -- see this module's docstring), but must be
        # verified against what the model ACTUALLY predicts after having
        # genuinely consumed draft_ids[:, 0 .. i-1] -- that's
        # verify_hidden's position (i-1), not position i: verify_hidden[:,
        # j, :] is the hidden state after consuming input positions 0..j,
        # i.e. it predicts position j+1, i.e. draft_ids[:, j+1].
        target_logits = [draft_logits[0]] + [verify_logits[:, j, :] for j in range(k_max - 1)]
        for i in range(k_max):
            accepted, correction = _verify_one_position(
                draft_ids[:, i], draft_logits[i], target_logits[i], temperature, greedy, generator)
            if accepted:
                n_accept += 1
                continue
            correction_id = correction
            break
        else:
            # every drafted token verified -- one bonus token, free, from
            # the verifier's own distribution at the position right after
            # the LAST draft token: verify_hidden[:, k_max-1, :] is the
            # hidden state after consuming all k_max draft tokens, i.e. it
            # predicts exactly the position right after them.
            correction_id = _sample_bonus(verify_logits[:, k_max - 1, :], temperature, greedy, generator)

        stats.proposed += k_max
        stats.accepted += n_accept

        if n_accept == k_max:
            # Nothing rejected: verify_cache already reflects exactly what
            # sequential generation would have produced for these k_max
            # tokens -- keep it. The bonus token is genuinely free ONLY
            # when max_new_tokens allows one more position; otherwise
            # adding it would overrun the caller's requested budget by one.
            cache = verify_cache
            if generated + k_max < max_new_tokens:
                new_tokens = torch.cat([draft_ids, correction_id], dim=1)
                hidden_next = model.forward_hidden(correction_id, cache=cache)
                last_hidden = hidden_next[:, -1, :]
            else:
                new_tokens = draft_ids
                last_hidden = verify_hidden[:, -1, :]  # still valid if a later call resumes with this cache
        else:
            # verify_cache reflects the FULL (partially wrong) draft and is
            # discarded; redo a small forward over just the accepted
            # prefix + correction token, from the untouched pre-draft
            # snapshot, to get a cache that's exactly correct.
            redo_ids = torch.cat([draft_ids[:, :n_accept], correction_id], dim=1)
            cache = pre_draft_cache
            hidden_redo = model.forward_hidden(redo_ids, cache=cache)
            new_tokens = redo_ids
            last_hidden = hidden_redo[:, -1, :]

        out = torch.cat([out, new_tokens], dim=1)
        generated += new_tokens.shape[1]

        if eos_token_id is not None and bool((new_tokens == eos_token_id).any().item()):
            break

    if stats.proposed:
        logger.info(f"speculative_generate: {stats.accepted}/{stats.proposed} draft tokens accepted "
                     f"({stats.acceptance_rate:.1%})")
    return out, cache, stats
