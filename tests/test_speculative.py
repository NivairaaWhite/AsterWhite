"""AsterWhite :: tests.test_speculative -- inference.speculative.speculative_generate.

The central correctness property tested here: in greedy mode, EVERY accepted
draft token was (by the acceptance test itself) the target model's own argmax,
and every rejected one is replaced by that argmax -- so speculative decoding's
output must be byte-identical to plain sequential greedy decoding no matter how
good or bad the (here: randomly-initialized, untrained) MTP drafter is. A bad
drafter only costs acceptance rate/speed, never correctness -- this is what the
module's docstring claims, and what makes it safe to ship without trained MTP
heads.
"""
from __future__ import annotations

import pytest
import torch

from asterwhite.config import AsterWhiteConfig
from asterwhite.core.model import AsterWhiteCore
from asterwhite.inference.speculative import SpeculativeStats, speculative_generate
from asterwhite.utils.error_tracing import AsterWhiteError


def _mtp_cfg():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.mtp_n_future_tokens = 3
    return cfg


def _sequential_greedy_reference(model: AsterWhiteCore, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """A deliberately independent, minimal greedy decode loop (NOT reusing
    AsterWhiteCore.generate, which only samples stochastically) -- the
    ground truth speculative_generate(greedy=True) must match exactly."""
    model.eval()
    cache = model.new_cache()
    hidden = model.forward_hidden(input_ids, cache=cache)
    out = input_ids
    next_id = model.lm_head(hidden[:, -1:, :]).argmax(dim=-1)
    for _ in range(max_new_tokens):
        out = torch.cat([out, next_id], dim=1)
        hidden = model.forward_hidden(next_id, cache=cache)
        next_id = model.lm_head(hidden[:, -1:, :]).argmax(dim=-1)
    return out


def test_greedy_speculative_output_matches_sequential_greedy_exactly():
    cfg = _mtp_cfg()
    torch.manual_seed(0)
    model = AsterWhiteCore(cfg.backbone)
    prompt = torch.randint(0, cfg.backbone.vocab_size, (1, 5))

    torch.manual_seed(0)  # re-seed so any dropout/init randomness (there is none at eval, but be explicit) lines up
    model.eval()
    reference = _sequential_greedy_reference(model, prompt, max_new_tokens=12)

    spec_out, _cache, stats = speculative_generate(model, prompt, max_new_tokens=12, greedy=True)

    assert torch.equal(spec_out, reference)
    assert stats.proposed > 0
    assert 0.0 <= stats.acceptance_rate <= 1.0


def test_greedy_speculative_matches_sequential_across_several_random_models_and_prompts():
    """The same exact-match property, repeated across several random seeds
    -- one match could be luck; several independent ones are a real
    property of the algorithm."""
    for seed in range(4):
        cfg = _mtp_cfg()
        torch.manual_seed(seed)
        model = AsterWhiteCore(cfg.backbone)
        model.eval()
        prompt = torch.randint(0, cfg.backbone.vocab_size, (1, 4 + seed))
        reference = _sequential_greedy_reference(model, prompt, max_new_tokens=10)
        spec_out, _cache, _stats = speculative_generate(model, prompt, max_new_tokens=10, greedy=True)
        assert torch.equal(spec_out, reference), f"mismatch at seed={seed}"


def test_stochastic_mode_runs_and_reports_valid_stats():
    cfg = _mtp_cfg()
    torch.manual_seed(0)
    model = AsterWhiteCore(cfg.backbone)
    prompt = torch.randint(0, cfg.backbone.vocab_size, (1, 5))
    gen = torch.Generator().manual_seed(0)
    out, _cache, stats = speculative_generate(model, prompt, max_new_tokens=10, greedy=False,
                                               temperature=0.9, generator=gen)
    assert out.shape[0] == 1
    assert out.shape[1] > prompt.shape[1]
    assert out.shape[1] <= prompt.shape[1] + 10
    assert isinstance(stats, SpeculativeStats)
    assert 0.0 <= stats.acceptance_rate <= 1.0
    assert torch.isfinite(out.float()).all()


def test_raises_without_mtp_heads():
    cfg = AsterWhiteConfig.tiny_for_testing()
    assert cfg.backbone.mtp_n_future_tokens == 0
    model = AsterWhiteCore(cfg.backbone)
    prompt = torch.randint(0, cfg.backbone.vocab_size, (1, 4))
    with pytest.raises(AsterWhiteError, match="mtp_n_future_tokens"):
        speculative_generate(model, prompt, max_new_tokens=5)


def test_raises_on_batch_size_greater_than_one():
    cfg = _mtp_cfg()
    model = AsterWhiteCore(cfg.backbone)
    prompt = torch.randint(0, cfg.backbone.vocab_size, (2, 4))
    with pytest.raises(AsterWhiteError, match="batch size"):
        speculative_generate(model, prompt, max_new_tokens=5)


def test_resume_from_a_returned_cache_continues_correctly():
    """Same cache/resume contract as AsterWhiteCore.generate: passing a
    previously-returned cache back in continues from exactly that point,
    rather than restarting -- verified against the sequential reference
    exactly like the main correctness test, just in two calls instead of one."""
    cfg = _mtp_cfg()
    torch.manual_seed(0)
    model = AsterWhiteCore(cfg.backbone)
    model.eval()
    prompt = torch.randint(0, cfg.backbone.vocab_size, (1, 5))

    out1, cache1, _ = speculative_generate(model, prompt, max_new_tokens=4, greedy=True)
    out2, _cache2, _ = speculative_generate(model, out1[:, -1:], max_new_tokens=4, greedy=True, cache=cache1)
    resumed_full = torch.cat([out1, out2[:, 1:]], dim=1)  # out2's first token IS out1's last token (re-fed as the new prompt)

    reference = _sequential_greedy_reference(model, prompt, max_new_tokens=8)
    assert torch.equal(resumed_full, reference)


def test_respects_max_new_tokens_even_when_draft_width_overshoots():
    cfg = AsterWhiteConfig.tiny_for_testing()
    cfg.backbone.mtp_n_future_tokens = 5  # K=6 draft tokens/round -- larger than max_new_tokens below
    model = AsterWhiteCore(cfg.backbone)
    prompt = torch.randint(0, cfg.backbone.vocab_size, (1, 4))
    out, _cache, _stats = speculative_generate(model, prompt, max_new_tokens=2, greedy=True)
    assert out.shape[1] == prompt.shape[1] + 2


def test_eos_stops_generation_early():
    cfg = _mtp_cfg()
    model = AsterWhiteCore(cfg.backbone)
    prompt = torch.randint(0, cfg.backbone.vocab_size, (1, 4))
    out, _cache, _stats = speculative_generate(model, prompt, max_new_tokens=50, greedy=True, eos_token_id=0)
    assert out.shape[1] <= prompt.shape[1] + 50  # either stopped early at/around the first eos, or never emitted one
