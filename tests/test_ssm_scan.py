"""AsterWhite :: tests.test_ssm_scan -- the chunked linear-recurrence scan
(core.ssm.chunked_linear_recurrence) must be the SAME function as the
original per-time-step loop, just computed in fewer sequential steps."""
from __future__ import annotations

import pytest
import torch

from asterwhite.core.ssm import SelectiveSSM, chunked_linear_recurrence, sequential_linear_recurrence


def _inputs(B=2, T=37, D=5, N=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(B, T, D, N, generator=g) * 0.98 + 0.01      # decays in (0, 1), like exp(dt * A) with A < 0
    u = torch.randn(B, T, D, N, generator=g)
    h0 = torch.randn(B, D, N, generator=g)
    return a, u, h0


@pytest.mark.parametrize("T,chunk", [(37, 8), (64, 16), (5, 32), (100, None), (1, None), (17, 17), (30, 1), (31, 2)])
def test_chunked_matches_sequential_for_every_length_and_chunk_size(T, chunk):
    a, u, h0 = _inputs(T=T)
    ref = sequential_linear_recurrence(a, u, h0)
    got = chunked_linear_recurrence(a, u, h0, chunk_size=chunk)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_chunked_gradients_match_sequential():
    a, u, h0 = _inputs(B=2, T=50, D=4, N=3)
    for t in (a, u, h0):
        t.requires_grad_()
    w = torch.randn(2, 50, 4, 3)
    g_ref = torch.autograd.grad((sequential_linear_recurrence(a, u, h0) * w).sum(), [a, u, h0])
    g_new = torch.autograd.grad((chunked_linear_recurrence(a, u, h0, 7) * w).sum(), [a, u, h0])
    for r, n in zip(g_ref, g_new):
        assert torch.allclose(r, n, atol=1e-5, rtol=1e-4)


def test_padding_does_not_leak_into_the_final_state():
    """T=10, chunk=4 pads two steps (a=1, u=0); the returned final state must be
    the state after the 10th REAL step, not a padded one."""
    a, u, h0 = _inputs(T=10)
    got = chunked_linear_recurrence(a, u, h0, chunk_size=4)
    assert got.shape[1] == 10
    assert torch.allclose(got[:, -1], sequential_linear_recurrence(a, u, h0)[:, -1], atol=1e-5)


def test_selective_ssm_scan_independent_of_chunk_size():
    torch.manual_seed(0)
    m = SelectiveSSM(16, state_dim=4)
    B, T, D = 2, 23, m.d_inner
    x, dt = torch.randn(B, T, D), torch.rand(B, T, D) + 0.05
    Bs, Cs = torch.randn(B, T, 4), torch.randn(B, T, 4)
    y1, h1 = m.scan(x, dt, Bs, Cs, chunk_size=1)
    for q in (None, 4, 23):
        y2, h2 = m.scan(x, dt, Bs, Cs, chunk_size=q)
        assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4)
        assert torch.allclose(h1, h2, atol=1e-4, rtol=1e-4)


def test_streaming_with_state_equals_one_full_pass():
    """Feeding the sequence in pieces while carrying ssm_state (how generation
    uses it) must equal processing it whole."""
    torch.manual_seed(1)
    m = SelectiveSSM(16, state_dim=4).eval()
    x = torch.randn(1, 40, 16)
    with torch.no_grad():
        full = m(x)
        state: dict = {}
        parts = [m(x[:, i:i + n], ssm_state=state) for i, n in ((0, 25), (25, 1), (26, 14))]
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-4, rtol=1e-4)
