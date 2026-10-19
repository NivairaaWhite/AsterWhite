"""
AsterWhite :: core.ssm

A from-scratch selective state-space block (Mamba-style S6 mechanism):
input-dependent (A, B, C, dt) give the recurrence content-addressable
"selection", letting the model compress arbitrarily long context into a
fixed-size hidden state -- O(T) instead of attention's O(T^2), and O(1)
memory per generated token when `ssm_state` is carried across calls. This
is the long-context complement to attention in the hybrid backbone, and the
main lever for scaling sequential processing toward very long documents
without the context window itself growing.

The scan itself (`SelectiveSSM.scan`) is a plain-PyTorch CHUNKED scan: the
recurrence h_t = a_t * h_{t-1} + u_t is solved exactly (not approximated) by
scanning inside every chunk in parallel, carrying each chunk's end state
across chunk boundaries, and correcting each chunk's states for the state it
started from -- ~2*sqrt(T) sequential steps instead of T, which is what
makes this block usable for training on CPU (a per-time-step Python loop ran
~60x slower than attention here). `sequential_linear_recurrence` is the
original step-by-step loop, kept as the reference the tests compare against.
A fused parallel-scan CUDA kernel implementing the same (A, B, C, dt)
interface can still be dropped in behind `SelectiveSSM.scan(...)` for
large-batch GPU throughput without touching any other file.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequential_linear_recurrence(a: torch.Tensor, u: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Reference: h_t = a_t * h_{t-1} + u_t, one time step at a time.
    a, u: (B, T, D, N); h0: (B, D, N). Returns every state, (B, T, D, N)."""
    h = h0
    states = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + u[:, t]
        states.append(h)
    return torch.stack(states, dim=1)


def chunked_linear_recurrence(a: torch.Tensor, u: torch.Tensor, h0: torch.Tensor,
                              chunk_size: Optional[int] = None) -> torch.Tensor:
    """Same result as sequential_linear_recurrence (exact up to float
    re-association), in ~chunk_size + T/chunk_size sequential steps.

    1. Inside every chunk, scan from a ZERO state, all chunks at once
       (chunk_size steps), also tracking the running product P_t of the
       decays a within the chunk -- the factor by which whatever state the
       chunk STARTED with has decayed by step t.
    2. Carry the true state across chunk boundaries (T/chunk_size steps):
       start_{c+1} = P_end_c * start_c + localstate_end_c.
    3. Every state is then  local_t + P_t * start_chunk  -- linearity of the
       recurrence.
    Padding (a=1, u=0) is appended when T is not a multiple of chunk_size;
    it leaves the state unchanged and is trimmed from the output.
    chunk_size=None picks ~sqrt(T), the minimizer of chunk + T/chunk."""
    B, T, D, N = a.shape
    if chunk_size is None:
        chunk_size = max(1, int(T ** 0.5))
    chunk_size = max(1, min(chunk_size, T))
    if chunk_size == 1 or T <= 1:
        return sequential_linear_recurrence(a, u, h0)

    pad = (-T) % chunk_size
    if pad:
        a = F.pad(a, (0, 0, 0, 0, 0, pad), value=1.0)
        u = F.pad(u, (0, 0, 0, 0, 0, pad), value=0.0)
    n_chunks = (T + pad) // chunk_size
    a = a.reshape(B, n_chunks, chunk_size, D, N)
    u = u.reshape(B, n_chunks, chunk_size, D, N)

    local = a.new_zeros(B, n_chunks, D, N)
    decay = torch.ones_like(local)
    local_states, decays = [], []
    for t in range(chunk_size):
        local = a[:, :, t] * local + u[:, :, t]
        decay = a[:, :, t] * decay
        local_states.append(local)
        decays.append(decay)
    local_states = torch.stack(local_states, dim=2)      # (B, C, Q, D, N)
    decays = torch.stack(decays, dim=2)                   # (B, C, Q, D, N)

    h = h0
    starts = []
    for c in range(n_chunks):
        starts.append(h)
        h = decays[:, c, -1] * h + local_states[:, c, -1]
    starts = torch.stack(starts, dim=1)                   # (B, C, D, N)

    states = local_states + decays * starts.unsqueeze(2)
    return states.reshape(B, n_chunks * chunk_size, D, N)[:, :T]


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, state_dim: int = 16, conv_kernel: int = 4, expand: int = 2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel, groups=d_inner, padding=0)
        self.x_proj = nn.Linear(d_inner, state_dim * 2 + 1, bias=False)  # -> (B_ssm, C_ssm, dt)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))   # (d_inner, state_dim); A itself kept negative via -exp()
        self.D = nn.Parameter(torch.ones(d_inner))  # skip connection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def scan(self, x: torch.Tensor, dt: torch.Tensor, B_ssm: torch.Tensor, C_ssm: torch.Tensor,
             h0: Optional[torch.Tensor] = None, chunk_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """x, dt: (B,T,d_inner). B_ssm, C_ssm: (B,T,state_dim). h0: (B,d_inner,state_dim) or None.
        Returns (y: (B,T,d_inner), h_final: (B,d_inner,state_dim)).
        chunk_size: see chunked_linear_recurrence (None -> ~sqrt(T); 1 -> the plain per-step loop)."""
        Bsz, T, D = x.shape
        A = -torch.exp(self.A_log)                        # (D,N), strictly negative -> stable recurrence
        dA = torch.exp(dt.unsqueeze(-1) * A)               # (B,T,D,N) decay per step
        dBx = dt.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)   # (B,T,D,N) input per step
        h0 = h0 if h0 is not None else x.new_zeros(Bsz, D, self.state_dim)
        states = chunked_linear_recurrence(dA, dBx, h0, chunk_size=chunk_size)   # (B,T,D,N)
        y = (states * C_ssm.unsqueeze(2)).sum(-1)          # (B,T,D)
        return y, states[:, -1]

    def forward(self, x: torch.Tensor, ssm_state: Optional[dict] = None) -> torch.Tensor:
        B = x.shape[0]
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        if ssm_state and "conv_buf" in ssm_state:
            x_in_padded = torch.cat([ssm_state["conv_buf"], x_in], dim=1)
        else:
            pad = x_in.new_zeros(B, self.conv_kernel - 1, self.d_inner)
            x_in_padded = torch.cat([pad, x_in], dim=1)

        x_conv = self.conv1d(x_in_padded.transpose(1, 2)).transpose(1, 2)  # exactly length T
        x_conv = F.silu(x_conv)
        if ssm_state is not None:
            k = self.conv_kernel - 1
            ssm_state["conv_buf"] = x_in_padded[:, -k:, :] if k > 0 else x_in.new_zeros(B, 0, self.d_inner)

        proj = self.x_proj(x_conv)
        B_ssm, C_ssm, dt_raw = torch.split(proj, [self.state_dim, self.state_dim, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))

        h0 = ssm_state["h"] if (ssm_state and "h" in ssm_state) else None
        y, h_final = self.scan(x_conv, dt, B_ssm, C_ssm, h0=h0)
        if ssm_state is not None:
            ssm_state["h"] = h_final.detach()

        y = y + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(y)
