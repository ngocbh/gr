# Naive (pure-PyTorch) Kimi Delta Attention (KDA).
#
# `naive_recurrent_kda` and `naive_chunk_kda` are copied verbatim from
# flash-linear-attention (fla/ops/kda/naive.py), MIT license,
# (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li:
#   https://github.com/fla-org/flash-linear-attention
#
# They implement the KDA recurrence with plain torch ops (no Triton / tilelang),
# so they run on CPU, are autograd-differentiable, and serve as a readable
# reference for the fused `chunk_kda` kernel. `naive_chunk_kda` uses the
# chunk-parallel UT/WY-transform formulation (intra-chunk inverse + inter-chunk
# state carry); `naive_recurrent_kda` is the token-by-token reference.
#
# This module also adds `kda_gate` (the pure-torch decay-gate activation the
# kernel fuses via `use_gate_in_kernel=True`) and `KimiDeltaAttentionNaive`, a
# drop-in token mixer that reuses fla's `KimiDeltaAttention` projections/params
# but routes the recurrence through `naive_chunk_kda`.
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    r"""Token-by-token KDA reference. See module docstring for provenance.

    Shapes: q,k ``[B,T,H,K]``; v ``[B,T,HV,V]``; g ``[B,T,HV,K]`` (log-space
    per-dim decay); beta ``[B,T,HV]`` (scalar erase gate). Returns ``(o, S)``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale   # [B, T, HV, K]
    k = k.repeat_interleave(G, dim=2)           # [B, T, HV, K]

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    r"""Chunk-parallel KDA reference (UT/WY transform). See module docstring.

    Shapes match :func:`naive_recurrent_kda`. ``chunk_size`` must divide ``T``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0

    # Rearrange into chunks: [B, head, NT, BT, ...]
    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [v, g, beta]]
    # Expand q/k to value head dim for GVA: [B, H, ...] -> [B, HV, ...]
    q = q.repeat_interleave(G, dim=1) * scale  # [B, HV, NT, BT, K]
    k = k.repeat_interleave(G, dim=1)          # [B, HV, NT, BT, K]
    g = g.cumsum(-2)

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    # Akk uses k (expanded to HV) and g (per value head)
    A = torch.zeros(*g.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        # clamp(max=0): a no-op on the retained (causal, c>=i) entries -- g is
        # monotonically decreasing so g-g_i<=0 there -- but it kills the fp32
        # exp overflow on the to-be-masked (c<i) entries, which would otherwise
        # feed 0*inf=NaN into the backward. Mirrors the kernel's overflow guard.
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).clamp(max=0).exp(), k_i)
    A = A * beta[..., None]

    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        # [B, HV, BT, ...]
        q_i = q[:, :, i]      # [B, HV, BT, K]
        k_i = k[:, :, i]      # [B, HV, BT, K]
        u_i = u[:, :, i]        # [B, HV, BT, V]
        g_i = g[:, :, i]        # [B, HV, BT, K]
        w_i = w[:, :, i]        # [B, HV, BT, K]
        # Aqk: per value head (q from qk head, g from value head, k from qk head)
        Aqk = torch.zeros(B, HV, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            # clamp(max=0): no-op on retained (c>=j) entries, overflow guard on
            # the masked (c<j) ones -- see the Akk loop above.
            Aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).clamp(max=0).exp(), k_j)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S


def kda_gate(g: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor | None = None) -> torch.Tensor:
    """Pure-torch KDA decay gate (mirrors fla `naive_kda_gate`, the activation
    the kernel fuses via ``use_gate_in_kernel=True``):

        g = -exp(A_log) * softplus(g + dt_bias)

    Args: ``g`` ``[..., H, K]`` (raw f_proj output), ``A_log`` ``[H]``,
    ``dt_bias`` ``[H*K]``. Returns fp32 log-space decay ``[..., H, K]``.
    """
    H = g.shape[-2]
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    return -A_log.view(H, 1).float().exp() * F.softplus(g.float())


# Layer import is deferred to the end so the core reference functions above stay
# importable without pulling in fla.layers.
from fla.layers import KimiDeltaAttention  # noqa: E402


class KimiDeltaAttentionNaive(KimiDeltaAttention):
    """KDA token mixer that routes the recurrence through the pure-torch
    :func:`naive_chunk_kda` instead of fla's Triton `chunk_kda`.

    Subclasses fla's :class:`KimiDeltaAttention`, so it shares the exact same
    parameters (projections, short conv, A_log, dt_bias, o_norm, g_proj, o_proj)
    and is weight-compatible with it. Only the forward recurrence differs, which
    (a) makes it a correctness reference and (b) sidesteps the H200/Triton-3.4
    gated-backward bug (no tilelang needed) -- at a large speed cost.

    Supports the training forward path (``attention_mask=None``, no cache).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        assert attention_mask is None and not use_cache, (
            "KimiDeltaAttentionNaive only supports the training forward path "
            "(attention_mask=None, use_cache=False)."
        )
        assert self.num_v_heads == self.num_heads, "naive KDA assumes num_v_heads == num_heads (no GVA)."

        if self.use_short_conv:
            q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False)
            k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cache=None, output_final_state=False)
            v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states).sigmoid()

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Preprocessing the kernel does internally (use_qk_l2norm_in_kernel /
        # use_gate_in_kernel): l2-normalize q,k and build the log-space decay gate.
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        g = kda_gate(g, self.A_log, self.dt_bias)

        # Run the recurrence in fp32 (mirror the fla kernel's autocast_custom_fwd /
        # the diagonal-KLA naive ops): under bf16-mixed autocast the internal
        # .float() casts don't protect the einsums, so the UT/WY build + carry would
        # run at bf16 (~5e-3). Keeping this fp32 makes the naive KDA baseline eat the
        # same precision as naive KLA, so the KLA-vs-KDA comparison stays apples-to-apples.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            o, _ = naive_chunk_kda(q, k, v, g, beta, output_final_state=False)

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o, None, past_key_values
