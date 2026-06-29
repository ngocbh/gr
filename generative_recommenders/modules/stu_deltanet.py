# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict
"""Memory-augmented HSTU block (``STU_DELTANET``) -- Design B (gated delta rule).

One HSTU block where the spatial aggregation is augmented with a long-term
memory read and a learned gate (Sec. "Memory-Augmented Spatial Aggregation"):

    Z_short = HSTU local-window attention   (the swappable slot from STU_PYTORCH)
    Z_long  = Psi_t(Q_t)                     (gated delta-rule fast-weight read)
    alpha   = sigmoid(g([Q, U, hist_len]) + b)
    Y       = compute_output( (1-alpha) Z_short + alpha Z_long , gated by U )

``Z_short`` is the Triton HSTU attention restricted to a local window
(``max_attn_len``); ``Z_long`` recovers the long-range history the window drops,
compressed into a per-(layer, head) fast-weight state ``A in R^{d_v x d_m}``
updated by the canonical gated delta rule (Yang et al.)

    A_i = gamma_i A_{i-1} (I - beta_i kbar_i kbar_i^T) + beta_i vbar_i kbar_i^T,
    Psi_t = W_O^Psi A_t qbar_t.

The bias of the gate ``g`` is initialised strongly negative so ``alpha ~= 0`` at
init: the block then reduces to the (windowed) HSTU baseline, enabling
conservative warm-start atop a pretrained backbone. Candidate/target positions
are read-only (they never write to ``A``); only user-history positions update it.

The delta recurrence runs at train time via ``flash-linear-attention``'s
chunked Triton kernel (:func:`gated_delta_fla`, varlen ``cu_seqlens`` mode over
the jagged batch). A pure-PyTorch chunked scan (:func:`gated_delta_chunk`) and a
sequential reference (:func:`gated_delta_sequential`) implement the same
operator and are used to validate the kernel.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from generative_recommenders.common import (
    dense_to_jagged,
    HammerKernel,
    jagged_to_padded_dense,
)
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig
from generative_recommenders.ops.hstu_attention import hstu_mha
from generative_recommenders.ops.pytorch.pt_hstu_linear import (
    pytorch_hstu_compute_output,
)
from torch.autograd.profiler import record_function

# Chunk size for the parallel delta scan. Trades GPU memory (O(C^2) per chunk
# pairwise terms) against the number of sequential chunk steps (L / C).
_DELTA_CHUNK_SIZE: int = 64


def gated_delta_sequential(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Reference (slow) gated delta rule. Loops over time; used to validate the
    chunked scan.

    Shapes: ``q, k`` ``[N, L, d_m]`` (already L2-normalized), ``v`` ``[N, L,
    d_v]``, ``beta, gamma`` ``[N, L]``. Returns ``o`` ``[N, L, d_v]`` where
    ``o_t = A_t q_t`` and the canonical Gated DeltaNet recurrence (Yang et al.)
    ``A_t = gamma_t A_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T`` -- the
    decay multiplies the whole previous-state transition, matching ``fla``'s
    ``chunk_gated_delta_rule``.
    """
    N, L, d_m = q.shape
    d_v = v.shape[-1]
    A = torch.zeros(N, d_v, d_m, dtype=q.dtype, device=q.device)
    outs = []
    for t in range(L):
        k_t = k[:, t]  # [N, d_m]
        q_t = q[:, t]
        v_t = v[:, t]  # [N, d_v]
        Ak = torch.einsum("nvd,nd->nv", A, k_t)  # [N, d_v] = A_{t-1} k_t
        u_t = beta[:, t, None] * (v_t - gamma[:, t, None] * Ak)  # [N, d_v]
        A = gamma[:, t, None, None] * A + torch.einsum("nv,nd->nvd", u_t, k_t)
        outs.append(torch.einsum("nvd,nd->nv", A, q_t))
    return torch.stack(outs, dim=1)


def gated_delta_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    chunk_size: int = _DELTA_CHUNK_SIZE,
) -> torch.Tensor:
    """Chunked parallel gated delta rule (the de-decay / UT-transform form).

    Same signature/semantics as :func:`gated_delta_sequential` but processes the
    sequence in chunks of ``chunk_size``: within a chunk the per-position writes
    are solved jointly via a unit-lower-triangular system, and only the
    inter-chunk state is carried sequentially (``L / chunk_size`` steps).

    All math runs in float32 for stability; output is cast back to ``q.dtype``.
    """
    orig_dtype = q.dtype
    # Run in float32 at minimum (cast up bf16/fp16); preserve float64 so the
    # numerical test can validate apples-to-apples against the fp64 reference.
    cdtype = torch.promote_types(q.dtype, torch.float32)
    q = q.to(cdtype)
    k = k.to(cdtype)
    v = v.to(cdtype)
    beta = beta.to(cdtype)
    gamma = gamma.to(cdtype)

    N, L, d_m = q.shape
    d_v = v.shape[-1]
    C = chunk_size
    pad = (C - L % C) % C
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        # Pad gamma with 1.0 (log 0) so padded steps neither decay nor write
        # (beta is padded with 0, so no write regardless).
        gamma = F.pad(gamma, (0, pad), value=1.0)
    Lp = L + pad
    n_chunks = Lp // C

    qc = q.view(N, n_chunks, C, d_m)
    kc = k.view(N, n_chunks, C, d_m)
    vc = v.view(N, n_chunks, C, d_v)
    bc = beta.view(N, n_chunks, C)
    logg = torch.log(gamma.clamp_min(1e-12)).view(N, n_chunks, C)

    eye = torch.eye(C, dtype=q.dtype, device=q.device)
    strict_lower = torch.tril(torch.ones(C, C, dtype=q.dtype, device=q.device), -1)
    lower_incl = torch.tril(torch.ones(C, C, dtype=q.dtype, device=q.device), 0)
    li_bool = lower_incl.bool()

    S = torch.zeros(N, d_v, d_m, dtype=q.dtype, device=q.device)
    out_chunks = []
    for c in range(n_chunks):
        q_ = qc[:, c]  # [N, C, d_m]
        k_ = kc[:, c]
        v_ = vc[:, c]  # [N, C, d_v]
        b_ = bc[:, c]  # [N, C]
        lg = logg[:, c]  # [N, C]

        c_cum = torch.cumsum(lg, dim=1)  # c_i = sum_{0..i} log gamma  [N, C]
        lam_i = torch.exp(c_cum)  # Lambda_i           [N, C]

        # Lambda_i / Lambda_j   [N, C, C]  (index [i, j]).  Mask the exponent
        # before exp() so masked-out (upper) entries can't overflow to inf and
        # produce 0*inf = nan under strong decay.
        e_ij = c_cum.unsqueeze(-1) - c_cum.unsqueeze(1)
        decay_ij = torch.exp(torch.where(li_bool, e_ij, torch.zeros_like(e_ij)))

        kk = torch.einsum("ncd,nmd->ncm", k_, k_)  # [i, j] = k_i . k_j
        qk = torch.einsum("ncd,nmd->ncm", q_, k_)  # [i, j] = q_i . k_j

        # Canonical gated delta (decay on the whole transition):
        # T_{ij} = beta_i (Lambda_i/Lambda_j)(k_i . k_j), strictly lower
        T = b_.unsqueeze(-1) * decay_ij * kk * strict_lower
        # RHS_i = beta_i v_i - beta_i Lambda_i (S k_i)
        Sk = torch.einsum("nvd,ncd->ncv", S, k_)  # [N, C, d_v]
        rhs = b_.unsqueeze(-1) * v_ - (b_ * lam_i).unsqueeze(-1) * Sk
        # Solve (I + T) U = RHS  (unit lower-triangular)
        U = torch.linalg.solve_triangular(
            eye + T, rhs, upper=False, unitriangular=True
        )  # [N, C, d_v]

        # o_i = Lambda_i (S q_i) + sum_{j<=i} (Lambda_i/Lambda_j)(q_i . k_j) u_j
        Sq = torch.einsum("nvd,ncd->ncv", S, q_)
        P = decay_ij * qk * lower_incl
        o_ = lam_i.unsqueeze(-1) * Sq + torch.einsum("ncm,nmv->ncv", P, U)
        out_chunks.append(o_)

        # State carry: A_end = Lambda_{C-1} S + sum_j (Lambda_{C-1}/Lambda_j) u_j k_j^T
        lam_last = lam_i[:, -1]  # [N]
        u_dec = U * torch.exp(c_cum[:, -1:] - c_cum).unsqueeze(-1)
        S = lam_last[:, None, None] * S + torch.einsum("ncv,ncd->nvd", u_dec, k_)

    out = torch.cat(out_chunks, dim=1)[:, :L]  # [N, L, d_v]
    return out.to(orig_dtype)


_FLA_AUTOTUNE_PRUNED: bool = False


def _prune_fla_autotune() -> None:
    """Drop the triton-3.2-incompatible autotune configs from fla's WY-repr
    kernel (``recompute_w_u_fwd_kernel``).

    fla 0.5.x targets triton >= 3.3. On triton 3.2 a few ``num_warps`` /
    ``num_stages`` variants of that kernel fail to compile (blank
    ``CompilationError`` at the ``tl.dot``), and triton's autotuner compiles
    *every* candidate config to benchmark it -- so one bad config crashes the
    whole call. We pin the kernel to the single config confirmed to compile +
    run (fwd & bwd) at model scale: ``num_warps=2, num_stages=2``. Idempotent;
    a no-op once fla/triton are on compatible versions and pruning is empty.
    """
    global _FLA_AUTOTUNE_PRUNED
    if _FLA_AUTOTUNE_PRUNED:
        return
    try:
        import fla.ops.gated_delta_rule.wy_fast as wy

        node = wy.recompute_w_u_fwd_kernel  # Heuristics -> CachedAutotuner
        while node is not None and not hasattr(node, "configs"):
            node = getattr(node, "fn", None)
        if node is not None:
            safe = [
                c for c in node.configs
                if c.num_warps == 2 and c.num_stages == 2
            ]
            if safe:
                node.configs = safe
    except Exception:
        pass  # best-effort; if fla internals change, fall back to defaults
    _FLA_AUTOTUNE_PRUNED = True


def gated_delta_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fast gated delta rule via ``flash-linear-attention``'s chunked Triton
    kernel. Same operator as :func:`gated_delta_sequential`.

    Two layouts:
      * batched: ``q, k`` ``[N, L, H, d_m]``, ``v`` ``[N, L, H, d_v]``,
        ``beta, gamma`` ``[N, L, H]``; ``cu_seqlens`` is ``None``.
      * varlen (packed): ``q, k`` ``[1, T, H, d_m]`` etc. with ``cu_seqlens``
        ``[B+1]`` int32 segment offsets (each segment scanned independently).

    ``q, k`` must be L2-normalized; ``beta, gamma`` in ``(0, 1)``. ``scale=1.0``
    (plain dot product) to match the reference. Returns ``o`` shaped like ``v``.
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    _prune_fla_autotune()
    # Unify q/k/v dtype. Under AMP the L2-normalized q,k come back fp32 while the
    # un-normalized v stays bf16; that K(fp32)/V(bf16) mix makes fla's WY-repr
    # tl.dot fail to compile on triton 3.2. Both all-bf16 and all-fp32 compile.
    cdt = v.dtype
    q = q.to(cdt)
    k = k.to(cdt)
    g = torch.log(gamma.clamp_min(1e-12)).to(torch.float32)  # fla wants log-decay
    o, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta.to(torch.float32),
        scale=1.0,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    return o


class STULayerDeltaNet(STULayer):
    """HSTU block with windowed attention (``Z_short``) + gated-delta long memory
    (``Z_long``) blended by a conservative-init gate. See module docstring."""

    def __init__(self, config: STULayerConfig, is_inference: bool = False) -> None:
        super().__init__(config=config, is_inference=is_inference)
        H = self._num_heads
        d_qk = self._attention_dim
        d_hidden = self._hidden_dim
        d_m = d_qk  # memory key dim
        d_v = d_hidden  # memory value dim
        self._delta_dm: int = d_m
        self._delta_dv: int = d_v

        def _proj(in_dim: int, out_dim: int) -> torch.nn.Parameter:
            w = torch.nn.Parameter(torch.empty(H, in_dim, out_dim))
            torch.nn.init.xavier_uniform_(w)
            return w

        # Psi projections (per head): W_K, W_Q: d_qk -> d_m ; W_V: d_hidden -> d_v ;
        # W_O: d_v -> d_hidden.
        self._psi_wk: torch.nn.Parameter = _proj(d_qk, d_m)
        self._psi_wq: torch.nn.Parameter = _proj(d_qk, d_m)
        self._psi_wv: torch.nn.Parameter = _proj(d_hidden, d_v)
        self._psi_wo: torch.nn.Parameter = _proj(d_v, d_hidden)
        # gamma (decay) and beta (write) gates: linear over k, per head.
        # gamma bias init positive -> gamma ~= sigmoid(3) ~= 0.95 so the memory
        # retains over long spans at init (and keeps the chunk-scan dynamic range
        # small for numerical stability).
        self._gamma_w: torch.nn.Parameter = torch.nn.Parameter(torch.zeros(H, d_qk))
        self._gamma_b: torch.nn.Parameter = torch.nn.Parameter(
            torch.full((H,), 3.0)
        )
        self._beta_w: torch.nn.Parameter = torch.nn.Parameter(torch.zeros(H, d_qk))
        self._beta_b: torch.nn.Parameter = torch.nn.Parameter(torch.zeros(H))

        # Blend gate g_l over [Q, U, hist_len] -> scalar; bias init negative so
        # alpha ~= 0 at start (reduces to windowed HSTU).
        gate_in = self._gate_input_dim()
        self._gate_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(gate_in)
        )
        self._gate_bias: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(-4.0))

    def _gate_input_dim(self) -> int:
        """Width of the blend-gate input vector. Base: ``[Q, U, hist_frac]``."""
        return (
            self._num_heads * self._attention_dim
            + self._num_heads * self._hidden_dim
            + 1
        )

    def _delta_write_mask(
        self,
        total: int,
        x_offsets: torch.Tensor,
        x_lengths: torch.Tensor,
        num_targets: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Boolean ``[total, 1]`` mask: which tokens write to the delta state A.

        Base (``STU_DELTANET``): every user-history position writes -- all tokens
        strictly before the candidate block. Candidates/targets/padding are
        read-only. The recent window is therefore covered by *both* the windowed
        attention and the delta memory (overlapping). ``STUpDeltaNet`` overrides
        this to make the split non-overlapping.
        """
        lengths = x_offsets[1:] - x_offsets[:-1]  # [B]
        token_start = torch.repeat_interleave(x_offsets[:-1], lengths)  # [total]
        pos_in_seq = torch.arange(total, device=device) - token_start
        hist_len = (x_lengths - num_targets).clamp_min(0)  # [B]
        hist_per_token = torch.repeat_interleave(hist_len, lengths)  # [total]
        return (pos_in_seq < hist_per_token).unsqueeze(1)  # [total, 1]

    def _delta_long(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x_offsets: torch.Tensor,
        x_lengths: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Z_long [total, H*d_hidden] via the gated delta rule over the
        per-user history (candidate/target positions are read-only)."""
        dtype = q.dtype
        H = self._num_heads
        total = q.shape[0]

        # Per-head Psi projections (jagged, [total, H, .]).
        kbar = torch.einsum("thd,hde->the", k, self._psi_wk.to(dtype))
        qbar = torch.einsum("thd,hde->the", q, self._psi_wq.to(dtype))
        vbar = torch.einsum("thd,hde->the", v, self._psi_wv.to(dtype))
        kbar = F.normalize(kbar, p=2, dim=-1)
        qbar = F.normalize(qbar, p=2, dim=-1)
        gamma = torch.sigmoid(
            torch.einsum("thd,hd->th", k, self._gamma_w.to(dtype))
            + self._gamma_b.to(dtype)
        )  # [total, H]
        beta = torch.sigmoid(
            torch.einsum("thd,hd->th", k, self._beta_w.to(dtype))
            + self._beta_b.to(dtype)
        )  # [total, H]

        # Which tokens write to the delta state A (overridable per variant:
        # STU_DELTANET = all history; STUpDeltaNet = only history older than W).
        write = self._delta_write_mask(
            total, x_offsets, x_lengths, num_targets, q.device
        )  # [total, 1] bool
        beta = torch.where(write, beta, torch.zeros_like(beta))
        gamma = torch.where(write, gamma, torch.ones_like(gamma))

        # fla varlen: pack as [1, total, H, .] with cu_seqlens = x_offsets, so
        # each user's history is scanned as an independent sequence -- no padding.
        o = gated_delta_fla(
            q=qbar.unsqueeze(0),
            k=kbar.unsqueeze(0),
            v=vbar.unsqueeze(0),
            beta=beta.unsqueeze(0),
            gamma=gamma.unsqueeze(0),
            cu_seqlens=x_offsets.to(torch.int32),
        ).squeeze(0)  # [total, H, d_v]

        # Readout W_O: d_v -> d_hidden, per head; result is already jagged.
        zo = torch.einsum("thd,hde->the", o.to(dtype), self._psi_wo.to(dtype))
        z_long = zo.reshape(total, H * self._hidden_dim)  # [total, H*d_hidden]
        return z_long

    def _build_gate_input(
        self,
        q: torch.Tensor,
        u: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Per-token blend-gate input. Base: ``[Q, U, hist_frac]`` where
        ``hist_frac = hist_len / max_seq_len`` is broadcast to every token.
        ``STUpDeltaNet`` overrides this to also condition on the candidate."""
        q_flat = q.reshape(q.shape[0], self._num_heads * self._attention_dim)
        hist_frac = (
            dense_to_jagged(
                jagged_to_padded_dense(
                    (x_lengths - num_targets)
                    .clamp_min(0)
                    .to(dtype)
                    .unsqueeze(1)
                    .expand(-1, max_seq_len)
                    .reshape(-1, 1),
                    [x_offsets],
                    [max_seq_len],
                    0.0,
                ),
                [x_offsets],
            )
            / float(max_seq_len)
        )
        return torch.cat([q_flat, u, hist_frac], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = x.dtype
        with record_function("## deltanet_uvqk ##"):
            normed_x = F.layer_norm(
                x,
                normalized_shape=(x.shape[-1],),
                weight=self._input_norm_weight.to(dtype),
                bias=self._input_norm_bias.to(dtype),
                eps=1e-6,
            )
            uvqk = torch.addmm(
                self._uvqk_beta.to(dtype), normed_x, self._uvqk_weight.to(dtype)
            )
            u, v, q, k = torch.split(
                uvqk,
                [
                    self._hidden_dim * self._num_heads,
                    self._hidden_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=1,
            )
            u = F.silu(u)
            q = q.view(-1, self._num_heads, self._attention_dim)
            k = k.view(-1, self._num_heads, self._attention_dim)
            v = v.view(-1, self._num_heads, self._hidden_dim)

        # Z_short: windowed HSTU attention via the Triton kernel (max_attn_len set
        # by --max-attn-len; the only O(N*window) step, never materializes [N,N]).
        with record_function("## deltanet_zshort (triton) ##"):
            z_short = hstu_mha(
                max_seq_len=max_seq_len,
                alpha=self._attn_alpha,
                q=q,
                k=k,
                v=v,
                seq_offsets=x_offsets,
                causal=self._causal,
                dropout_pr=0.0,
                training=False,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                sort_by_length=self._sort_by_length,
                kernel=HammerKernel.TRITON,
            ).view(-1, self._hidden_dim * self._num_heads)

        # Z_long: gated delta-rule long-term memory read.
        with record_function("## deltanet_zlong ##"):
            z_long = self._delta_long(
                q=q,
                k=k,
                v=v,
                x_offsets=x_offsets,
                x_lengths=x_lengths,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
            )

        # Blend gate alpha = sigmoid(g([Q, U, hist_len, ...]) + b), conservative init.
        with record_function("## deltanet_gate ##"):
            gate_in = self._build_gate_input(
                q=q,
                u=u,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                dtype=dtype,
            )
            alpha = torch.sigmoid(
                gate_in @ self._gate_weight.to(dtype) + self._gate_bias.to(dtype)
            ).unsqueeze(1)  # [total, 1]
            blended = (1.0 - alpha) * z_short + alpha * z_long

        with record_function("## deltanet_compute_output ##"):
            return pytorch_hstu_compute_output(
                attn=blended,
                u=u,
                x=x,
                norm_weight=self._output_norm_weight.to(dtype),
                norm_bias=self._output_norm_bias.to(dtype),
                output_weight=self._output_weight.to(dtype),
                eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                training=self.training,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="none",
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
            )


class STULayerpDeltaNet(STULayerDeltaNet):
    """``STUpDeltaNet`` -- the "plus" variant with a *non-overlapping* split:

    The windowed attention (``Z_short``, ``max_attn_len=W``) handles the most
    recent ``W`` positions before each user's candidate block; the gated-delta
    memory (``Z_long``) summarizes *only* the older history the window drops, i.e.
    positions ``[0, hist_len - (W - num_targets)) == [0, x_lengths - W)``. Unlike
    the base :class:`STULayerDeltaNet` (where the memory writes the full history
    and so overlaps the window), here attention and memory partition the sequence.

    The blend gate additionally conditions on the candidate (per-user mean of the
    candidate-block query, broadcast to every token), so how much long memory to
    mix in can depend on what is being scored.
    """

    def _delta_write_mask(
        self,
        total: int,
        x_offsets: torch.Tensor,
        x_lengths: torch.Tensor,
        num_targets: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Write only history older than the attention window: positions
        ``[0, x_lengths - W)`` (the last ``W`` tokens -- recent history + the
        candidate block -- are covered by attention and are read-only here)."""
        lengths = x_offsets[1:] - x_offsets[:-1]  # [B]
        token_start = torch.repeat_interleave(x_offsets[:-1], lengths)  # [total]
        pos_in_seq = torch.arange(total, device=device) - token_start
        hist_len = (x_lengths - num_targets).clamp_min(0)  # [B]
        # cut = hist_len - (W - num_targets) = x_lengths - W; clamp into [0, hist_len]
        # so candidates never write and a non-positive W degrades to full history.
        cut = torch.minimum((x_lengths - self._max_attn_len).clamp_min(0), hist_len)
        cut_per_token = torch.repeat_interleave(cut, lengths)  # [total]
        return (pos_in_seq < cut_per_token).unsqueeze(1)  # [total, 1]

    def _gate_input_dim(self) -> int:
        # Base [Q, U, hist_frac] plus the per-user candidate query (H*d_qk).
        return super()._gate_input_dim() + self._num_heads * self._attention_dim

    def _candidate_query(
        self,
        q: torch.Tensor,
        x_offsets: torch.Tensor,
        x_lengths: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Per-user mean of the candidate-block query, broadcast to every token
        of that user. Shape ``[total, H*d_qk]``."""
        total = q.shape[0]
        d = self._num_heads * self._attention_dim
        q_flat = q.reshape(total, d)
        lengths = x_offsets[1:] - x_offsets[:-1]  # [B]
        B = lengths.shape[0]
        token_start = torch.repeat_interleave(x_offsets[:-1], lengths)
        pos_in_seq = torch.arange(total, device=q.device) - token_start
        hist_len = (x_lengths - num_targets).clamp_min(0)
        hist_per_token = torch.repeat_interleave(hist_len, lengths)
        cand_mask = pos_in_seq >= hist_per_token  # candidate tokens [total]
        user_id = torch.repeat_interleave(
            torch.arange(B, device=q.device), lengths
        )
        cand_sum = torch.zeros(B, d, dtype=q_flat.dtype, device=q.device)
        cand_sum.index_add_(0, user_id[cand_mask], q_flat[cand_mask])
        denom = num_targets.clamp_min(1).to(q_flat.dtype).unsqueeze(1)  # [B, 1]
        cand_mean = cand_sum / denom  # [B, d]
        return cand_mean[user_id]  # [total, d]

    def _build_gate_input(
        self,
        q: torch.Tensor,
        u: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = super()._build_gate_input(
            q, u, x_lengths, x_offsets, max_seq_len, num_targets, dtype
        )
        cand = self._candidate_query(q, x_offsets, x_lengths, num_targets)
        return torch.cat([base, cand], dim=1)


class STUStackDeltaNet(STU):
    """Sequential stack of :class:`STULayerDeltaNet` blocks (vanilla residual)."""

    def __init__(
        self,
        stu_list: List[STULayerDeltaNet],
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=stu_list)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self._stu_layers:
            x = layer(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
        return x

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "STUStackDeltaNet does not support incremental/cached decoding."
        )


class STUStackpDeltaNet(STUStackDeltaNet):
    """Sequential stack of :class:`STULayerpDeltaNet` blocks (vanilla residual).

    Identical plumbing to :class:`STUStackDeltaNet`; exists as a distinct type so
    the model factory can select the "plus" (non-overlapping split) variant.
    """
