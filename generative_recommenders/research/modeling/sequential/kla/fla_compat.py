"""fla-version compatibility shims.

The pinned fla (v0.5.1, deliberately held back -- HEAD/0.5.2 dropped
USE_CUDA_GRAPH) exposes a ``chunk_gla_fwd_o_gk`` *Python wrapper* whose
signature predates the ``use_exp2`` / ``transpose_state_layout`` keywords that
this repo's vendored GDN-2 / KDA / Kalman chunk ops pass to it (they were
written against a newer fla where the wrapper gained both). Calling the pinned
wrapper therefore raises ``TypeError: ... unexpected keyword argument
'use_exp2'`` on the very first chunk forward -- crash-looping any run that hits
a repo chunk kernel (gdn2, kda_naive, diag_kla/kalman, gdn2-path kda).

The underlying Triton *kernel* ``chunk_gla_fwd_kernel_o`` is unchanged between
these fla versions and already exposes everything we need, so we only
re-implement the thin wrapper here (reusing fla's installed kernel) and map the
two extra keywords onto its existing behaviour:

  * ``transpose_state_layout`` -> the kernel's ``STATE_V_FIRST`` constexpr.
    Both select the ``[V, K]`` state layout instead of ``[K, V]``. Verified
    against the vendored producer ``chunk_gated_delta_rule_fwd_h``
    (lit_gpt/gdn2_ops/chunk_kda.py), which allocates ``h`` as ``[B,NT,H,V,K]``
    when ``transpose_state_layout=True`` and ``[B,NT,H,K,V]`` otherwise --
    exactly the layouts the o-kernel reads under ``STATE_V_FIRST`` True/False.

  * ``use_exp2`` -> the o-kernel accumulates with ``exp2`` unconditionally
    (``b_q * exp2(b_g)``; ``g`` is pre-scaled by ``RCP_LN2`` upstream). There is
    no ``exp`` path in this kernel, so only ``use_exp2=True`` is representable;
    it is asserted rather than silently ignored.

Point the repo's chunk ops at ``from generative_recommenders.research.modeling.sequential.kla.fla_compat import
chunk_gla_fwd_o_gk`` instead of ``from fla.ops.gla.chunk import ...`` so they no
longer depend on the fla wrapper's signature drift.
"""
from __future__ import annotations

import torch
import triton

# Reuse fla's actual (installed) kernel + helper -- unchanged across the version
# drift; only the Python wrapper's signature changed.
from fla.ops.gla.chunk import chunk_gla_fwd_kernel_o
from fla.ops.utils import prepare_chunk_indices


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    # --- newer-fla keywords the repo's vendored chunk ops pass through ---
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
) -> torch.Tensor:
    # The o-kernel hardcodes exp2 (g is pre-scaled by RCP_LN2 by the callers), so
    # only use_exp2=True is faithfully representable with the pinned kernel.
    assert use_exp2, (
        "lit_gpt.fla_compat.chunk_gla_fwd_o_gk supports only use_exp2=True "
        "(fla v0.5.1's chunk_gla_fwd_kernel_o accumulates with exp2)"
    )
    # Newer fla renamed state_v_first -> transpose_state_layout; identical
    # meaning ([V,K] vs [K,V] state layout, selected by the STATE_V_FIRST
    # constexpr). Honour either spelling.
    state_v_first = state_v_first or transpose_state_layout

    # --- body verbatim from fla v0.5.1 chunk_gla_fwd_o_gk ---
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Please ensure zeros, since vllm will use padding v
    o = torch.zeros_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * HV)
    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return o
