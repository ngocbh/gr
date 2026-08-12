# Fused chunk kernel for the general Kalman delta-rule (`chunk_kalman`).
#
# Recurrence (S in R^{K x V}, INDEPENDENT write key kappa and read key k):
#     S_t = (I - kappa_t k_t^T) D_t S_{t-1} + kappa_t v_t^T ,    o_t = q_t^T S_t
# with D_t = diag(alpha_t) per-channel decay. This is exactly the PyTorch
# `_memory_chunk_scan` (lit_gpt/kla_ops/exact_scan.py) as a fused kernel.
#
# Design: docs/plans/2026-07-29-kalman-chunk-kernel-design.md
# Review (C1/C2 decay fix): docs/plans/kalman-chunk-kernel-design-review-1.md
#
# Milestone 1 (forward only): the intra WY stage is computed in PyTorch; the
# three heavy stages are the REUSED Triton kernels imported from chunk_kda.py
# (`chunk_gated_delta_rule_fwd_h` for the inter-chunk recurrence) and fla
# (`chunk_gla_fwd_o_gk` for the output). A pure-PyTorch reference forward
# (`_forward_pytorch`, mirrors `_memory_chunk_scan` at chunk_size=64) locks the
# entry/scale/l2norm/initial_state contract.
#
# CRITICAL decay convention (review C1/C2): the h-kernel applies the chunk-end
# decay A_end only to the OLD state, NOT to the fed write key. So the write key
# fed to the h-kernel's `k` slot must pre-bake BOTH the down-normalization and
# A_end:  kappa_fed = kappa * exp2(gk_last - gk_cum)  (== A_end/A * kappa).
# The read key carries  exp2(gk_cum) (== A). Two DISTINCT factors.

from __future__ import annotations

import math
import os

import torch

# Reused heavy kernels (frozen references; imported, never edited).
from generative_recommenders.research.modeling.sequential.kla.gdn2_ops.chunk_kda import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_bwd_dhu,
    chunk_kda_bwd_dAv,
    chunk_local_cumsum,
    RCP_LN2,
)
from generative_recommenders.research.modeling.sequential.kla.fla_compat import chunk_gla_fwd_o_gk  # fla v0.5.1 wrapper lacks use_exp2/transpose_state_layout
from fla.modules.l2norm import l2norm_fwd, l2norm_bwd

# TI1: forked Triton forward intra (WY) stage. Replaces the fp64 PyTorch
# `_intra_pytorch` when `intra="triton"`. `_intra_pytorch` stays as the fp64
# oracle (intra="pytorch") and the TI2 backward reference.
# TI2: forked Triton BACKWARD intra kernels (wy_dqkg + bwd_intra) for the
# full-Triton backward path (KALMAN_BWD=triton_intra).
from generative_recommenders.research.modeling.sequential.kla.kla_ops.kalman_intra_triton import (
    _intra_triton,
    kalman_recompute_w_u_fwd,
    kalman_bwd_wy_dqkg_fused,
    kalman_bwd_intra,
)


_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Optional per-sub-kernel BACKWARD profiling (env KALMAN_BWD_PROFILE=1).
#
# When enabled, _backward_triton_intra records a PAIR of CUDA timing events
# around each backward sub-kernel (recompute_fwd, dAv_einsum, dhu, wy_dqkg,
# bwd_intra, reverse_cumsum) and appends (label, start_evt, end_evt) to
# _KALMAN_BWD_PROFILE_EVENTS. A profiler (scripts/analyses/bench_kalman_kda_gdn2_bwd.py)
# clears the list, runs N backward passes, cuda-synchronizes, then reads
# start.elapsed_time(end) per label. DEFAULT OFF: a single dict lookup per
# region, no events created, so production backward is unaffected.
# ---------------------------------------------------------------------------
_KALMAN_BWD_PROFILE_EVENTS: list = []


def _kal_prof_start():
    if os.environ.get("KALMAN_BWD_PROFILE", "0") != "1":
        return None
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def _kal_prof_end(label, start_ev):
    if start_ev is None:
        return
    end_ev = torch.cuda.Event(enable_timing=True)
    end_ev.record()
    _KALMAN_BWD_PROFILE_EVENTS.append((label, start_ev, end_ev))


def _hp(x: torch.Tensor) -> torch.Tensor:
    """Promote to at least fp32 (matches exact_scan._hp)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


# ---------------------------------------------------------------------------
# PyTorch intra (WY) stage.
#
# Produces, per chunk, the tensors the reused Triton kernels consume:
#   w         = Abar @ (k * exp2(gk_cum))         (read key, "w" operand of h-kernel)
#   u         = Abar @ v                          (plain v, NO beta; "v" operand)
#   kappa_fed = kappa * exp2(gk_last - gk_cum)    (write key, "k" operand of h-kernel)
#   Aqk       = tril(q~ . kappa~, incl) * scale   (query x write, "A" of gla output)
# where Abar = (I + tril(k_tilde . kappa_tilde, -1))^{-1}.
#
# Within a chunk we work with the decay-normalized keys k_tilde = k*A,
# kappa_tilde = kappa/A (A = exp2(gk_cum)). Note k_tilde . kappa_tilde = (k*A).(kappa/A)
# -- the A factors cancel per pair; that is what the gla-output `A=Aqk` and the WY
# inverse operate on. Mirrors exact_scan._memory_chunk_scan lines 218-238.
# ---------------------------------------------------------------------------
def _intra_pytorch(q, k, kappa, v, g_cumsum, scale, chunk_size):
    """PyTorch WY intra stage feeding the reused Triton h + output kernels.

    Args (all [B, T, H, *], contiguous):
        q, k, kappa : [B, T, H, K]   read q, read key k, write key kappa
        v           : [B, T, H, V]
        g_cumsum    : [B, T, H, K]   base-2 within-chunk cumulative log-decay
                                     (= chunk_local_cumsum(g, RCP_LN2))
        scale       : float          applied to q (baked into Aqk)
    Returns (padded to a whole number of chunks along T):
        w        : [B, Tp, H, K]
        u        : [B, Tp, H, V]
        kappa_fed: [B, Tp, H, K]
        Aqk      : [B, Tp, H, BT]   (BT = chunk_size)
        Tp, nc, BT
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    dt = torch.promote_types(q.dtype, torch.float32)
    # DEEP-DECAY FIX: the decay-sensitive intra math forms A = exp2(gk_cum) and
    # its reciprocal kappa/A (== kappa * exp2(-gk_cum)). Over a 64-token chunk with
    # per-token |g|~1 the base-2 cumulative decay reaches ~-90 => 1/A ~ 2^90 ~ 1e27.
    # That is finite in fp32 for the (paired) forward matmul, but autograd computes
    # SEPARATE grads for k_t and kap_t where the 1/A intermediate overflows fp32
    # (max ~3.4e38) once the cumsum grows a bit deeper -> inf -> NaN in dg. fp64
    # (max ~1.8e308) holds the 1/A intermediates in both forward AND backward, so
    # dg stays finite. The intra tensors are small ([B,H,nc,BT,BT/K/V]) so fp64 here
    # is cheap. We compute the whole WY intra in fp64 and cast the outputs back to dt.
    ct = torch.float64  # compute dtype for the decay-sensitive intra math

    q, k, kappa, v = (x.to(ct) for x in (q, k, kappa, v))
    gc = g_cumsum.to(ct)

    pad = (-T) % BT
    if pad:
        zK = q.new_zeros(B, pad, H, K)
        q = torch.cat([q, zK], 1)
        k = torch.cat([k, zK], 1)
        kappa = torch.cat([kappa, zK], 1)
        v = torch.cat([v, v.new_zeros(B, pad, H, V)], 1)
        # padded tokens sit at the tail of the final chunk; keys are zeroed so
        # they contribute nothing to the pairwise (row-col) score terms. BUT the
        # chunk-end cumulative decay `gk_last = gcc[:,:,:,-1:]` (baked into
        # kappa_fed = kappa*exp2(gk_last-gk_cum)) reads this LAST slot. Zero-padding
        # made gk_last=0 for the partial last chunk instead of the real chunk-end
        # g_cumsum[T-1], inflating kappa_fed by exp2(-g_cumsum[T-1]) and corrupting
        # final_state. Pad with a FLAT continuation (repeat g_cumsum[:, T-1]) so
        # gk_last == the real last-token cumulative decay (matches _forward_pytorch,
        # which pads the RAW g so its cumsum stays flat past T-1).
        gc = torch.cat([gc, gc[:, -1:].expand(B, pad, H, K)], 1)
    Tp = T + pad
    nc = Tp // BT

    # reshape to [B, H, nc, BT, *]
    def r(x, last):
        return x.view(B, nc, BT, H, last).permute(0, 3, 1, 2, 4).contiguous()

    qc = r(q, K)
    kc = r(k, K)
    kpc = r(kappa, K)
    vc = r(v, V)
    gcc = r(gc, K)  # base-2 cumulative decay [B,H,nc,BT,K]

    A = torch.exp2(gcc)   # within-chunk cumulative decay (base-2 == exp(cumsum logalpha))
    k_t = kc * A          # read key k~ = k * A
    kap_t = kpc / A       # write key kappa~ = kappa / A
    q_t = qc * A          # query q~ = q * A

    tril_s = torch.tril(q.new_ones(BT, BT), -1)   # strictly lower
    tril_i = torch.tril(q.new_ones(BT, BT))       # lower incl diag

    # M = tril(k~ . kappa~, -1) -- asymmetric (two different keys)
    M = torch.einsum("bhnck,bhnsk->bhncs", k_t, kap_t) * tril_s
    IpM = torch.eye(BT, device=q.device, dtype=ct) + M
    # Abar = (I + M)^{-1}
    eyeC = torch.eye(BT, device=q.device, dtype=ct).expand(B, H, nc, BT, BT)
    Abar = torch.linalg.solve_triangular(IpM, eyeC, upper=False, unitriangular=True)

    # w = Abar @ (k * A)   (read key, carries exp2(gk_cum))
    w = torch.einsum("bhncs,bhnsk->bhnck", Abar, k_t)
    # u = Abar @ v         (plain v, no beta)
    u = torch.einsum("bhncs,bhnsv->bhncv", Abar, vc)

    # write key fed to h-kernel's `k` slot: kappa * exp2(gk_last - gk_cum)
    gk_last = gcc[:, :, :, -1:, :]                 # [B,H,nc,1,K]
    kappa_fed = kpc * torch.exp2(gk_last - gcc)    # [B,H,nc,BT,K]

    # Aqk = tril_incl(q~ . kappa~) * scale
    Aqk = torch.einsum("bhnck,bhnsk->bhncs", q_t, kap_t) * tril_i * scale  # [B,H,nc,BT,BT]

    # back to [B, Tp, H, *]; cast the fp64 intra outputs back to the working
    # dtype dt (the decay-sensitive 1/A math is confined to fp64 above; downstream
    # reused kernels consume dt/bf16 exactly as before this deep-decay fix).
    def unr(x, last):
        return x.permute(0, 2, 3, 1, 4).contiguous().view(B, Tp, H, last).to(dt)

    w = unr(w, K)
    u = unr(u, V)
    kappa_fed = unr(kappa_fed, K)
    Aqk = unr(Aqk, BT)
    return w, u, kappa_fed, Aqk, Tp, nc, BT


def _forward_triton(q, k, kappa, v, g, scale, initial_state, output_final_state, chunk_size, intra="triton", return_intra_cache=False):
    """M1 forward: intra WY stage -> reused Triton h-recurrence + gla output.

    intra: "triton" (forked Triton WY kernels, default) or "pytorch" (fp64
           `_intra_pytorch` oracle). Both feed the same reused h + gla kernels.

    return_intra_cache (RO-1): when True (triton intra only) also return the bf16
           (Aqk, Abar) intra tensors so the autograd Function can `save_for_backward`
           them and run a RECOMPUTE-ONLY backward (mirrors GDN-2 caching Aqk/Akk).

    q,k,kappa: [B,T,H,K] (l2norm already applied by caller if requested)
    v:         [B,T,H,V]
    g:         [B,T,H,K]  natural log-decay (same convention as chunk_kda's g)
    initial_state: [N,H,K,V] fp32 or None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    input_dtype = q.dtype
    kdtype = torch.bfloat16   # heavy kernels: bf16 matmul in / fp32 accumulate (KDA)

    # Stage 1: base-2 within-chunk cumulative decay (identical to KDA).
    g_cumsum = chunk_local_cumsum(g, scale=RCP_LN2, chunk_size=chunk_size)  # [B,T,H,K] fp32

    # Stage 2: WY intra (Triton fork, or fp64 PyTorch oracle).
    Abar = None
    if intra == "triton":
        if return_intra_cache:
            # RO-1: also build Abar (the WY inverse) to cache; skip the Mraw buffer.
            w, u, kappa_fed, Aqk, Tp, nc, _, Abar = _intra_triton(
                q, k, kappa, v, g_cumsum, scale, chunk_size, kernel_dtype=kdtype,
                return_internals=True, return_M=False,
            )
        else:
            w, u, kappa_fed, Aqk, Tp, nc, _ = _intra_triton(
                q, k, kappa, v, g_cumsum, scale, chunk_size, kernel_dtype=kdtype
            )
    elif intra == "pytorch":
        if return_intra_cache:
            raise ValueError("return_intra_cache requires intra='triton' (Abar cache)")
        w, u, kappa_fed, Aqk, Tp, nc, _ = _intra_pytorch(
            q, k, kappa, v, g_cumsum, scale, chunk_size
        )
    else:
        raise ValueError(f"unknown intra {intra!r}")

    # The reused kernels operate on the (possibly padded) length Tp; pad q and
    # g_cumsum to Tp too, pass T=Tp implicitly (via tensor shape), then slice.
    pad = Tp - T
    if pad:
        q_p = torch.cat([q, q.new_zeros(B, pad, H, K)], 1)
        # FLAT-continuation pad for g_cumsum (NOT zeros): the h-kernel reads the
        # chunk's LAST slot (Tp-1 for the partial last chunk) as the chunk-end decay
        # that carries the state to final_state (h_{c+1}=diag(exp2(gk_last))h_c+...).
        # Zero-padding made gk_last=0 => exp2(0)=1 => NO decay on the last chunk =>
        # wrong final_state at T%64!=0. Repeat g_cumsum[:, T-1] so the h-kernel's
        # carry decay == the real last-token cumulative decay == the decay baked
        # into kappa_fed (invariant; see _intra_pytorch / _intra_triton).
        g_p = torch.cat([g_cumsum, g_cumsum[:, -1:].expand(B, pad, H, K)], 1)
    else:
        q_p, g_p = q, g_cumsum

    w_k = w.to(kdtype)
    u_k = u.to(kdtype)
    kappa_fed_k = kappa_fed.to(kdtype)
    q_k = q_p.to(kdtype)
    Aqk_k = Aqk.to(kdtype)
    g_k = g_p  # fp32 decay

    # Stage 3: inter-chunk state recurrence (REUSED). Route:
    #   w  <- read key term  (w operand)
    #   u  <- plain-v term   (v operand)
    #   k  <- kappa_fed      (write key)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kappa_fed_k,
        w=w_k,
        u=u_k,
        gk=g_k,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        use_exp2=True,
        transpose_state_layout=False,
    )

    # Stage 4: output (REUSED). A=Aqk (scale baked in); raw q; v_new from h-kernel.
    o = chunk_gla_fwd_o_gk(
        q=q_k,
        v=v_new,
        g=g_k,
        A=Aqk_k,
        h=h,
        scale=scale,
        chunk_size=chunk_size,
        use_exp2=True,
        transpose_state_layout=False,
    )
    o = o[:, :T].to(input_dtype)
    if return_intra_cache:
        # Aqk_k (bf16, padded, post-scale, tril-masked) + Abar (bf16, padded WY
        # inverse) are exactly what the recompute-only backward consumes.
        return o, final_state, Aqk_k, Abar
    return o, final_state


def _forward_pytorch(q, k, kappa, v, g, scale, initial_state, output_final_state, chunk_size):
    """Pure-PyTorch reference forward (mirrors exact_scan._memory_chunk_scan at
    chunk_size). Locks the entry/scale/l2norm/initial_state contract.

    g: natural log-decay [B,T,H,K]; alpha = exp(g).
    initial_state: [N,H,K,V] fp32 (N==B for equal-length) or None -- KDA convention.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    input_dtype = q.dtype
    dt = torch.promote_types(q.dtype, torch.float32)
    # DEEP-DECAY FIX (mirrors _intra_pytorch): the chunk scan forms A = exp(cumsum g)
    # and its reciprocal kappa/A. Under deep decay 1/A overflows fp32 in the SEPARATE
    # autograd grads of k_t and kap_t (finite forward, NaN dg). Compute the whole
    # decay-sensitive scan in fp64 (max ~1.8e308 holds the ~1e27 1/A intermediates in
    # fwd AND bwd), then cast o/final_state back. The intra/state tensors are small so
    # fp64 is cheap. NOTE dt is retained only for the padded-input path's typing hints;
    # ct drives every decay-touching tensor below.
    ct = torch.float64  # compute dtype for the decay-sensitive scan

    q_, k_, v_, kap_ = (x.to(ct).permute(0, 2, 1, 3).contiguous() for x in (q, k, v, kappa))  # [B,H,T,*]
    g_ = g.to(ct).permute(0, 2, 1, 3).contiguous()  # natural log-decay [B,H,T,K]
    q_ = q_ * scale

    C = min(int(chunk_size), T)
    pad = (-T) % C
    if pad:
        zK = q_.new_zeros(B, H, pad, K)
        q_ = torch.cat([q_, zK], 2)
        k_ = torch.cat([k_, zK], 2)
        kap_ = torch.cat([kap_, zK], 2)
        g_ = torch.cat([g_, g_.new_zeros(B, H, pad, K)], 2)  # log-decay 0 => alpha 1 (no-op)
        v_ = torch.cat([v_, v_.new_zeros(B, H, pad, V)], 2)
    Tp = T + pad
    nc = Tp // C
    qc, kc, vc, kpc, gc = (x.view(B, H, nc, C, -1) for x in (q_, k_, v_, kap_, g_))

    A = gc.cumsum(dim=3).exp()  # within-chunk cumulative decay = exp(cumsum logalpha)
    k_t = kc * A
    kap_t = kpc / A
    q_t = qc * A
    tril_s = torch.tril(q_.new_ones(C, C), -1)
    tril_i = torch.tril(q_.new_ones(C, C))
    M = torch.einsum("bhnck,bhnsk->bhncs", k_t, kap_t) * tril_s
    Pqk = torch.einsum("bhnck,bhnsk->bhncs", q_t, kap_t) * tril_i
    IpM = torch.eye(C, device=q_.device, dtype=ct) + M
    A_end = A[:, :, :, -1]

    S = q_.new_zeros(B, H, K, V)
    if initial_state is not None:
        # initial_state is [N,H,K,V] (N==B), KDA convention -- use per-batch directly.
        S = S + initial_state.to(ct)
    outs = []
    for c in range(nc):
        Bc = vc[:, :, c] - torch.einsum("bhck,bhkv->bhcv", k_t[:, :, c], S)
        U = torch.linalg.solve_triangular(IpM[:, :, c], Bc, upper=False, unitriangular=True)
        Oc = torch.einsum("bhck,bhkv->bhcv", q_t[:, :, c], S) + torch.einsum("bhcs,bhsv->bhcv", Pqk[:, :, c], U)
        outs.append(Oc)
        S = A_end[:, :, c][..., None] * (S + torch.einsum("bhsk,bhsv->bhkv", kap_t[:, :, c], U))
    o = torch.cat(outs, dim=2)[:, :, :T].permute(0, 2, 1, 3).contiguous().to(input_dtype)  # [B,T,H,V]
    # final_state cast back to the working dtype dt (fp32) -- matches the pre-fix
    # contract (the h-kernel path returns an fp32 state); the deep-decay fp64 is
    # confined to the scan above.
    final_state = S.to(dt) if output_final_state else None
    return o, final_state


def chunk_kalman(
    q: torch.Tensor,
    k: torch.Tensor,
    kappa: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    backend: str = "triton",
    chunk_size: int = 64,
    intra: str = "triton",
):
    r"""General chunked Kalman delta-rule with independent write (kappa) and read (k) keys.

        S_t = (I - kappa_t k_t^T) diag(alpha_t) S_{t-1} + kappa_t v_t^T,   o_t = q_t^T S_t

    Args:
        q, k, kappa : [B, T, H, K]   read query, read key, write key.
        v           : [B, T, H, V]
        g           : [B, T, H, K]   forget gate in NATURAL log space (alpha = exp(g));
                                     same convention as chunk_kda's g.
        scale       : q scale (default K**-0.5).
        initial_state : [N, H, K, V] fp32 (N == B for equal-length) or None.
        output_final_state : whether to return the final [N,H,K,V] state.
        use_qk_l2norm_in_kernel : l2-normalize q and READ k only (kappa left raw).
        backend     : "triton" (Triton/PyTorch intra + reused Triton h/output; default)
                      or "pytorch" (pure-PyTorch reference).
        chunk_size  : 64.
        intra       : WY intra stage for backend="triton": "triton" (forked Triton
                      kernels, default) or "pytorch" (fp64 `_intra_pytorch` oracle).

    Returns:
        (o, final_state) with o [B,T,H,V]; final_state [N,H,K,V] fp32 or None.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Differentiable triton path: if any input requires grad, route through the
    # autograd Function (fast triton forward + Phase-A recompute backward). The
    # pure functional paths below stay intact as the grad oracle / no-grad forward.
    if backend == "triton" and chunk_size == 64 and torch.is_grad_enabled():
        needs_grad = any(
            t is not None and t.requires_grad
            for t in (q, k, kappa, v, g, initial_state)
        )
        if needs_grad:
            return ChunkKalmanFunction.apply(
                q,
                k,
                kappa,
                v,
                g,
                scale,
                initial_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
                intra,
            )

    if use_qk_l2norm_in_kernel:
        if backend == "triton":
            q, _ = l2norm_fwd(q)
            k, _ = l2norm_fwd(k)
        else:
            q = torch.nn.functional.normalize(_hp(q), dim=-1).to(q.dtype)
            k = torch.nn.functional.normalize(_hp(k), dim=-1).to(k.dtype)

    if backend == "pytorch":
        return _forward_pytorch(
            q, k, kappa, v, g, scale, initial_state, output_final_state, chunk_size
        )
    elif backend == "triton":
        return _forward_triton(
            q, k, kappa, v, g, scale, initial_state, output_final_state, chunk_size, intra=intra
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")


class ChunkKalmanFunction(torch.autograd.Function):
    """Autograd wrapper for `chunk_kalman`.

    Forward runs the fast triton path. Backward (Phase A, milestone B1) recomputes
    the output through the DIFFERENTIABLE pytorch backend under ``torch.enable_grad``
    and returns ``torch.autograd.grad`` -- the exact grad oracle. This is slow at
    long T (a full PyTorch recompute) but correct, and gives the dk/dkappa split for
    free (autograd differentiates the independent k/kappa uses in the intra stage).

    CRITICAL (round-2 review CI-A): recompute through the ``chunk_kalman`` WRAPPER
    (differentiable ``F.normalize``), NOT bare ``_forward_pytorch`` -- else the
    l2norm Jacobian is dropped and dq/dk are wrong when l2norm is on (production runs
    it on).
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        kappa,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        intra="triton",
    ):
        # Run the l2norm + fast triton forward INLINE (mirrors chunk_kalman's
        # triton branch) rather than calling chunk_kalman(backend="triton") --
        # that would re-enter this Function once chunk_kalman routes grad-requiring
        # inputs through ChunkKalmanFunction.apply (infinite recursion).
        if use_qk_l2norm_in_kernel:
            q_n, _ = l2norm_fwd(q)
            k_n, _ = l2norm_fwd(k)
        else:
            q_n, k_n = q, k
        # RO-1: cache the bf16 (Aqk, Abar) intra tensors so the triton_intra backward
        # is RECOMPUTE-ONLY (mirror GDN-2 caching Aqk/Akk). Only the triton intra path
        # produces Abar; the fp64 pytorch intra falls back to a cache-free backward.
        if intra == "triton":
            o, final_state, Aqk_cache, Abar_cache = _forward_triton(
                q_n, k_n, kappa, v, g, scale, initial_state, output_final_state, 64,
                intra=intra, return_intra_cache=True,
            )
        else:
            o, final_state = _forward_triton(
                q_n, k_n, kappa, v, g, scale, initial_state, output_final_state, 64, intra=intra
            )
            Aqk_cache, Abar_cache = None, None
        # Save the RAW (pre-l2norm) inputs for the recompute-in-PyTorch backward;
        # the pytorch wrapper re-applies l2norm so its Jacobian is captured. Aqk_cache
        # + Abar_cache (bf16 intra WY tensors) feed the recompute-only triton backward.
        ctx.save_for_backward(q, k, kappa, v, g, Aqk_cache, Abar_cache)
        ctx.scale = scale
        ctx.output_final_state = output_final_state
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = 64
        # Stash initial_state separately (a Tensor or None; may need a grad).
        ctx.initial_state = initial_state
        ctx.has_initial_state = initial_state is not None
        ctx.initial_state_needs_grad = (
            initial_state is not None and initial_state.requires_grad
        )
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht):
        # Backward dispatch (env KALMAN_BWD selects the path; DEFAULT = triton_intra):
        #   KALMAN_BWD unset / "triton_intra" / "triton" / "ti2"
        #       -> TI3 DEFAULT: full-Triton backward (forked wy_dqkg + bwd_intra).
        #          Fastest; verified element-wise vs the hybrid fp64 reference.
        #   KALMAN_BWD=hybrid / phase_b / b
        #       -> fp64-PyTorch intra-autograd hybrid (TRUSTED element-wise reference).
        #   KALMAN_BWD=phase_a / a / oracle
        #       -> slow fp64 full-recompute oracle (grad gate reference).
        # Forward default: intra="triton" (see _forward_triton / chunk_kalman).
        mode = os.environ.get("KALMAN_BWD", "triton_intra").lower()
        if mode in ("phase_a", "phasea", "a", "oracle"):
            return ChunkKalmanFunction._backward_phase_a(ctx, do, dht)
        if mode in ("hybrid", "phase_b", "phaseb", "b"):
            return ChunkKalmanFunction._backward_hybrid(ctx, do, dht)
        return ChunkKalmanFunction._backward_triton_intra(ctx, do, dht)

    # ------------------------------------------------------------------
    # Phase A (milestone B1): recompute the full pytorch forward under
    # enable_grad and autograd.grad it. Exact but slow (O(T) serial fp64
    # scan with a triangular solve per chunk). Kept as the grad ORACLE.
    # ------------------------------------------------------------------
    @staticmethod
    def _backward_phase_a(ctx, do, dht):
        q, k, kappa, v, g = ctx.saved_tensors[:5]  # (Aqk/Abar cache unused: full recompute)
        scale = ctx.scale
        chunk_size = ctx.chunk_size
        flag = ctx.use_qk_l2norm_in_kernel
        want_final_state = dht is not None

        # Recompute under enable_grad through the DIFFERENTIABLE pytorch WRAPPER.
        # Clone saved inputs as fresh leaves so autograd builds a clean graph and we
        # do not disturb the outer graph's saved tensors.
        with torch.enable_grad():
            q2 = q.detach().clone().requires_grad_(True)
            k2 = k.detach().clone().requires_grad_(True)
            kap2 = kappa.detach().clone().requires_grad_(True)
            v2 = v.detach().clone().requires_grad_(True)
            g2 = g.detach().clone().requires_grad_(True)

            inputs = [q2, k2, kap2, v2, g2]
            if ctx.initial_state_needs_grad:
                is2 = ctx.initial_state.detach().clone().requires_grad_(True)
                inputs.append(is2)
            elif ctx.has_initial_state:
                # provided but no grad wanted: pass through detached (no graph edge).
                is2 = ctx.initial_state.detach()
            else:
                is2 = None

            o2, fs2 = chunk_kalman(
                q2,
                k2,
                kap2,
                v2,
                g2,
                scale=scale,
                initial_state=is2,
                output_final_state=want_final_state,
                use_qk_l2norm_in_kernel=flag,
                backend="pytorch",
                chunk_size=chunk_size,
            )

            outputs = [o2]
            grad_outputs = [do]
            if want_final_state:
                outputs.append(fs2)
                grad_outputs.append(dht)

            grads = torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=grad_outputs,
                allow_unused=True,
            )

        # Map grads back to the forward input signature:
        #   (q, k, kappa, v, g, scale, initial_state, output_final_state,
        #    use_qk_l2norm_in_kernel)
        dq, dk, dkappa, dv, dg = grads[0], grads[1], grads[2], grads[3], grads[4]

        def _match(grad, ref):
            # allow_unused can return None; cast to input dtype/shape for safety.
            if grad is None:
                return None
            return grad.to(ref.dtype)

        dq = _match(dq, q)
        dk = _match(dk, k)
        dkappa = _match(dkappa, kappa)
        dv = _match(dv, v)
        dg = _match(dg, g)

        if ctx.initial_state_needs_grad:
            d_initial_state = _match(grads[5], ctx.initial_state)
        else:
            d_initial_state = None

        return (
            dq,
            dk,
            dkappa,
            dv,
            dg,
            None,  # scale
            d_initial_state,  # initial_state
            None,  # output_final_state
            None,  # use_qk_l2norm_in_kernel
            None,  # intra
        )

    # ------------------------------------------------------------------
    # Phase B (PB-mem): FAST hybrid backward. No O(T) PyTorch recompute.
    #
    #   * output bwd     : chunk-parallel einsums       -> dAqk, dv_new_out
    #                      (was chunk_kda_bwd_dAv (Triton); replaced because that
    #                       frozen kernel throws a CUDA illegal-memory-access during
    #                       autotune at B>=8,T=2048. KALMAN_DAV=triton restores it.)
    #   * state bwd      : chunk_gated_delta_rule_bwd_dhu (Triton) -> dh, dh0, dv2
    #   * h-input adjoint: chunk-parallel einsums from {dh, dv2} + fwd tensors
    #                      -> d(kappa_fed), d(w), d(u), dq_inter, dg_out, dg_state
    #   * intra bwd      : recompute _intra_pytorch under enable_grad (fp64,
    #                      CHUNK-PARALLEL, NOT the O(T) scan) and autograd.grad it
    #                      with cotangents (dw,du,dkappa_fed,dAqk) -> dq_intra, dk,
    #                      dkappa, dv, dg_intra  (the dk/dkappa split is FREE here).
    #   * cumsum reverse : chunk_local_cumsum(reverse=True) -> dg.
    #
    # Every boundary einsum was derived + verified bit-exact vs autograd through a
    # differentiable PyTorch h-recurrence (fp64) before implementation. The whole
    # path is chunk-parallel (heavy state work in the two Triton kernels) so it is
    # much faster than Phase A, and the intra autograd stays in fp64 so it is
    # deep-decay stable (never materializes 1/A in a way that overflows).
    # ------------------------------------------------------------------
    @staticmethod
    def _backward_hybrid(ctx, do, dht):
        q_raw, k_raw, kappa, v, g = ctx.saved_tensors[:5]  # (fp64 recompute: cache unused)
        scale = ctx.scale
        chunk_size = ctx.chunk_size
        flag = ctx.use_qk_l2norm_in_kernel
        want_final_state = dht is not None

        # CRITICAL: the incoming output cotangent `do` may be a broadcast/expanded or
        # otherwise non-contiguous tensor -- e.g. `o.sum().backward()` hands us a
        # 0-stride expand of a scalar 1.0. The reused Triton kernels (chunk_kda_bwd_dAv,
        # chunk_gated_delta_rule_bwd_dhu) address `do` with HARD-CODED contiguous strides
        # (H*V, 1), so a 0-stride `do` makes them read out-of-bounds garbage (~1e37) ->
        # overflow -> NaN in dk/dkappa/dv/dg (dq survives only because the einsum dAv path
        # materializes do via _hp). Real training feeds a contiguous do so the gates never
        # saw this; a broadcast do (o.sum().backward()) exposes it. Force contiguity here.
        do = do.contiguous()
        if dht is not None:
            dht = dht.contiguous()

        B, T, H, K = q_raw.shape
        V = v.shape[-1]
        in_dtype = q_raw.dtype
        kdtype = torch.bfloat16  # heavy kernels: bf16 in / fp32 accumulate (KDA)

        # ---- l2norm (read q,k only); capture rstd for the VJP ----
        if flag:
            q_n, q_rstd = l2norm_fwd(q_raw)
            k_n, k_rstd = l2norm_fwd(k_raw)
        else:
            q_n, k_n = q_raw, k_raw

        # ---- recompute forward tensors (cheap: 1 cumsum + chunk-parallel intra +
        # 1 Triton h-forward launch; NOT the O(T) pytorch scan). ----
        g_cumsum = chunk_local_cumsum(g, scale=RCP_LN2, chunk_size=chunk_size)  # [B,T,H,K] fp32

        # intra outputs (values only here; the differentiable recompute is below).
        with torch.no_grad():
            w, u, kappa_fed, Aqk, Tp, nc, BT = _intra_pytorch(
                q_n, k_n, kappa, v, g_cumsum, scale, chunk_size
            )
        pad = Tp - T
        if pad:
            q_p = torch.cat([q_n, q_n.new_zeros(B, pad, H, K)], 1)
            # FLAT-continuation pad (see _forward_triton): keeps the recomputed
            # chunk-end decay (gk_last for kappa_fed + the h-kernel carry) consistent
            # with the FIXED forward. Training uses dht=None so the last-chunk
            # contribution is zero (bit-unchanged grads); with dht it is now correct.
            g_p = torch.cat([g_cumsum, g_cumsum[:, -1:].expand(B, pad, H, K)], 1)
        else:
            q_p, g_p = q_n, g_cumsum

        w_k = w.to(kdtype)
        u_k = u.to(kdtype)
        kappa_fed_k = kappa_fed.to(kdtype)
        q_k = q_p.to(kdtype)
        Aqk_k = Aqk.to(kdtype)
        # dhu expects the query PRE-scaled by the per-channel decay exp2(gk_cum) (the
        # `qg` the GDN2/KDA orchestrator feeds it -- recompute_w_u_fwd_gdn2 stores
        # qg = q * exp2(gk_cum), chunk_gdn2.py:771). dhu then only applies `scale`.
        qg_k = (_hp(q_p) * torch.exp2(_hp(g_p))).to(kdtype)

        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kappa_fed_k,
            w=w_k,
            u=u_k,
            gk=g_p,
            initial_state=ctx.initial_state,
            output_final_state=False,
            chunk_size=chunk_size,
            use_exp2=True,
            transpose_state_layout=False,
        )
        # h: [B, nc, H, K, V] (state entering each chunk); v_new: [B, Tp, H, V].

        do_p = do
        if pad:
            do_p = torch.cat([do, do.new_zeros(B, pad, H, V)], 1)
        do_k = do_p.to(kdtype)

        # ---- output bwd: dAqk, dv_new_out ----
        # The gla output-intra term is  o_intra = tril_incl(Aqk) @ v_new, where Aqk
        # (from _intra_pytorch) already has `scale` baked in AND is tril-incl masked.
        # Its backward is two chunk-parallel einsums (per chunk, C = chunk_size):
        #     dv_new_out = tril_incl(Aqk)^T @ do    (cotangent wrt v_new)
        #     dAqk       = tril_incl(do @ v_new^T)  (cotangent wrt the POST-scale Aqk)
        #
        # This REPLACES the frozen Triton `chunk_kda_bwd_dAv`. That kernel threw a CUDA
        # illegal-memory-access at B>=8, T=2048 (surfacing during its autotune). ROOT CAUSE
        # (see the `do = do.contiguous()` note at the top of this method): a 0-stride
        # broadcast `do` (from o.sum().backward()) made the kernel's hard-coded-stride
        # block-pointer read out of bounds. The contiguity fix removes that crash for BOTH
        # backends; the einsum additionally removes the frozen-kernel autotune dependency
        # entirely and is exact-math-equivalent. `KALMAN_DAV=triton` restores the old path.
        #
        # NOTE on scale: `chunk_kda_bwd_dAv` returns dAqk PRE-scale (its kernel does the
        # `* scale`), so the Triton path divides by scale to recover the cotangent wrt
        # _intra_pytorch's post-scale Aqk. The einsum does NOT apply scale (do @ v_new^T
        # is directly the post-scale-Aqk cotangent), so there is NO /scale here.
        dav_backend = os.environ.get("KALMAN_DAV", "einsum").lower()
        if dav_backend == "triton":
            dAqk, dv_new_out = chunk_kda_bwd_dAv(
                q=q_k,
                k=kappa_fed_k,   # unused by dAv for dv/dAqk (k only shapes K); pass a K-tensor
                v=v_new,
                do=do_k,
                A=Aqk_k,
                scale=scale,
                chunk_size=chunk_size,
            )
            dAqk = _hp(dAqk) / scale  # -> cotangent wrt _intra_pytorch's (post-scale) Aqk
        else:
            C = BT
            # reshape to [B, nc, C, H, *]; c = query row, s = write-key col within chunk.
            Aqk_c = Aqk.reshape(B, nc, C, H, C)              # [b,n,c,h,s] fp32, tril-masked+scaled
            do_c_o = _hp(do_p).reshape(B, nc, C, H, V)       # [b,n,c,h,v]
            v_new_c = _hp(v_new).reshape(B, nc, C, H, V)     # [b,n,s,h,v]
            tril_incl = torch.tril(Aqk_c.new_ones(C, C))     # [c_row, s_col] incl diag
            # dv_new_out[b,n,s,h,v] = sum_c Aqk[c,s] * do[c,v]  (Aqk already tril-masked)
            dv_new_out = torch.einsum(
                "bnchs,bnchv->bnshv", Aqk_c, do_c_o
            ).reshape(B, Tp, H, V).to(kdtype)                # feed dhu at the same bf16 as before
            # dAqk[b,n,c,h,s] = tril_incl[c,s] * sum_v do[c,v] * v_new[s,v]  (mask over c,s dims)
            dAqk = torch.einsum("bnchv,bnshv->bnchs", do_c_o, v_new_c)
            dAqk = (dAqk * tril_incl[None, None, :, None, :]).reshape(B, Tp, H, C)  # post-scale-Aqk cotangent (fp32)

        # ---- state bwd (Triton): dh (per-chunk state adjoint carry), dh0, dv2 ----
        # dv2 = dv_new_out + kappa_fed @ dh  (the state-write read of v_new).
        dht_arg = dht if want_final_state else None
        dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
            q=qg_k,                # PRE-scaled query (q * exp2(gk_cum)); dhu adds `scale`
            k=kappa_fed_k,         # state-write key
            w=w_k,                 # read operand
            gk=g_p,
            h0=ctx.initial_state,
            dht=dht_arg,
            do=do_k,
            dv=dv_new_out,
            scale=scale,
            chunk_size=chunk_size,
            use_exp2=True,
            transpose_state_layout=False,
        )
        # dh: [B, nc, H, K, V]  (== the carry s_{c+1} entering chunk c in the derivation)
        # dv2: [B, Tp, H, V]    (== dv_new_full)



        # ---- h-input adjoints (chunk-parallel einsums, fp32) ----
        hp = _hp(h)                              # [B,nc,H,K,V]
        dhp = _hp(dh)                            # [B,nc,H,K,V]
        dv2p = _hp(dv2).view(B, nc, BT, H, V)    # [B,nc,C,H,V]
        v_newp = _hp(v_new).view(B, nc, BT, H, V)
        gcum_c = _hp(g_p).view(B, nc, BT, H, K)  # base-2 cum decay, padded
        gk_last = gcum_c[:, :, -1, :, :]         # [B,nc,H,K]

        # d(u) = dv2 ;  d(w) = -(dv2 @ h^T)  (v_new = u - w@h)
        du = dv2p                                                       # [B,nc,C,H,V]
        dw = -torch.einsum("bnchv,bnhkv->bnchk", dv2p, hp)             # [B,nc,C,H,K]
        # d(kappa_fed) = v_new @ dh^T  (h_{c+1} = D h_c + kappa_fed^T v_new)
        dkappa_fed = torch.einsum("bnchv,bnhkv->bnchk", v_newp, dhp)   # [B,nc,C,H,K]

        # output-inter dq: q_tilde = q * exp2(g_cum) * scale ; dq_tilde = do @ h^T
        do_c = _hp(do_p).view(B, nc, BT, H, V)
        dq_tilde = torch.einsum("bnchv,bnhkv->bnchk", do_c, hp)        # [B,nc,C,H,K]
        A_cum = torch.exp2(gcum_c)                                     # [B,nc,C,H,K]
        dq_inter = dq_tilde * A_cum * scale                           # grad wrt q (inter part)
        dg_out = _LN2 * _hp(q_p).view(B, nc, BT, H, K) * dq_inter      # dg_cumsum from q_tilde decay

        # state-decay dg: h_{c+1} = diag(exp2(gk_last_c)) h_c + ...  -> at last token.
        dg_state_last = _LN2 * torch.exp2(gk_last) * torch.einsum(
            "bnhkv,bnhkv->bnhk", hp, dhp
        )                                                             # [B,nc,H,K]

        # reshape the intra cotangents back to [B, Tp, H, *]
        def _flat_ck(x):  # [B,nc,C,H,K] -> [B,Tp,H,K]
            return x.reshape(B, nc * BT, H, x.shape[-1])

        dw_f = _flat_ck(dw)
        du_f = _flat_ck(du)
        dkappa_fed_f = _flat_ck(dkappa_fed)
        dAqk_f = dAqk  # [B, Tp, H, BT] (already _hp'd + de-scaled above)

        # ---- intra bwd via autograd (fp64, chunk-parallel) ----
        # feed fp64 leaves so the whole intra math + its VJP stay in fp64 (deep-decay
        # stable), and cotangents in fp64.  _intra_pytorch computes in fp64 internally.
        with torch.enable_grad():
            q2 = q_n.detach().to(torch.float64).requires_grad_(True)
            k2 = k_n.detach().to(torch.float64).requires_grad_(True)
            kap2 = kappa.detach().to(torch.float64).requires_grad_(True)
            v2 = v.detach().to(torch.float64).requires_grad_(True)
            gc2 = g_cumsum.detach().to(torch.float64).requires_grad_(True)
            w2, u2, kf2, Aqk2, _, _, _ = _intra_pytorch(
                q2, k2, kap2, v2, gc2, scale, chunk_size
            )
            # cotangents (slice to real length Tp of the intra outputs == padded).
            cot = [
                dw_f.to(torch.float64),
                du_f.to(torch.float64),
                dkappa_fed_f.to(torch.float64),
                dAqk_f.to(torch.float64),
            ]
            dq_intra, dk_read, dkappa, dv, dgc_intra = torch.autograd.grad(
                outputs=[w2, u2, kf2, Aqk2],
                inputs=[q2, k2, kap2, v2, gc2],
                grad_outputs=cot,
                allow_unused=True,
            )

        # ---- combine dq and dg_cumsum ----
        # The intra autograd grads (dq_intra,dk_read,dkappa,dv,dgc_intra) come out at
        # the UNPADDED length T (that is the length of the inputs fed to _intra_pytorch).
        # The h-boundary contributions (dq_inter, dg_out, dg_state) live at the padded
        # length Tp -> slice them to T before adding (padded tail is zero anyway).
        dq_inter_f = _flat_ck(dq_inter)[:, :T]  # [B,T,H,K]
        dg_out_f = _flat_ck(dg_out)[:, :T]      # [B,T,H,K]
        dq = dq_intra + dq_inter_f.to(torch.float64)

        # dg_cumsum = intra + output-inter decay + state decay (at last token).
        dg_state_full = dg_out.new_zeros(B, nc, BT, H, K)
        dg_state_full[:, :, -1, :, :] = dg_state_last
        dg_state_f = dg_state_full.reshape(B, nc * BT, H, K)[:, :T]  # [B,T,H,K]
        dg_cumsum = dgc_intra + dg_out_f.to(torch.float64) + dg_state_f.to(torch.float64)

        # ---- cumsum reverse: g_cumsum = RCP_LN2 * cumsum(g); map grad back to g.
        # d/dg = RCP_LN2 * reverse_cumsum(dg_cumsum). (mirror chunk_gdn2's reverse
        # cumsum; the RCP_LN2 factor is the base-2 conversion in the fwd cumsum.)
        dg = chunk_local_cumsum(
            dg_cumsum.to(torch.float32) * RCP_LN2, chunk_size=chunk_size, reverse=True
        )

        # ---- l2norm VJP (read q, k only) ----
        if flag:
            dq = l2norm_bwd(q_n.reshape(-1, K), q_rstd, dq.to(in_dtype).reshape(-1, K)).view(B, T, H, K)
            dk_read = l2norm_bwd(k_n.reshape(-1, K), k_rstd, dk_read.to(in_dtype).reshape(-1, K)).view(B, T, H, K)

        def _cast(x):
            return None if x is None else x.to(in_dtype)

        dq = _cast(dq)
        dk_read = _cast(dk_read)
        dkappa = _cast(dkappa)
        dv = _cast(dv)
        dg = _cast(dg)

        # ---- initial_state grad (from dhu's dh0) ----
        d_initial_state = None
        if ctx.initial_state_needs_grad:
            d_initial_state = dh0.to(ctx.initial_state.dtype) if dh0 is not None else None

        return (
            dq,
            dk_read,
            dkappa,
            dv,
            dg,
            None,  # scale
            d_initial_state,  # initial_state
            None,  # output_final_state
            None,  # use_qk_l2norm_in_kernel
            None,  # intra
        )

    # ------------------------------------------------------------------
    # TI2: FULL-Triton backward. Same chunk-parallel setup as _backward_hybrid
    # (dAv einsum + dhu reuse) but the h-input adjoints + fp64 intra-autograd are
    # REPLACED by the forked Triton backward intra kernels:
    #   * wy_dqkg fork  -> dq_inter, dk_read (w-decay), dkappa (kappa_fed),
    #                      dv, dg (partial), dM (WY-inverse VJP).
    #   * bwd_intra fork-> adds the within-chunk score-matrix VJP to
    #                      dq, dk_read, dkappa, dg (the M/Aqk two-key split).
    # Trusted reference = _backward_hybrid (element-wise gate, KALMAN_BWD=hybrid).
    # ------------------------------------------------------------------
    @staticmethod
    def _backward_triton_intra(ctx, do, dht):
        q_raw, k_raw, kappa, v, g, Aqk_cache, Abar_cache = ctx.saved_tensors
        scale = ctx.scale
        chunk_size = ctx.chunk_size
        flag = ctx.use_qk_l2norm_in_kernel
        want_final_state = dht is not None

        do = do.contiguous()
        if dht is not None:
            dht = dht.contiguous()

        B, T, H, K = q_raw.shape
        V = v.shape[-1]
        in_dtype = q_raw.dtype
        kdtype = torch.bfloat16

        _p_recompute = _kal_prof_start()
        # ---- l2norm (read q,k only); capture rstd for the VJP ----
        if flag:
            q_n, q_rstd = l2norm_fwd(q_raw)
            k_n, k_rstd = l2norm_fwd(k_raw)
        else:
            q_n, k_n = q_raw, k_raw

        # ---- recompute forward tensors (cheap; chunk-parallel) ----
        g_cumsum = chunk_local_cumsum(g, scale=RCP_LN2, chunk_size=chunk_size)  # [B,T,H,K] fp32

        # RO-1: REUSE the cached Aqk + Abar (WY inverse) from the forward instead of
        # re-running the full WY intra (token_parallel + inter_solve block-solve). The
        # backward now recomputes ONLY the cheap w/u/kappa_fed (+ qg) via
        # kalman_recompute_w_u_fwd and the h-forward -- exactly mirroring GDN-2's
        # recompute-only backward. The persisted bf16 Abar is bit-identical to a
        # re-derived one (deterministic kernels, same bf16 inputs), so grads DO NOT move.
        if Abar_cache is not None:
            Abar = Abar_cache
            Aqk_k = Aqk_cache               # post-scale, tril-masked Aqk (bf16, cached)
            Tp = Abar.shape[1]
            BT = Abar.shape[-1]
            nc = (Tp + BT - 1) // BT        # NO-PAD: Abar may be native T (Tp%BT!=0)
        else:
            # Fallback (forward used a non-triton intra -> no cache): rebuild Abar+Aqk
            # via the triton intra, still SKIPPING the Mraw debug buffer (RO-3).
            with torch.no_grad():
                _w, _u, _kf, Aqk_bf, Tp, nc, BT, Abar = _intra_triton(
                    q_n, k_n, kappa, v, g_cumsum, scale, chunk_size,
                    safe_gate=False, kernel_dtype=kdtype,
                    return_internals=True, return_M=False,
                )
            Aqk_k = Aqk_bf.to(kdtype)
        pad = Tp - T

        def _pad_T(x, last, dtype=None):
            if pad:
                x = torch.cat([x, x.new_zeros(B, pad, H, last)], 1)
            return x if dtype is None else x.to(dtype)

        q_p = _pad_T(q_n, K)
        # FLAT-continuation pad for g_cumsum (NOT the zero _pad_T used for the
        # keys/values): the forked wy_dqkg + recompute kernels read the chunk-end
        # decay from g at slot Tp-1 for the partial last chunk. Zero-padding would
        # make that read 0 (the same bug the forward had). Training uses dht=None so
        # the last-chunk term is zero (grads bit-unchanged); with dht it is correct.
        if pad:
            g_p = torch.cat([g_cumsum, g_cumsum[:, -1:].expand(B, pad, H, K)], 1)
        else:
            g_p = g_cumsum
        k_p = _pad_T(k_n, K)
        kappa_p = _pad_T(kappa, K)
        v_p = _pad_T(v, V)
        do_p = _pad_T(do, V)

        q_k = q_p.to(kdtype)
        k_k = k_p.to(kdtype)
        kappa_k = kappa_p.to(kdtype)
        v_k = v_p.to(kdtype)
        do_k = do_p.to(kdtype)

        # ---- recompute-only w/u/kappa_fed (+ qg via STORE_QG) from the cached Abar
        # (RO-1/RO-4). qg = q * exp2(gk_cum) is emitted IN-KERNEL (was a separate
        # PyTorch exp2 sweep); dhu consumes qg (it then only applies `scale`).
        w_k, u_k, kappa_fed_k, qg_k = kalman_recompute_w_u_fwd(
            k=k_k, kappa=kappa_k, v=v_k, A=Abar, gk=g_p, q=q_k,
        )

        # ---- recompute h / v_new (Triton h-forward) ----
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kappa_fed_k,
            w=w_k,
            u=u_k,
            gk=g_p,
            initial_state=ctx.initial_state,
            output_final_state=False,
            chunk_size=chunk_size,
            use_exp2=True,
            transpose_state_layout=False,
        )
        # h: [B, nc, H, K, V]; v_new: [B, Tp, H, V].
        _kal_prof_end("recompute_wu", _p_recompute)

        _p_dav = _kal_prof_start()
        # ---- output bwd via FUSED Triton dAv (RO-2): dAqk (PRE-scale-score cotangent,
        # i.e. already * scale) + dv_new_out. Replaces the fp32 einsum with GDN-2's own
        # chunk_kda_bwd_dAv kernel (the KALMAN_DAV=triton path). `do` is contiguous (top
        # of this method) so the 0-stride crash cannot recur. The kernel APPLIES scale
        # internally (its dA store is `* scale`), and bwd_intra consumes that pre-scale
        # -score cotangent directly -- matching the old einsum's `* scale`. q/k are only
        # shape sources for the kernel wrapper (unused in its body).
        dAqk, dv_new_out = chunk_kda_bwd_dAv(
            q=q_k,
            k=kappa_fed_k,
            v=v_new,
            do=do_k,
            A=Aqk_k,
            scale=scale,
            chunk_size=chunk_size,
        )
        dAqk = dAqk.to(torch.float32).contiguous()   # bwd_intra loads fp32 dAqk
        dv_new_out = dv_new_out.to(kdtype)
        _kal_prof_end("dAv_fused", _p_dav)

        _p_dhu = _kal_prof_start()
        # ---- state bwd (Triton): dh, dh0, dv(=du) ----
        dht_arg = dht if want_final_state else None
        dh, dh0, du = chunk_gated_delta_rule_bwd_dhu(
            q=qg_k,
            k=kappa_fed_k,
            w=w_k,
            gk=g_p,
            h0=ctx.initial_state,
            dht=dht_arg,
            do=do_k,
            dv=dv_new_out,
            scale=scale,
            chunk_size=chunk_size,
            use_exp2=True,
            transpose_state_layout=False,
        )
        _kal_prof_end("dhu", _p_dhu)

        _p_wy = _kal_prof_start()
        # ---- wy_dqkg fork: h-adjoints + WY VJP -> dq_inter, dk_read, dkappa, dv, dg, dM ----
        dq_w, dk_read, dkappa, dv, dg_w, dM = kalman_bwd_wy_dqkg_fused(
            q=q_k, k=k_k, kappa=kappa_k, v=v_k, v_new=v_new, g=g_p,
            A=Abar, h=h, do=do_k, dh=dh, dv=du, scale=scale, chunk_size=chunk_size,
        )
        _kal_prof_end("wy_dqkg", _p_wy)

        _p_bi = _kal_prof_start()
        # ---- bwd_intra fork: within-chunk score-matrix VJP (M/Aqk two-key split) ----
        dq, dk_read, dkappa, dg = kalman_bwd_intra(
            q=q_k, k=k_k, kappa=kappa_k, g=g_p,
            dAqk=dAqk, dAkk=dM,
            dq=dq_w, dk=dk_read, dkappa=dkappa, dg=dg_w,
            chunk_size=chunk_size, safe_gate=False,
        )
        _kal_prof_end("bwd_intra", _p_bi)

        _p_rc = _kal_prof_start()
        # ---- cumsum reverse: map dg_cumsum grad back to g (KDA convention, no scale) ----
        dg = chunk_local_cumsum(
            dg.to(torch.float32), chunk_size=chunk_size, reverse=True
        )
        _kal_prof_end("reverse_cumsum", _p_rc)

        # ---- slice to real length T ----
        dq = dq[:, :T]
        dk_read = dk_read[:, :T]
        dkappa = dkappa[:, :T]
        dv = dv[:, :T]
        dg = dg[:, :T]

        # ---- l2norm VJP (read q, k only) ----
        if flag:
            dq = l2norm_bwd(q_n.reshape(-1, K), q_rstd, dq.to(in_dtype).reshape(-1, K)).view(B, T, H, K)
            dk_read = l2norm_bwd(k_n.reshape(-1, K), k_rstd, dk_read.to(in_dtype).reshape(-1, K)).view(B, T, H, K)

        def _cast(x):
            return None if x is None else x.to(in_dtype)

        dq = _cast(dq)
        dk_read = _cast(dk_read)
        dkappa = _cast(dkappa)
        dv = _cast(dv)
        dg = _cast(dg)

        d_initial_state = None
        if ctx.initial_state_needs_grad:
            d_initial_state = dh0.to(ctx.initial_state.dtype) if dh0 is not None else None

        return (
            dq,
            dk_read,
            dkappa,
            dv,
            dg,
            None,  # scale
            d_initial_state,  # initial_state
            None,  # output_final_state
            None,  # use_qk_l2norm_in_kernel
            None,  # intra
        )



def chunk_kalman_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    kappa: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Differentiable entry: fast triton forward + Phase-A (recompute) backward.

    Returns (o, final_state). Use this in a training layer to get grads for
    q, k, kappa, v, g (and initial_state when it requires grad).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return ChunkKalmanFunction.apply(
        q,
        k,
        kappa,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
    )
