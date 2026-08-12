# Triton kernel(s) for the ISO-KLA write-gate scan (beta_t).
#
# ISO-KLA (paper appendix A.3) is a linear-attention mixer whose write gate
# beta_t is a SCALAR per (B, T, H) produced by an additive-Kalman scalar scan.
# The pure-PyTorch references are ``lit_gpt/kla_ops/iso_naive.py::_iso_beta_seq``
# (sequential) and ``_iso_beta_chunk`` (Hillis-Steele Mobius scan); they agree
# to ~1e-7 and are the CORRECTNESS ORACLES for this kernel.
#
# ------------------------------------------------------------------------------
# Kernel contract (Option A -- pure scan)
# ------------------------------------------------------------------------------
#   iso_beta_chunk(a, q, r, mu, *, out_dtype=None) -> beta
#   with  a, q, r, beta : [B, T, H]   and   mu : [H]
#
#   * a_t = mean_i(alpha_{t,i}^2) is PRECOMPUTED by the caller (the kernel does
#     NOT reduce alpha; it never sees alpha).
#   * PRECONDITION: k is L2-normalized  =>  ||k||^2 = 1. The kernel NEVER sees k.
#     This collapses u_t = ||k||^2 / r_t -> 1/r_t and the gain denominator
#     r_t + b_hat_t * ||k||^2 -> r_t + b_hat_t. ``assert_normalized`` documents
#     this for the layer/tests; the kernel itself takes only a, q, r, mu.
#
# ------------------------------------------------------------------------------
# Exact math the kernel computes (this IS _iso_beta_seq with ||k||^2 = 1)
# ------------------------------------------------------------------------------
#   c_{-1} = mu[h]                       # exclusive prefix info at t=0 (per head)
#   for t = 0..T-1:
#       b_hat  = a_t / c_{t-1} + q_t     # predicted covariance (EXCLUSIVE prefix)
#       beta_t = b_hat / (r_t + s*b_hat) # write gate  (since ||k||^2 = 1; s = info_scale)
#       c_t    = 1/b_hat + s/r_t         # posterior info (c_hat=1/b_hat, u_t = s/r_t)
#
# ``s = info_scale`` is a RUNTIME fp32 scalar (default 1.0 = pre-ablation IsoKLA,
# i.e. NO d_k standardization) threaded into every scale-carrying kernel via
# ``do_not_specialize`` so ONE compiled variant serves every s in {1, sqrt(d_k), d_k}.
#
# The scalar ``c`` is tracked in fp32 -- no matrix, no renorm needed for the
# sequential (scalar) form. ``c`` stays bounded/positive for the LAYER-REACHABLE
# input space, where ``q_t = q_min + softplus(.) > 0`` always: the additive q caps
# the growth of c, so no overflow/underflow over 16k. (The "q == 0 is fine" claim
# is FALSE in the non-layer-reachable adversarial ``q == 0, a < 1, r == r_min``
# regime -- there the predict step ``b_hat = a/c`` with q==0 lets c grow
# geometrically and it overflows fp32 around t~800. Softplus makes q>0 in the
# layer, so this edge is unreachable in practice.) The scan uses NO ``tl.dot``
# (scalar mults only), so there is no tf32/ieee matmul knob.
#
# This is the SIMPLE, CORRECT v0: grid (B*H,), one program per (b, h) stream,
# serial over T inside the program. Parallelism is only across B*H -- deliberately
# slow; a later task optimizes it. Correctness is non-negotiable.
from __future__ import annotations

import torch
import triton
import triton.language as tl


def assert_normalized(k: torch.Tensor, atol: float = 1e-3) -> None:
    """Assert ``k`` is L2-normalized along its last dim (``||k||^2 == 1``).

    Documents the kernel precondition at the layer/test boundary. The kernel
    itself never sees ``k`` -- it relies on ``||k||^2 == 1`` to collapse
    ``u_t = ||k||^2 / r_t -> 1/r_t`` and the gain denominator. Cheap; call it
    where ``k`` is available (not on the kernel hot path).
    """
    ksq = (k.float() * k.float()).sum(dim=-1)
    max_dev = (ksq - 1.0).abs().max().item()
    if max_dev > atol:
        raise AssertionError(
            f"iso_beta_chunk precondition violated: k is not L2-normalized "
            f"(max ||k||^2-1 = {max_dev:.3e} > atol={atol:.3e}). Normalize k with "
            f"F.normalize(k, p=2, dim=-1) before computing beta."
        )


# =============================================================================
# v0 KERNEL -- scalar Kalman info scan, one program per (b, h)
# -----------------------------------------------------------------------------
# For [B, T, H]-contiguous tensors, element (b, t, h) sits at offset
# ``b*T*H + t*H + h``; so for a fixed (b, h) the base is ``b*T*H + h`` and the
# per-token stride along T is ``H``. Each program owns one (b, h) stream, loads
# mu[h] once as the initial ``c``, and walks the tokens one at a time in a
# RUNTIME ``for t in range(T)`` loop (T is a runtime arg, so this is a real loop,
# NOT Python-unrolled -- which is what keeps it compilable at T=16384). Each
# iteration loads the three scalars a_t/q_t/r_t (bf16 or fp32) at the strided
# offset, casts to fp32, runs the scalar recurrence carrying ``c`` in an fp32
# register, and stores beta_t in the output dtype. Deliberately serial in T;
# parallelism is only across the B*H programs. This is the correctness anchor.
# =============================================================================
@triton.jit(do_not_specialize=["T", "info_scale"])
def _iso_beta_fwd_kernel(
    a_ptr,          # [B, T, H]  a_t = mean_i(alpha_{t,i}^2)
    q_ptr,          # [B, T, H]  additive process noise q_t
    r_ptr,          # [B, T, H]  observation noise r_t
    mu_ptr,         # [H]        info prior c_{-1}
    beta_ptr,       # [B, T, H]  OUTPUT write gate
    T,              # sequence length (runtime -> real loop, not unrolled)
    info_scale,     # info-increment scale s in u=s/r, gain denom r+s*b_hat (runtime fp32; default 1)
    H: tl.constexpr,
):
    pid = tl.program_id(0)          # in [0, B*H)
    i_b = pid // H
    i_h = pid % H

    # Base offset of this (b, h) stream; per-token stride along T is H.
    base = i_b * T * H + i_h

    # c_{-1} = mu[h]  (fp32)
    c = tl.load(mu_ptr + i_h).to(tl.float32)

    # Serial scalar recurrence, one token per iteration (runtime loop).
    for t in range(0, T):
        off = base + t * H          # scalar element offset for (b, t, h)

        a = tl.load(a_ptr + off).to(tl.float32)      # a_t
        q = tl.load(q_ptr + off).to(tl.float32)      # q_t
        r = tl.load(r_ptr + off).to(tl.float32)      # r_t

        b_hat = a / c + q                            # b_hat_t = a_t / c_{t-1} + q_t
        beta = b_hat / (r + info_scale * b_hat)      # beta_t   (||k||^2 = 1, s scales it)
        c = 1.0 / b_hat + info_scale / r             # posterior info c_t (u_t = s/r)

        tl.store(beta_ptr + off, beta.to(beta_ptr.dtype.element_ty))


@torch.compiler.disable
def _iso_beta_chunk_v0(
    a: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    mu: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    info_scale: float | None = None,
) -> torch.Tensor:
    """v0 -- serial-in-T scalar scan (grid ``(B*H,)``). Correctness anchor.

    Kept as the oracle/fallback for the faster v1 (chunked two-pass Mobius). Same
    contract and math as :func:`iso_beta_chunk`; see that docstring. ``info_scale``
    is the info-increment scale ``s`` (default ``None`` -> ``1.0``).
    """
    B, T, H = a.shape
    info_scale = 1.0 if info_scale is None else float(info_scale)   # s in u=s/r (default 1)
    if out_dtype is None:
        out_dtype = torch.float32
    beta = torch.empty((B, T, H), dtype=out_dtype, device=a.device)
    if T == 0:
        return beta
    grid = (B * H,)
    _iso_beta_fwd_kernel[grid](
        a, q, r, mu, beta,
        T, info_scale,
        H=H,
    )
    return beta


# =============================================================================
# v1 KERNELS -- chunked two-pass Mobius (reduce-then-scan). Attacks v0's serial-
# in-T weakness (only B*H programs): parallelism becomes B*H*NT.
#
# The information recursion c_t = (A_t c_{t-1}+B_t)/(C_t c_{t-1}+D_t) is a Mobius
# (linear-fractional) map, composed by 2x2 matmul. M_t = [[1+u_t q_t, u_t a_t],
# [q_t, a_t]] with u_t = 1/r_t (||k||^2=1). Mobius maps are scale-invariant
# (c=n/d), so partial products are renormalized by their max entry (exact + keeps
# the all-positive entries bounded) -- the analog of GDN2's anchored decay. c is
# tracked fp32 throughout (a prefix product needs fp32; bf16 mantissa too coarse).
#
#   Pass A (grid (NT, B*H)): each chunk computes its NET map M_chunk =
#       M_{end-1}...M_{start} (left-product), fp32 + running max-entry renorm.
#       Stores 4 floats [A,B,C,D] per (nt,b,h).
#   Pass B (grid (B*H,)):    serial over NT (tiny), thread a SCALAR carry c=mu
#       through the chunk maps; store the EXCLUSIVE-prefix c entering each chunk
#       (carry_c[nt] = c BEFORE applying M_chunk[nt]).
#   Pass C (grid (NT, B*H)): seed c=carry_c[nt], run the PURE scalar recurrence
#       over the chunk's BT tokens emitting beta (identical to v0's inner loop,
#       no matrix/renorm). Re-reads a,q,r.
#
# Verified bit-exact (<=1.1e-16, fp64) to the scalar recurrence for BT in
# {1,7,16,40,64} incl. non-divisor BT and tails.
# =============================================================================
@triton.jit(do_not_specialize=["T", "info_scale"])
def _iso_beta_passA_kernel(
    a_ptr,          # [B, T, H]
    q_ptr,          # [B, T, H]
    r_ptr,          # [B, T, H]
    cm_ptr,         # [B, NT, H, 4]  OUTPUT chunk net map [A,B,C,D]
    T,              # sequence length (runtime)
    info_scale,     # info-increment scale s in u=s/r (runtime fp32; default 1)
    NT: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    i_nt = tl.program_id(0)         # chunk index in [0, NT)
    pid = tl.program_id(1)          # (b, h) in [0, B*H)
    i_b = pid // H
    i_h = pid % H

    base = i_b * T * H + i_h        # base of this (b,h) stream; per-token stride H
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)

    # net map M_chunk, start = identity [[1,0],[0,1]]
    mA = 1.0
    mB = 0.0
    mC = 0.0
    mD = 1.0
    for t in range(t0, t1):
        off = base + t * H
        a = tl.load(a_ptr + off).to(tl.float32)
        q = tl.load(q_ptr + off).to(tl.float32)
        r = tl.load(r_ptr + off).to(tl.float32)
        u = info_scale / r                          # u_t = s ||k||^2 / r = s/r (||k||^2=1)
        # M_t = [[1+u q, u a],[q, a]]
        tA = 1.0 + u * q
        tB = u * a
        tC = q
        tD = a
        # left-multiply: M = M_t @ M   (row-major 2x2)
        nA = tA * mA + tB * mC
        nB = tA * mB + tB * mD
        nC = tC * mA + tD * mC
        nD = tC * mB + tD * mD
        # max-entry renorm (Mobius scale-invariant): keep entries bounded
        m = tl.maximum(tl.maximum(tl.abs(nA), tl.abs(nB)),
                       tl.maximum(tl.abs(nC), tl.abs(nD)))
        inv = 1.0 / tl.maximum(m, 1e-30)
        mA = nA * inv
        mB = nB * inv
        mC = nC * inv
        mD = nD * inv

    cm_base = (i_b * NT * H + i_nt * H + i_h) * 4
    tl.store(cm_ptr + cm_base + 0, mA)
    tl.store(cm_ptr + cm_base + 1, mB)
    tl.store(cm_ptr + cm_base + 2, mC)
    tl.store(cm_ptr + cm_base + 3, mD)


@triton.jit
def _iso_beta_passB_kernel(
    cm_ptr,         # [B, NT, H, 4]  chunk net maps [A,B,C,D]
    mu_ptr,         # [H]
    carry_ptr,      # [B, NT, H]  OUTPUT exclusive-prefix c entering each chunk
    NT: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(0)          # (b, h) in [0, B*H)
    i_b = pid // H
    i_h = pid % H

    c = tl.load(mu_ptr + i_h).to(tl.float32)   # c entering chunk 0 = mu[h]
    for nt in range(0, NT):
        # store EXCLUSIVE prefix (c before applying this chunk's map)
        carry_off = i_b * NT * H + nt * H + i_h
        tl.store(carry_ptr + carry_off, c)
        cm_base = (i_b * NT * H + nt * H + i_h) * 4
        mA = tl.load(cm_ptr + cm_base + 0)
        mB = tl.load(cm_ptr + cm_base + 1)
        mC = tl.load(cm_ptr + cm_base + 2)
        mD = tl.load(cm_ptr + cm_base + 3)
        n = mA * c + mB
        d = mC * c + mD
        c = n / d


@triton.jit(do_not_specialize=["T", "info_scale"])
def _iso_beta_passC_kernel(
    a_ptr,          # [B, T, H]
    q_ptr,          # [B, T, H]
    r_ptr,          # [B, T, H]
    carry_ptr,      # [B, NT, H]  exclusive-prefix c entering each chunk
    beta_ptr,       # [B, T, H]  OUTPUT
    cprev_ptr,      # [B, T, H]  OPTIONAL fp32 OUTPUT: c_{t-1} entering each step (SAVE_C)
    T,
    info_scale,     # info-increment scale s in u=s/r, gain denom r+s*b_hat (runtime fp32; default 1)
    NT: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    SAVE_C: tl.constexpr,   # if True, also store the exclusive-prefix c_{t-1} per token
):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H

    base = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)

    carry_off = i_b * NT * H + i_nt * H + i_h
    c = tl.load(carry_ptr + carry_off).to(tl.float32)   # seed = carry_c[nt]

    for t in range(t0, t1):
        off = base + t * H
        a = tl.load(a_ptr + off).to(tl.float32)
        q = tl.load(q_ptr + off).to(tl.float32)
        r = tl.load(r_ptr + off).to(tl.float32)
        if SAVE_C:
            # c currently holds c_{t-1} (exclusive prefix entering step t). Save it
            # fp32 so the parallel backward can read c_{t-1} without recomputing.
            tl.store(cprev_ptr + off, c)
        b_hat = a / c + q
        beta = b_hat / (r + info_scale * b_hat)          # ||k||^2 = 1 (s scales it)
        c = 1.0 / b_hat + info_scale / r                 # u_t = s/r
        tl.store(beta_ptr + off, beta.to(beta_ptr.dtype.element_ty))


# Default chunk length for v1 (tokens per program). 256 keeps NT small enough for
# the short Pass-B serial carry (16k/256 = 64) while giving B*H*NT programs.
_ISO_BT_DEFAULT = 256


@torch.compiler.disable
def _iso_beta_chunk_v1(
    a: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    mu: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    BT: int = _ISO_BT_DEFAULT,
    return_cprev: bool = False,
    info_scale: float | None = None,
) -> torch.Tensor:
    """v1 -- chunked two-pass Mobius scan. Same contract/math as :func:`iso_beta_chunk`.

    Parallelism B*H*NT (vs v0's B*H); only the short Pass-B carry is serial. ``BT``
    is the tokens-per-chunk tile (exposed for the BT>T test case). ``info_scale`` is
    the info-increment scale ``s`` (default ``None`` -> ``1.0``).

    If ``return_cprev`` is True, also returns the fp32 exclusive-prefix info state
    ``cprev`` ``[B,T,H]`` (``cprev[b,t,h] = c_{t-1}``, ``= mu`` at ``t=0``) that the
    PARALLEL backward reads (state-handling option A, save-c). Pass C already holds
    ``c_{t-1}`` in a register at the top of each step, so this is a free extra store.
    Returns ``beta`` (or ``(beta, cprev)`` when ``return_cprev``).
    """
    B, T, H = a.shape
    info_scale = 1.0 if info_scale is None else float(info_scale)   # s in u=s/r (default 1)
    if out_dtype is None:
        out_dtype = torch.float32
    beta = torch.empty((B, T, H), dtype=out_dtype, device=a.device)
    cprev = (torch.empty((B, T, H), dtype=torch.float32, device=a.device)
             if return_cprev else None)
    if T == 0:
        return (beta, cprev) if return_cprev else beta

    NT = (T + BT - 1) // BT
    # fp32 scratch: chunk maps [B,NT,H,4] and exclusive-prefix carries [B,NT,H].
    cm = torch.empty((B, NT, H, 4), dtype=torch.float32, device=a.device)
    carry = torch.empty((B, NT, H), dtype=torch.float32, device=a.device)

    gridAC = (NT, B * H)
    _iso_beta_passA_kernel[gridAC](a, q, r, cm, T, info_scale, NT=NT, H=H, BT=BT)
    _iso_beta_passB_kernel[(B * H,)](cm, mu, carry, NT=NT, H=H)
    # cprev may be None; Triton ignores an unused ptr arg when SAVE_C is False.
    _iso_beta_passC_kernel[gridAC](
        a, q, r, carry, beta, cprev if cprev is not None else beta,
        T, info_scale, NT=NT, H=H, BT=BT, SAVE_C=return_cprev,
    )
    return (beta, cprev) if return_cprev else beta


# Kernel-version selector: "v1" (fast chunked two-pass, default) or "v0" (serial
# anchor). Env override for A/B + oracle-compare: ISO_BETA_KERNEL in {v0,v1}.
import os as _os  # noqa: E402

_ISO_BETA_KERNEL = _os.environ.get("ISO_BETA_KERNEL", "v1").lower()


# =============================================================================
# BACKWARD -- serial reverse VJP of the beta scan (Task B1: simple + correct).
#
# Derivation: docs/plans/2026-07-24-iso-beta-backward-derivation.md (2-round
# confirmed). Given output cotangents ``dbeta_t`` (final state c_{T-1} is NOT an
# output => its cotangent is 0), emit ``da,dq,dr`` per (b,t,h) and ``dmu`` per h.
#
# Per stream (b, h), reverse t = T-1 -> 0, carrying the state-adjoint s = c_bar_t
# (fp32), init s = 0. With cprev = c_{t-1} (mu if t==0), b_hat_t = a_t/cprev + q_t,
# denom_t = r_t + b_hat_t:
#     b_bar_t = dbeta_t*(r_t/denom^2)  +  s*(-1/b_hat^2)
#     dr_t    = dbeta_t*(-b_hat/denom^2) + s*(-1/r_t^2)
#     da_t    = b_bar_t / cprev
#     dq_t    = b_bar_t
#     c_bar_{t-1} = b_bar_t*(-a_t/cprev^2)   ;   s <- c_bar_{t-1}
# After t=0: dmu[h] += s (= c_bar_{-1}); mu is shared across the batch per head, so
# every (b,h) program atomic-adds its c_bar_{-1} into dmu[h].
#
# State handling: OPTION B (recompute). Each (b,h) program first re-runs the
# forward scalar scan FORWARD, writing c_{t-1} into an fp32 scratch buffer
# ``cprev_ptr[b,t,h]``, then walks the reverse scan reading that buffer. One
# program per (b,h); serial in T inside the program (grid (B*H,)). No tl.dot, no
# renorm. fp32 scan; bf16 I/O for dbeta in / da,dq,dr out (fp32 anchor variant
# too). Deliberately slow -- Task B2 parallelizes via the affine two-pass.
# =============================================================================
@triton.jit(do_not_specialize=["T", "info_scale"])
def _iso_beta_bwd_kernel(
    a_ptr,          # [B, T, H]  a_t
    q_ptr,          # [B, T, H]  q_t
    r_ptr,          # [B, T, H]  r_t
    mu_ptr,         # [H]        info prior c_{-1}
    dbeta_ptr,      # [B, T, H]  INPUT cotangent dL/dbeta_t
    cprev_ptr,      # [B, T, H]  fp32 SCRATCH: c_{t-1} recomputed forward-order
    da_ptr,         # [B, T, H]  OUTPUT dL/da_t
    dq_ptr,         # [B, T, H]  OUTPUT dL/dq_t
    dr_ptr,         # [B, T, H]  OUTPUT dL/dr_t
    dmu_ptr,        # [H]        OUTPUT dL/dmu[h]  (fp32, atomic-accumulated over b)
    T,              # sequence length (runtime -> real loop, not unrolled)
    info_scale,     # info-increment scale s in u=s/r, gain denom r+s*b_hat (runtime fp32; default 1)
    H: tl.constexpr,
):
    pid = tl.program_id(0)          # in [0, B*H)
    i_b = pid // H
    i_h = pid % H

    # Base offset of this (b, h) stream; per-token stride along T is H.
    base = i_b * T * H + i_h

    # ---- forward recompute: fill cprev[t] = c_{t-1} (exclusive prefix info) ----
    c = tl.load(mu_ptr + i_h).to(tl.float32)     # c_{-1} = mu[h]
    for t in range(0, T):
        off = base + t * H
        tl.store(cprev_ptr + off, c)             # c_{t-1} entering step t
        a = tl.load(a_ptr + off).to(tl.float32)
        q = tl.load(q_ptr + off).to(tl.float32)
        r = tl.load(r_ptr + off).to(tl.float32)
        b_hat = a / c + q
        c = 1.0 / b_hat + info_scale / r         # posterior info c_t (u_t = s/r)

    # ---- reverse VJP scan: t = T-1 -> 0, carry s = c_bar_t (init c_bar_{T-1}=0) ----
    s = 0.0                                       # state-adjoint from the future (fp32)
    for t in range(T - 1, -1, -1):
        off = base + t * H
        cprev = tl.load(cprev_ptr + off).to(tl.float32)   # c_{t-1} (= mu at t=0)
        a = tl.load(a_ptr + off).to(tl.float32)
        q = tl.load(q_ptr + off).to(tl.float32)
        r = tl.load(r_ptr + off).to(tl.float32)
        db = tl.load(dbeta_ptr + off).to(tl.float32)      # dbeta_t

        b_hat = a / cprev + q
        denom = r + info_scale * b_hat                     # gain denom (s scales b_hat*||k||^2)
        inv_denom2 = 1.0 / (denom * denom)

        # adjoint of b_hat_t: from beta_t (via gain) and from c_t (via 1/b_hat)
        b_bar = db * (r * inv_denom2) + s * (-1.0 / (b_hat * b_hat))
        # grad of r_t: from beta_t and from c_t (s/r term)
        dr = db * (-b_hat * inv_denom2) + s * (-info_scale / (r * r))
        # grads of a_t, q_t (from b_hat = a/cprev + q)
        da = b_bar / cprev
        dq = b_bar
        # adjoint pushed to previous state c_{t-1} (becomes s for step t-1)
        s = b_bar * (-a / (cprev * cprev))

        tl.store(da_ptr + off, da.to(da_ptr.dtype.element_ty))
        tl.store(dq_ptr + off, dq.to(dq_ptr.dtype.element_ty))
        tl.store(dr_ptr + off, dr.to(dr_ptr.dtype.element_ty))

    # after t=0, s = c_bar_{-1} = adjoint of mu for this (b,h); mu shared over batch
    # per head => sum over b via atomic add into dmu[h] (fp32 accumulator).
    tl.atomic_add(dmu_ptr + i_h, s)


@torch.compiler.disable
def _iso_beta_bwd_serial(
    a: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    mu: torch.Tensor,
    dbeta: torch.Tensor,
    *,
    grad_dtype: torch.dtype | None = None,
    info_scale: float | None = None,
):
    """Serial reverse VJP of the beta scan (grid ``(B*H,)``). Correctness anchor.

    Given ``dbeta = dL/dbeta`` ``[B,T,H]`` returns ``(da, dq, dr, dmu)`` with
    ``da/dq/dr : [B,T,H]`` and ``dmu : [H]``. Recomputes ``c_{t-1}`` forward
    (option B) into an fp32 scratch, then walks the reverse affine scan. ``dmu`` is
    always fp32 (atomic accumulator); ``da/dq/dr`` are ``grad_dtype`` (default the
    dtype of ``a`` so grads match input dtype for autograd). ``info_scale`` is the
    info-increment scale ``s`` (default ``None`` -> ``1.0``).
    """
    B, T, H = a.shape
    if grad_dtype is None:
        grad_dtype = a.dtype
    info_scale = 1.0 if info_scale is None else float(info_scale)   # s in u=s/r (default 1)
    da = torch.empty((B, T, H), dtype=grad_dtype, device=a.device)
    dq = torch.empty((B, T, H), dtype=grad_dtype, device=a.device)
    dr = torch.empty((B, T, H), dtype=grad_dtype, device=a.device)
    # dmu must be fp32 (atomic-add accumulator) and zeroed before accumulation.
    dmu = torch.zeros((H,), dtype=torch.float32, device=a.device)
    if T == 0:
        return da, dq, dr, dmu.to(mu.dtype)

    # fp32 scratch for the recomputed exclusive-prefix c_{t-1} (option B).
    cprev = torch.empty((B, T, H), dtype=torch.float32, device=a.device)

    grid = (B * H,)
    _iso_beta_bwd_kernel[grid](
        a, q, r, mu, dbeta, cprev,
        da, dq, dr, dmu,
        T, info_scale,
        H=H,
    )
    return da, dq, dr, dmu.to(mu.dtype)


# =============================================================================
# BACKWARD v1 -- PARALLEL affine two-pass reverse VJP (Task B2). Mirrors the
# FORWARD v1 chunked two-pass, but the reverse scan is AFFINE (not Mobius), so it
# is SIMPLER -- 2 scalars per chunk and NO renorm needed (derivation §4, round-2
# confirmed: q>0 caps c so the segment product stays <= ~6.5e13 << fp32-max).
#
# The state-adjoint recurrence (derivation §4) is
#     s_{t-1} = A_t * s_t + B_t ,   s_t = c_bar_t   (the adjoint ENTERING step t),
#     A_t =  a_t / (c_{t-1}^2 * b_hat_t^2),
#     B_t = -(a_t / c_{t-1}^2) * dbeta_t * r_t / denom_t^2,
# processed reverse t=T-1 -> 0, init c_bar_{T-1}=0 (final state is not an output).
# Per token, grads use c_bar_t (the value entering step t, from the future):
#     b_bar_t = dbeta_t*(r_t/denom^2)  -  c_bar_t/b_hat^2
#     da_t = b_bar_t/c_{t-1} ; dq_t = b_bar_t ; dr_t = dbeta_t*(-b_hat/denom^2) - c_bar_t/r_t^2
#     dmu[h] = sum_b c_bar_{-1}(b,h).
#
# Affine maps compose associatively  (A2,B2) o (A1,B1) = (A2*A1, A2*B1 + B2)  so
# the reverse scan uses the SAME reduce-then-scan structure as the forward v1:
#   Pass A (grid (NT,B*H)): per chunk, compose (A_t,B_t) IN REVERSE token order
#       into the chunk net map (A_chunk,B_chunk) that sends the adjoint entering
#       the chunk from its HIGH end -> the adjoint exiting at its LOW end
#       (= c_bar_{t0-1}, handed to the next-lower chunk). Store 2 fp32/chunk.
#   Pass B (grid (B*H,)):    serial over chunks IN REVERSE (nt=NT-1->0), carry the
#       scalar adjoint s (init 0 at the top). Store carry_bwd[nt] = s BEFORE the
#       chunk's net map (the adjoint c_bar entering the chunk's highest token).
#   Pass C (grid (NT,B*H)):  seed s=carry_bwd[nt], reverse-walk the chunk (t=t1-1
#       -> t0): use s as c_bar_t to emit b_bar_t,da_t,dq_t,dr_t, then s<-A_t*s+B_t.
#       At chunk 0, after t=0, s=c_bar_{-1}; atomic-add into dmu[h].
#
# State handling: OPTION A (save-c). The forward v1 optionally emits cprev[b,t,h]=
# c_{t-1} (Pass C already holds it in a register); the autograd.Function saves it
# and both backward passes READ it (no forward recompute inside the backward). All
# scans fp32; bf16 I/O for dbeta in / da,dq,dr out. Grid B*H*NT (vs serial B*H) --
# only the tiny Pass-B carry is serial. Kept GATED behind _iso_beta_bwd_serial as
# the oracle/fallback (ISO_BETA_BWD env / kwarg).
# =============================================================================
@triton.jit(do_not_specialize=["T", "info_scale"])
def _iso_beta_bwd_passA_kernel(
    a_ptr,          # [B, T, H]
    q_ptr,          # [B, T, H]
    r_ptr,          # [B, T, H]
    dbeta_ptr,      # [B, T, H]  cotangent dL/dbeta
    cprev_ptr,      # [B, T, H]  fp32: c_{t-1} entering each step (saved by forward)
    ab_ptr,         # [B, NT, H, 2]  OUTPUT chunk net affine map [A_chunk, B_chunk]
    T,
    info_scale,     # info-increment scale s in gain denom r+s*b_hat (runtime fp32; default 1)
    NT: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    i_nt = tl.program_id(0)         # chunk index in [0, NT)
    pid = tl.program_id(1)          # (b, h) in [0, B*H)
    i_b = pid // H
    i_h = pid % H

    base = i_b * T * H + i_h        # base of this (b,h) stream; per-token stride H
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)

    # Compose the chunk's affine maps IN REVERSE token order. Net map applied to
    # "s entering the chunk from the HIGH end" -> "s exiting at the LOW end".
    # Start = identity affine (A=1, B=0).
    A_net = 1.0
    B_net = 0.0
    for t in range(t1 - 1, t0 - 1, -1):
        off = base + t * H
        a = tl.load(a_ptr + off).to(tl.float32)
        r = tl.load(r_ptr + off).to(tl.float32)
        db = tl.load(dbeta_ptr + off).to(tl.float32)
        cprev = tl.load(cprev_ptr + off).to(tl.float32)     # c_{t-1}
        q = tl.load(q_ptr + off).to(tl.float32)
        b_hat = a / cprev + q
        denom = r + info_scale * b_hat                       # gain denom (s scales b_hat*||k||^2)
        inv_cprev2 = 1.0 / (cprev * cprev)
        # A_t = a/(c_{t-1}^2 b_hat^2) ; B_t = -(a/c_{t-1}^2)*dbeta*r/denom^2
        A_t = a * inv_cprev2 / (b_hat * b_hat)
        B_t = -(a * inv_cprev2) * db * r / (denom * denom)
        # left-compose: (A_net,B_net) <- (A_t,B_t) o (A_net,B_net) = (A_t*A_net, A_t*B_net + B_t)
        A_net = A_t * A_net
        B_net = A_t * B_net + B_t

    ab_base = (i_b * NT * H + i_nt * H + i_h) * 2
    tl.store(ab_ptr + ab_base + 0, A_net)
    tl.store(ab_ptr + ab_base + 1, B_net)


@triton.jit
def _iso_beta_bwd_passB_kernel(
    ab_ptr,         # [B, NT, H, 2]  chunk net affine maps [A_chunk, B_chunk]
    carry_ptr,      # [B, NT, H]  OUTPUT adjoint s entering each chunk's HIGH end
    NT: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(0)          # (b, h) in [0, B*H)
    i_b = pid // H
    i_h = pid % H

    s = 0.0                         # c_bar entering the TOP chunk = c_bar_{T-1} = 0
    for nt in range(NT - 1, -1, -1):
        # store the adjoint ENTERING this chunk (before its net map is applied)
        carry_off = i_b * NT * H + nt * H + i_h
        tl.store(carry_ptr + carry_off, s)
        ab_base = (i_b * NT * H + nt * H + i_h) * 2
        A_chunk = tl.load(ab_ptr + ab_base + 0)
        B_chunk = tl.load(ab_ptr + ab_base + 1)
        s = A_chunk * s + B_chunk   # adjoint entering the next-LOWER chunk


@triton.jit(do_not_specialize=["T", "info_scale"])
def _iso_beta_bwd_passC_kernel(
    a_ptr,          # [B, T, H]
    q_ptr,          # [B, T, H]
    r_ptr,          # [B, T, H]
    dbeta_ptr,      # [B, T, H]  cotangent dL/dbeta
    cprev_ptr,      # [B, T, H]  fp32: c_{t-1} entering each step
    carry_ptr,      # [B, NT, H]  adjoint s entering each chunk's HIGH end
    da_ptr,         # [B, T, H]  OUTPUT dL/da
    dq_ptr,         # [B, T, H]  OUTPUT dL/dq
    dr_ptr,         # [B, T, H]  OUTPUT dL/dr
    dmu_ptr,        # [H]        OUTPUT dL/dmu (fp32, atomic over b) -- chunk 0 only
    T,
    info_scale,     # info-increment scale s in gain denom r+s*b_hat, c->r term (runtime fp32; default 1)
    NT: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H

    base = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)

    carry_off = i_b * NT * H + i_nt * H + i_h
    s = tl.load(carry_ptr + carry_off).to(tl.float32)   # c_bar entering the HIGH token

    # reverse-walk the chunk: s currently holds c_bar_t at the top of each iter.
    for t in range(t1 - 1, t0 - 1, -1):
        off = base + t * H
        a = tl.load(a_ptr + off).to(tl.float32)
        q = tl.load(q_ptr + off).to(tl.float32)
        r = tl.load(r_ptr + off).to(tl.float32)
        db = tl.load(dbeta_ptr + off).to(tl.float32)
        cprev = tl.load(cprev_ptr + off).to(tl.float32)   # c_{t-1} (= mu at t=0)

        b_hat = a / cprev + q
        denom = r + info_scale * b_hat                     # gain denom (s scales b_hat*||k||^2)
        inv_denom2 = 1.0 / (denom * denom)

        # per-token grads use c_bar_t = s (the value entering step t, from future)
        b_bar = db * (r * inv_denom2) + s * (-1.0 / (b_hat * b_hat))
        dr = db * (-b_hat * inv_denom2) + s * (-info_scale / (r * r))
        da = b_bar / cprev
        dq = b_bar

        tl.store(da_ptr + off, da.to(da_ptr.dtype.element_ty))
        tl.store(dq_ptr + off, dq.to(dq_ptr.dtype.element_ty))
        tl.store(dr_ptr + off, dr.to(dr_ptr.dtype.element_ty))

        # push adjoint to previous state: s <- A_t*s + B_t = c_bar_{t-1}
        inv_cprev2 = 1.0 / (cprev * cprev)
        A_t = a * inv_cprev2 / (b_hat * b_hat)
        B_t = -(a * inv_cprev2) * db * r * inv_denom2
        s = A_t * s + B_t

    # chunk 0 finishes at t=0, so s is now c_bar_{-1} = adjoint of mu for this (b,h).
    # mu is shared over the batch per head => atomic-add sum_b into dmu[h].
    if i_nt == 0:
        tl.atomic_add(dmu_ptr + i_h, s)


@torch.compiler.disable
def _iso_beta_bwd_v1(
    a: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    mu: torch.Tensor,
    dbeta: torch.Tensor,
    cprev: torch.Tensor,
    *,
    grad_dtype: torch.dtype | None = None,
    BT: int = _ISO_BT_DEFAULT,
    info_scale: float | None = None,
):
    """Parallel affine two-pass reverse VJP (grid ``B*H*NT``). Faster path for B2.

    Same result/contract as :func:`_iso_beta_bwd_serial` but parallel across chunks
    (only the short Pass-B carry is serial). Requires the forward's saved fp32
    ``cprev`` ``[B,T,H]`` (``= c_{t-1}``, option A save-c) -- no forward recompute
    inside the backward. Returns ``(da, dq, dr, dmu)``; ``dmu`` fp32 accumulator in
    ``mu``'s dtype. All scans fp32; no renorm (derivation §4). ``info_scale`` is the
    info-increment scale ``s`` (default ``None`` -> ``1.0``); MUST match the ``s`` the
    forward used to produce ``cprev``.
    """
    B, T, H = a.shape
    if grad_dtype is None:
        grad_dtype = a.dtype
    info_scale = 1.0 if info_scale is None else float(info_scale)   # s in gain denom (default 1)
    da = torch.empty((B, T, H), dtype=grad_dtype, device=a.device)
    dq = torch.empty((B, T, H), dtype=grad_dtype, device=a.device)
    dr = torch.empty((B, T, H), dtype=grad_dtype, device=a.device)
    dmu = torch.zeros((H,), dtype=torch.float32, device=a.device)
    if T == 0:
        return da, dq, dr, dmu.to(mu.dtype)

    NT = (T + BT - 1) // BT
    # fp32 scratch: per-chunk affine maps [B,NT,H,2] and per-chunk high-end adjoints.
    ab = torch.empty((B, NT, H, 2), dtype=torch.float32, device=a.device)
    carry = torch.empty((B, NT, H), dtype=torch.float32, device=a.device)

    gridAC = (NT, B * H)
    _iso_beta_bwd_passA_kernel[gridAC](a, q, r, dbeta, cprev, ab, T, info_scale, NT=NT, H=H, BT=BT)
    _iso_beta_bwd_passB_kernel[(B * H,)](ab, carry, NT=NT, H=H)
    _iso_beta_bwd_passC_kernel[gridAC](
        a, q, r, dbeta, cprev, carry, da, dq, dr, dmu, T, info_scale, NT=NT, H=H, BT=BT
    )
    return da, dq, dr, dmu.to(mu.dtype)


# Backward-version selector: "v1" (parallel affine two-pass, default) or "serial"
# (the B1 anchor). Env override for A/B + oracle-compare: ISO_BETA_BWD in
# {v1, serial}. Kwarg ``bwd_kernel`` on iso_beta_chunk overrides the env.
_ISO_BETA_BWD = _os.environ.get("ISO_BETA_BWD", "v1").lower()


# =============================================================================
# autograd.Function -- wraps forward (v1/v0 dispatch) + backward (parallel v1 /
# serial) so iso_beta_chunk is differentiable end-to-end w.r.t. (a, q, r, mu).
#
# The backward version is chosen at FORWARD time (needs to know whether to save
# cprev): "v1" (parallel affine two-pass, DEFAULT) has the forward emit + stash
# the fp32 exclusive-prefix ``cprev`` = c_{t-1} (option A save-c); "serial" (the
# B1 anchor/oracle) stashes nothing and RECOMPUTES c in the backward (option B).
# dbeta arrives in the output dtype; da/dq/dr come back in each input's dtype. The
# final state c_{T-1} is not an output, so there is no second cotangent to seed.
# =============================================================================
class _IsoBetaChunkFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, q, r, mu, out_dtype, kernel, bwd_kernel, info_scale):
        ver = (kernel or _ISO_BETA_KERNEL).lower()
        bwd = (bwd_kernel or _ISO_BETA_BWD).lower()
        save_c = (bwd != "serial")   # parallel v1 backward needs the saved cprev
        cprev = None
        if ver == "v0":
            beta = _iso_beta_chunk_v0(a, q, r, mu, out_dtype=out_dtype, info_scale=info_scale)
            # v0 has no cprev output; fall back to serial (recompute) backward.
            save_c = False
            bwd = "serial"
        elif save_c:
            beta, cprev = _iso_beta_chunk_v1(a, q, r, mu, out_dtype=out_dtype,
                                             return_cprev=True, info_scale=info_scale)
        else:
            beta = _iso_beta_chunk_v1(a, q, r, mu, out_dtype=out_dtype, info_scale=info_scale)
        ctx.bwd = bwd
        ctx.info_scale = info_scale
        if cprev is not None:
            ctx.save_for_backward(a, q, r, mu, cprev)
        else:
            ctx.save_for_backward(a, q, r, mu)
        return beta

    @staticmethod
    def backward(ctx, dbeta):
        saved = ctx.saved_tensors
        a, q, r, mu = saved[0], saved[1], saved[2], saved[3]
        cprev = saved[4] if len(saved) > 4 else None
        needs = ctx.needs_input_grad  # (a, q, r, mu, out_dtype, kernel, bwd_kernel, info_scale)
        # dbeta comes in the beta out_dtype; the kernel upcasts to fp32 internally.
        if ctx.bwd == "serial" or cprev is None:
            da, dq, dr, dmu = _iso_beta_bwd_serial(
                a.contiguous(), q.contiguous(), r.contiguous(),
                mu.contiguous(), dbeta.contiguous(),
                info_scale=ctx.info_scale,
            )
        else:
            da, dq, dr, dmu = _iso_beta_bwd_v1(
                a.contiguous(), q.contiguous(), r.contiguous(),
                mu.contiguous(), dbeta.contiguous(), cprev.contiguous(),
                info_scale=ctx.info_scale,
            )
        # Respect needs_input_grad (return None where no grad is required). Extra
        # None x4 for the non-tensor args (out_dtype, kernel, bwd_kernel, info_scale).
        return (
            da if needs[0] else None,
            dq if needs[1] else None,
            dr if needs[2] else None,
            dmu if needs[3] else None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def iso_beta_chunk(
    a: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    mu: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    kernel: str | None = None,
    bwd_kernel: str | None = None,
    info_scale: float | None = None,
) -> torch.Tensor:
    """Triton ISO-KLA write-gate scan: ``beta_t`` from the additive-Kalman scan.

    Computes, per (b, h) stream and token t (with ``c_{-1} = mu[h]``):

        b_hat  = a_t / c_{t-1} + q_t
        beta_t = b_hat / (r_t + s * b_hat)      # requires ||k||^2 = 1 (see precondition)
        c_t    = 1 / b_hat + s / r_t

    where ``s = info_scale`` (default 1.0). Bit-exact (fp32) to
    :func:`lit_gpt.kla_ops.iso_naive._iso_beta_seq` when ``k`` is L2-normalized.
    Dispatches to **v1** (chunked two-pass Mobius, parallel B*H*NT) by default; **v0**
    (serial-in-T anchor) is kept as an oracle/fallback and is selectable via
    ``kernel="v0"`` or the ``ISO_BETA_KERNEL`` env var.

    **Differentiable** w.r.t. ``(a, q, r, mu)`` via :class:`_IsoBetaChunkFn`
    (autograd.Function). The BACKWARD dispatches to **v1** (parallel affine two-pass
    reverse VJP, :func:`_iso_beta_bwd_v1`, DEFAULT) which uses the forward's saved
    fp32 ``cprev`` = c_{t-1} (option A save-c); or **serial** (:func:`_iso_beta_bwd_serial`,
    the B1 anchor/oracle) which recomputes ``c`` (option B). Select via
    ``bwd_kernel="serial"``/``"v1"`` or the ``ISO_BETA_BWD`` env var. The final
    state ``c_{T-1}`` is not an output, so it carries no cotangent.

    Args:
        a: ``[B, T, H]`` -- ``a_t = mean_i(alpha_{t,i}^2)`` (precomputed by caller).
        q: ``[B, T, H]`` -- additive process noise ``q_t``.
        r: ``[B, T, H]`` -- observation noise ``r_t`` (``>= r_min``).
        mu: ``[H]`` -- info prior ``c_{-1}`` (per head).
        out_dtype: dtype of the returned ``beta``. Default fp32 (correctness
            anchor). Pass ``torch.bfloat16`` for the real path (matches how KDA
            emits beta into ``chunk_kda``). Compute is ALWAYS fp32 regardless.
        kernel: ``"v0"`` / ``"v1"`` forward override (else the ``ISO_BETA_KERNEL`` env, def v1).
        bwd_kernel: ``"serial"`` / ``"v1"`` backward override (else ``ISO_BETA_BWD`` env, def v1).
        info_scale: info-increment scale ``s`` (effective standardized ``||k~||^2``)
            in ``beta = b_hat/(r + s*b_hat)`` and ``c = 1/b_hat + s/r``. Default
            ``None`` -> ``1.0``, bit-identical to the pre-ablation IsoKLA (``||k||^2=1``,
            no d_k standardization). The IsoKLA info-scale ablation passes ``s`` in
            ``{1, sqrt(d_k), d_k}``. Always forwarded to the kernels as a runtime fp32
            scalar (kernels ``do_not_specialize`` it), so one compiled variant serves
            every scale.

    Returns:
        ``beta`` : ``[B, T, H]`` in ``out_dtype`` (default fp32).
    """
    assert a.shape == q.shape == r.shape, (
        f"a/q/r must share shape [B,T,H]; got {tuple(a.shape)}, "
        f"{tuple(q.shape)}, {tuple(r.shape)}"
    )
    assert a.dim() == 3, f"a/q/r must be 3-D [B,T,H]; got {a.dim()}-D"
    B, T, H = a.shape
    assert mu.dim() == 1 and mu.shape[0] == H, (
        f"mu must be [H]={[H]}; got {tuple(mu.shape)}"
    )
    assert a.is_cuda and q.is_cuda and r.is_cuda and mu.is_cuda, (
        "iso_beta_chunk requires CUDA tensors (Triton is GPU-only)."
    )
    # info-increment scale s in beta = b_hat/(r + s*b_hat), c = 1/b_hat + s/r.
    # Default s = 1.0 (bit-identical to the pre-ablation IsoKLA); the info-scale
    # ablation sweeps s in {1, sqrt(d_k), d_k}.
    info_scale = 1.0 if info_scale is None else float(info_scale)

    # Contiguity is required for the flat [B,T,H] offset arithmetic in the kernels.
    a = a.contiguous()
    q = q.contiguous()
    r = r.contiguous()
    mu = mu.contiguous()

    ver = (kernel or _ISO_BETA_KERNEL).lower()
    bwd = (bwd_kernel or _ISO_BETA_BWD).lower()
    # Route through the autograd.Function so beta is differentiable w.r.t.
    # (a, q, r, mu). Forward dispatches v0/v1 by ``ver``; backward is the parallel
    # v1 (default) or serial reverse VJP by ``bwd``. ``out_dtype`` defaults to fp32.
    return _IsoBetaChunkFn.apply(a, q, r, mu, out_dtype, ver, bwd, info_scale)

