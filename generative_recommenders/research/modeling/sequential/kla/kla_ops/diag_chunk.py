# Triton kernel(s) for the diagonal-KLA per-channel Kalman gain kappa (and beta_ch).
#
# Diagonal KLA (paper Prop 3.1) is the delta-rule family with an ADAPTIVE,
# ANISOTROPIC diagonal-information gain kappa_t in place of the fixed write
# direction beta_t * k_t used by DeltaNet/GDN/KDA. kappa depends ONLY on the
# per-channel information scan (not on the memory state S_t), so it is precomputed
# once and fed to the GDN-2 memory kernel. The pure-PyTorch references are
# ``lit_gpt/kla_ops/naive.py::_kla_kappa_seq`` (sequential) and ``_kla_kappa``
# (Hillis-Steele Mobius scan); they agree to ~1e-16 (fp64) and are the CORRECTNESS
# ORACLES for this kernel.
#
# This is the PER-CHANNEL analog of the ISO scalar-beta kernel
# (``lit_gpt/kla_ops/iso_chunk.py``): K INDEPENDENT Mobius scans (one per key
# channel) PLUS a per-token cross-channel denominator reduction. Unlike ISO there
# is NO ``||k||^2 == 1`` simplification -- the diagonal gain uses per-channel
# ``k_i^2`` throughout, so the kernel takes ``k`` directly.
#
# ------------------------------------------------------------------------------
# Kernel contract
# ------------------------------------------------------------------------------
#   kla_kappa_chunk(k, alpha, omega, r, mu, *, out_dtype=None) -> (kappa, beta_ch)
#   with  k, alpha, omega : [B, T, H, K]
#         r  : [B, T, H] | [H] | scalar   (broadcast to [B,T,H] by the wrapper)
#         mu : [H] | scalar               (broadcast to [H] by the wrapper)
#         kappa, beta_ch : [B, T, H, K]
#
# ------------------------------------------------------------------------------
# Exact math the kernel computes (this IS _kla_kappa_seq, per channel i)
# ------------------------------------------------------------------------------
#   c_{i,-1} = mu[h]                                  # per-channel info prior (same mu per head)
#   for t = 0..T-1:
#       p_hat_i = alpha_{i,t}^2 / c_{i,t-1} + omega_{i,t}     # per-channel predicted uncertainty (EXCLUSIVE prefix)
#       denom   = r_t + sum_i p_hat_i k_{i,t}^2           # per-TOKEN reduction over K (the ONLY cross-channel coupling)
#       kappa_i     = p_hat_i k_{i,t} / denom
#       beta_ch_i   = p_hat_i / denom                     # kappa = beta_ch * k
#       c_{i,t} = 1/p_hat_i + d_k k_{i,t}^2 / r_t         # per-channel posterior info (d_k = K)
#
# ------------------------------------------------------------------------------
# Why [n, d] with renorm, NOT a raw scalar c (CRITICAL -- design C1)
# ------------------------------------------------------------------------------
# The scalar posterior info c_t = 1/p_hat + k^2/r is bounded only because omega>0 caps
# it near 1/omega. Per channel there are K x more chances for a tiny-omega channel (very
# negative qn_proj bias => omega ~ 0+), where c grows geometrically across the predict
# step and OVERFLOWS fp32 to inf around t~124. So the kernel does NOT track a raw
# scalar c. It tracks the per-channel Mobius state ``[n, d]`` (``c = n/d``) with a
# per-step MAX-ENTRY RENORM: the information recursion
#   c_t = ((1 + u omega) c_{t-1} + u alpha^2) / (omega c_{t-1} + alpha^2),   u = d_k k^2/r  (d_k=K)
# is a Mobius (linear-fractional) map, applied by 2x2 matmul on ``[n, d]``:
#   M_t = [[1 + u omega, u alpha^2], [omega, alpha^2]]
#   n' = (1 + u omega) n + (u alpha^2) d ;  d' = omega n + alpha^2 d ;  then renorm by max(|n'|,|d'|).
# Mobius maps are scale-invariant (c = n/d unchanged by the renorm), and the renorm
# keeps ``n, d`` bounded by construction -- so even when ``c = n/d`` is legitimately
# huge (tiny omega, d -> 0), ``n`` and ``d`` THEMSELVES stay finite. This is the
# oracle's own form (``_kla_kappa``) and is overflow-proof.
#
# No ``tl.dot`` (scalar per-channel mults + a K-reduction only) => no tf32/ieee
# matmul knob. fp32 compute, bf16 I/O; ``out_dtype`` flag (fp32 anchor / bf16 real
# path). The kernel is a chunked two-pass per-channel Mobius scan (reduce-then-scan):
# parallelism is B*H*NT (chunks) with the K channels vectorized inside each program.
from __future__ import annotations

import torch
import triton
import triton.language as tl


# =============================================================================
# Chunked two-pass per-channel Mobius (reduce-then-scan). Parallelism B*H*NT
# (with the K channels vectorized inside each program). This is the PER-CHANNEL
# analog of the ISO scalar-beta kernel (iso_chunk.py::_iso_beta_pass{A,B,C}); the
# two structural differences from ISO are (1) K INDEPENDENT Mobius scans per (b,h)
# instead of one scalar scan, held as [BK] register vectors, and (2) the per-token
# cross-channel ``denom = r + sum_i p_hat_i k_i^2`` K-reduction that Pass C does
# in-program.
# CRITICAL (round-2-verified crux): the carry AND the in-chunk scan track the
# per-channel Mobius state ``[n, d]`` (c = n/d) with per-step max-entry renorm --
# NOT a scalar ``c``. A scalar carry/scan goes inf/NaN the moment a chunk overflows
# for a tiny-omega channel (c grows geometrically across the predict step and blows fp32
# to inf around t~124; over BT=256 steps a scalar in-chunk scan overflows too). The
# [n,d] form is scale-invariant (c=n/d unchanged by renorm) and overflow-proof by
# construction -- n, d themselves stay finite even when c=n/d is legitimately huge
# (tiny omega, d -> 0). fp32 scan, bf16 I/O; no ``tl.dot`` (scalar per-channel mults +
# a K-reduction), so no tf32/ieee matmul knob.
#
#   Pass A (grid (NT, B*H)): each chunk composes its NET per-channel map M_chunk =
#       M_{end-1}...M_{start} (left-product), fp32 + running max-entry renorm, K
#       channels vectorized. Stores 4 fp32/channel [A,B,C,D] -> [B,NT,H,K,4]
#       (per-chunk only, NT << T, so this scratch is small).
#   Pass B (grid (B*H,)): serial over NT (tiny -- 16k/256 = 64), thread the
#       per-channel [n,d] carry (init [mu,1]) through each chunk's net map WITH
#       renorm; store the EXCLUSIVE-prefix [n,d] entering each chunk -> [B,NT,H,K,2].
#   Pass C (grid (NT, B*H)): seed [n,d] from the carry, run the [n,d] renorm scan
#       over the chunk's BT tokens; per token c_excl=n/d, p_hat=a2*d/n+omega,
#       denom=r+sum_K(p_hat k^2) (in-program K-reduction), kappa=p_hat*k/denom,
#       beta_ch=p_hat/denom; then advance [n,d] <- M_t.[n,d] + renorm. Re-reads
#       k,alpha,omega. Optionally stores the exclusive-prefix [n,d] (SAVE_STATE, for G2).
#
# Verified bit-close (fp32, <=1e-6 vs fp64) to the oracle for BT in
# {1,...} incl. non-divisor BT and tails.
# =============================================================================
@triton.jit(do_not_specialize=["T", "info_scale"])
def _kla_kappa_passA_kernel(
    k_ptr,          # [B, T, H, K]  key
    a_ptr,          # [B, T, H, K]  alpha
    omega_ptr,          # [B, T, H, K]  omega_t
    r_ptr,          # [B, T, H]     r_t (broadcast to per-token by wrapper)
    cm_ptr,         # [B, NT, H, K, 4]  OUTPUT per-chunk net map [A,B,C,D]
    T,              # sequence length (runtime)
    info_scale,     # info-increment scale s in u = s*k^2/r (runtime fp32; default d_k=K)
    NT: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    i_nt = tl.program_id(0)         # chunk index in [0, NT)
    pid = tl.program_id(1)          # (b, h) in [0, B*H)
    i_b = pid // H
    i_h = pid % H

    i_k = tl.arange(0, BK)
    mask = i_k < K

    base = i_b * T * H * K + i_h * K        # base of this (b,h) stream; per-token stride H*K
    base_r = i_b * T * H + i_h              # base of the [B,T,H] r tensor; per-token stride H
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)

    # net per-channel map M_chunk, start = identity [[1,0],[0,1]] on every channel.
    mA = tl.full((BK,), 1.0, tl.float32)
    mB = tl.full((BK,), 0.0, tl.float32)
    mC = tl.full((BK,), 0.0, tl.float32)
    mD = tl.full((BK,), 1.0, tl.float32)
    for t in range(t0, t1):
        off = base + t * H * K + i_k
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)          # scalar
        a2 = a * a
        u = (info_scale * k * k) / r                     # u = s*k^2/r  (s default d_k = K)
        # M_t = [[1+u omega, u a2],[omega, a2]]  (per channel)
        tA = 1.0 + u * omega
        tB = u * a2
        tC = omega
        tD = a2
        # left-multiply: M = M_t @ M
        nA = tA * mA + tB * mC
        nB = tA * mB + tB * mD
        nC = tC * mA + tD * mC
        nD = tC * mB + tD * mD
        # per-channel max-entry renorm (Mobius scale-invariant): keep entries bounded.
        m = tl.maximum(tl.maximum(tl.abs(nA), tl.abs(nB)),
                       tl.maximum(tl.abs(nC), tl.abs(nD)))
        inv = 1.0 / tl.maximum(m, 1e-30)
        mA = nA * inv
        mB = nB * inv
        mC = nC * inv
        mD = nD * inv

    # store [B, NT, H, K, 4] contiguous: element (b,nt,h,i,j) at
    # ((((i_b*NT + i_nt)*H + i_h)*K + i)*4 + j).
    cm_base = (((i_b * NT + i_nt) * H + i_h) * K) * 4
    tl.store(cm_ptr + cm_base + i_k * 4 + 0, mA, mask=mask)
    tl.store(cm_ptr + cm_base + i_k * 4 + 1, mB, mask=mask)
    tl.store(cm_ptr + cm_base + i_k * 4 + 2, mC, mask=mask)
    tl.store(cm_ptr + cm_base + i_k * 4 + 3, mD, mask=mask)


@triton.jit
def _kla_kappa_passB_kernel(
    cm_ptr,         # [B, NT, H, K, 4]  per-chunk net maps [A,B,C,D]
    mu_ptr,         # [H]
    carry_ptr,      # [B, NT, H, K, 2]  OUTPUT exclusive-prefix [n,d] entering each chunk
    NT: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)          # (b, h) in [0, B*H)
    i_b = pid // H
    i_h = pid % H

    i_k = tl.arange(0, BK)
    mask = i_k < K

    # per-channel [n,d] carry, init [mu,1]  =>  c_0 = n/d = mu[h] on every channel.
    mu = tl.load(mu_ptr + i_h).to(tl.float32)
    cn = tl.full((BK,), 0.0, tl.float32) + mu
    cd = tl.full((BK,), 1.0, tl.float32)
    for nt in range(0, NT):
        # store the EXCLUSIVE prefix ([n,d] before applying this chunk's map).
        carry_base = (((i_b * NT + nt) * H + i_h) * K) * 2
        tl.store(carry_ptr + carry_base + i_k * 2 + 0, cn, mask=mask)
        tl.store(carry_ptr + carry_base + i_k * 2 + 1, cd, mask=mask)

        cm_base = (((i_b * NT + nt) * H + i_h) * K) * 4
        mA = tl.load(cm_ptr + cm_base + i_k * 4 + 0, mask=mask, other=0.0)
        mB = tl.load(cm_ptr + cm_base + i_k * 4 + 1, mask=mask, other=0.0)
        mC = tl.load(cm_ptr + cm_base + i_k * 4 + 2, mask=mask, other=0.0)
        mD = tl.load(cm_ptr + cm_base + i_k * 4 + 3, mask=mask, other=0.0)
        # advance [n,d] <- M_chunk . [n,d]  (per channel), then renorm. NOT a scalar
        # c=n/d: threading a scalar overflows once a chunk map is ill-conditioned at
        # tiny omega; [n,d] with renorm stays finite (round-2-verified crux).
        nn = mA * cn + mB * cd
        nd = mC * cn + mD * cd
        m = tl.maximum(tl.abs(nn), tl.abs(nd))
        inv = 1.0 / tl.maximum(m, 1e-30)
        cn = nn * inv
        cd = nd * inv


@triton.jit(do_not_specialize=["T", "info_scale"])
def _kla_kappa_passC_kernel(
    k_ptr,          # [B, T, H, K]
    a_ptr,          # [B, T, H, K]
    omega_ptr,          # [B, T, H, K]
    r_ptr,          # [B, T, H]
    carry_ptr,      # [B, NT, H, K, 2]  exclusive-prefix [n,d] entering each chunk
    kappa_ptr,      # [B, T, H, K]  OUTPUT
    beta_ptr,       # [B, T, H, K]  OUTPUT
    n_ptr,          # [B, T, H, K]  OPTIONAL fp32 OUTPUT (SAVE_STATE): [n] entering step t
    d_ptr,          # [B, T, H, K]  OPTIONAL fp32 OUTPUT (SAVE_STATE): [d] entering step t
    T,
    info_scale,     # info-increment scale s in u = s*k^2/r (runtime fp32; default d_k=K)
    NT: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    SAVE_STATE: tl.constexpr,
):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H

    i_k = tl.arange(0, BK)
    mask = i_k < K

    base = i_b * T * H * K + i_h * K
    base_r = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)

    # seed [n,d] from the Pass-B exclusive-prefix carry for this chunk.
    carry_base = (((i_b * NT + i_nt) * H + i_h) * K) * 2
    n = tl.load(carry_ptr + carry_base + i_k * 2 + 0, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(carry_ptr + carry_base + i_k * 2 + 1, mask=mask, other=1.0).to(tl.float32)

    for t in range(t0, t1):
        off = base + t * H * K + i_k
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)

        if SAVE_STATE:
            # store the EXCLUSIVE-prefix [n,d] (before advancing) -- G2 asserts
            # isfinite(n) AND isfinite(d) SEPARATELY, never c=n/d (legit non-finite
            # as d -> 0 at tiny omega). Masked lanes store 0 (finite, ignored).
            tl.store(n_ptr + off, n, mask=mask)
            tl.store(d_ptr + off, d, mask=mask)

        a2 = a * a
        # c_excl = n/d (exclusive prefix). p_hat = alpha^2/c_excl + omega = alpha^2*d/n + omega.
        p_hat = a2 * d / n + omega
        ksq = k * k
        # denom = r + sum_i(p_hat_i k_i^2) -- in-program K-reduction (C3 invariant:
        # ONE program owns all K of a token, so the cross-channel sum is a tl.sum).
        denom = r + tl.sum(tl.where(mask, p_hat * ksq, 0.0), axis=0)
        kappa = p_hat * k / denom
        beta_ch = p_hat / denom
        tl.store(kappa_ptr + off, kappa.to(kappa_ptr.dtype.element_ty), mask=mask)
        tl.store(beta_ptr + off, beta_ch.to(beta_ptr.dtype.element_ty), mask=mask)

        # advance [n,d] <- M_t . [n,d], then max-entry renorm (overflow-proof).
        u = (info_scale * ksq) / r                       # u = s*k^2/r  (s default d_k = K)
        n_new = (1.0 + u * omega) * n + (u * a2) * d
        d_new = omega * n + a2 * d
        m = tl.maximum(tl.abs(n_new), tl.abs(d_new))
        inv = 1.0 / tl.maximum(m, 1e-30)
        n = n_new * inv
        d = d_new * inv


# Default chunk length (tokens per program). 256 keeps NT small enough for
# the short Pass-B serial carry (16k/256 = 64) while giving B*H*NT programs; mirrors
# the ISO beta default (_ISO_BT_DEFAULT).
_KLA_BT_DEFAULT = 256


@torch.compiler.disable
def _kla_kappa_chunk_v1(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r_bt: torch.Tensor,   # [B,T,H] fp32 (pre-broadcast by the public wrapper)
    mu_h: torch.Tensor,   # [H] fp32 (pre-broadcast by the public wrapper)
    *,
    out_dtype: torch.dtype,
    save_state: bool = False,
    info_scale: float | None = None,
    BT: int = _KLA_BT_DEFAULT,
):
    """Chunked two-pass per-channel Mobius scan (grid ``B*H*NT``, K vectorized).

    Same contract/math as :func:`kla_kappa_chunk`; parallelism ``B*H*NT`` (only the
    short Pass-B carry is serial). ``r_bt``/``mu_h`` are already broadcast to
    ``[B,T,H]``/``[H]`` fp32 by the public wrapper (single kernel path). ``BT`` is
    the tokens-per-chunk tile (exposed for the BT>T / tail tests).

    Returns ``(kappa, beta_ch)`` in ``out_dtype``; or ``(kappa, beta_ch, n, d)``
    (``n,d`` always fp32) when ``save_state`` (the exclusive-prefix Mobius state for
    the G2 numerical-safety gate).
    """
    B, T, H, K = k.shape
    info_scale = float(K) if info_scale is None else float(info_scale)  # s in u=s*k^2/r (default d_k=K)
    kappa = torch.empty((B, T, H, K), dtype=out_dtype, device=k.device)
    beta_ch = torch.empty((B, T, H, K), dtype=out_dtype, device=k.device)
    n_buf = (torch.empty((B, T, H, K), dtype=torch.float32, device=k.device)
             if save_state else None)
    d_buf = (torch.empty((B, T, H, K), dtype=torch.float32, device=k.device)
             if save_state else None)
    if T == 0:
        if save_state:
            return kappa, beta_ch, n_buf, d_buf
        return kappa, beta_ch

    NT = (T + BT - 1) // BT
    BK = triton.next_power_of_2(K)
    # fp32 scratch: per-chunk net maps [B,NT,H,K,4] and exclusive-prefix [n,d]
    # carries [B,NT,H,K,2] (per-chunk only -- NT << T -- so this is small).
    cm = torch.empty((B, NT, H, K, 4), dtype=torch.float32, device=k.device)
    carry = torch.empty((B, NT, H, K, 2), dtype=torch.float32, device=k.device)

    gridAC = (NT, B * H)
    _kla_kappa_passA_kernel[gridAC](
        k, alpha, omega, r_bt, cm, T, info_scale, NT=NT, H=H, K=K, BK=BK, BT=BT,
    )
    _kla_kappa_passB_kernel[(B * H,)](
        cm, mu_h, carry, NT=NT, H=H, K=K, BK=BK,
    )
    _kla_kappa_passC_kernel[gridAC](
        k, alpha, omega, r_bt, carry,
        kappa, beta_ch,
        n_buf if n_buf is not None else kappa,   # unused ptr when SAVE_STATE False
        d_buf if d_buf is not None else beta_ch,
        T, info_scale, NT=NT, H=H, K=K, BK=BK, BT=BT, SAVE_STATE=save_state,
    )
    if save_state:
        return kappa, beta_ch, n_buf, d_buf
    return kappa, beta_ch


# =============================================================================
# BACKWARD -- serial reverse VJP of the per-channel Kalman-gain scan (Task KB2:
# the SLOW correctness anchor). Mirrors the ISO serial backward
# (iso_chunk.py::_iso_beta_bwd_kernel), but per-channel: K channels are held as
# [BK] fp32 register vectors, and there are THREE per-token cross-channel
# K-reductions (D, G, and the dr scan-sum) done with masked tl.sum.
#
# Derivation: docs/plans/2026-07-25-kla-kappa-backward-derivation.md (§3 gain VJP
# with per-token D,G; §4 serial reverse affine scan A_t/B_t, two-path dk/dr, dmu).
# This kernel computes EXACTLY what lit_gpt/kla_ops/naive.py::_kla_kappa_bwd_ref
# does (BG2 gates it fp32 <= 1e-6). Given output cotangents dkappa, dbeta_ch
# ([B,T,H,K]; the final state c_{T-1} is NOT an output => no cotangent), it emits
# dk,dalpha,domega ([B,T,H,K]), dr ([B,T,H]) and dmu ([H], batch+channel sum).
#
# State handling: RECOMPUTE (no forward buffer saved across the fwd->bwd boundary).
# Because the reverse scan needs the forward EXCLUSIVE-prefix c_{i,t-1} (= mu at
# t=0) while walking t=T-1 -> 0, each program first FORWARD-scans the per-channel
# Mobius [n,d] (the forward's own form) storing per-token c_excl=n/d into a
# TRANSIENT GLOBAL fp32 scratch c_excl_scratch[B,T,H,K] (allocated in `backward`,
# freed after), then reverse-reads it. A length-T register/local array is NOT
# possible (T is a runtime arg) -- exactly the pattern ISO uses with its `cprev`
# scratch. The [n,d] recompute (NOT a raw scalar c) is overflow-proof for the
# tiny-omega regime, so this anchor stays finite at T=16384 (BG2 asserts it).
#
# fp32 compute; bf16 I/O for dkappa,dbeta_ch in and dk,dalpha,domega,dr out; dmu
# fp32 (atomic accumulator). No tl.dot (scalar per-channel mults + a K-reduction),
# so no tf32/ieee matmul knob. Grid (B*H,); serial in T inside the program.
# Deliberately slow -- Task KB3 parallelizes it via the affine two-pass.
# =============================================================================
@triton.jit(do_not_specialize=["T", "info_scale"])
def _kla_kappa_bwd_serial_kernel(
    k_ptr,          # [B, T, H, K]  key
    a_ptr,          # [B, T, H, K]  alpha
    omega_ptr,      # [B, T, H, K]  omega_t
    r_ptr,          # [B, T, H]     r_t (broadcast to per-token by wrapper)
    mu_ptr,         # [H]           info prior c_{i,-1}
    dkappa_ptr,     # [B, T, H, K]  INPUT cotangent dL/dkappa
    dbeta_ptr,      # [B, T, H, K]  INPUT cotangent dL/dbeta_ch
    c_excl_ptr,     # [B, T, H, K]  fp32 SCRATCH: c_{i,t-1} recomputed forward-order
    dk_ptr,         # [B, T, H, K]  OUTPUT dL/dk
    dalpha_ptr,     # [B, T, H, K]  OUTPUT dL/dalpha
    domega_ptr,     # [B, T, H, K]  OUTPUT dL/domega
    dr_ptr,         # [B, T, H]     OUTPUT dL/dr_t
    dmu_ptr,        # [H]           OUTPUT dL/dmu[h]  (fp32, atomic over b AND channels)
    T,              # sequence length (runtime -> real loop, not unrolled)
    info_scale,     # info-increment scale s in u = s*k^2/r (runtime fp32; default d_k=K)
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)          # (b, h) in [0, B*H)
    i_b = pid // H
    i_h = pid % H

    i_k = tl.arange(0, BK)
    mask = i_k < K

    base = i_b * T * H * K + i_h * K        # base of this (b,h) stream; per-token stride H*K
    base_r = i_b * T * H + i_h              # base of the [B,T,H] r tensor; per-token stride H
    Kf = info_scale                         # info-increment scale s (default d_k=K; float runtime)

    # ---- forward recompute (Mobius [n,d]): fill c_excl[t] = c_{i,t-1} (exclusive prefix).
    # [n,d] = [mu,1] => c_{i,-1} = mu (same per channel/head). Store c=n/d BEFORE each
    # advance, then M_t.[n,d] with max-entry renorm (overflow-proof; NOT a scalar c). ----
    mu = tl.load(mu_ptr + i_h).to(tl.float32)
    cn = tl.zeros((BK,), tl.float32) + mu
    cd = tl.full((BK,), 1.0, tl.float32)
    for t in range(0, T):
        off = base + t * H * K + i_k
        tl.store(c_excl_ptr + off, cn / cd, mask=mask)         # c_{i,t-1} = n/d
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)     # scalar
        a2 = a * a
        u = (Kf * k * k) / r                                   # u = d_k k^2/r
        n_new = (1.0 + u * omega) * cn + (u * a2) * cd
        d_new = omega * cn + a2 * cd
        m = tl.maximum(tl.abs(n_new), tl.abs(d_new))
        inv = 1.0 / tl.maximum(m, 1e-30)
        cn = n_new * inv
        cd = d_new * inv

    # ---- reverse VJP scan: t = T-1 -> 0, carry per-channel state-adjoint s = c_bar_{i,t}
    # (init c_bar_{i,T-1}=0, final state not an output). ----
    s = tl.zeros((BK,), tl.float32)
    for t in range(T - 1, -1, -1):
        off = base + t * H * K + i_k
        cprev = tl.load(c_excl_ptr + off, mask=mask, other=1.0).to(tl.float32)   # c_{i,t-1} (= mu at t=0)
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)     # scalar r_t
        dki = tl.load(dkappa_ptr + off, mask=mask, other=0.0).to(tl.float32)     # dkappa_i
        dbi = tl.load(dbeta_ptr + off, mask=mask, other=0.0).to(tl.float32)      # dbeta_ch_i

        a2 = a * a
        ksq = k * k
        p_hat = a2 / cprev + omega                             # predicted uncertainty [BK]
        # --- §3 per-token cross-channel reductions D and G (masked tl.sum over K) ---
        D = r + tl.sum(tl.where(mask, p_hat * ksq, 0.0), axis=0)     # r_t + sum_j p_hat_j k_j^2 (scalar)
        kap = p_hat * k / D                                    # kappa_i (recomputed)
        beta = p_hat / D                                       # beta_i (recomputed)
        G = tl.sum(tl.where(mask, dki * kap + dbi * beta, 0.0), axis=0)   # sum_i(dk_i kap_i + db_i beta_i)
        # --- §3 gain-path adjoints ---
        Pbar_gain = (dki * k + dbi) / D - ksq * G / D          # adjoint of p_hat_m (gain)  [BK]
        kbar_gain = dki * beta - 2.0 * p_hat * k * G / D       # gain-path grad of k_m
        rbar_gain = -G / D                                     # gain-path grad of r_t (scalar)
        # --- §4 total adjoint of p_hat (gain consumer + info-update consumer c_t = 1/p_hat + u) ---
        Pbar_tot = Pbar_gain + s * (-1.0 / (p_hat * p_hat))    # [BK]
        # --- local input grads at (m,t) ---
        domega = Pbar_tot
        da2 = Pbar_tot / cprev                                 # da2_m = Pbar_tot / cprev
        dalpha = 2.0 * a * da2                                 # dalpha = 2 alpha da2  (IMPL-CARRY #3)
        dk = kbar_gain + s * (2.0 * Kf * k / r)                # gain path + info-update(u) path
        # dr: gain path (per-token scalar) + info-update scan-sum (K-reduction)
        dr_scan = -(Kf / (r * r)) * tl.sum(tl.where(mask, s * ksq, 0.0), axis=0)   # -(K/r^2) sum_m s_m k_m^2
        dr = rbar_gain + dr_scan                               # scalar
        # --- emit grads (bf16/fp32 I/O per out ptr dtype) ---
        tl.store(dk_ptr + off, dk.to(dk_ptr.dtype.element_ty), mask=mask)
        tl.store(dalpha_ptr + off, dalpha.to(dalpha_ptr.dtype.element_ty), mask=mask)
        tl.store(domega_ptr + off, domega.to(domega_ptr.dtype.element_ty), mask=mask)
        tl.store(dr_ptr + base_r + t * H, dr.to(dr_ptr.dtype.element_ty))
        # --- push adjoint to the previous state c_{m,t-1} ---
        s = Pbar_tot * (-a2 / (cprev * cprev))                 # c_bar_{i,t-1}
        # IMPL-CARRY #1: padded lanes have cprev=0 => p_hat=inf => s NaN; zero them so
        # the next iteration's reductions/store stay finite (real configs have K==BK).
        s = tl.where(mask, s, 0.0)

    # after t=0, s = c_bar_{i,-1} (adjoint of mu via channel m). mu[h] seeds every channel
    # of every batch element => dmu[h] = sum_b sum_m c_bar_{m,-1}. Sum over channels here,
    # atomic-add over b into dmu[h] (fp32 accumulator).
    dmu = tl.sum(tl.where(mask, s, 0.0), axis=0)
    tl.atomic_add(dmu_ptr + i_h, dmu)


@torch.compiler.disable
def _kla_kappa_bwd_serial(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r_bt: torch.Tensor,   # [B,T,H] fp32 (pre-broadcast)
    mu_h: torch.Tensor,   # [H] fp32 (pre-broadcast)
    dkappa: torch.Tensor,
    dbeta_ch: torch.Tensor,
    *,
    grad_dtype: torch.dtype | None = None,
    info_scale: float | None = None,
):
    """Serial reverse VJP of the per-channel Kalman-gain scan (grid ``(B*H,)``).

    Correctness anchor (slow). Given the output cotangents ``dkappa``/``dbeta_ch``
    ``[B,T,H,K]`` returns ``(dk, dalpha, domega, dr, dmu)`` with ``dk/dalpha/domega
    : [B,T,H,K]``, ``dr : [B,T,H]`` and ``dmu : [H]``. Recomputes ``c_{i,t-1}``
    forward via the overflow-safe Mobius ``[n,d]`` scan into a transient fp32 scratch
    ``c_excl_scratch[B,T,H,K]`` (allocated here, freed on return), then walks the
    reverse affine scan. ``dmu`` is always an fp32 atomic accumulator (returned in
    ``mu``'s dtype); ``dk/dalpha/domega/dr`` are ``grad_dtype`` (default the dtype of
    ``k`` so grads match the autograd input dtype). Computes exactly
    :func:`lit_gpt.kla_ops.diag_naive._kla_kappa_bwd_ref`.
    """
    B, T, H, K = k.shape
    if grad_dtype is None:
        grad_dtype = k.dtype
    info_scale = float(K) if info_scale is None else float(info_scale)  # s in u=s*k^2/r (default d_k=K)
    dk = torch.empty((B, T, H, K), dtype=grad_dtype, device=k.device)
    dalpha = torch.empty((B, T, H, K), dtype=grad_dtype, device=k.device)
    domega = torch.empty((B, T, H, K), dtype=grad_dtype, device=k.device)
    dr = torch.empty((B, T, H), dtype=grad_dtype, device=k.device)
    # dmu must be fp32 (atomic-add accumulator) and zeroed before accumulation.
    dmu = torch.zeros((H,), dtype=torch.float32, device=k.device)
    if T == 0:
        return dk, dalpha, domega, dr, dmu.to(mu_h.dtype)

    BK = triton.next_power_of_2(K)
    # transient fp32 scratch for the recomputed exclusive-prefix c_{i,t-1} (recompute).
    c_excl = torch.empty((B, T, H, K), dtype=torch.float32, device=k.device)

    grid = (B * H,)
    _kla_kappa_bwd_serial_kernel[grid](
        k, alpha, omega, r_bt, mu_h, dkappa, dbeta_ch, c_excl,
        dk, dalpha, domega, dr, dmu,
        T, info_scale, H=H, K=K, BK=BK,
    )
    return dk, dalpha, domega, dr, dmu.to(mu_h.dtype)


# =============================================================================
# BACKWARD v1 -- PARALLEL affine two-pass reverse VJP (Task KB3: the SPEED task).
# Mirrors the FORWARD chunked two-pass (Pass A/B/C) but REVERSED. The forward is a
# per-channel MOBIUS scan; the reverse state-adjoint recurrence is AFFINE (derivation
# §5, R1+R2 verified renorm-free fp32), so it composes with 2 scalars/channel and
# needs NO renorm -- exactly the ISO affine backward (iso_chunk.py::_iso_beta_bwd_*),
# but ×K channels held as [BK] register vectors PLUS the three per-token cross-channel
# K-reductions (D, G, sum_m s_m k_m^2) done with masked tl.sum in Pass C'.
#
# Reverse recurrence (derivation §5), per channel m, with cprev = c_{m,t-1} (=mu at
# t=0), p_hat = a2/cprev + omega, and the gain adjoint Pbar_gain (needs the per-token
# D, G reductions):
#     s_{t-1} = A_t * s_t + B_t ,   s_t = c_bar_t  (adjoint ENTERING step t),
#     A_t =  a2 / (cprev^2 * p_hat^2),
#     B_t = -Pbar_gain * a2 / cprev^2.
# init c_bar_{T-1}=0 (final state not an output). Affine maps compose associatively
# (A2,B2) o (A1,B1) = (A2*A1, A2*B1 + B2), so the reverse scan uses the SAME
# reduce-then-scan structure as the forward.
#
# STATE HANDLING -- RECOMPUTE via a SHARED per-token fp32 scratch (a deliberate,
# strictly-better deviation from the impl-plan's "SRAM c_excl[BT,BK] tile rebuilt
# per pass"): a chunk-parallel FILL pass (grid (NT,B*H)) forward-scans each chunk
# from its Pass-B [n,d] entry carry and writes c_excl[b,t,h,K]=c_{t-1} to a TRANSIENT
# GLOBAL fp32 buffer (allocated in `backward`, freed after). Pass A' and Pass C' then
# READ that buffer -- the c_excl is materialized ONCE, so the impl-plan's "double
# rescan is unavoidable" (which held only under an SRAM-only constraint) is AVOIDED.
# Memory is O(T) [B,T,H,K] -- IDENTICAL to the serial anchor's own scratch (which
# already ships), so zero regression; and Triton dynamic-indexing of a [BT,BK]
# register tile (needed for the reverse-read of a forward scan) spills badly, so the
# global buffer is also the cleaner idiom. Grid stays (NT,B*H) for every per-token
# pass => full chunk parallelism (the actual KB3 goal), vs the serial anchor's B*H.
#
#   FILL  (grid (NT,B*H)): seed [n,d] from the Pass-B exclusive-prefix carry, run the
#         forward Mobius [n,d] renorm scan over the chunk, store c_excl=n/d per token.
#         (The overflow-safe [n,d] form -- NOT a raw scalar c -- keeps c_excl finite
#         at tiny omega / T=16384, exactly as the forward + serial anchor do.)
#   Pass A' (grid (NT,B*H)): per chunk, read c_excl + (dkappa,dbeta_ch), recompute
#         p_hat/D/G/Pbar_gain/A_t/B_t, and compose the chunk NET affine map
#         (A_chunk,B_chunk) per channel IN REVERSE token order (fp32). Store 2
#         fp32/channel -> [B,NT,H,K,2].
#   Pass B' (grid (B*H,)): serial over NT IN REVERSE (nt=NT-1->0), thread the scalar
#         per-channel carry s (init 0 at the top). Store the adjoint ENTERING each
#         chunk's HIGH token (carry_bwd[nt] = s BEFORE the chunk net map).
#   Pass C' (grid (NT,B*H)): seed s from carry_bwd[nt], reverse-walk the chunk
#         (t=t1-1 -> t0) using s = c_bar_t to emit domega,dalpha,dk,dr (with the three
#         masked K-reductions), then push s <- A_t*s + B_t. Mask s=0 on padded lanes.
#         At chunk 0, after t=0, s = c_bar_{-1}; atomic-add sum_m s_m into dmu[h].
#
# All scans/adjoints/reductions fp32; bf16 I/O for dkappa,dbeta_ch in / dk,dalpha,
# domega,dr out (dmu fp32 atomic). No tl.dot (scalar per-channel mults + K-reductions)
# => no tf32/ieee matmul knob. Kept GATED behind the serial anchor as the oracle/
# fallback (KLA_KAPPA_BWD env / kwarg). BG3 gates parallel==serial==ref (fp32<=1e-6).
# =============================================================================
@triton.jit(do_not_specialize=["T", "info_scale"])
def _kla_kappa_bwd_fill_kernel(
    k_ptr,          # [B, T, H, K]  key
    a_ptr,          # [B, T, H, K]  alpha
    omega_ptr,      # [B, T, H, K]  omega_t
    r_ptr,          # [B, T, H]     r_t (broadcast to per-token by wrapper)
    carry_ptr,      # [B, NT, H, K, 2]  exclusive-prefix [n,d] entering each chunk (forward Pass B)
    c_excl_ptr,     # [B, T, H, K]  fp32 OUTPUT: c_{i,t-1} (exclusive prefix), per token
    T,
    info_scale,     # info-increment scale s in u = s*k^2/r (runtime fp32; default d_k=K)
    NT: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H

    i_k = tl.arange(0, BK)
    mask = i_k < K

    base = i_b * T * H * K + i_h * K
    base_r = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)

    # seed [n,d] from the forward Pass-B exclusive-prefix carry for this chunk.
    carry_base = (((i_b * NT + i_nt) * H + i_h) * K) * 2
    n = tl.load(carry_ptr + carry_base + i_k * 2 + 0, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(carry_ptr + carry_base + i_k * 2 + 1, mask=mask, other=1.0).to(tl.float32)

    for t in range(t0, t1):
        off = base + t * H * K + i_k
        # store the EXCLUSIVE prefix c_{i,t-1} = n/d BEFORE advancing (= mu at global t=0).
        tl.store(c_excl_ptr + off, n / d, mask=mask)
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
        a2 = a * a
        u = (info_scale * k * k) / r                           # u = s*k^2/r  (s default d_k = K)
        n_new = (1.0 + u * omega) * n + (u * a2) * d
        d_new = omega * n + a2 * d
        m = tl.maximum(tl.abs(n_new), tl.abs(d_new))
        inv = 1.0 / tl.maximum(m, 1e-30)
        n = n_new * inv
        d = d_new * inv


@triton.jit(do_not_specialize=["T"])
def _kla_kappa_bwd_passA_kernel(
    k_ptr,          # [B, T, H, K]
    a_ptr,          # [B, T, H, K]
    omega_ptr,      # [B, T, H, K]
    r_ptr,          # [B, T, H]
    dkappa_ptr,     # [B, T, H, K]  cotangent dL/dkappa
    dbeta_ptr,      # [B, T, H, K]  cotangent dL/dbeta_ch
    c_excl_ptr,     # [B, T, H, K]  fp32: c_{i,t-1} (from FILL)
    ab_ptr,         # [B, NT, H, K, 2]  OUTPUT chunk net affine map [A_chunk, B_chunk] per channel
    T,
    NT: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H

    i_k = tl.arange(0, BK)
    mask = i_k < K

    base = i_b * T * H * K + i_h * K
    base_r = i_b * T * H + i_h
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)

    # Compose the chunk's per-channel affine maps IN REVERSE token order. Net map sends
    # "s entering the chunk from the HIGH end" -> "s exiting at the LOW end" (= c_bar_{t0-1}).
    # Start = identity affine (A=1, B=0) on every channel.
    A_net = tl.full((BK,), 1.0, tl.float32)
    B_net = tl.full((BK,), 0.0, tl.float32)
    for t in range(t1 - 1, t0 - 1, -1):
        off = base + t * H * K + i_k
        cprev = tl.load(c_excl_ptr + off, mask=mask, other=1.0).to(tl.float32)   # c_{i,t-1} (= mu at t=0)
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)     # scalar r_t
        dki = tl.load(dkappa_ptr + off, mask=mask, other=0.0).to(tl.float32)
        dbi = tl.load(dbeta_ptr + off, mask=mask, other=0.0).to(tl.float32)

        a2 = a * a
        ksq = k * k
        p_hat = a2 / cprev + omega                             # [BK]
        # §3 per-token cross-channel reductions D, G (masked tl.sum over K).
        D = r + tl.sum(tl.where(mask, p_hat * ksq, 0.0), axis=0)
        kap = p_hat * k / D
        beta = p_hat / D
        G = tl.sum(tl.where(mask, dki * kap + dbi * beta, 0.0), axis=0)
        # §3 gain-path adjoint of p_hat_m (only Pbar_gain is needed for A_t/B_t).
        Pbar_gain = (dki * k + dbi) / D - ksq * G / D          # [BK]
        # §5 affine map coefficients A_t, B_t (renorm-free fp32).
        inv_cprev2 = 1.0 / (cprev * cprev)
        A_t = a2 * inv_cprev2 / (p_hat * p_hat)                # a2/(cprev^2 p_hat^2)
        B_t = -Pbar_gain * a2 * inv_cprev2                     # -Pbar_gain a2/cprev^2
        # left-compose: (A_net,B_net) <- (A_t,B_t) o (A_net,B_net) = (A_t*A_net, A_t*B_net + B_t)
        A_net = A_t * A_net
        B_net = A_t * B_net + B_t

    ab_base = (((i_b * NT + i_nt) * H + i_h) * K) * 2
    tl.store(ab_ptr + ab_base + i_k * 2 + 0, A_net, mask=mask)
    tl.store(ab_ptr + ab_base + i_k * 2 + 1, B_net, mask=mask)


@triton.jit
def _kla_kappa_bwd_passB_kernel(
    ab_ptr,         # [B, NT, H, K, 2]  chunk net affine maps [A_chunk, B_chunk] per channel
    carry_ptr,      # [B, NT, H, K]  OUTPUT adjoint s entering each chunk's HIGH end (per channel)
    NT: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)          # (b, h) in [0, B*H)
    i_b = pid // H
    i_h = pid % H

    i_k = tl.arange(0, BK)
    mask = i_k < K

    # c_bar entering the TOP chunk = c_bar_{T-1} = 0 on every channel.
    s = tl.zeros((BK,), tl.float32)
    for nt in range(NT - 1, -1, -1):
        # store the adjoint ENTERING this chunk (before its net map is applied).
        carry_base = ((i_b * NT + nt) * H + i_h) * K
        tl.store(carry_ptr + carry_base + i_k, s, mask=mask)
        ab_base = (((i_b * NT + nt) * H + i_h) * K) * 2
        A_chunk = tl.load(ab_ptr + ab_base + i_k * 2 + 0, mask=mask, other=0.0)
        B_chunk = tl.load(ab_ptr + ab_base + i_k * 2 + 1, mask=mask, other=0.0)
        s = A_chunk * s + B_chunk   # adjoint entering the next-LOWER chunk


@triton.jit(do_not_specialize=["T", "info_scale"])
def _kla_kappa_bwd_passC_kernel(
    k_ptr,          # [B, T, H, K]
    a_ptr,          # [B, T, H, K]
    omega_ptr,      # [B, T, H, K]
    r_ptr,          # [B, T, H]
    dkappa_ptr,     # [B, T, H, K]  cotangent dL/dkappa
    dbeta_ptr,      # [B, T, H, K]  cotangent dL/dbeta_ch
    c_excl_ptr,     # [B, T, H, K]  fp32: c_{i,t-1} (from FILL)
    carry_ptr,      # [B, NT, H, K]  adjoint s entering each chunk's HIGH end
    dk_ptr,         # [B, T, H, K]  OUTPUT dL/dk
    dalpha_ptr,     # [B, T, H, K]  OUTPUT dL/dalpha
    domega_ptr,     # [B, T, H, K]  OUTPUT dL/domega
    dr_ptr,         # [B, T, H]     OUTPUT dL/dr_t
    dmu_ptr,        # [H]           OUTPUT dL/dmu (fp32, atomic over b) -- chunk 0 only
    T,
    info_scale,     # info-increment scale s in u = s*k^2/r (runtime fp32; default d_k=K)
    NT: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    i_nt = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H
    i_h = pid % H

    i_k = tl.arange(0, BK)
    mask = i_k < K

    base = i_b * T * H * K + i_h * K
    base_r = i_b * T * H + i_h
    Kf = info_scale                         # info-increment scale s (default d_k=K; float runtime)
    t0 = i_nt * BT
    t1 = tl.minimum(t0 + BT, T)

    carry_base = ((i_b * NT + i_nt) * H + i_h) * K
    s = tl.load(carry_ptr + carry_base + i_k, mask=mask, other=0.0).to(tl.float32)  # c_bar entering HIGH token

    # reverse-walk the chunk: s currently holds c_bar_t at the top of each iteration.
    for t in range(t1 - 1, t0 - 1, -1):
        off = base + t * H * K + i_k
        cprev = tl.load(c_excl_ptr + off, mask=mask, other=1.0).to(tl.float32)   # c_{i,t-1} (= mu at t=0)
        k = tl.load(k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
        omega = tl.load(omega_ptr + off, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base_r + t * H).to(tl.float32)     # scalar r_t
        dki = tl.load(dkappa_ptr + off, mask=mask, other=0.0).to(tl.float32)
        dbi = tl.load(dbeta_ptr + off, mask=mask, other=0.0).to(tl.float32)

        a2 = a * a
        ksq = k * k
        p_hat = a2 / cprev + omega                             # [BK]
        # §3 per-token cross-channel reductions D, G.
        D = r + tl.sum(tl.where(mask, p_hat * ksq, 0.0), axis=0)
        kap = p_hat * k / D
        beta = p_hat / D
        G = tl.sum(tl.where(mask, dki * kap + dbi * beta, 0.0), axis=0)
        # §3 gain-path adjoints.
        Pbar_gain = (dki * k + dbi) / D - ksq * G / D          # [BK]
        kbar_gain = dki * beta - 2.0 * p_hat * k * G / D       # gain-path grad of k_m
        rbar_gain = -G / D                                     # gain-path grad of r_t (scalar)
        # §4 total adjoint of p_hat: gain consumer + info-update consumer (c_t = 1/p_hat + u).
        # Uses s = c_bar_t (the value ENTERING step t, from the future) -- the inclusive
        # state-adjoint, per the derivation §5 caveat (emit s_t BEFORE the push).
        Pbar_tot = Pbar_gain + s * (-1.0 / (p_hat * p_hat))    # [BK]
        # local input grads at (m,t).
        domega = Pbar_tot
        da2 = Pbar_tot / cprev
        dalpha = 2.0 * a * da2                                 # IMPL-CARRY #3: emit dalpha=2 alpha da2
        dk = kbar_gain + s * (2.0 * Kf * k / r)                # gain path + info-update(u) path
        # dr: gain path (per-token scalar) + info-update scan-sum (K-reduction).
        dr_scan = -(Kf / (r * r)) * tl.sum(tl.where(mask, s * ksq, 0.0), axis=0)
        dr = rbar_gain + dr_scan                               # scalar
        tl.store(dk_ptr + off, dk.to(dk_ptr.dtype.element_ty), mask=mask)
        tl.store(dalpha_ptr + off, dalpha.to(dalpha_ptr.dtype.element_ty), mask=mask)
        tl.store(domega_ptr + off, domega.to(domega_ptr.dtype.element_ty), mask=mask)
        tl.store(dr_ptr + base_r + t * H, dr.to(dr_ptr.dtype.element_ty))
        # push adjoint to the previous state c_{m,t-1}:  s <- A_t*s + B_t = c_bar_{t-1}.
        inv_cprev2 = 1.0 / (cprev * cprev)
        A_t = a2 * inv_cprev2 / (p_hat * p_hat)
        B_t = -Pbar_gain * a2 * inv_cprev2
        s = A_t * s + B_t
        # IMPL-CARRY #1: padded lanes have cprev=0 => p_hat=inf => s NaN; zero them so
        # the next iteration's reductions/store stay finite (real configs have K==BK).
        s = tl.where(mask, s, 0.0)

    # chunk 0 finishes at t=0, so s is now c_bar_{i,-1} = adjoint of mu for this (b,h).
    # mu[h] seeds every channel of every batch element => dmu[h] = sum_b sum_m c_bar_{m,-1}.
    if i_nt == 0:
        dmu = tl.sum(tl.where(mask, s, 0.0), axis=0)
        tl.atomic_add(dmu_ptr + i_h, dmu)


@torch.compiler.disable
def _kla_kappa_bwd_parallel(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r_bt: torch.Tensor,   # [B,T,H] fp32 (pre-broadcast)
    mu_h: torch.Tensor,   # [H] fp32 (pre-broadcast)
    dkappa: torch.Tensor,
    dbeta_ch: torch.Tensor,
    *,
    grad_dtype: torch.dtype | None = None,
    info_scale: float | None = None,
    BT: int = _KLA_BT_DEFAULT,
):
    """Parallel affine two-pass reverse VJP of the per-channel Kalman-gain scan.

    Same result/contract as :func:`_kla_kappa_bwd_serial` (returns
    ``(dk, dalpha, domega, dr, dmu)``) but parallel across chunks -- grid ``(NT,B*H)``
    for every per-token pass, only the short Pass-B carry is serial. RECOMPUTES the
    forward exclusive-prefix ``c_{i,t-1}`` via the forward Pass A/B (net Mobius maps +
    chunk-entry ``[n,d]`` carry) followed by a chunk-parallel FILL into a transient
    global fp32 ``c_excl[B,T,H,K]`` (allocated here, freed on return; the SAME memory
    the serial anchor uses). Pass A'/B'/C' then run the AFFINE reverse scan (derivation
    §5, renorm-free fp32). ``dmu`` fp32 accumulator in ``mu``'s dtype. Computes exactly
    :func:`lit_gpt.kla_ops.diag_naive._kla_kappa_bwd_ref`.
    """
    B, T, H, K = k.shape
    if grad_dtype is None:
        grad_dtype = k.dtype
    info_scale = float(K) if info_scale is None else float(info_scale)  # s in u=s*k^2/r (default d_k=K)
    dk = torch.empty((B, T, H, K), dtype=grad_dtype, device=k.device)
    dalpha = torch.empty((B, T, H, K), dtype=grad_dtype, device=k.device)
    domega = torch.empty((B, T, H, K), dtype=grad_dtype, device=k.device)
    dr = torch.empty((B, T, H), dtype=grad_dtype, device=k.device)
    dmu = torch.zeros((H,), dtype=torch.float32, device=k.device)
    if T == 0:
        return dk, dalpha, domega, dr, dmu.to(mu_h.dtype)

    NT = (T + BT - 1) // BT
    BK = triton.next_power_of_2(K)
    # fp32 scratch:
    #  * cm    [B,NT,H,K,4] forward per-chunk net Mobius maps (Pass A)
    #  * carry [B,NT,H,K,2] forward chunk-entry [n,d] carry (Pass B)
    #  * c_excl[B,T,H,K]    recomputed exclusive-prefix c_{i,t-1} (FILL; O(T), == serial anchor)
    #  * ab    [B,NT,H,K,2] backward per-chunk net AFFINE maps [A_chunk,B_chunk] (Pass A')
    #  * cbwd  [B,NT,H,K]   backward per-chunk high-end adjoint s (Pass B')
    cm = torch.empty((B, NT, H, K, 4), dtype=torch.float32, device=k.device)
    carry = torch.empty((B, NT, H, K, 2), dtype=torch.float32, device=k.device)
    c_excl = torch.empty((B, T, H, K), dtype=torch.float32, device=k.device)
    ab = torch.empty((B, NT, H, K, 2), dtype=torch.float32, device=k.device)
    cbwd = torch.empty((B, NT, H, K), dtype=torch.float32, device=k.device)

    gridAC = (NT, B * H)
    # --- recompute c_excl: forward Pass A (net maps) + Pass B (entry carry) + FILL ---
    _kla_kappa_passA_kernel[gridAC](
        k, alpha, omega, r_bt, cm, T, info_scale, NT=NT, H=H, K=K, BK=BK, BT=BT,
    )
    _kla_kappa_passB_kernel[(B * H,)](
        cm, mu_h, carry, NT=NT, H=H, K=K, BK=BK,
    )
    _kla_kappa_bwd_fill_kernel[gridAC](
        k, alpha, omega, r_bt, carry, c_excl, T, info_scale, NT=NT, H=H, K=K, BK=BK, BT=BT,
    )
    # --- reverse affine two-pass: Pass A' (compose) -> Pass B' (serial carry) -> Pass C' (emit) ---
    _kla_kappa_bwd_passA_kernel[gridAC](
        k, alpha, omega, r_bt, dkappa, dbeta_ch, c_excl, ab,
        T, NT=NT, H=H, K=K, BK=BK, BT=BT,
    )
    _kla_kappa_bwd_passB_kernel[(B * H,)](
        ab, cbwd, NT=NT, H=H, K=K, BK=BK,
    )
    _kla_kappa_bwd_passC_kernel[gridAC](
        k, alpha, omega, r_bt, dkappa, dbeta_ch, c_excl, cbwd,
        dk, dalpha, domega, dr, dmu,
        T, info_scale, NT=NT, H=H, K=K, BK=BK, BT=BT,
    )
    return dk, dalpha, domega, dr, dmu.to(mu_h.dtype)


# Backward-version selector: "parallel" (the KB3 two-pass, default once BG3 green) or
# "serial" (the KB2 anchor / oracle / fallback). Env override for A/B + oracle-compare:
# KLA_KAPPA_BWD in {serial, parallel}. Kwarg ``bwd_kernel`` overrides the env.
import os as _os  # noqa: E402

_KLA_KAPPA_BWD = _os.environ.get("KLA_KAPPA_BWD", "parallel").lower()


# =============================================================================
# autograd.Function -- wraps the forward (chunked two-pass, returns kappa,beta_ch)
# + the backward (parallel affine two-pass by default; serial anchor as oracle/
# fallback) so kla_kappa_chunk is differentiable end-to-end w.r.t. (k, alpha, omega,
# r_bt, mu_h). RECOMPUTE state strategy: nothing is saved ACROSS the fwd->bwd
# boundary except the five differentiable inputs; the backward re-derives c_{i,t-1}
# itself.
#
# dkappa/dbeta_ch arrive in the outputs' dtype; dk/dalpha/domega/dr come back in
# each input's dtype (dmu fp32-accumulated, cast to mu's dtype). The final state
# c_{T-1} is not an output, so there is no third cotangent to seed.
# =============================================================================
class _KlaKappaChunkFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, alpha, omega, r_bt, mu_h, out_dtype, bwd_kernel, info_scale):
        kappa, beta_ch = _kla_kappa_chunk_v1(
            k, alpha, omega, r_bt, mu_h, out_dtype=out_dtype, info_scale=info_scale,
        )
        ctx.bwd = (bwd_kernel or _KLA_KAPPA_BWD).lower()
        ctx.info_scale = info_scale
        ctx.save_for_backward(k, alpha, omega, r_bt, mu_h)
        return kappa, beta_ch

    @staticmethod
    def backward(ctx, dkappa, dbeta_ch):
        k, alpha, omega, r_bt, mu_h = ctx.saved_tensors
        needs = ctx.needs_input_grad  # (k, alpha, omega, r_bt, mu_h, out_dtype, bwd_kernel, info_scale)
        # dkappa/dbeta_ch come in the outputs' dtype; the kernel upcasts to fp32.
        # (either may be None if only one output feeds the loss.)
        dkappa = (torch.zeros_like(k) if dkappa is None else dkappa).contiguous()
        dbeta_ch = (torch.zeros_like(k) if dbeta_ch is None else dbeta_ch).contiguous()
        if ctx.bwd == "parallel":
            dk, dalpha, domega, dr, dmu = _kla_kappa_bwd_parallel(
                k.contiguous(), alpha.contiguous(), omega.contiguous(),
                r_bt.contiguous(), mu_h.contiguous(), dkappa, dbeta_ch,
                info_scale=ctx.info_scale,
            )
        else:
            dk, dalpha, domega, dr, dmu = _kla_kappa_bwd_serial(
                k.contiguous(), alpha.contiguous(), omega.contiguous(),
                r_bt.contiguous(), mu_h.contiguous(), dkappa, dbeta_ch,
                info_scale=ctx.info_scale,
            )
        # Respect needs_input_grad (None where no grad is required). Extra None x3 for
        # the non-tensor args (out_dtype, bwd_kernel, info_scale).
        return (
            dk if needs[0] else None,
            dalpha if needs[1] else None,
            domega if needs[2] else None,
            dr if needs[3] else None,
            dmu if needs[4] else None,
            None,
            None,
            None,
        )


def _bcast_r(r: torch.Tensor | float, B: int, T: int, H: int, device, dtype) -> torch.Tensor:
    """Broadcast ``r`` (scalar / [H] / [B,T,H]) to a contiguous ``[B,T,H]`` tensor.

    Keeps the kernel single-path (S6): it always reads a per-token ``r`` at
    ``base_r + t*H``. fp32 (the compute dtype); the kernel upcasts anyway.
    """
    if not torch.is_tensor(r):
        return torch.full((B, T, H), float(r), dtype=torch.float32, device=device)
    r = r.to(torch.float32)
    if r.dim() == 0:                       # scalar tensor
        return r.expand(B, T, H).contiguous()
    if r.dim() == 1:                       # [H] per-head
        assert r.shape[0] == H, f"r [H] must be [{H}]; got {tuple(r.shape)}"
        return r.view(1, 1, H).expand(B, T, H).contiguous()
    assert r.shape == (B, T, H), f"r must be [B,T,H]={[B,T,H]}, [H], or scalar; got {tuple(r.shape)}"
    return r.contiguous()


def _bcast_mu(mu: torch.Tensor | float, H: int, device) -> torch.Tensor:
    """Broadcast ``mu`` (scalar / [H]) to a contiguous ``[H]`` fp32 tensor."""
    if not torch.is_tensor(mu):
        return torch.full((H,), float(mu), dtype=torch.float32, device=device)
    mu = mu.to(torch.float32)
    if mu.dim() == 0:
        return mu.expand(H).contiguous()
    assert mu.dim() == 1 and mu.shape[0] == H, f"mu must be [H]={[H]} or scalar; got {tuple(mu.shape)}"
    return mu.contiguous()


@torch.compiler.disable
def kla_kappa_chunk(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor | float = 1.0,
    mu: torch.Tensor | float = 1.0,
    *,
    out_dtype: torch.dtype | None = None,
    initial_info: torch.Tensor | None = None,
    save_state: bool = False,
    info_scale: float | None = None,
):
    """Triton diagonal-KLA per-channel Kalman gain scan: ``(kappa, beta_ch)``.

    Computes, per (b, h) stream, key channel i, and token t (with ``c_{i,-1}=mu[h]``):

        p_hat_i   = alpha_{i,t}^2 / c_{i,t-1} + omega_{i,t}          # predicted uncertainty (exclusive prefix)
        denom     = r_t + sum_i p_hat_i k_{i,t}^2                # per-token K-reduction (cross-channel)
        kappa_i   = p_hat_i k_{i,t} / denom
        beta_ch_i = p_hat_i / denom                              # kappa = beta_ch * k
        c_{i,t}   = 1/p_hat_i + d_k k_{i,t}^2 / r_t              # posterior info (d_k = K)

    Bit-close (fp32) to :func:`lit_gpt.kla_ops.diag_naive._kla_kappa`
    (``return_beta_ch=True``), which is fp64-verified vs ``_kla_kappa_seq``. The
    per-channel Mobius state is tracked as ``[n, d]`` (``c = n/d``) with per-step
    max-entry renorm -- overflow-proof for the tiny-``omega`` regime where the raw
    scalar ``c`` blows fp32 to ``inf`` (design C1).

    Runs a chunked two-pass per-channel Mobius scan (parallel ``B*H*NT`` with the K
    channels vectorized; only the short Pass-B carry is serial).

    **Differentiable** w.r.t. ``(k, alpha, omega, r, mu)`` via :class:`_KlaKappaChunkFn`
    (autograd.Function) -- but ONLY on the plain forward path (``initial_info is None``
    and ``not save_state``) taken when ``torch.is_grad_enabled()``. The backward is the
    PARALLEL affine two-pass reverse VJP (:func:`_kla_kappa_bwd_parallel`, KB3, DEFAULT)
    with the serial reverse VJP (:func:`_kla_kappa_bwd_serial`, the KB2 anchor) kept as
    the oracle/fallback; select via the ``KLA_KAPPA_BWD`` env var (``parallel`` default;
    ``serial`` = anchor). Under ``no_grad`` / with ``save_state`` / ``initial_info`` it
    calls the two-pass kernel directly (forward-only, no ``grad_fn``).

    Args:
        k, alpha, omega: ``[B, T, H, K]`` (bf16 or fp32). ``k`` is used directly
            (per-channel ``k_i^2``); the layer l2-norms ``k`` upstream but the
            kernel does not rely on it.
        r: observation noise ``[B,T,H]`` (per-token) / ``[H]`` (per-head) / scalar
            (``>= r_min``). Broadcast to ``[B,T,H]`` internally (single kernel path).
        mu: info prior ``[H]`` (per-head) or scalar. Broadcast to ``[H]`` internally.
        out_dtype: dtype of the returned ``kappa``/``beta_ch``. Default fp32
            (correctness anchor). Pass ``torch.bfloat16`` for the real path. Compute
            is ALWAYS fp32 regardless.
        initial_info: NOT supported (state-passing/decoding is out of scope); must
            be ``None``. The kernel seeds ``c_0 = mu`` only.
        save_state: if True, also returns the fp32 exclusive-prefix Mobius state
            ``(n, d)`` ``[B,T,H,K]`` (``= [mu, 1]`` at ``t=0``) for the G2 numerical
            -safety gate (assert ``isfinite(n) AND isfinite(d)`` SEPARATELY -- never
            ``c=n/d``, which is legitimately non-finite as ``d -> 0``).
        info_scale: info-increment scale ``s`` in the posterior-info update
            ``c_t = 1/p_hat + s * k^2 / r`` (``u = s * k^2 / r``). Default ``None`` ->
            ``float(K)`` (= ``d_k``), bit-identical to the pre-ablation hardcoded ``K``.
            The DiagKLA info-scale ablation passes ``s`` in ``{1, sqrt(K), K}``. Always
            forwarded to the kernels as a runtime fp32 scalar (kernels
            ``do_not_specialize`` it), so one compiled variant serves every scale.

    Returns:
        ``(kappa, beta_ch)`` each ``[B,T,H,K]`` in ``out_dtype`` (default fp32); or
        ``(kappa, beta_ch, n, d)`` when ``save_state`` (``n, d`` always fp32).
    """
    assert initial_info is None, (
        "kla_kappa_chunk does not support initial_info (state-passing/decoding is "
        "out of scope); it seeds c_0 = mu only."
    )
    assert k.shape == alpha.shape == omega.shape, (
        f"k/alpha/omega must share shape [B,T,H,K]; got {tuple(k.shape)}, "
        f"{tuple(alpha.shape)}, {tuple(omega.shape)}"
    )
    assert k.dim() == 4, f"k/alpha/omega must be 4-D [B,T,H,K]; got {k.dim()}-D"
    B, T, H, K = k.shape
    # info-increment scale s in the posterior-info update c_t = 1/p_hat + s*k^2/r
    # (u = s*k^2/r). Default s = d_k = K (bit-identical to the pre-ablation hardcoded K);
    # the DiagKLA info-scale ablation sweeps s in {1, sqrt(K), K}.
    info_scale = float(K) if info_scale is None else float(info_scale)
    # K must be a power of 2: BK=next_power_of_2(K); when K<BK the padded lanes are
    # masked in the loads/stores/denom, but Pass C seeds a masked lane with n=0 =>
    # p_hat=a2*d/n divides by 0 in-register (contained by the store masks, but only
    # truly absent when there are NO padded lanes). head_k_dim is a power of 2 in all
    # configs; enforce it so a non-pow2 K fails loudly rather than computing masked NaN.
    assert (K & (K - 1)) == 0, f"kla_kappa_chunk requires K a power of 2; got K={K}"
    assert k.is_cuda and alpha.is_cuda and omega.is_cuda, (
        "kla_kappa_chunk requires CUDA tensors (Triton is GPU-only)."
    )
    if torch.is_tensor(r):
        assert r.is_cuda, "r must be a CUDA tensor (or a python scalar)."
    if torch.is_tensor(mu):
        assert mu.is_cuda, "mu must be a CUDA tensor (or a python scalar)."

    if out_dtype is None:
        out_dtype = torch.float32

    # Contiguity is required for the flat [B,T,H,K] offset arithmetic in the kernels.
    k = k.contiguous()
    alpha = alpha.contiguous()
    omega = omega.contiguous()
    # Broadcast r -> [B,T,H] and mu -> [H] so the kernels are single-path (S6).
    r_bt = _bcast_r(r, B, T, H, k.device, k.dtype)
    mu_h = _bcast_mu(mu, H, k.device)

    # Route through the differentiable autograd.Function ONLY when grads are actually
    # needed AND this is the plain (non-state, no-init) forward the Function supports.
    # Otherwise call the two-pass kernel directly -- preserving the forward-only
    # save_state / no_grad / out_dtype callers and the existing forward tests unchanged.
    if torch.is_grad_enabled() and initial_info is None and not save_state:
        return _KlaKappaChunkFn.apply(k, alpha, omega, r_bt, mu_h, out_dtype, None, info_scale)

    return _kla_kappa_chunk_v1(
        k, alpha, omega, r_bt, mu_h,
        out_dtype=out_dtype, save_state=save_state, info_scale=info_scale,
    )

