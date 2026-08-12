# Fused Triton FORWARD kernel for the ExactKLA dense-covariance Kalman gain kappa (G1).
#
# Computes the exact anisotropic Kalman gain
#     kappa_t = P_hat_t k_t / (r_eff + k_t^T P_hat_t k_t)
# for the DENSE-covariance Kalman filter, token-serially, carrying the full
# K x K covariance P in registers. This replaces the fp64 compute-bound
# `exact_kla_gains_scan` (PyTorch chunked LFT block scan) as the producer of
# kappa for the `chunk_kalman` memory kernel.
#
# Recurrence (mirrors lit_gpt/kla_ops/exact_naive.py::naive_recurrent_exact_kla
# EXACTLY, joseph=True code default; D_t=diag(alpha_t), Omega_t=diag(omega_t)):
#     predict:  P_hat_t = D_t P_{t-1} D_t + Omega_t
#     gain:     kappa_t = P_hat_t k_t / (r_eff + k_t^T P_hat_t k_t)
#     update:   P_t = (I - kappa_t k_t^T) P_hat_t (I - kappa_t k_t^T)^T
#                     + r_eff kappa_t kappa_t^T          (Joseph, PSD)
# with r_eff = r_t / d_k when dk_calibration (d_k = K), else r_eff = r_t, and
# P_0 = mu^-1 I.  The predict step is elementwise (P scaled by alpha (x) alpha,
# plus diag(omega) on the diagonal -- NO matmul); the gain needs one matvec
# P_hat @ k (P_hat symmetric => k^T P_hat = (P_hat k)^T); the Joseph update is
# two rank-structured outer-product recombinations. Everything runs in fp32.
#
# ============================ CONVENTIONS ====================================
#   * This kernel takes RAW alpha (the transition D=diag(alpha)), NOT log-alpha.
#     `chunk_kalman` (the downstream memory kernel) takes NATURAL-log g = log(alpha).
#     Keep this straight at the call site: gain_recurrent(alpha=...) but
#     chunk_kalman(g=alpha.log()).  The gain kernel emits RAW kappa; decay
#     pre-baking of the write key stays inside chunk_kalman.
#   * Output kappa is [B, T, H, K], matching exact_kla_gains_scan.
#
# ============================== SCOPE (v1) ===================================
#   * K = 64 ONLY.  P is a [K, K] fp32 register tile = 4096 elements at K=64,
#     comparable to the fused_recurrent_gdn2 template's [BK, BV] tile. At K=128
#     that is 16384 fp32 registers (4x), which spills -- a K=128 version needs a
#     TILED / shared-memory P layout (see the note below). THE REAL TRAINING DIM
#     IS K=128, so this v1 forward kernel is a stepping stone, not yet the
#     production path.
#   * grid = (B * H): one program per (batch, head), serial over T.
#   * Forward only.  The autograd.Function backward raises NotImplementedError
#     (that is milestone G2). Use exact_kla_gains_scan / autograd through
#     naive_recurrent_exact_kla for gradients until G2 lands.
#
# --------------------------- K=128 note --------------------------------------
# A [128,128] fp32 tile does not fit in registers per-program. Options for a
# future K=128 kernel: (a) tile P into [BK, BK] sub-blocks with the predict /
# matvec / Joseph passes looping over sub-blocks (P still fully materialized in
# SRAM, not registers); (b) 2 programs per head splitting the K axis, exchanging
# the P_hat@k partials through shared memory. Either way the O(K^2) state is the
# hard wall -- unchanged algebra, different memory layout.
#
# Fork honesty: only the SKELETON transfers from fused_recurrent_gdn2.py (grid
# = programs over batch*head, per-token T-loop, state carried in registers,
# block-pointer load/store). The entire K x K predict / matvec / Joseph inner
# body is new (GDN-2 carries a [K,V] state with a rank-1 update; this carries a
# [K,K] covariance with a Riccati update).
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _hp(x: torch.Tensor) -> torch.Tensor:
    """Promote to at least fp32 while preserving fp64 (matches exact_naive._hp)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


# Default gain backward backend (env GAIN_RECURRENT_BWD overrides).
#   "split" (DEFAULT, recompute -- SAFE AT ANY DEPTH): the backward re-forwards (Kernel A) into a
#     TRANSIENT [B*H,T,K,K] pp_full, reverse-scans (Kernel B), frees it -- only ONE ~4.3 GB (K=64)
#     buffer is live at a time, serialized per layer.
#   "stash" (Lever 1, OPT-IN): the FORWARD stashes pp_full so the backward skips the re-forward and
#     runs ONLY Kernel B. FASTER (gain-bwd 34->13 ms) BUT the stash is DEPTH-MULTIPLIED: in standard
#     autograd all N layer forwards complete before any backward, so N x stash_GB buffers COEXIST at
#     the fwd/bwd boundary (16-layer K=64 ~= 69 GB; 4x at K=128). Enable only when N_layers x stash_GB
#     fits the memory budget (e.g. noisy_mqar 2-layer K=64 = 8.6 GB -- fits; deep / K=128 -> keep split).
#   "phaseB": alias for "split".  "fused": the old single fat kernel.  "phaseA": O(T) PyTorch oracle.
_DEFAULT_BWD = "split"


# =============================================================================
# FORWARD KERNEL
# -----------------------------------------------------------------------------
# One program per (batch, head). Carries P [K, K] fp32 in registers as a 2-D
# tile and walks tokens serially. Per token:
#   load k_t, alpha_t, omega_t, r_t
#   predict:  P = alpha[:,None] * P * alpha[None,:] ; P += diag(omega)
#   matvec :  Pk[i] = sum_j P[i,j] k[j]              (P symmetric)
#   denom  :  r_eff + sum_i k[i] Pk[i]
#   kappa  :  Pk / denom  -> store
#   Joseph :  J = I - kappa (x) k ; P = J P J^T + r_eff kappa (x) kappa
# The [K,K] intermediates (J, J@P) are held as register tiles; the two matmuls
# in the Joseph update are done with tl.sum over a broadcast axis (K=64 keeps
# the K x K x K work resident), all in fp32.
# =============================================================================
@triton.jit(do_not_specialize=["T"])
def gain_recurrent_fwd_kernel(
    k_ptr,           # [B, T, H, K]
    alpha_ptr,       # [B, T, H, K]  (RAW alpha, not log)
    omega_ptr,       # [B, T, H, K]
    r_ptr,           # [B, T, H]     (already broadcast to per-token per-head)
    kappa_ptr,       # [B, T, H, K]  (output)
    inv_mu_ptr,      # [H]           (1/mu per head; P_0 = inv_mu * I)
    pp_full_ptr,     # [B*H, T, K, K]  P_prev stash (written per token ONLY if STORE_PPREV)
    r_eff_scale: tl.constexpr,   # 1/d_k if dk_calibration else 1.0
    T: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    SYMMETRIZE: tl.constexpr,    # re-symmetrize P each step (fp32 hygiene)
    STORE_PPREV: tl.constexpr,   # Lever 1: also stash P_prev[t] so the bwd can drop Kernel A
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    # 2-D masks for the [K, K] covariance tile.
    mask_row = mask_k[:, None]
    mask_col = mask_k[None, :]
    mask_kk = mask_row & mask_col
    # Identity tile (fp32) for the Joseph update.
    eye = tl.where(o_k[:, None] == o_k[None, :], 1.0, 0.0).to(tl.float32)

    # P_0 = mu^-1 I  (per head).
    inv_mu = tl.load(inv_mu_ptr + i_h).to(tl.float32)
    b_P = eye * inv_mu   # [BK, BK] fp32, off-diagonal + padded rows/cols are 0

    # Base pointers into token 0 of this (batch, head). Layout [B, T, H, K]:
    #   offset(b, t, h, :) = ((b*T + t)*H + h)*K
    base_kk = ((i_b * T + 0) * H + i_h) * K
    p_k = k_ptr + base_kk + o_k
    p_a = alpha_ptr + base_kk + o_k
    p_o = omega_ptr + base_kk + o_k
    p_kap = kappa_ptr + base_kk + o_k
    # r is [B, T, H]:  offset(b, t, h) = (b*T + t)*H + h
    p_r = r_ptr + (i_b * T + 0) * H + i_h
    # Lever 1 (STASH-FROM-FORWARD): the covariance P entering each token (pre-predict) is
    # EXACTLY what the bwd re-forward (gain_recurrent_reforward_kernel, Kernel A) recomputes.
    # When STORE_PPREV, write it once per token to pp_full [B*H, T, K, K] so the backward can
    # skip the re-forward and reverse-scan straight off this HBM buffer. Grad-gated in the
    # launcher (only when an input needs grad), so inference stays lean (no 4.3 GB stash).
    if STORE_PPREV:
        p_pp = pp_full_ptr + i_bh * T * K * K + o_k[:, None] * K + o_k[None, :]

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)      # [BK]
        b_a = tl.load(p_a, mask=mask_k, other=0.0).to(tl.float32)      # [BK]
        b_om = tl.load(p_o, mask=mask_k, other=0.0).to(tl.float32)     # [BK]
        b_r = tl.load(p_r).to(tl.float32)                             # scalar
        r_eff = b_r * r_eff_scale

        # stash P_prev (= P entering this token, pre-predict) -- bit-identical to Kernel A.
        if STORE_PPREV:
            tl.store(p_pp, b_P.to(pp_full_ptr.dtype.element_ty), mask=mask_kk)

        # --- predict: P_hat = diag(alpha) P diag(alpha) + diag(omega) ---
        b_P = b_a[:, None] * b_P * b_a[None, :]
        # add omega on the diagonal only (padded lanes stay 0 via mask_kk)
        b_P = b_P + eye * b_om[None, :]
        b_P = tl.where(mask_kk, b_P, 0.0)

        # --- gain: matvec Pk = P_hat @ k (P_hat symmetric), denom, kappa ---
        b_Pk = tl.sum(b_P * b_k[None, :], axis=1)                     # [BK]
        b_Pk = tl.where(mask_k, b_Pk, 0.0)
        denom = r_eff + tl.sum(b_k * b_Pk, axis=0)                    # scalar
        b_kap = b_Pk / denom                                         # [BK]
        b_kap = tl.where(mask_k, b_kap, 0.0)
        tl.store(p_kap, b_kap.to(kappa_ptr.dtype.element_ty), mask=mask_k)

        # --- Joseph covariance update: P = J P_hat J^T + r_eff kappa kappa^T ---
        # J = I - kappa (x) k                                          [BK, BK]
        b_J = eye - b_kap[:, None] * b_k[None, :]
        # Two K x K matmuls via tl.dot with IEEE (full-fp32) accumulation. A
        # broadcasted 3-D tl.sum (materializing a [K,K,K] tile) was numerically
        # lossy here (~2e-3 vs fp64, 1000x worse than a genuine fp32 recurrence);
        # tl.dot(input_precision="ieee") matches fp32 to ~1e-6. Padded rows/cols
        # of b_P and b_J are already zero, so the K->BK padding contributes 0.
        b_JP = tl.dot(b_J, b_P, input_precision="ieee")               # J @ P    [BK, BK]
        b_JPJt = tl.dot(b_JP, tl.trans(b_J), input_precision="ieee")  # (JP) J^T [BK, BK]
        b_P = b_JPJt + r_eff * (b_kap[:, None] * b_kap[None, :])
        if SYMMETRIZE:
            # Joseph form is analytically symmetric; fp32 rounding breaks it
            # slightly. Averaging with the transpose halves the antisymmetric
            # rounding error and keeps P PSD-clean for the next matvec.
            b_P = 0.5 * (b_P + tl.trans(b_P))
        b_P = tl.where(mask_kk, b_P, 0.0)

        p_k += H * K
        p_a += H * K
        p_o += H * K
        p_kap += H * K
        p_r += H
        if STORE_PPREV:
            p_pp += K * K


# =============================================================================
# BACKWARD (Phase B) -- Triton reverse covariance-adjoint recurrence.
# -----------------------------------------------------------------------------
# Two kernels:
#   (1) gain_recurrent_ckpt_fwd_kernel -- re-runs the forward carrying P in
#       registers and STORES P at the entry of every C-block (a checkpoint
#       P_{c*C-1}, i.e. the covariance ENTERING block c, pre-predict) to
#       ckpt_ptr [B, H, nc, K, K]. nc = cdiv(T, C). This is the cheap forward
#       pass whose checkpoints let the reverse loop re-forward each block.
#   (2) gain_recurrent_bwd_kernel -- grid (B*H). Walks C-blocks in REVERSE.
#       For each block it (a) loads the block's checkpoint P, (b) RE-FORWARDS
#       the <=C steps of the block with the EXACT forward op-order (same ieee
#       dots + SYMMETRIZE) storing per-token P_prev and P_hat to a per-program
#       HBM scratch pp_ptr/ph_ptr [B*H, C, K, K] (HBM-per-block-P: we do NOT
#       hold two K x K tiles in registers at K=64 -- the carried adjoint s is
#       the only resident K x K tile; P_prev/P_hat are streamed from HBM),
#       then (c) reverse-scans the covariance adjoint over the block using the
#       stored intermediates, emitting dk, dalpha, domega, dr per token and
#       accumulating d(P_0) into dP0_ptr at the very first token (t==0) for dmu.
#
# The reverse recurrence is the exact adjoint verified (fp64) to ~1e-15 vs
# autograd through _gain_recurrent_pytorch. Per reverse step (carry s = dP_t):
#   symmetrize s (adjoint of symmetric P; factor-2 on off-diagonals)
#   Joseph  P=J Phat J^T + r_eff kappa kappa^T:
#       gJ    = (s+s^T) J Phat          (= 2 s J Phat, s symmetric)
#       gPhat = J^T s J
#       dkap += 2 r_eff (s kappa)
#       dr_eff = kappa^T s kappa
#       dkap += -gJ k ;  dk += -gJ^T kappa
#   gain    kappa=Pk/denom, Pk=Phat k, denom=r_eff + k^T Phat k:
#       dPk    = dkap / denom
#       ddenom = -(dkap . kappa) / denom
#       dr_eff += ddenom
#       dk    += ddenom * 2 Pk            (denom's explicit k, Phat fixed)
#       gPhat += dPk k^T                  (Pk = Phat k)
#       gPhat += ddenom * k k^T           (denom = r_eff + k^T Phat k)
#       dk    += Phat dPk                 (Phat symmetric)
#   predict Phat=diag(a) P diag(a)+diag(om):
#       dom    = diag(gPhat)
#       dP_prev = a[:,None] gPhat a[None,:]    (-> carried s for next reverse step)
#       da     = sum_j gPhat_ij P_ij a_j  +  sum_i gPhat_ij P_ij a_i   (TWO terms)
#   emit dk, da, dom, dr=dr_eff/dk_cal ; s <- dP_prev ; at t==0 add s to dP0.
# =============================================================================
@triton.jit(do_not_specialize=["T"])
def gain_recurrent_ckpt_fwd_kernel(
    k_ptr,           # [B, T, H, K]
    alpha_ptr,       # [B, T, H, K]  (RAW alpha)
    omega_ptr,       # [B, T, H, K]
    r_ptr,           # [B, T, H]
    ckpt_ptr,        # [B, H, nc, K, K]  (P entering each C-block, pre-predict)
    inv_mu_ptr,      # [H]
    r_eff_scale: tl.constexpr,
    T: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    C: tl.constexpr,        # checkpoint block length
    NC: tl.constexpr,       # number of checkpoints = cdiv(T, C)
    SYMMETRIZE: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    mask_kk = mask_k[:, None] & mask_k[None, :]
    eye = tl.where(o_k[:, None] == o_k[None, :], 1.0, 0.0).to(tl.float32)

    inv_mu = tl.load(inv_mu_ptr + i_h).to(tl.float32)
    b_P = eye * inv_mu

    base_kk = (i_b * T * H + i_h) * K
    p_k = k_ptr + base_kk + o_k
    p_a = alpha_ptr + base_kk + o_k
    p_o = omega_ptr + base_kk + o_k
    p_r = r_ptr + (i_b * T) * H + i_h
    # ckpt base for (i_b, i_h): [B, H, NC, K, K]
    ckpt_base = (i_b * H + i_h) * NC * K * K

    for t in range(0, T):
        # store checkpoint at block entries t = 0, C, 2C, ...  (P entering the block)
        if (t % C) == 0:
            i_c = t // C
            p_ck = ckpt_ptr + ckpt_base + i_c * K * K + o_k[:, None] * K + o_k[None, :]
            tl.store(p_ck, b_P.to(ckpt_ptr.dtype.element_ty), mask=mask_kk)

        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0.0).to(tl.float32)
        b_om = tl.load(p_o, mask=mask_k, other=0.0).to(tl.float32)
        b_r = tl.load(p_r).to(tl.float32)
        r_eff = b_r * r_eff_scale

        # predict
        b_P = b_a[:, None] * b_P * b_a[None, :]
        b_P = b_P + eye * b_om[None, :]
        b_P = tl.where(mask_kk, b_P, 0.0)
        # gain
        b_Pk = tl.sum(b_P * b_k[None, :], axis=1)
        b_Pk = tl.where(mask_k, b_Pk, 0.0)
        denom = r_eff + tl.sum(b_k * b_Pk, axis=0)
        b_kap = tl.where(mask_k, b_Pk / denom, 0.0)
        # Joseph
        b_J = eye - b_kap[:, None] * b_k[None, :]
        b_JP = tl.dot(b_J, b_P, input_precision="ieee")
        b_JPJt = tl.dot(b_JP, tl.trans(b_J), input_precision="ieee")
        b_P = b_JPJt + r_eff * (b_kap[:, None] * b_kap[None, :])
        if SYMMETRIZE:
            b_P = 0.5 * (b_P + tl.trans(b_P))
        b_P = tl.where(mask_kk, b_P, 0.0)

        p_k += H * K
        p_a += H * K
        p_o += H * K
        p_r += H


# =============================================================================
# DE-SPILL (GD-A/GD-C, 2026-07-30): the reverse kernel below was register-spill-
# limited (~1384 spills, ~808 ms/layer @ B32/T2046). GD0 probe finding: ptxas
# pinned n_regs=32 (targeting occupancy) even though grid = B*H <= 132 SMs means
# occupancy is ALREADY 1 CTA/SM -- so there was NO occupancy to trade, yet
# everything beyond 32 regs/thread spilled. The fix is FREE: force a higher
# per-thread register budget via `maxnreg` (bounded by maxnreg*num_warps*32 <=
# 65536 regs/SM). This barely changes the static spill COUNT (~1384 -> ~1380) but
# lifts n_regs 32 -> up to 255, so ~8x more of the working set stays resident and
# the kernel runs ~2x faster.
#
#   * num_warps=16 + maxnreg=128 is BIT-IDENTICAL to the old kernel (delta=0;
#     maxnreg changes only register ALLOCATION, not compute order) -- a safe
#     ~1.7x. Lowering num_warps allows a bigger maxnreg (fewer threads => more
#     regs/thread) and is faster still, at a ~2e-6 grad delta (fp32 reduction
#     reassociation across a different warp layout).
#   * The OPTIMAL (num_warps, maxnreg) depends on the GRID SIZE (B*H), NOT on any
#     kernel constexpr: measured B8/grid=32 -> w8/255 (161 ms, 2.1x); B32/grid=128
#     -> w4/255 (351 ms, 2.3x). Because B is invisible to @triton.autotune (it is
#     not a kernel arg) AND autotune's L2-flushed do_bench mis-ranks these close
#     configs run-to-run (non-deterministic 354 vs 408 ms), we DROP autotune and
#     pick the config DETERMINISTICALLY from the grid size in the launcher
#     (_resolve_bwd_config). num_stages had no measurable effect (kept =1).
#   * Env override GAIN_RECURRENT_BWD_WARPS (+ optional GAIN_RECURRENT_BWD_MAXNREG)
#     forces a single config (WARPS=16 with NO maxnreg reproduces the pre-de-spill
#     kernel exactly; used by the bit-identical A/B gate). The two forward kernels
#     do not spill badly and stay at num_warps=4.
# =============================================================================
def _resolve_bwd_config(grid_size: int):
    """Return (num_warps, maxnreg) for the reverse kernel. Env override wins;
    else pick deterministically from the grid size (see the note above)."""
    w_env = os.environ.get("GAIN_RECURRENT_BWD_WARPS")
    if w_env is not None:
        m_env = os.environ.get("GAIN_RECURRENT_BWD_MAXNREG")
        return int(w_env), (int(m_env) if m_env is not None else None)
    # deterministic default: fewer warps (bigger maxnreg) win once the grid nearly
    # saturates the SMs; more warps hide latency when the grid is small.
    if grid_size >= 64:
        return 4, 255
    return 8, 255


# Split-backward (Kernel A / Kernel B) launch configs. Same de-spill mechanism as
# the fat kernel: grid = B*H <= ~132 SMs => 1 CTA/SM, no occupancy to trade, so lift
# the per-thread register budget with maxnreg to keep the working set resident. These
# defaults are grid-tuned by scripts/analyses/bench_gain_split_config.py; env overrides
# (GAIN_RECURRENT_REFWD_WARPS/_MAXNREG, GAIN_RECURRENT_REVSCAN_WARPS/_MAXNREG) force a
# single config for the config sweep / A-B measurement.
def _resolve_reforward_config(grid_size: int):
    """(num_warps, maxnreg) for Kernel A (lean re-forward + store). It carries one
    K x K tile + the two ieee Joseph dots (the ckpt-fwd working set)."""
    w_env = os.environ.get("GAIN_RECURRENT_REFWD_WARPS")
    if w_env is not None:
        m_env = os.environ.get("GAIN_RECURRENT_REFWD_MAXNREG")
        return int(w_env), (int(m_env) if m_env is not None else None)
    return 4, 255


def _resolve_revscan_config(grid_size: int):
    """(num_warps, maxnreg) for Kernel B (dot-free reverse-scan adjoint). It carries
    one K x K adjoint tile + K-vectors -- no dots, so it never spills badly. Sweep
    (bench_gain_split.py @grid=128) picks num_warps=8: the serial per-token adjoint
    chain is latency-bound, so more warps hide it (w8 ~12.5 ms vs w4 ~26 ms)."""
    w_env = os.environ.get("GAIN_RECURRENT_REVSCAN_WARPS")
    if w_env is not None:
        m_env = os.environ.get("GAIN_RECURRENT_REVSCAN_MAXNREG")
        return int(w_env), (int(m_env) if m_env is not None else None)
    return 8, 255


@triton.jit(do_not_specialize=["T"])
def gain_recurrent_bwd_kernel(
    k_ptr,           # [B, T, H, K]
    alpha_ptr,       # [B, T, H, K]
    omega_ptr,       # [B, T, H, K]
    r_ptr,           # [B, T, H]
    dkappa_ptr,      # [B, T, H, K]  (incoming grad of the gain output)
    ckpt_ptr,        # [B, H, nc, K, K]
    dk_ptr,          # [B, T, H, K]  (out)
    dalpha_ptr,      # [B, T, H, K]  (out)
    domega_ptr,      # [B, T, H, K]  (out)
    dr_ptr,          # [B, T, H]     (out)
    dP0_ptr,         # [B, H, K, K]  (out; d(P_0), reduced to dmu on host)
    pp_scratch_ptr,  # [B*H, C, K, K]  per-program P_prev scratch (P_hat recomputed)
    r_eff_scale: tl.constexpr,
    T: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    C: tl.constexpr,
    NC: tl.constexpr,
    SYMMETRIZE: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    mask_kk = mask_k[:, None] & mask_k[None, :]
    eye = tl.where(o_k[:, None] == o_k[None, :], 1.0, 0.0).to(tl.float32)

    base_kk = (i_b * T * H + i_h) * K              # token-0 base for [B,T,H,K]
    base_r = (i_b * T) * H + i_h                    # token-0 base for [B,T,H]
    ckpt_base = (i_b * H + i_h) * NC * K * K
    scr_base = i_bh * C * K * K                     # per-program scratch base

    # carried covariance adjoint s = d(P_t), init 0 (no final-P grad in training).
    b_s = tl.zeros([BK, BK], dtype=tl.float32)

    # Walk C-blocks in reverse.
    for i_c in range(NC - 1, -1, -1):
        t0 = i_c * C
        # block length (last block may be short)
        # ---- (a) load checkpoint P entering the block ----
        p_ck = ckpt_ptr + ckpt_base + i_c * K * K + o_k[:, None] * K + o_k[None, :]
        b_P = tl.load(p_ck, mask=mask_kk, other=0.0).to(tl.float32)

        # ---- (b) RE-FORWARD the block, store P_prev per token (P_hat is cheaply
        #      recomputed from P_prev in the reverse pass -- one predict, no dot -- so
        #      we halve scratch and one live tile). ----
        for j in range(0, C):
            t = t0 + j
            if t < T:
                p_k = k_ptr + base_kk + t * H * K + o_k
                p_a = alpha_ptr + base_kk + t * H * K + o_k
                p_o = omega_ptr + base_kk + t * H * K + o_k
                b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
                b_a = tl.load(p_a, mask=mask_k, other=0.0).to(tl.float32)
                b_om = tl.load(p_o, mask=mask_k, other=0.0).to(tl.float32)
                b_r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
                r_eff = b_r * r_eff_scale

                # store P_prev (= P entering this token, pre-predict)
                p_pp = pp_scratch_ptr + scr_base + j * K * K + o_k[:, None] * K + o_k[None, :]
                tl.store(p_pp, b_P.to(pp_scratch_ptr.dtype.element_ty), mask=mask_kk)

                # predict (EXACT forward op order)
                b_Ph = b_a[:, None] * b_P * b_a[None, :]
                b_Ph = b_Ph + eye * b_om[None, :]
                b_Ph = tl.where(mask_kk, b_Ph, 0.0)

                # gain
                b_Pk = tl.sum(b_Ph * b_k[None, :], axis=1)
                b_Pk = tl.where(mask_k, b_Pk, 0.0)
                denom = r_eff + tl.sum(b_k * b_Pk, axis=0)
                b_kap = tl.where(mask_k, b_Pk / denom, 0.0)
                # Joseph -> advance b_P to P_t (same ieee + symmetrize as forward)
                b_J = eye - b_kap[:, None] * b_k[None, :]
                b_JP = tl.dot(b_J, b_Ph, input_precision="ieee")
                b_JPJt = tl.dot(b_JP, tl.trans(b_J), input_precision="ieee")
                b_P = b_JPJt + r_eff * (b_kap[:, None] * b_kap[None, :])
                if SYMMETRIZE:
                    b_P = 0.5 * (b_P + tl.trans(b_P))
                b_P = tl.where(mask_kk, b_P, 0.0)

        # ---- (c) reverse-scan the adjoint over the block ----
        for j in range(C - 1, -1, -1):
            t = t0 + j
            if t < T:
                p_k = k_ptr + base_kk + t * H * K + o_k
                p_a = alpha_ptr + base_kk + t * H * K + o_k
                p_o = omega_ptr + base_kk + t * H * K + o_k
                b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
                b_a = tl.load(p_a, mask=mask_k, other=0.0).to(tl.float32)
                b_om = tl.load(p_o, mask=mask_k, other=0.0).to(tl.float32)
                b_r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
                r_eff = b_r * r_eff_scale
                b_dkap = tl.load(dkappa_ptr + base_kk + t * H * K + o_k,
                                 mask=mask_k, other=0.0).to(tl.float32)

                # reload P_prev from scratch; recompute P_hat (one predict, no dot).
                p_pp = pp_scratch_ptr + scr_base + j * K * K + o_k[:, None] * K + o_k[None, :]
                b_Pprev = tl.load(p_pp, mask=mask_kk, other=0.0).to(tl.float32)
                b_Ph = b_a[:, None] * b_Pprev * b_a[None, :] + eye * b_om[None, :]
                b_Ph = tl.where(mask_kk, b_Ph, 0.0)

                # recompute gain intermediates (cheap; avoids storing them)
                b_Pk = tl.sum(b_Ph * b_k[None, :], axis=1)
                b_Pk = tl.where(mask_k, b_Pk, 0.0)
                denom = r_eff + tl.sum(b_k * b_Pk, axis=0)
                b_kap = tl.where(mask_k, b_Pk / denom, 0.0)

                # symmetrize carried adjoint s = d(P_t)
                b_s = 0.5 * (b_s + tl.trans(b_s))

                # ---- Joseph VJP (matvec / rank-structured -- NO K x K dots) ----
                # gJ appears ONLY via gJ k and gJ^T kappa; both collapse to matvecs
                # because gJ = 2 s J Phat and J = I - kappa k^T, s & Phat symmetric:
                #   gJ k       = 2 s J Pk        (Pk = Phat k)
                #   gJ^T kappa = 2 Phat J^T (s kappa)
                # and gPhat_joseph = J^T s J = s - k (s k)^T - (s k) k^T + (k^T s k) k k^T
                # (all rank-structured updates to s -- no matmul).
                b_sk = tl.sum(b_s * b_kap[None, :], axis=1)                # s kappa  [BK]
                ksk = tl.sum(b_kap * b_sk, axis=0)                          # kappa^T s kappa
                # dkap += -gJ k = -2 s (J Pk) ; J Pk = Pk - kappa (k.Pk)
                b_JPk = b_Pk - b_kap * tl.sum(b_k * b_Pk, axis=0)
                b_dkap = b_dkap - 2.0 * tl.sum(b_s * b_JPk[None, :], axis=1)
                # dk += -gJ^T kappa = -2 Phat (J^T sk) ; J^T sk = sk - k (kappa.sk)
                b_Jtsk = b_sk - b_k * ksk
                b_dk = -2.0 * tl.sum(b_Ph * b_Jtsk[None, :], axis=1)
                # r_eff kappa kappa^T term
                b_dkap = b_dkap + 2.0 * r_eff * b_sk
                dr_eff = ksk
                # gPhat (Joseph, rank-structured), then extended by the gain VJP below.
                b_gPhat = (b_s
                           - b_k[:, None] * b_sk[None, :]
                           - b_sk[:, None] * b_k[None, :]
                           + ksk * (b_k[:, None] * b_k[None, :]))

                # ---- gain VJP ----
                inv_denom = 1.0 / denom
                b_dPk = b_dkap * inv_denom
                ddenom = -tl.sum(b_dkap * b_kap, axis=0) * inv_denom
                dr_eff = dr_eff + ddenom
                b_dk = b_dk + ddenom * (2.0 * b_Pk)
                b_gPhat = b_gPhat + b_dPk[:, None] * b_k[None, :]          # dPk k^T
                b_gPhat = b_gPhat + ddenom * (b_k[:, None] * b_k[None, :]) # denom's k k^T
                b_dk = b_dk + tl.sum(b_Ph * b_dPk[None, :], axis=1)        # Phat dPk (sym)
                b_dk = tl.where(mask_k, b_dk, 0.0)

                # ---- predict VJP ----
                b_gPhat = tl.where(mask_kk, b_gPhat, 0.0)
                b_dom = tl.sum(tl.where(o_k[:, None] == o_k[None, :], b_gPhat, 0.0), axis=0)
                b_dom = tl.where(mask_k, b_dom, 0.0)
                # da: TWO terms. row_i = sum_j gPhat_ij P_ij a_j ; col_j = sum_i gPhat_ij P_ij a_i
                gPP = b_gPhat * b_Pprev
                b_da = tl.sum(gPP * b_a[None, :], axis=1) + tl.sum(gPP * b_a[:, None], axis=0)
                b_da = tl.where(mask_k, b_da, 0.0)
                # dP_prev -> carried s for the next (earlier) reverse step
                b_s = b_a[:, None] * b_gPhat * b_a[None, :]
                b_s = tl.where(mask_kk, b_s, 0.0)

                # ---- emit ----
                tl.store(dk_ptr + base_kk + t * H * K + o_k, b_dk.to(dk_ptr.dtype.element_ty), mask=mask_k)
                tl.store(dalpha_ptr + base_kk + t * H * K + o_k, b_da.to(dalpha_ptr.dtype.element_ty), mask=mask_k)
                tl.store(domega_ptr + base_kk + t * H * K + o_k, b_dom.to(domega_ptr.dtype.element_ty), mask=mask_k)
                tl.store(dr_ptr + base_r + t * H, (dr_eff * r_eff_scale).to(dr_ptr.dtype.element_ty))

                # at the very first token, s now holds d(P_0) -> accumulate for dmu
                if t == 0:
                    p_dP0 = dP0_ptr + (i_b * H + i_h) * K * K + o_k[:, None] * K + o_k[None, :]
                    tl.store(p_dP0, b_s.to(dP0_ptr.dtype.element_ty), mask=mask_kk)


# =============================================================================
# PROFILING VARIANT (measurement only -- default/production path is BYTE-UNCHANGED)
# -----------------------------------------------------------------------------
# `gain_recurrent_bwd_kernel_prof` is an EXACT copy of `gain_recurrent_bwd_kernel`
# with two extra compile-time (constexpr) gates used ONLY by the profiling script
# scripts/analyses/profile_gain_bwd.py. It is DEAD CODE for the production launcher
# (`gain_recurrent_bwd` never references it), so the production kernel above and its
# emitted PTX are untouched. The gates:
#   * PROF_REFONLY : keep the per-block RE-FORWARD (pass (b)) but SKIP the reverse-
#       scan adjoint (pass (c)) entirely. Grads are NOT written (garbage) -- this is
#       a STOPWATCH to isolate re-forward cost. reverse-scan_ms = full - refonly.
#   * PROF_TF32    : flip the re-forward's two Joseph tl.dot()s from ieee -> tf32 to
#       quantify how much the full-fp32 (ieee) accumulation costs. Reverse-scan (c)
#       uses tl.sum matvecs only (NO tl.dot), so the only dots in this kernel are the
#       two in the re-forward -- this A/B measures exactly their cost.
# Both default False => the branches are compile-time dead-code-eliminated, so with
# both False this kernel compiles to the SAME body as the production kernel.
# =============================================================================
@triton.jit(do_not_specialize=["T"])
def gain_recurrent_bwd_kernel_prof(
    k_ptr,
    alpha_ptr,
    omega_ptr,
    r_ptr,
    dkappa_ptr,
    ckpt_ptr,
    dk_ptr,
    dalpha_ptr,
    domega_ptr,
    dr_ptr,
    dP0_ptr,
    pp_scratch_ptr,
    r_eff_scale: tl.constexpr,
    T: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    C: tl.constexpr,
    NC: tl.constexpr,
    SYMMETRIZE: tl.constexpr,
    PROF_REFONLY: tl.constexpr = False,
    PROF_TF32: tl.constexpr = False,
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    mask_kk = mask_k[:, None] & mask_k[None, :]
    eye = tl.where(o_k[:, None] == o_k[None, :], 1.0, 0.0).to(tl.float32)

    base_kk = (i_b * T * H + i_h) * K
    base_r = (i_b * T) * H + i_h
    ckpt_base = (i_b * H + i_h) * NC * K * K
    scr_base = i_bh * C * K * K

    b_s = tl.zeros([BK, BK], dtype=tl.float32)

    for i_c in range(NC - 1, -1, -1):
        t0 = i_c * C
        p_ck = ckpt_ptr + ckpt_base + i_c * K * K + o_k[:, None] * K + o_k[None, :]
        b_P = tl.load(p_ck, mask=mask_kk, other=0.0).to(tl.float32)

        # ---- (b) RE-FORWARD ----
        for j in range(0, C):
            t = t0 + j
            if t < T:
                p_k = k_ptr + base_kk + t * H * K + o_k
                p_a = alpha_ptr + base_kk + t * H * K + o_k
                p_o = omega_ptr + base_kk + t * H * K + o_k
                b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
                b_a = tl.load(p_a, mask=mask_k, other=0.0).to(tl.float32)
                b_om = tl.load(p_o, mask=mask_k, other=0.0).to(tl.float32)
                b_r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
                r_eff = b_r * r_eff_scale

                p_pp = pp_scratch_ptr + scr_base + j * K * K + o_k[:, None] * K + o_k[None, :]
                tl.store(p_pp, b_P.to(pp_scratch_ptr.dtype.element_ty), mask=mask_kk)

                b_Ph = b_a[:, None] * b_P * b_a[None, :]
                b_Ph = b_Ph + eye * b_om[None, :]
                b_Ph = tl.where(mask_kk, b_Ph, 0.0)

                b_Pk = tl.sum(b_Ph * b_k[None, :], axis=1)
                b_Pk = tl.where(mask_k, b_Pk, 0.0)
                denom = r_eff + tl.sum(b_k * b_Pk, axis=0)
                b_kap = tl.where(mask_k, b_Pk / denom, 0.0)
                b_J = eye - b_kap[:, None] * b_k[None, :]
                if PROF_TF32:
                    b_JP = tl.dot(b_J, b_Ph, input_precision="tf32")
                    b_JPJt = tl.dot(b_JP, tl.trans(b_J), input_precision="tf32")
                else:
                    b_JP = tl.dot(b_J, b_Ph, input_precision="ieee")
                    b_JPJt = tl.dot(b_JP, tl.trans(b_J), input_precision="ieee")
                b_P = b_JPJt + r_eff * (b_kap[:, None] * b_kap[None, :])
                if SYMMETRIZE:
                    b_P = 0.5 * (b_P + tl.trans(b_P))
                b_P = tl.where(mask_kk, b_P, 0.0)

        # ---- (c) reverse-scan the adjoint over the block ----
        if not PROF_REFONLY:
            for j in range(C - 1, -1, -1):
                t = t0 + j
                if t < T:
                    p_k = k_ptr + base_kk + t * H * K + o_k
                    p_a = alpha_ptr + base_kk + t * H * K + o_k
                    p_o = omega_ptr + base_kk + t * H * K + o_k
                    b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
                    b_a = tl.load(p_a, mask=mask_k, other=0.0).to(tl.float32)
                    b_om = tl.load(p_o, mask=mask_k, other=0.0).to(tl.float32)
                    b_r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
                    r_eff = b_r * r_eff_scale
                    b_dkap = tl.load(dkappa_ptr + base_kk + t * H * K + o_k,
                                     mask=mask_k, other=0.0).to(tl.float32)

                    p_pp = pp_scratch_ptr + scr_base + j * K * K + o_k[:, None] * K + o_k[None, :]
                    b_Pprev = tl.load(p_pp, mask=mask_kk, other=0.0).to(tl.float32)
                    b_Ph = b_a[:, None] * b_Pprev * b_a[None, :] + eye * b_om[None, :]
                    b_Ph = tl.where(mask_kk, b_Ph, 0.0)

                    b_Pk = tl.sum(b_Ph * b_k[None, :], axis=1)
                    b_Pk = tl.where(mask_k, b_Pk, 0.0)
                    denom = r_eff + tl.sum(b_k * b_Pk, axis=0)
                    b_kap = tl.where(mask_k, b_Pk / denom, 0.0)

                    b_s = 0.5 * (b_s + tl.trans(b_s))

                    b_sk = tl.sum(b_s * b_kap[None, :], axis=1)
                    ksk = tl.sum(b_kap * b_sk, axis=0)
                    b_JPk = b_Pk - b_kap * tl.sum(b_k * b_Pk, axis=0)
                    b_dkap = b_dkap - 2.0 * tl.sum(b_s * b_JPk[None, :], axis=1)
                    b_Jtsk = b_sk - b_k * ksk
                    b_dk = -2.0 * tl.sum(b_Ph * b_Jtsk[None, :], axis=1)
                    b_dkap = b_dkap + 2.0 * r_eff * b_sk
                    dr_eff = ksk
                    b_gPhat = (b_s
                               - b_k[:, None] * b_sk[None, :]
                               - b_sk[:, None] * b_k[None, :]
                               + ksk * (b_k[:, None] * b_k[None, :]))

                    inv_denom = 1.0 / denom
                    b_dPk = b_dkap * inv_denom
                    ddenom = -tl.sum(b_dkap * b_kap, axis=0) * inv_denom
                    dr_eff = dr_eff + ddenom
                    b_dk = b_dk + ddenom * (2.0 * b_Pk)
                    b_gPhat = b_gPhat + b_dPk[:, None] * b_k[None, :]
                    b_gPhat = b_gPhat + ddenom * (b_k[:, None] * b_k[None, :])
                    b_dk = b_dk + tl.sum(b_Ph * b_dPk[None, :], axis=1)
                    b_dk = tl.where(mask_k, b_dk, 0.0)

                    b_gPhat = tl.where(mask_kk, b_gPhat, 0.0)
                    b_dom = tl.sum(tl.where(o_k[:, None] == o_k[None, :], b_gPhat, 0.0), axis=0)
                    b_dom = tl.where(mask_k, b_dom, 0.0)
                    gPP = b_gPhat * b_Pprev
                    b_da = tl.sum(gPP * b_a[None, :], axis=1) + tl.sum(gPP * b_a[:, None], axis=0)
                    b_da = tl.where(mask_k, b_da, 0.0)
                    b_s = b_a[:, None] * b_gPhat * b_a[None, :]
                    b_s = tl.where(mask_kk, b_s, 0.0)

                    tl.store(dk_ptr + base_kk + t * H * K + o_k, b_dk.to(dk_ptr.dtype.element_ty), mask=mask_k)
                    tl.store(dalpha_ptr + base_kk + t * H * K + o_k, b_da.to(dalpha_ptr.dtype.element_ty), mask=mask_k)
                    tl.store(domega_ptr + base_kk + t * H * K + o_k, b_dom.to(domega_ptr.dtype.element_ty), mask=mask_k)
                    tl.store(dr_ptr + base_r + t * H, (dr_eff * r_eff_scale).to(dr_ptr.dtype.element_ty))

                    if t == 0:
                        p_dP0 = dP0_ptr + (i_b * H + i_h) * K * K + o_k[:, None] * K + o_k[None, :]
                        tl.store(p_dP0, b_s.to(dP0_ptr.dtype.element_ty), mask=mask_kk)


# =============================================================================
# SPLIT BACKWARD (GD-B, 2026-07-30): decouple the re-forward from the reverse scan.
# -----------------------------------------------------------------------------
# Profiling (scripts/analyses/profile_gain_bwd.py, job 1583819) found the fat
# `gain_recurrent_bwd_kernel` above is dominated by its per-block RE-FORWARD (pass b:
# the two ieee Joseph dots J@P / (JP)@J^T): 263 ms of a 345 ms bwd @B32/T2046/H4/K64.
# The SAME ieee dots cost only ~20 ms in the lean `gain_recurrent_ckpt_fwd_kernel`
# -- a ~12.8x register-pressure amplification from running them INSIDE the fat kernel
# (which also carries the K x K adjoint `s` + the reverse-scan working set). The
# reverse-scan itself (pass c) is only ~63 ms; parallel-scanning it addresses <=18%.
#
# The lever is to get the re-forward OUT of the fat kernel by SPLITTING the backward
# into two lean kernels, so neither is register-starved:
#   * Kernel A (`gain_recurrent_reforward_kernel`): a lean re-forward that mirrors the
#     ckpt-fwd kernel EXACTLY (same predict / gain / ieee-Joseph / symmetrize op-order)
#     but STORES the per-token P_prev (the covariance ENTERING each token, pre-predict)
#     to a full HBM scratch pp_full[B*H, T, K, K] fp32 (~4.3 GB at K=64). It carries
#     only the one K x K covariance tile => the ieee dots run at ckpt-fwd speed (~20 ms),
#     not 263 ms. This REPLACES both the checkpoint kernel and the fat kernel's pass (b)
#     (we store every P_prev instead of block checkpoints, so no re-forward-from-ckpt).
#   * Kernel B (`gain_recurrent_revscan_kernel`): the fat kernel's reverse-scan (pass c)
#     with pass (b) removed -- it READS P_prev from pp_full and recomputes P_hat with a
#     single elementwise predict (NO dot). Dot-free matvec adjoint => no register
#     amplification. Same reverse-scan math + op-order + (i_c, j) iteration as the fat
#     kernel, so grads are element-wise identical (bit-identical when the warp layout
#     matches; else <=~fp32 roundoff from tl.sum reassociation).
#
# The stored P_prev EXACTLY equals what the fat kernel re-forwards, so this is a
# PERF-only refactor. The old fused kernel is kept behind GAIN_RECURRENT_BWD=fused as
# the A/B oracle. Estimate ~20 ms (A) + ~50-63 ms (B) => ~4-4.5x from 345 ms. K=64 only
# (the pp_full scratch is 4x at K=128 -- verify the memory budget there before enabling).
# =============================================================================
@triton.jit(do_not_specialize=["T"])
def gain_recurrent_reforward_kernel(
    k_ptr,           # [B, T, H, K]
    alpha_ptr,       # [B, T, H, K]  (RAW alpha)
    omega_ptr,       # [B, T, H, K]
    r_ptr,           # [B, T, H]
    pp_full_ptr,     # [B*H, T, K, K]  P_prev per token (P entering each token, pre-predict)
    inv_mu_ptr,      # [H]
    r_eff_scale: tl.constexpr,
    T: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    SYMMETRIZE: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    mask_kk = mask_k[:, None] & mask_k[None, :]
    eye = tl.where(o_k[:, None] == o_k[None, :], 1.0, 0.0).to(tl.float32)

    inv_mu = tl.load(inv_mu_ptr + i_h).to(tl.float32)
    b_P = eye * inv_mu

    base_kk = (i_b * T * H + i_h) * K
    p_k = k_ptr + base_kk + o_k
    p_a = alpha_ptr + base_kk + o_k
    p_o = omega_ptr + base_kk + o_k
    p_r = r_ptr + (i_b * T) * H + i_h
    # pp_full base for this program: [B*H, T, K, K]; advance by K*K each token.
    p_pp = pp_full_ptr + i_bh * T * K * K + o_k[:, None] * K + o_k[None, :]

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0.0).to(tl.float32)
        b_om = tl.load(p_o, mask=mask_k, other=0.0).to(tl.float32)
        b_r = tl.load(p_r).to(tl.float32)
        r_eff = b_r * r_eff_scale

        # store P_prev (= P entering this token, pre-predict) -- EXACTLY what the fat
        # kernel's pass (b) stashes, but for every token to full HBM.
        tl.store(p_pp, b_P.to(pp_full_ptr.dtype.element_ty), mask=mask_kk)

        # predict (EXACT forward op order)
        b_Ph = b_a[:, None] * b_P * b_a[None, :]
        b_Ph = b_Ph + eye * b_om[None, :]
        b_Ph = tl.where(mask_kk, b_Ph, 0.0)
        # gain
        b_Pk = tl.sum(b_Ph * b_k[None, :], axis=1)
        b_Pk = tl.where(mask_k, b_Pk, 0.0)
        denom = r_eff + tl.sum(b_k * b_Pk, axis=0)
        b_kap = tl.where(mask_k, b_Pk / denom, 0.0)
        # Joseph -> advance b_P to P_t (same ieee + symmetrize as forward)
        b_J = eye - b_kap[:, None] * b_k[None, :]
        b_JP = tl.dot(b_J, b_Ph, input_precision="ieee")
        b_JPJt = tl.dot(b_JP, tl.trans(b_J), input_precision="ieee")
        b_P = b_JPJt + r_eff * (b_kap[:, None] * b_kap[None, :])
        if SYMMETRIZE:
            b_P = 0.5 * (b_P + tl.trans(b_P))
        b_P = tl.where(mask_kk, b_P, 0.0)

        p_k += H * K
        p_a += H * K
        p_o += H * K
        p_r += H
        p_pp += K * K


@triton.jit(do_not_specialize=["T"])
def gain_recurrent_revscan_kernel(
    k_ptr,           # [B, T, H, K]
    alpha_ptr,       # [B, T, H, K]
    omega_ptr,       # [B, T, H, K]
    r_ptr,           # [B, T, H]
    dkappa_ptr,      # [B, T, H, K]
    pp_full_ptr,     # [B*H, T, K, K]  P_prev per token (from Kernel A)
    dk_ptr,          # [B, T, H, K]  (out)
    dalpha_ptr,      # [B, T, H, K]  (out)
    domega_ptr,      # [B, T, H, K]  (out)
    dr_ptr,          # [B, T, H]     (out)
    dP0_ptr,         # [B, H, K, K]  (out; d(P_0))
    r_eff_scale: tl.constexpr,
    T: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    C: tl.constexpr,
    NC: tl.constexpr,
    SYMMETRIZE: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    mask_kk = mask_k[:, None] & mask_k[None, :]
    eye = tl.where(o_k[:, None] == o_k[None, :], 1.0, 0.0).to(tl.float32)

    base_kk = (i_b * T * H + i_h) * K
    base_r = (i_b * T) * H + i_h
    pp_base = i_bh * T * K * K

    # carried covariance adjoint s = d(P_t), init 0 (no final-P grad in training).
    b_s = tl.zeros([BK, BK], dtype=tl.float32)

    # Walk tokens in REVERSE (nested (i_c, j) exactly mirrors the fat kernel's pass (c),
    # so b_s accumulates in the identical order). No re-forward, no dots -- P_prev is READ
    # from pp_full[t] (Kernel A) and P_hat is one elementwise predict.
    for i_c in range(NC - 1, -1, -1):
        t0 = i_c * C
        for j in range(C - 1, -1, -1):
            t = t0 + j
            if t < T:
                p_k = k_ptr + base_kk + t * H * K + o_k
                p_a = alpha_ptr + base_kk + t * H * K + o_k
                p_o = omega_ptr + base_kk + t * H * K + o_k
                b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
                b_a = tl.load(p_a, mask=mask_k, other=0.0).to(tl.float32)
                b_om = tl.load(p_o, mask=mask_k, other=0.0).to(tl.float32)
                b_r = tl.load(r_ptr + base_r + t * H).to(tl.float32)
                r_eff = b_r * r_eff_scale
                b_dkap = tl.load(dkappa_ptr + base_kk + t * H * K + o_k,
                                 mask=mask_k, other=0.0).to(tl.float32)

                # read P_prev from HBM (Kernel A); recompute P_hat (one predict, no dot).
                p_pp = pp_full_ptr + pp_base + t * K * K + o_k[:, None] * K + o_k[None, :]
                b_Pprev = tl.load(p_pp, mask=mask_kk, other=0.0).to(tl.float32)
                b_Ph = b_a[:, None] * b_Pprev * b_a[None, :] + eye * b_om[None, :]
                b_Ph = tl.where(mask_kk, b_Ph, 0.0)

                # recompute gain intermediates (cheap; avoids storing them)
                b_Pk = tl.sum(b_Ph * b_k[None, :], axis=1)
                b_Pk = tl.where(mask_k, b_Pk, 0.0)
                denom = r_eff + tl.sum(b_k * b_Pk, axis=0)
                b_kap = tl.where(mask_k, b_Pk / denom, 0.0)

                # symmetrize carried adjoint s = d(P_t)
                b_s = 0.5 * (b_s + tl.trans(b_s))

                # ---- Joseph VJP (matvec / rank-structured -- NO K x K dots) ----
                b_sk = tl.sum(b_s * b_kap[None, :], axis=1)                # s kappa  [BK]
                ksk = tl.sum(b_kap * b_sk, axis=0)                          # kappa^T s kappa
                b_JPk = b_Pk - b_kap * tl.sum(b_k * b_Pk, axis=0)
                b_dkap = b_dkap - 2.0 * tl.sum(b_s * b_JPk[None, :], axis=1)
                b_Jtsk = b_sk - b_k * ksk
                b_dk = -2.0 * tl.sum(b_Ph * b_Jtsk[None, :], axis=1)
                b_dkap = b_dkap + 2.0 * r_eff * b_sk
                dr_eff = ksk
                b_gPhat = (b_s
                           - b_k[:, None] * b_sk[None, :]
                           - b_sk[:, None] * b_k[None, :]
                           + ksk * (b_k[:, None] * b_k[None, :]))

                # ---- gain VJP ----
                inv_denom = 1.0 / denom
                b_dPk = b_dkap * inv_denom
                ddenom = -tl.sum(b_dkap * b_kap, axis=0) * inv_denom
                dr_eff = dr_eff + ddenom
                b_dk = b_dk + ddenom * (2.0 * b_Pk)
                b_gPhat = b_gPhat + b_dPk[:, None] * b_k[None, :]          # dPk k^T
                b_gPhat = b_gPhat + ddenom * (b_k[:, None] * b_k[None, :]) # denom's k k^T
                b_dk = b_dk + tl.sum(b_Ph * b_dPk[None, :], axis=1)        # Phat dPk (sym)
                b_dk = tl.where(mask_k, b_dk, 0.0)

                # ---- predict VJP ----
                b_gPhat = tl.where(mask_kk, b_gPhat, 0.0)
                b_dom = tl.sum(tl.where(o_k[:, None] == o_k[None, :], b_gPhat, 0.0), axis=0)
                b_dom = tl.where(mask_k, b_dom, 0.0)
                gPP = b_gPhat * b_Pprev
                b_da = tl.sum(gPP * b_a[None, :], axis=1) + tl.sum(gPP * b_a[:, None], axis=0)
                b_da = tl.where(mask_k, b_da, 0.0)
                b_s = b_a[:, None] * b_gPhat * b_a[None, :]
                b_s = tl.where(mask_kk, b_s, 0.0)

                # ---- emit ----
                tl.store(dk_ptr + base_kk + t * H * K + o_k, b_dk.to(dk_ptr.dtype.element_ty), mask=mask_k)
                tl.store(dalpha_ptr + base_kk + t * H * K + o_k, b_da.to(dalpha_ptr.dtype.element_ty), mask=mask_k)
                tl.store(domega_ptr + base_kk + t * H * K + o_k, b_dom.to(domega_ptr.dtype.element_ty), mask=mask_k)
                tl.store(dr_ptr + base_r + t * H, (dr_eff * r_eff_scale).to(dr_ptr.dtype.element_ty))

                if t == 0:
                    p_dP0 = dP0_ptr + (i_b * H + i_h) * K * K + o_k[:, None] * K + o_k[None, :]
                    tl.store(p_dP0, b_s.to(dP0_ptr.dtype.element_ty), mask=mask_kk)


# =============================================================================
# ORCHESTRATION
# =============================================================================

def _resolve_r(r, B, T, H, device, dtype):
    """Broadcast r (scalar / [H] / [B,T,H]) to a contiguous [B, T, H] fp32 tensor."""
    if torch.is_tensor(r):
        r = r.to(device=device, dtype=dtype)
        if r.dim() == 0:
            r = r.expand(B, T, H)
        elif r.dim() == 1:
            if r.shape[0] != H:
                raise ValueError(f"r as [H] must have H={H}, got {tuple(r.shape)}")
            r = r.view(1, 1, H).expand(B, T, H)
        elif r.dim() == 3:
            if tuple(r.shape) != (B, T, H):
                raise ValueError(f"r as [B,T,H] must be {(B, T, H)}, got {tuple(r.shape)}")
        else:
            raise ValueError(f"r must be scalar / [H] / [B,T,H], got {tuple(r.shape)}")
    else:
        r = torch.full((B, T, H), float(r), device=device, dtype=dtype)
    return r.contiguous()


def _resolve_inv_mu(mu, H, device, dtype):
    """Broadcast mu (scalar / [H]) to a contiguous [H] fp32 tensor of 1/mu."""
    if torch.is_tensor(mu):
        mu = mu.to(device=device, dtype=dtype)
        if mu.dim() == 0:
            mu = mu.expand(H)
        elif mu.dim() == 1:
            if mu.shape[0] != H:
                raise ValueError(f"mu as [H] must have H={H}, got {tuple(mu.shape)}")
        else:
            raise ValueError(f"mu must be scalar / [H], got {tuple(mu.shape)}")
        inv_mu = 1.0 / mu
    else:
        inv_mu = torch.full((H,), 1.0 / float(mu), device=device, dtype=dtype)
    return inv_mu.contiguous()


def _gain_recurrent_pytorch(k, alpha, omega, r, mu=1.0, dk_calibration=True):
    r"""Differentiable pure-PyTorch kappa-only recurrence -- the Phase-A backward ORACLE.

    Runs the EXACT per-token dense-covariance gain recurrence and returns ONLY the gain
    ``kappa [B, T, H, K]`` (no memory S, no read q/v). The gain math is byte-for-byte the
    gain sub-computation inside :func:`lit_gpt.kla_ops.exact_naive.naive_recurrent_exact_kla`
    (Joseph covariance update, ``joseph=True`` code default):

        predict:  P_hat = diag(alpha) P diag(alpha) + diag(omega)
        gain:     kappa = P_hat k / (r_eff + k^T P_hat k)
        update:   P = (I - kappa k^T) P_hat (I - kappa k^T)^T + r_eff kappa kappa^T

    with ``r_eff = r / d_k`` (d_k = K) when ``dk_calibration`` else ``r_eff = r``, and
    ``P_0 = mu^-1 I``. Because ``naive_recurrent_exact_kla`` returns the layer OUTPUT ``o``
    (which folds q, v and the memory recurrence), it cannot supply grads of the GAIN alone;
    this kappa-only twin is the differentiable oracle we autograd through for the fused
    kernel's Phase-A backward. It is anchored to the frozen reference in the gate (its kappa
    drives ``_memory_read`` to the frozen ``naive_recurrent_exact_kla`` output, and passes a
    fp64 ``gradcheck``).

    Everything runs at the promoted precision of the inputs (fp32 min, fp64 preserved), so it
    doubles as the fp64 gradcheck target. NO scale / l2norm here -- kappa is a pure function
    of (k, alpha, omega, r, mu) (unlike the memory kernel).

    Args mirror :func:`gain_recurrent`: ``k, alpha, omega`` are ``[B,T,H,K]`` (alpha RAW),
    ``r`` scalar / ``[H]`` / ``[B,T,H]``, ``mu`` scalar / ``[H]``.
    """
    B, T, H, K = k.shape
    k, alpha, omega = (_hp(x) for x in (k, alpha, omega))
    dk = float(K) if dk_calibration else 1.0

    # r -> [B,T,H]
    if torch.is_tensor(r):
        r = _hp(r)
        if r.dim() == 0:
            r = r.expand(B, T, H)
        elif r.dim() == 1:
            if r.shape[0] != H:
                raise ValueError(f"r as [H] must have H={H}, got {tuple(r.shape)}")
            r = r.view(1, 1, H).expand(B, T, H)
        elif r.dim() != 3:
            raise ValueError(f"r must be scalar / [H] / [B,T,H], got {tuple(r.shape)}")
    else:
        r = k.new_full((B, T, H), float(r))

    # P_0 = mu^-1 I (per head)
    if torch.is_tensor(mu):
        inv_mu = (1.0 / _hp(mu)).view(1, H, 1, 1)
    else:
        inv_mu = 1.0 / float(mu)
    eye_k = torch.eye(K, dtype=k.dtype, device=k.device)
    P = (eye_k.view(1, 1, K, K) * (inv_mu if not torch.is_tensor(inv_mu) else inv_mu)).expand(B, H, K, K).clone()

    kappas = []
    for t in range(T):
        a, om, kt = alpha[:, t], omega[:, t], k[:, t]                 # [B,H,K]
        r_eff = r[:, t] / dk                                          # [B,H]

        # --- predict ---
        P = a[..., :, None] * P * a[..., None, :]                     # diag(a) P diag(a)
        P = P + torch.diag_embed(om)                                  # + diag(omega)

        # --- gain (exact anisotropic Kalman gain) ---
        Pk = (P * kt[..., None, :]).sum(-1)                           # P_hat k   [B,H,K]
        denom = r_eff + (kt * Pk).sum(-1)                             # r_eff + k^T P_hat k  [B,H]
        kappa = Pk / denom[..., None]                                 # [B,H,K]
        kappas.append(kappa)

        # --- covariance update (Joseph form, code default) ---
        J = eye_k - kappa[..., :, None] * kt[..., None, :]           # I - kappa k^T  [B,H,K,K]
        JP = torch.einsum("bhij,bhjl->bhil", J, P)
        P = torch.einsum("bhil,bhml->bhim", JP, J) + r_eff[..., None, None] * (
            kappa[..., :, None] * kappa[..., None, :]
        )

    return torch.stack(kappas, dim=1)                                 # [B,T,H,K]


def gain_recurrent_fwd(k, alpha, omega, r, mu=1.0, dk_calibration=True, store_pprev=False):
    """Launch the fused forward gain kernel. Returns kappa [B, T, H, K] fp32.

    k, alpha, omega: [B, T, H, K] (alpha is RAW, not log). r: scalar / [H] / [B,T,H].
    mu: scalar / [H]. K must be 64 (v1). All compute is fp32.

    If ``store_pprev`` (Lever 1), the kernel ALSO stashes the per-token P_prev (the covariance
    entering each token, pre-predict) to a ``[B*H, T, K, K]`` fp32 HBM buffer (~4.3 GB @T2046)
    and returns ``(kappa, pp_full)``. This is EXACTLY what the bwd re-forward (Kernel A) would
    recompute, so the backward can read it and drop Kernel A. Off by default (inference is lean).
    """
    assert k.shape == alpha.shape == omega.shape, "k, alpha, omega must share [B,T,H,K]"
    B, T, H, K = k.shape
    assert K == 64, (
        f"gain_recurrent v1 supports K=64 only (got K={K}); K=128 (real training "
        f"dim) needs a tiled/shared-mem P layout -- see the module docstring."
    )
    assert k.is_cuda, "gain_recurrent requires CUDA tensors"

    device = k.device
    # fp32 compute throughout; cast inputs to contiguous fp32.
    k32 = k.to(torch.float32).contiguous()
    a32 = alpha.to(torch.float32).contiguous()
    o32 = omega.to(torch.float32).contiguous()
    r32 = _resolve_r(r, B, T, H, device, torch.float32)
    inv_mu = _resolve_inv_mu(mu, H, device, torch.float32)

    kappa = torch.empty(B, T, H, K, device=device, dtype=torch.float32)
    if store_pprev:
        pp_full = torch.empty(B * H, T, K, K, device=device, dtype=torch.float32)
        pp_arg = pp_full
    else:
        pp_full = None
        pp_arg = kappa   # unused dummy pointer (STORE_PPREV=False dead-codes the store)

    r_eff_scale = (1.0 / float(K)) if dk_calibration else 1.0
    BK = triton.next_power_of_2(K)
    # P is analytically symmetric; re-symmetrizing each step halves the fp32
    # antisymmetric rounding drift (default ON). Toggle off via env for A/B.
    symmetrize = os.environ.get("GAIN_RECURRENT_SYMMETRIZE", "1") != "0"
    grid = (B * H,)
    gain_recurrent_fwd_kernel[grid](
        k32,
        a32,
        o32,
        r32,
        kappa,
        inv_mu,
        pp_arg,
        r_eff_scale=r_eff_scale,
        T=T,
        H=H,
        K=K,
        BK=BK,
        SYMMETRIZE=symmetrize,
        STORE_PPREV=store_pprev,
        num_warps=4,
    )
    if store_pprev:
        return kappa, pp_full
    return kappa


def gain_recurrent_bwd(k, alpha, omega, r, dkappa, mu=1.0, dk_calibration=True,
                       C: int | None = None, pp_full=None):
    """Phase-B Triton reverse covariance-adjoint backward for the gain kernel.

    Given the incoming gain-output grad ``dkappa [B,T,H,K]`` and the SAME inputs the
    forward saw, returns ``(dk, dalpha, domega, dr, dP0)`` where

      * ``dk, dalpha, domega`` : ``[B,T,H,K]`` fp32,
      * ``dr``               : ``[B,T,H]`` fp32 (host-side selects it out if r was scalar),
      * ``dP0``              : ``[B,H,K,K]`` fp32 = d(P_0); ``dmu`` is reduced from it on host.

    Backends (env ``GAIN_RECURRENT_BWD``, default ``split``):
      * ``split`` / ``phaseB`` (DEFAULT): run Kernel A (re-forward + store all ``P_prev`` to a fresh
        TRANSIENT ``[B*H,T,K,K]`` HBM scratch, freed on return) then Kernel B. Only one ~4.3 GB
        (K=64) buffer live per layer -> SAFE AT ANY DEPTH. This is also the A/B oracle for stash.
      * ``stash`` (Lever 1, OPT-IN): the FORWARD already stashed every per-token ``P_prev`` (passed
        in via ``pp_full`` from :func:`gain_recurrent_fwd(..., store_pprev=True)`), so the backward
        runs ONLY Kernel B (the dot-free reverse-scan); Kernel A is ELIDED. Faster, but the stash is
        DEPTH-MULTIPLIED (N_layers coexist). If ``pp_full`` is None (forward did not stash),
        transparently falls back to running Kernel A first (== the split path).
      * ``fused``: the OLD single fat reverse kernel (checkpoint-forward + per-block re-forward
        + reverse-scan). All three are deep-decay-SAFE (the Joseph map is contractive). K=64 only.
    """
    assert k.shape == alpha.shape == omega.shape, "k, alpha, omega must share [B,T,H,K]"
    B, T, H, K = k.shape
    assert K == 64, f"gain_recurrent_bwd supports K=64 only (got K={K})."
    assert k.is_cuda, "gain_recurrent_bwd requires CUDA tensors"

    device = k.device
    k32 = k.to(torch.float32).contiguous()
    a32 = alpha.to(torch.float32).contiguous()
    o32 = omega.to(torch.float32).contiguous()
    r32 = _resolve_r(r, B, T, H, device, torch.float32)
    inv_mu = _resolve_inv_mu(mu, H, device, torch.float32)
    dkap32 = dkappa.to(torch.float32).contiguous()

    r_eff_scale = (1.0 / float(K)) if dk_calibration else 1.0
    BK = triton.next_power_of_2(K)
    symmetrize = os.environ.get("GAIN_RECURRENT_SYMMETRIZE", "1") != "0"
    if C is None:
        C = int(os.environ.get("GAIN_RECURRENT_CKPT_C", "32"))
    C = max(1, min(int(C), T if T > 0 else 1))
    NC = (T + C - 1) // C
    grid = (B * H,)

    dk = torch.empty(B, T, H, K, device=device, dtype=torch.float32)
    da = torch.empty(B, T, H, K, device=device, dtype=torch.float32)
    do_om = torch.empty(B, T, H, K, device=device, dtype=torch.float32)
    dr = torch.empty(B, T, H, device=device, dtype=torch.float32)
    dP0 = torch.zeros(B, H, K, K, device=device, dtype=torch.float32)

    mode = os.environ.get("GAIN_RECURRENT_BWD", _DEFAULT_BWD).lower()
    if mode == "fused":
        # ---- OLD fused path (kept as the A/B oracle): ckpt-fwd + fat reverse kernel.
        ckpt = torch.empty(B, H, NC, K, K, device=device, dtype=torch.float32)
        pp_scr = torch.empty(B * H, C, K, K, device=device, dtype=torch.float32)
        gain_recurrent_ckpt_fwd_kernel[grid](
            k32, a32, o32, r32, ckpt, inv_mu,
            r_eff_scale=r_eff_scale, T=T, H=H, K=K, BK=BK, C=C, NC=NC,
            SYMMETRIZE=symmetrize, num_warps=4,
        )
        # De-spilled fat reverse kernel. (num_warps, maxnreg) chosen deterministically
        # from the grid size (see _resolve_bwd_config); GAIN_RECURRENT_BWD_WARPS
        # (+ GAIN_RECURRENT_BWD_MAXNREG) forces a single config for the A/B gate.
        num_warps, maxnreg = _resolve_bwd_config(B * H)
        extra = {} if maxnreg is None else {"maxnreg": maxnreg}
        gain_recurrent_bwd_kernel[grid](
            k32, a32, o32, r32, dkap32, ckpt,
            dk, da, do_om, dr, dP0, pp_scr,
            r_eff_scale=r_eff_scale, T=T, H=H, K=K, BK=BK, C=C, NC=NC,
            SYMMETRIZE=symmetrize,
            num_warps=num_warps, num_stages=1, **extra,
        )
        return dk, da, do_om, dr, dP0

    # ---- Reverse-scan path (stash / split / phaseB). Kernel B (lean dot-free reverse-scan)
    # reads per-token P_prev from pp_full [B*H, T, K, K] fp32 (~4.3 GB @ K=64/T2046). In the
    # STASH default the forward already produced pp_full (Kernel A is skipped -- Lever 1); when
    # pp_full is None (split/phaseB, or a forward that did not stash) we run Kernel A to make it.
    use_stash = (mode == "stash") and (pp_full is not None)
    if not use_stash:
        # split / phaseB (or stash-fallback): re-forward + store all P_prev with Kernel A.
        pp_full = torch.empty(B * H, T, K, K, device=device, dtype=torch.float32)
        w_a, m_a = _resolve_reforward_config(B * H)
        extra_a = {} if m_a is None else {"maxnreg": m_a}
        gain_recurrent_reforward_kernel[grid](
            k32, a32, o32, r32, pp_full, inv_mu,
            r_eff_scale=r_eff_scale, T=T, H=H, K=K, BK=BK,
            SYMMETRIZE=symmetrize, num_warps=w_a, num_stages=1, **extra_a,
        )
    w_b, m_b = _resolve_revscan_config(B * H)
    extra_b = {} if m_b is None else {"maxnreg": m_b}
    gain_recurrent_revscan_kernel[grid](
        k32, a32, o32, r32, dkap32, pp_full,
        dk, da, do_om, dr, dP0,
        r_eff_scale=r_eff_scale, T=T, H=H, K=K, BK=BK, C=C, NC=NC,
        SYMMETRIZE=symmetrize, num_warps=w_b, num_stages=1, **extra_b,
    )
    return dk, da, do_om, dr, dP0


class _GainRecurrentFunction(torch.autograd.Function):
    """Autograd wrapper: forward runs the fused Triton kernel; backward is Phase A (G2).

    Forward runs the fast fused Triton gain kernel. Backward (Phase A, milestone G2)
    recomputes the gain through the DIFFERENTIABLE pure-PyTorch kappa-only recurrence
    (:func:`_gain_recurrent_pytorch`) under ``torch.enable_grad`` and returns
    ``torch.autograd.grad`` -- the exact grad oracle for d{k, alpha, omega, r, mu}. This is
    slow at long T (a full O(T) PyTorch recompute) but correct; Phase B (a Triton reverse
    covariance-adjoint) is the later speed milestone.


    Why not autograd ``naive_recurrent_exact_kla``? That frozen reference returns the layer
    OUTPUT ``o`` (folding q, v and the memory recurrence), not the gain -- so it cannot give
    grads of kappa alone. ``_gain_recurrent_pytorch`` is the kappa-only differentiable twin
    (identical gain math), anchored to the frozen reference in the gate.

    kappa has NO scale / l2norm (unlike the memory kernel), so this is simpler: kappa is a
    pure function of (k, alpha, omega, r, mu).
    """

    @staticmethod
    def forward(ctx, k, alpha, omega, r, mu, dk_calibration):
        # Lever 1 (STASH-FROM-FORWARD): under the "stash" backend, have the forward ALSO write
        # every per-token P_prev to HBM so the backward can drop the re-forward (Kernel A). Only
        # do so when a grad is actually needed (any input requires grad) -- inside an
        # autograd.Function.forward grad mode is off, so torch.is_grad_enabled() is unreliable;
        # ctx.needs_input_grad is the correct signal. Inference (no grad) stays lean (no stash).
        mode = os.environ.get("GAIN_RECURRENT_BWD", _DEFAULT_BWD).lower()
        stash = (mode == "stash") and any(ctx.needs_input_grad)
        if stash:
            kappa, pp_full = gain_recurrent_fwd(
                k, alpha, omega, r, mu=mu, dk_calibration=dk_calibration, store_pprev=True)
        else:
            kappa = gain_recurrent_fwd(k, alpha, omega, r, mu=mu, dk_calibration=dk_calibration)
            pp_full = None
        # Save inputs for the recompute-in-PyTorch backward. r and mu may be tensors OR
        # python scalars; save_for_backward only accepts tensors, so stash tensors there
        # and non-tensor scalars on ctx. Track which are tensors to align the returned grads.
        r_is_tensor = torch.is_tensor(r)
        mu_is_tensor = torch.is_tensor(mu)
        saved = [k, alpha, omega]
        if r_is_tensor:
            saved.append(r)
        if mu_is_tensor:
            saved.append(mu)
        ctx.save_for_backward(*saved)
        ctx.r_is_tensor = r_is_tensor
        ctx.mu_is_tensor = mu_is_tensor
        ctx.r_scalar = None if r_is_tensor else r
        ctx.mu_scalar = None if mu_is_tensor else mu
        ctx.dk_calibration = dk_calibration
        # The forward's P_prev stash (an intermediate, not an input/output) -- held on ctx until
        # backward consumes it, then freed. ~4.3 GB @ K=64/T2046 (the Lever-1 memory tradeoff).
        ctx.pp_full = pp_full
        return kappa

    @staticmethod
    def backward(ctx, dkappa):
        saved = list(ctx.saved_tensors)
        k, alpha, omega = saved[0], saved[1], saved[2]
        idx = 3
        if ctx.r_is_tensor:
            r_saved = saved[idx]
            idx += 1
        else:
            r_saved = ctx.r_scalar
        if ctx.mu_is_tensor:
            mu_saved = saved[idx]
            idx += 1
        else:
            mu_saved = ctx.mu_scalar

        # Default (stash/split/fused): fast Triton reverse covariance-adjoint. Phase A (the
        # trusted O(T) PyTorch recompute) is kept as an env-toggled oracle/fallback:
        #   GAIN_RECURRENT_BWD=phaseA -> use the recompute path (the gate's reference).
        # In the "stash" default the forward already produced ctx.pp_full, so gain_recurrent_bwd
        # skips Kernel A and runs only the reverse-scan (Kernel B).
        backend = os.environ.get("GAIN_RECURRENT_BWD", _DEFAULT_BWD).lower()
        if backend != "phasea" and dkappa.is_cuda and k.shape[-1] == 64:
            pp_full = getattr(ctx, "pp_full", None)
            dk, da, do_om, dr_full, dP0 = gain_recurrent_bwd(
                k, alpha, omega, r_saved, dkappa, mu=mu_saved,
                dk_calibration=ctx.dk_calibration, pp_full=pp_full,
            )
            ctx.pp_full = None   # free the ~4.3 GB stash promptly
            dk = dk.to(k.dtype)
            da = da.to(alpha.dtype)
            do_om = do_om.to(omega.dtype)
            if ctx.r_is_tensor:
                # reduce dr_full [B,T,H] back to r_saved's shape (scalar/[H]/[B,T,H]).
                dr = _reduce_to_shape(dr_full, r_saved).to(r_saved.dtype)
            else:
                dr = None
            if ctx.mu_is_tensor:
                # dmu[h] = sum_b (-tr(dP0[b,h]) / mu[h]^2).  P_0 = mu^-1 I.
                B, H = dP0.shape[0], dP0.shape[1]
                tr_dP0 = torch.diagonal(dP0, dim1=-2, dim2=-1).sum(-1)   # [B,H]
                mu_h = _hp(mu_saved).reshape(H)
                dmu_full = (-tr_dP0 / (mu_h.view(1, H) ** 2)).sum(0)     # [H]
                dmu = _reduce_to_shape(dmu_full, mu_saved).to(mu_saved.dtype)
            else:
                dmu = None
            return dk, da, do_om, dr, dmu, None

        # Phase A -- recompute kappa through the DIFFERENTIABLE kappa-only recurrence. Free the
        # forward's stash first: if the forward ran under "stash" but the backward is on phaseA,
        # the (unused) ~4.3 GB pp_full would otherwise linger across this slowest path.
        ctx.pp_full = None
        return _gain_phase_a_backward(ctx, dkappa, k, alpha, omega, r_saved, mu_saved)


def _reduce_to_shape(grad, ref):
    """Sum-reduce a [B,T,H]-broadcast grad back to ref's shape (scalar/[H]/[B,T,H]),
    or a [H]-broadcast grad back to scalar/[H]. Mirrors autograd's broadcast-VJP."""
    if not torch.is_tensor(ref):
        return grad.sum()
    if ref.dim() == 0:
        return grad.sum()
    if grad.dim() == 3 and ref.dim() == 1:      # [B,T,H] -> [H]
        return grad.sum(dim=(0, 1))
    if grad.dim() == 1 and ref.dim() == 1:      # [H] -> [H]
        return grad
    if tuple(grad.shape) == tuple(ref.shape):
        return grad
    # generic: sum leading dims until shapes match
    while grad.dim() > ref.dim():
        grad = grad.sum(0)
    return grad


def _gain_phase_a_backward(ctx, dkappa, k, alpha, omega, r_saved, mu_saved):
    """Phase-A backward: exact grads via autograd through the pure-PyTorch kappa recurrence.
    Slow (O(T) recompute) but the trusted correctness oracle; kept as fallback + gate ref."""
    with torch.enable_grad():
        k2 = k.detach().clone().requires_grad_(True)
        a2 = alpha.detach().clone().requires_grad_(True)
        o2 = omega.detach().clone().requires_grad_(True)
        inputs = [k2, a2, o2]

        if ctx.r_is_tensor:
            r2 = r_saved.detach().clone().requires_grad_(True)
            inputs.append(r2)
        else:
            r2 = r_saved  # python scalar -> no grad edge

        if ctx.mu_is_tensor:
            mu2 = mu_saved.detach().clone().requires_grad_(True)
            inputs.append(mu2)
        else:
            mu2 = mu_saved  # python scalar -> no grad edge

        kappa2 = _gain_recurrent_pytorch(
            k2, a2, o2, r2, mu=mu2, dk_calibration=ctx.dk_calibration
        )

        grads = torch.autograd.grad(
            outputs=kappa2,
            inputs=inputs,
            grad_outputs=dkappa,
            allow_unused=True,
        )

    # Align grads to forward's signature: (k, alpha, omega, r, mu, dk_calibration).
    def _match(grad, ref):
        if grad is None:
            return None
        return grad.to(ref.dtype)

    dk = _match(grads[0], k)
    da = _match(grads[1], alpha)
    do_om = _match(grads[2], omega)
    gi = 3
    if ctx.r_is_tensor:
        dr = _match(grads[gi], r_saved)
        gi += 1
    else:
        dr = None  # scalar r -> no grad
    if ctx.mu_is_tensor:
        dmu = _match(grads[gi], mu_saved)
        gi += 1
    else:
        dmu = None  # scalar mu -> no grad

    return dk, da, do_om, dr, dmu, None  # None for dk_calibration


def gain_recurrent(k, alpha, omega, r, mu=1.0, dk_calibration=True):
    r"""Fused Triton forward for the dense-covariance Kalman gain kappa.

    Signature mirrors :func:`lit_gpt.kla_ops.exact_scan.exact_kla_gains_scan`.

    Args:
        k, alpha, omega : [B, T, H, K]  (K=64 in v1). ``alpha`` is RAW alpha
            (transition D=diag(alpha)), NOT log-alpha.
        r  : observation noise -- scalar / [H] / [B, T, H].
        mu : information prior -- scalar / [H]. ``P_0 = mu^-1 I``.
        dk_calibration : if True, ``r_eff = r / d_k`` (d_k = K); else ``r_eff = r``.

    Returns:
        kappa [B, T, H, K] fp32 -- the exact anisotropic Kalman gain.

    Differentiable: backward defaults to the fast Triton reverse covariance-adjoint recurrence
    (:func:`gain_recurrent_bwd`), returning the exact grads d{k, alpha, omega, r, mu} (r/mu grads
    only when they are tensors). Backends via ``GAIN_RECURRENT_BWD``:
      * ``split`` (DEFAULT): the backward re-forwards into a TRANSIENT pp_full then reverse-scans --
        only one ~4.3 GB (K=64) buffer live at a time, SAFE AT ANY DEPTH.
      * ``stash`` (Lever 1, OPT-IN): the FORWARD stashes pp_full so the backward runs only the
        reverse-scan (Kernel A elided) -- faster, but the stash is DEPTH-MULTIPLIED (N_layers x
        stash_GB coexist at the fwd/bwd boundary in standard autograd). Enable only when that fits
        the memory budget (noisy_mqar 2-layer K=64 = 8.6 GB fits; deep / K=128 -> keep split).
      * ``fused`` the old fat kernel; ``phaseA`` the trusted O(T) PyTorch recompute (the oracle).
    No scale / l2norm on kappa, so grads are wrt the raw (k, alpha, omega, r, mu).
    """
    return _GainRecurrentFunction.apply(k, alpha, omega, r, mu, dk_calibration)


__all__ = [
    "gain_recurrent",
    "gain_recurrent_fwd",
    "gain_recurrent_fwd_kernel",
    "gain_recurrent_bwd",
    "gain_recurrent_bwd_kernel",
    "gain_recurrent_ckpt_fwd_kernel",
    "gain_recurrent_bwd_kernel_prof",  # profiling-only variant (see note above)
    "gain_recurrent_reforward_kernel",  # split path: Kernel A (lean re-forward + store)
    "gain_recurrent_revscan_kernel",    # split path: Kernel B (lean dot-free reverse-scan)
    "_resolve_r",
    "_resolve_inv_mu",
    "_resolve_bwd_config",
    "_resolve_reforward_config",
    "_resolve_revscan_config",
    "_gain_recurrent_pytorch",
]
