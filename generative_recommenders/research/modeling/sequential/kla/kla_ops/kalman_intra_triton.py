# Forward Triton intra (WY) stage for `chunk_kalman` (TI1).
#
# This module forks FOUR of KDA's forward intra kernels (chunk_kda.py) to thread
# the INDEPENDENT write key `kappa` on the COLUMN side of every score-matrix
# build and to DROP the scalar write-strength gate `beta` (Kalman absorbs the
# write strength into `kappa`). It replaces the fp64 PyTorch `_intra_pytorch`
# with a KDA-class Triton path.
#
# Frozen ref (fork-by-copy, never edited): lit_gpt/gdn2_ops/chunk_kda.py.
#   - chunk_kda_fwd_kernel_intra_token_parallel  (diag M/Aqk builder)
#   - chunk_kda_fwd_kernel_intra_sub_chunk       (safe_gate diag builder)
#   - chunk_kda_fwd_kernel_inter_solve_fused     (off-diag builder + WY solve)
#   - recompute_w_u_fwd_kda_kernel               (w/u/kappa_fed)
#
# The math (per chunk; A = exp2(gk_cum) within-chunk cumulative decay):
#   k~ = k*A (read), kappa~ = kappa/A (write), q~ = q*A.
#   M       = tril(k~ . kappa~^T, -1)          (asymmetric, two keys)
#   Abar    = (I + M)^{-1}
#   w       = Abar @ (k * exp2(gk_cum))        (read key)
#   u       = Abar @ v                         (plain v, NO beta)
#   kappa_fed = kappa * exp2(gk_last - gk_cum) (write key, NO beta)
#   Aqk     = tril_incl(q~ . kappa~^T) * scale
# The bounded pairwise decay exp2(g_row - g_col) <= 1 is inherited from KDA
# (BC-anchored); the fork NEVER forms 1/A -> deep-decay stable, matching the
# fp64 oracle exactly there.
#
# Every forked kernel copies its @triton.autotune config list VERBATIM from KDA.

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2, gather
from fla.utils import (
    IS_GATHER_SUPPORTED,
    IS_NVIDIA_HOPPER,
    IS_TF32_SUPPORTED,
    autotune_cache_kwargs,
    check_shared_mem,
)

# Same precision knob as chunk_kda.py:2643 (tf32 on H200 for the WY solve dots).
if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')

# Verbatim from chunk_kda.py:3661-3663 (the wy_dqkg backward autotune tiling
# lists). The forked backward wy_dqkg kernel copies KDA's @triton.autotune config
# list verbatim incl. the Hopper WGMMA (BK==32, num_warps==4) exclusion (CI-6/CI-B4).
BK_LIST = [32, 64] if check_shared_mem() else [16, 32]
# RO-5: add BV=32 so the wy_dqkg autotuner can pick GDN-2's tiling. GDN-2's
# wy_dqkg uses BV in [32,64] (chunk_gdn2.py:1239) because the EXTRA kappa operand
# (Kalman/GDN-2 thread a second key on the column side) raises register pressure,
# so a smaller BV often wins over the KDA default [64,128]. Superset keeps 128 as
# an option; autotune picks the fastest. Non-ampere path left as KDA's [16,32].
BV_LIST = [32, 64, 128] if check_shared_mem('ampere') else [16, 32]
NUM_WARPS_WY = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]
# RO-6: Hopper-restrict the bwd_intra num_warps to [1,2,4] (drop 8), mirroring
# GDN-2's NUM_WARPS_INTRA (chunk_gdn2.py:80,:1438). num_warps=8 never wins the
# within-chunk score-matrix VJP on Hopper and only widens the autotune search.
NUM_WARPS_INTRA = [1, 2, 4] if IS_NVIDIA_HOPPER else [1, 2, 4, 8]


# =============================================================================
# (1) Diagonal M/Aqk builder -- FORK of chunk_kda_fwd_kernel_intra_token_parallel
#     row = read k (M) / read q (Aqk); DISTINCT column load -> write kappa; drop beta.
# =============================================================================
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'N'])
def chunk_kalman_fwd_kernel_intra_token_parallel(
    q,
    k,
    kappa,
    g,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    q += bos * H*K
    k += bos * H*K
    kappa += bos * H*K
    g += bos * H*K
    Aqk += bos * H*BT
    Akk += bos * H*BC

    BK: tl.constexpr = triton.next_power_of_2(K)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = (i_hg * BH + o_h) < H
    m_k = o_k < K

    p_q = tl.make_block_ptr(q + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    # [BH, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)  # read key row (NO beta)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        # COLUMN operand -> WRITE key kappa (was read key k in KDA).
        p_kappaj = tl.make_block_ptr(kappa + j * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        p_gj = tl.make_block_ptr(g + j * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        # [BH, BK]
        b_kappaj = tl.load(p_kappaj, boundary_check=(0, 1)).to(tl.float32)
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        b_kappagj = b_kappaj * exp2(b_g - b_gj)

        b_kappagj = tl.where(m_k[None, :], b_kappagj, 0.0)
        # [BH]
        b_Aqk = tl.sum(b_q * b_kappagj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kappagj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

        tl.store(Aqk + i_t * H*BT + (i_hg * BH + o_h) * BT + j % BT, b_Aqk.to(Aqk.dtype.element_ty), mask=m_h)
        tl.store(Akk + i_t * H*BC + (i_hg * BH + o_h) * BC + j - i_ts, b_Akk.to(Akk.dtype.element_ty), mask=m_h)


def kalman_fwd_intra_token_parallel(q, k, kappa, gk, Aqk, Akk, scale,
                                    cu_seqlens=None, chunk_size=64, sub_chunk_size=16):
    B, T, H, K = q.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    def grid(meta): return (B * T, triton.cdiv(H, meta['BH']))
    chunk_kalman_fwd_kernel_intra_token_parallel[grid](
        q=q, k=k, kappa=kappa, g=gk, Aqk=Aqk, Akk=Akk, scale=scale,
        cu_seqlens=cu_seqlens, N=N, T=T, H=H, K=K, BT=BT, BC=BC,
    )
    return Aqk, Akk


# =============================================================================
# (2) safe_gate diagonal builder -- FORK of chunk_kda_fwd_kernel_intra_sub_chunk
#     one key was loaded ONCE as both row+col -> add a SEPARATE kappa column load;
#     drop beta.
# =============================================================================
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kalman_fwd_kernel_intra_sub_chunk(
    q,
    k,
    kappa,
    g,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    kappa = kappa + (bos * H + i_h) * K
    g = g + (bos * H + i_h) * K
    Aqk = Aqk + (bos * H + i_h) * BT
    Akk = Akk + (bos * H + i_h) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_kappa = tl.make_block_ptr(kappa, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))          # read key (row)
    b_kappa = tl.load(p_kappa, boundary_check=(0, 1))  # write key (col)
    b_g = tl.load(p_g, boundary_check=(0, 1))

    if USE_GATHER:
        b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
    else:
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H*K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    b_gm = (b_g - b_gn).to(tl.float32)

    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

    # column operand -> WRITE key kappa (was read key k in KDA).
    b_kappagt = tl.trans(b_kappa * b_gk)

    b_Aqk = tl.dot(b_q * b_gq, b_kappagt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kappagt)  # drop beta (was * b_beta[:, None])

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (H*BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    # forward substitution (symmetry-agnostic; inverts diagonal M block) -- REUSE
    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * H*BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# (3) off-diagonal M/Aqk builder + WY block-solve
#     -- FORK of chunk_kda_fwd_kernel_inter_solve_fused
#     transposed COLUMN loads b_kgt -> write kappa (separate loads for col
#     sub-chunks 0,1,2); rows stay read k / read q; drop beta row-scale.
#     REUSE the block-solve arithmetic verbatim.
# =============================================================================
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'STORE_M': lambda args: args['Mraw'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kalman_fwd_kernel_inter_solve_fused(
    q,
    k,
    kappa,
    g,
    Aqk,
    Akkd,
    Akk,
    Mraw,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
    STORE_M: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    kappa += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC
    if STORE_M:
        Mraw += (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    ################################################################################
    # off-diagonal blocks (rows=read k / read q; columns=write kappa)
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        # sub-chunk 0 is only ever a COLUMN -> load the WRITE key kappa.
        p_kappa0 = tl.make_block_ptr(kappa, (T, K), (H*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_kappa0 = tl.load(p_kappa0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_kappa1 = tl.make_block_ptr(kappa, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            # [BC, BK]
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)          # row (Aqk)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)          # row (Akk)
            b_kappa1 = tl.load(p_kappa1, boundary_check=(0, 1)).to(tl.float32)  # col (write)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * H*K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]  column -> write kappa (sub-chunk 0)
            b_kgt = tl.trans(b_kappa0 * exp2(b_gn1[None, :] - b_g0))
            # [BC, BC]
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_k2 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_kappa2 = tl.make_block_ptr(kappa, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_g2 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                # [BC, BK]
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)          # row
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)          # row
                b_kappa2 = tl.load(p_kappa2, boundary_check=(0, 1)).to(tl.float32)  # col
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                # [BK]
                b_gn2 = tl.load(g + i_tc2 * H*K + o_k, mask=m_k, other=0).to(tl.float32)
                # [BC, BK]
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                # [BK, BC]  column -> write kappa (sub-chunk 0)
                b_kgt = tl.trans(b_kappa0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                # [BK, BC]  column -> write kappa (sub-chunk 1)
                b_kgt = tl.trans(b_kappa1 * exp2(b_gn2[None, :] - b_g1))
                # [BC, BC]
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_k3 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_g3 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    # [BC, BK]
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)  # row
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)  # row
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    # [BK]
                    b_gn3 = tl.load(g + i_tc3 * H*K + o_k, mask=m_k, other=0).to(tl.float32)
                    # [BC, BK]
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    # [BK, BC]  column -> write kappa (sub-chunk 0)
                    b_kgt = tl.trans(b_kappa0 * exp2(b_gn3[None, :] - b_g0))
                    # [BC, BC]
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]  column -> write kappa (sub-chunk 1)
                    b_kgt = tl.trans(b_kappa1 * exp2(b_gn3[None, :] - b_g1))
                    # [BC, BC]
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]  column -> write kappa (sub-chunk 2)
                    b_kgt = tl.trans(b_kappa2 * exp2(b_gn3[None, :] - b_g2))
                    # [BC, BC]
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

    ################################################################################
    # save off-diagonal Aqk blocks and prepare Akk (drop beta row-scale)
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    # DEBUG (STORE_M): dump the RAW strictly-lower M (diag blocks from Akkd +
    # off-diag register blocks) BEFORE the WY solve overwrites Akk with the
    # inverse. Only valid for USE_SAFE_GATE=False (Akkd holds raw diag M there;
    # in safe_gate it holds the pre-inverted diagonal). Enables a conditioning-
    # free element-wise M gate vs the fp64 oracle (plan Stage A).
    if STORE_M:
        # diag blocks come from Akkd (torch.empty -> uninitialized UPPER triangle);
        # mask strictly-lower before the dump so the debug buffer is clean.
        m_low = o_i[:, None] > o_i[None, :]
        b_M00 = tl.where(m_low, b_Ai00, 0.0)
        b_M11 = tl.where(m_low, b_Ai11, 0.0)
        b_M22 = tl.where(m_low, b_Ai22, 0.0)
        b_M33 = tl.where(m_low, b_Ai33, 0.0)
        p_M00 = tl.make_block_ptr(Mraw, (T, BT), (H*BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
        p_M11 = tl.make_block_ptr(Mraw, (T, BT), (H*BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
        p_M22 = tl.make_block_ptr(Mraw, (T, BT), (H*BT, 1), (i_tc2, 2*BC), (BC, BC), (1, 0))
        p_M33 = tl.make_block_ptr(Mraw, (T, BT), (H*BT, 1), (i_tc3, 3*BC), (BC, BC), (1, 0))
        p_M10 = tl.make_block_ptr(Mraw, (T, BT), (H*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        p_M20 = tl.make_block_ptr(Mraw, (T, BT), (H*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_M21 = tl.make_block_ptr(Mraw, (T, BT), (H*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        p_M30 = tl.make_block_ptr(Mraw, (T, BT), (H*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_M31 = tl.make_block_ptr(Mraw, (T, BT), (H*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_M32 = tl.make_block_ptr(Mraw, (T, BT), (H*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
        tl.store(p_M00, b_M00.to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M11, b_M11.to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M22, b_M22.to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M33, b_M33.to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M10, b_Akk10.to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M20, b_Akk20.to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M21, b_Akk21.to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M30, b_Akk30.to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M31, b_Akk31.to(Mraw.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_M32, b_Akk32.to(Mraw.dtype.element_ty), boundary_check=(0, 1))

    ################################################################################
    # forward substitution on diagonals (REUSE, symmetry-agnostic)
    ################################################################################
    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2*BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2*BC + 2, min(3*BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
            b_a22 = tl.where(o_i < i - 2*BC, b_a22, 0.)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2*BC)[:, None], b_a22, b_Ai22)
        for i in range(3*BC + 2, min(4*BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
            b_a33 = tl.where(o_i < i - 3*BC, b_a33, 0.)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3*BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals (REUSE)
    ################################################################################
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    ################################################################################
    # store full Akk_inv to Akk (REUSE)
    ################################################################################
    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, 2*BC), (BC, BC), (1, 0))
    p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 3*BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# (4) recompute w/u/kappa_fed -- FORK of recompute_w_u_fwd_kda_kernel
#     load TWO keys: w = Abar@(k*exp2(gk_cum)) [read]; u = Abar@v [drop beta];
#     kappa_fed = kappa*exp2(gk_last-gk_cum) [write, no beta].
#     RO-4: optional STORE_QG path emits qg = q*exp2(gk_cum) in-kernel (mirror
#     GDN-2's recompute_w_u_fwd_gdn2_kernel :765-772), so the backward no longer
#     needs a separate PyTorch exp2 sweep. qg is None on the forward (not needed).
# =============================================================================
@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kalman_kernel(
    q,
    k,
    kappa,
    qg,
    kg,
    v,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_u = tl.dot(b_A, b_v)  # u = Abar @ v (drop beta)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))          # read key

        p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb = b_k * exp2(b_gk)  # (drop beta) w = Abar @ (k * exp2(gk_cum))

        if STORE_QG:
            # qg = q * exp2(gk_cum) (mirror GDN-2 :765-772); dhu consumes this.
            p_q = tl.make_block_ptr(q + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_qg = tl.make_block_ptr(qg + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))

        # kappa_fed = kappa * exp2(gk_last - gk_cum)  (write key, no beta)
        last_idx = min(i_t * BT + BT, T) - 1
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.).to(tl.float32)
        p_kappa = tl.make_block_ptr(kappa + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_kappa = tl.load(p_kappa, boundary_check=(0, 1))
        b_kg = b_kappa * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
        p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def kalman_recompute_w_u_fwd(k, kappa, v, A, gk, q=None, cu_seqlens=None, chunk_indices=None):
    """Recompute (w, u, kappa_fed[, qg]) from the solved Abar (=A) and gated inputs.

    Mirrors GDN-2's recompute_w_u_fwd_gdn2: given the cached WY inverse Abar, rebuild
    only the cheap w/u/kappa_fed (and optionally qg) WITHOUT re-running the WY solve.
    When ``q`` is provided, the STORE_QG kernel path also emits qg = q*exp2(gk_cum)
    and the return is (w, u, kappa_fed, qg); otherwise (w, u, kappa_fed).
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(kappa)
    qg = torch.empty_like(q) if q is not None else None
    recompute_w_u_fwd_kalman_kernel[(NT, B*H)](
        q=q, k=k, kappa=kappa, qg=qg, kg=kg, v=v, w=w, u=u, A=A, gk=gk,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV,
    )
    if q is not None:
        return w, u, kg, qg
    return w, u, kg


# =============================================================================
# Orchestrator: build (w, u, kappa_fed, Aqk, [Abar]) via the forked kernels.
# Mirrors chunk_kda_fwd_intra structure. Operates at native length T (KDA
# handles arbitrary T via boundary checks); the padded-Tp drop-in wrapper is
# `_intra_triton` below.
# =============================================================================
def kalman_fwd_intra(q, k, kappa, v, gk, scale, chunk_size=64, safe_gate=False, return_M=False):
    """Returns (w, u, kappa_fed, Aqk, Abar[, Mraw]).

    q,k,kappa: [B,T,H,K]; v: [B,T,H,V]; gk: [B,T,H,K] base-2 within-chunk cumsum.
    Aqk: [B,T,H,BT] (scale baked in, tril_incl in the used region).
    Abar: [B,T,H,BT] the WY inverse (I+M)^{-1} (lower-tri, unit diag).
    Mraw: [B,T,H,BT] raw strictly-lower score matrix M (debug; safe_gate=False only).
    """
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16

    dt = k.dtype
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=dt)
    # Akk holds Abar after the solve; zero-init (kernels write only lower blocks).
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=dt)
    # fp32 buffer for the diagonal 16x16 M blocks (precision for the solve).
    Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)
    Mraw = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32) if return_M else None

    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)

    # Step 1: diagonal M blocks -> Akkd (fp32); Aqk diagonal sub-blocks.
    if safe_gate:
        grid = (NT, NC, B * H)
        BK = triton.next_power_of_2(K)
        chunk_kalman_fwd_kernel_intra_sub_chunk[grid](
            q=q, k=k, kappa=kappa, g=gk, Aqk=Aqk, Akk=Akkd, scale=scale,
            cu_seqlens=None, chunk_indices=None,
            T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, USE_GATHER=IS_GATHER_SUPPORTED,
        )
    else:
        kalman_fwd_intra_token_parallel(
            q=q, k=k, kappa=kappa, gk=gk, Aqk=Aqk, Akk=Akkd, scale=scale,
            cu_seqlens=None, chunk_size=BT, sub_chunk_size=BC,
        )

    # Step 2: fused off-diagonal M/Aqk + WY block-solve -> Aqk, Akk(=Abar).
    grid = (NT, B * H)
    chunk_kalman_fwd_kernel_inter_solve_fused[grid](
        q=q, k=k, kappa=kappa, g=gk, Aqk=Aqk, Akkd=Akkd, Akk=Akk, Mraw=Mraw, scale=scale,
        cu_seqlens=None, chunk_indices=None,
        T=T, H=H, K=K, BT=BT, BC=BC, USE_SAFE_GATE=safe_gate,
    )

    # Step 3: w / u / kappa_fed.
    w, u, kappa_fed = kalman_recompute_w_u_fwd(
        k=k, kappa=kappa, v=v, A=Akk, gk=gk, cu_seqlens=None, chunk_indices=None,
    )
    if return_M:
        return w, u, kappa_fed, Aqk, Akk, Mraw
    return w, u, kappa_fed, Aqk, Akk


def _intra_triton(q, k, kappa, v, g_cumsum, scale, chunk_size,
                  *, safe_gate=False, kernel_dtype=torch.bfloat16,
                  return_internals=False, return_M=True):
    """Drop-in Triton replacement for `_intra_pytorch`.

    Same signature/return as kalman_chunk._intra_pytorch: pads to a whole number
    of chunks Tp and returns (w, u, kappa_fed, Aqk, Tp, nc, BT). The forked
    Triton kernels run in `kernel_dtype` (bf16 in production; fp32 for the tight
    gate). With return_internals, also returns Abar (the WY inverse) so the caller
    can CACHE it for a recompute-only backward (RO-1); when return_M is also True
    the raw strictly-lower Mraw debug buffer is returned too (for the M/Abar gate).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size

    # PERF (WY-intra padding lever): the forked intra kernels (token_parallel /
    # inter_solve / recompute) AND the reused h/gla kernels all handle a PARTIAL last
    # chunk at native length T via boundary_check + `last_idx = min(i_t*BT+BT, T) - 1`
    # (this is exactly how the frozen chunk_kda runs -- it never pads). The old path
    # padded T->Tp with torch.cat, which materializes full [B,Tp,H,*] copies of q, k,
    # kappa, v AND g_cumsum every call. At T%64!=0 (e.g. T=2046) those copies + the
    # follow-on `.contiguous()`/cast roughly DOUBLE the intra time and add a second
    # pad in `_forward_triton` -> chunk_kalman fwd ~1.5-1.67x chunk_kda (measured;
    # profile_kalman_stages.py). Running at native T recovers KDA parity (~1.05x). The
    # math is unchanged (min(...,T) already reads the true chunk-end decay for kappa_fed
    # and the h-kernel carry). KALMAN_INTRA_PAD=1 restores the torch.cat path (A/B +
    # oracle fallback).
    if os.environ.get("KALMAN_INTRA_PAD", "0") == "1":
        pad = (-T) % BT
        if pad:
            zK = q.new_zeros(B, pad, H, K)
            q = torch.cat([q, zK], 1)
            k = torch.cat([k, zK], 1)
            kappa = torch.cat([kappa, zK], 1)
            v = torch.cat([v, v.new_zeros(B, pad, H, V)], 1)
            # FLAT-continuation pad for g_cumsum (NOT zeros): recompute_w_u_fwd_kalman_kernel
            # bakes kappa_fed = kappa*exp2(gk_last-gk_cum) where gk_last is read at the
            # chunk's LAST slot (last_idx = min(i_t*BT+BT, Tp)-1 = Tp-1 for the partial
            # last chunk). Zero-padding made gk_last=0 => kappa_fed inflated by
            # exp2(-g_cumsum[T-1]) => wrong final_state at T%64!=0. Repeat g_cumsum[:, T-1]
            # so the read hits the real chunk-end cumulative decay (matches the h-kernel
            # carry decay fed from _forward_triton's g_p; the two MUST agree).
            g_cumsum = torch.cat([g_cumsum, g_cumsum[:, -1:].expand(B, pad, H, K)], 1)
        Tp = T + pad
        nc = Tp // BT
    else:
        # NO-PAD: run at native T (KDA-style). Tp == T; nc = number of (possibly
        # partial) chunks. The kernels' min(...,T) + boundary_check handle the tail.
        Tp = T
        nc = triton.cdiv(T, BT)

    qc = q.to(kernel_dtype).contiguous()
    kc = k.to(kernel_dtype).contiguous()
    kpc = kappa.to(kernel_dtype).contiguous()
    vc = v.to(kernel_dtype).contiguous()
    gc = g_cumsum.to(torch.float32).contiguous()

    if return_internals:
        if return_M:
            w, u, kappa_fed, Aqk, Abar, Mraw = kalman_fwd_intra(
                qc, kc, kpc, vc, gc, scale, chunk_size=BT, safe_gate=safe_gate, return_M=True,
            )
            return w, u, kappa_fed, Aqk, Tp, nc, BT, Abar, Mraw
        # RO-1 fast path: build Abar (cache-for-backward) but skip the Mraw buffer.
        w, u, kappa_fed, Aqk, Abar = kalman_fwd_intra(
            qc, kc, kpc, vc, gc, scale, chunk_size=BT, safe_gate=safe_gate, return_M=False,
        )
        return w, u, kappa_fed, Aqk, Tp, nc, BT, Abar

    w, u, kappa_fed, Aqk, Abar = kalman_fwd_intra(
        qc, kc, kpc, vc, gc, scale, chunk_size=BT, safe_gate=safe_gate, return_M=False,
    )
    return w, u, kappa_fed, Aqk, Tp, nc, BT


# =============================================================================
# BACKWARD Triton intra (TI2). Forks the TWO KDA backward intra kernels to emit
# INDEPENDENT dk (read) / dkappa (write) grads and DROP beta. Trusted reference =
# `_backward_hybrid` fp64 autograd in kalman_chunk.py (element-wise gate).
#
# (5) wy_dqkg fork -- FORK of chunk_kda_bwd_kernel_wy_dqkg_fused (chunk_kda.py:3758).
#     Emits inter-chunk h-adjoints + WY-inverse VJP (dM) + w/u VJP. Kalman edits:
#       * UN-MERGE its own dk/dkappa (KDA merges at :3908): the state-write grad
#         `v_new@dh` (·exp2(gn-g)) -> dkappa; the w-read-decay grad `b_dkgb·A`
#         -> dk_read. Two SEPARATE outputs (CI-R2-1).
#       * key routing: read key k in {M rows via w, w=Abar@(k*A)}; write key kappa
#         in {kappa_fed = kappa*exp2(gn-g)}. dg write-decay term routes through
#         kappa: `-b_kappa*b_dkappa` and `sum(kappa*dkappa)` (CI-B3, :3907).
#       * drop beta (CI-R2-2): dv2 no *beta (:3884); drop db (:3885/:3901/:3926);
#         b_gb = A not A*beta (:3890); dg w-term `b_kg*b_dkgb` no *beta (:3907);
#         dA columns no *beta (:3918).
#     Copy the @triton.autotune config list VERBATIM incl. the Hopper WGMMA guard.
# =============================================================================
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BV_LIST
        for num_warps in NUM_WARPS_WY
        for num_stages in [2, 3, 4]
        if not (IS_NVIDIA_HOPPER and BK == 32 and num_warps == 4)
    ],
    key=['BT', 'TRANSPOSE_STATE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kalman_bwd_kernel_wy_dqkg_fused(
    q,
    k,
    kappa,
    v,
    v_new,
    g,
    A,
    h,
    do,
    dh,
    dq,
    dk,       # dk_read (w-read-decay grad; un-merged)
    dkappa,   # dkappa (kappa_fed state-write grad; un-merged)
    dv,       # in: du (u cotangent from dhu)
    dv2,      # out: real dv = Abar^T @ du
    dg,
    dA,       # out: dM (WY-inverse VJP cotangent)
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    kappa += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K*V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K*V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dkappa += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    dA += (bos * H + i_h) * BT

    # (drop beta load)
    p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))          # read key rows
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dkappa = tl.zeros([BT, BK], dtype=tl.float32)   # state-write (kappa_fed) grad
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            p_v_new = tl.make_block_ptr(v_new, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            if TRANSPOSE_STATE:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            # [BT, BV]
            b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            # [BT, BV]
            b_dv = tl.load(p_dv, boundary_check=(0, 1))

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dkappa += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))   # d(kappa_fed) = v_new @ dh
            b_dw += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))
            tl.debug_barrier()  # DO NOT REMOVE THIS LINE!
            if i_k == 0:
                p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

                b_v = tl.load(p_v, boundary_check=(0, 1))

                b_dA += tl.dot(b_dv, tl.trans(b_v))    # u = Abar @ v (drop beta)

                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb                          # dv = Abar^T @ du (drop beta)
                # (drop db from u path)

                tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp                                # A (drop beta)
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dkappa = b_dkappa * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)   # kappa_fed decay

        b_kg = b_k * b_gk_exp                          # k * A (read key)

        b_dw = -b_dw.to(b_A.dtype)
        b_dA += tl.dot(b_dw, tl.trans(b_kg.to(b_A.dtype)))   # w = Abar @ (k*A)

        b_dkgb = tl.dot(b_A, b_dw)                     # Abar^T @ dw : read-key decay grad
        # (drop db from w path)

        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_kappa = tl.make_block_ptr(kappa, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kappa = tl.load(p_kappa, boundary_check=(0, 1))     # write key rows (dg routing)
        b_kdk = b_kappa * b_dkappa                     # route write-decay dg through kappa
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb   # drop *beta on w-term

        b_dk = b_dkgb * b_gb                           # dk_read (read-key decay; un-merged)

        p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dkappa = tl.make_block_ptr(dkappa, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dkappa, b_dkappa.to(p_dkappa.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)                      # drop *beta on dA columns
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)

    p_dA = tl.make_block_ptr(dA, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
    # (drop db store)


def kalman_bwd_wy_dqkg_fused(q, k, kappa, v, v_new, g, A, h, do, dh, dv, scale,
                             cu_seqlens=None, chunk_size=64, chunk_indices=None):
    """Kalman fork of chunk_kda_bwd_wy_dqkg_fused. Returns
    (dq, dk_read, dkappa, dv, dg, dA=dM). dv IN = du (dhu output); dv OUT = real dv.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)       # dk_read
    dkappa = torch.empty_like(k, dtype=torch.float)   # dkappa (kappa_fed part)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    grid = (NT, B * H)
    chunk_kalman_bwd_kernel_wy_dqkg_fused[grid](
        q=q,
        k=k,
        kappa=kappa,
        v=v,
        v_new=v_new,
        g=g,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dkappa=dkappa,
        dv=dv,
        dv2=dv2,
        dg=dg,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        TRANSPOSE_STATE=False,
    )
    return dq, dk, dkappa, dv2, dg, dA


# =============================================================================
# (6) bwd_intra fork -- FORK of chunk_kda_bwd_kernel_intra (chunk_kda.py:2988).
#     The within-chunk score-matrix VJP; PRIMARY dk/dkappa split. Kalman edits:
#       * b_dk2 (dAkk=dM ROWS) -> dk_read ; b_dkt (dAqk+dAkk COLS) -> dkappa. Keep
#         SEPARATE (delete the `b_dk2 += b_dkt` merge, chunk_kda.py:3245).
#       * load the COLUMN forward operand from kappa (was read k): the row-pass
#         col operands (:3063 off-diag, :3113 safe diag, :3122 else) -> kappa.
#       * dg two-key routing (:3243): `b_dk2*b_k - b_dkt*b_kappa` (row keeps read
#         k; col routes through kappa) (CI-B3).
#       * drop beta (CI-R2-2): no db (:3134/:3144); no `b_dk2*=b_b` (:3135); the
#         b_dkt ROW read-key operands stay raw k (drop :3165/:3176/:3215/:3226 beta).
#       * load/add wy's dk_read into b_dk2 and wy's dkappa into b_dkt (:3244 becomes
#         two loads; CI-R2-1 two-tensor threading).
#     Autotune key ["BK","NC","BT"] copied verbatim; no Hopper WGMMA guard needed.
# =============================================================================
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_INTRA
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_kalman_bwd_kernel_intra(
    q,
    k,
    kappa,
    g,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,        # in: dk_read (from wy)
    dk2,       # out: dk_read
    dkappa,    # in: dkappa (from wy)
    dkappa2,   # out: dkappa
    dg,
    dg2,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    kappa += (bos * H + i_h) * K
    g += (bos * H + i_h) * K

    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dkappa += (bos * H + i_h) * K
    dkappa2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dg2 += (bos * H + i_h) * K

    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    # (drop beta load)

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * H*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            # COLUMN operand -> WRITE key kappa (was read k in KDA).
            p_kappa = tl.make_block_ptr(kappa, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_kappa = tl.load(p_kappa, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kappag = b_kappa * exp2(b_gn - b_gk)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            # [BC, BK]
            b_dq2 += tl.dot(b_dAqk, b_kappag)
            b_dk2 += tl.dot(b_dAkk, b_kappag)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * H*BT + i_i * BC
    p_kappaj = kappa + i_ti * H*K + o_k    # COLUMN operand -> kappa (else branch)
    p_gkj = g + i_ti * H*K + o_k

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))          # read key row (i_ti)

    if SAFE_GATE:
        # separate WRITE key kappa column load for the diagonal block.
        p_kappa_diag = tl.make_block_ptr(kappa, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_kappa_diag = tl.load(p_kappa_diag, boundary_check=(0, 1))
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H*K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]

        p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        b_dAqk_diag_qk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_qk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_qk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
        b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.)
        exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.)
        exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.)

        b_kappa_exp_diag_qk = b_kappa_diag * exp_neg_b_g_diag_qk    # column -> kappa
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_kappa_exp_diag_qk) * exp_b_g_diag_qk
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_kappa_exp_diag_qk) * exp_b_g_diag_qk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC]
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            # [BK]  column -> kappa
            b_kappaj = tl.load(p_kappaj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] >= j
            # [BC, BK]
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kappaj[None, :] * b_gqk, 0.)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kappaj[None, :] * b_gqk, 0.)

            p_kappaj += H*K
            p_gkj += H*K

    # (drop b_db = sum(b_dk2*b_k) and b_dk2 *= b_b)

    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(dq2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    # (drop db store)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * H*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            # [BC, BK]  ROW operands: query q, read key k (drop beta)
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k_row = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_q * tl.where(m_j[:, None], b_gkn, 0)
            b_kg_row = b_k_row * tl.where(m_j[:, None], b_gkn, 0)   # drop beta
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dkt += tl.dot(b_dAqk, b_qg)
            b_dkt += tl.dot(b_dAkk, b_kg_row)
        b_dkt *= exp2(b_gn - b_g)
    o_dA = i_ti * H*BT + i_i * BC + o_i
    p_qj = q + i_ti * H*K + o_k
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * H*K + o_k

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H*K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))

        p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H*BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H*BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        b_dAqk_diag_kk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_kk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_kk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.)
        b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.)
        # ensure numerical stability
        b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn, 0.)
        exp_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.)
        exp_neg_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.)

        b_q_exp = b_q * exp_b_g_diag_kk
        b_k_exp = b_k * exp_b_g_diag_kk               # read key ROW (drop beta)

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp) * exp_neg_b_g_diag_kk
        b_dkt += tl.dot(b_dAkk_diag_kk, b_k_exp) * exp_neg_b_g_diag_kk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC,]
            b_dAqk = tl.load(dAqk + o_dA + j * H*BT)
            b_dAkk = tl.load(dAkk + o_dA + j * H*BT)
            # [BK,]  ROW operands: query q, read key k (drop beta)
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kj_row = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kj_row[None, :] * b_gkq, 0.)

            p_qj += H*K
            p_kj += H*K
            p_gkj += H*K

    # WRITE key kappa row at i_ti for the dg column-routing.
    p_kappa_row = tl.make_block_ptr(kappa, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_kappa_row = tl.load(p_kappa_row, boundary_check=(0, 1))

    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(dk2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dkappa = tl.make_block_ptr(dkappa, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dkappa2 = tl.make_block_ptr(dkappa2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg2 = tl.make_block_ptr(dg2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))

    # dg two-key routing (:3243): row term keeps read k; col term routes kappa.
    b_dg2 += b_dk2 * b_k - b_dkt * b_kappa_row + tl.load(p_dg, boundary_check=(0, 1))
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))          # dk_read += wy dk_read
    b_dkt += tl.load(p_dkappa, boundary_check=(0, 1))      # dkappa += wy dkappa (NO merge)

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dkappa2, b_dkt.to(p_dkappa2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))


def kalman_bwd_intra(q, k, kappa, g, dAqk, dAkk, dq, dk, dkappa, dg,
                     cu_seqlens=None, chunk_indices=None, chunk_size=64, safe_gate=False):
    """Kalman fork of chunk_kda_bwd_intra. Adds the within-chunk score-matrix VJP
    to (dq, dk_read, dkappa, dg). Returns (dq, dk_read, dkappa, dg)."""
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(q)
    dk2 = torch.empty_like(k)
    dkappa2 = torch.empty_like(k)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    grid = (NK * NC, NT, B * H)
    chunk_kalman_bwd_kernel_intra[grid](
        q=q,
        k=k,
        kappa=kappa,
        g=g,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dkappa=dkappa,
        dkappa2=dkappa2,
        dg=dg,
        dg2=dg2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        SAFE_GATE=safe_gate,
        USE_GATHER=IS_GATHER_SUPPORTED,
    )
    return dq2, dk2, dkappa2, dg2
