# Parallel (associative-scan) EXACT Kalman Linear Attention gains.
#
# The dense-covariance Kalman recurrence P_{t-1} -> P_hat_t -> kappa_t -> P_t is, per the paper's
# derivation (derivations.tex, "Use covariance for prediction"), an ASSOCIATIVE scan: each token's
# covariance map is a matrix-fractional (linear-fractional / Moebius) transform, and these compose by
# multiplying 2K x 2K block matrices. So the sequential O(T)-depth loop in
# `naive_recurrent_exact_kla` can be replaced by an O(log T)-depth prefix scan over 2K x 2K blocks.
#
# Per token (D_t = diag(alpha_t), Omega_t = diag(omega_t), r_eff = r_t/d_k):
#   predict  P_hat_t = D_t P_{t-1} D_t + Omega_t         <=> LFT block  U_t = [[D, Omega D^-1], [0, D^-1]]
#   measure  P_t     = (I - kappa_t k_t^T) P_hat_t       <=> LFT block  W_t = [[I, 0], [(1/r_eff) k k^T, I]]
# where an LFT block G=[[A,B],[C,D]] acts on a covariance P as  G . P = (A P + B)(C P + D)^-1.
# The token map is M_t = W_t @ U_t; the prefix product G_t = M_t @ ... @ M_1 (G_0 = I) gives
# P_t = G_t . P_0, and the *predicted* covariance needed for the gain is P_hat_t = (U_t @ G_{t-1}) . P_0.
# Then kappa_t = P_hat_t k_t / (r_eff + k_t^T P_hat_t k_t) -- the exact dense anisotropic gain, in
# log-depth. This is the dense generalization of the per-channel 2x2 Moebius scan used by diagonal KLA.
#
# Numerical note: the LFT is scale-invariant (G and s*G act identically), but the raw prefix products
# blow up because U carries D^-1 = 1/alpha > 1. We RENORMALIZE each (sub)product by its max-abs entry
# after every scan step -- this cannot change the induced map, but it is only a SCALE fix: the
# within-chunk product's CONDITION NUMBER still grows ~ (1/alpha_min)^(C-1), so once that exceeds the
# compute dtype's dynamic-range budget the renorm rounds the map's small (coupling) entries to zero and
# the composed LFT is WRONG (not just imprecise) -> garbage gains. There is no scalar-renorm cure for
# this (it is intrinsic Riccati-scan stiffness); the only fix is to keep the chunk length C small enough
# that the condition number stays representable. `_safe_cov_chunk` picks that C from alpha_min + dtype
# (calibrated in scripts/analyses/_calib_scan_chunk.py); the same bound guards the memory chunk scan,
# whose kappa/A term has the identical 1/alpha blow-up. Compute in fp32+ (fp64 in the CPU verifier).
from __future__ import annotations

import math
import warnings

import torch

_EPS = 1e-30
_WARNED_UNSAFE_CHUNK: set = set()   # dedup the explicit-unsafe-chunk warning (was per-batch spammy)


def _hp(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.promote_types(x.dtype, torch.float32))


def _renorm(G: torch.Tensor) -> torch.Tensor:
    """Divide each [.., 2K, 2K] block by its max-abs entry (LFT is scale-invariant)."""
    s = G.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(_EPS)
    return G / s


def _safe_cov_chunk(alpha: torch.Tensor, compute_dtype) -> int:
    """Largest within-chunk length C whose LFT prefix product stays numerically EXACT.

    The chunk's 2K x 2K prefix product accumulates D^-1 = 1/alpha, so its condition number
    ~ (1/alpha_min)^(C-1). We keep that under a dtype-calibrated budget so the max-abs `_renorm`
    (a single scalar per 2K x 2K block) never rounds the map's small entries away. Calibrated
    (scripts/analyses/_calib_scan_chunk.py, K=64/128, T=256) to stay >=~2x inside the error tolerance
    for alpha_min down to 0.01; for smaller alpha C shrinks toward 1 (a chunk of 1 is a plain
    sequential step -> no LFT product -> exact). NOTE the bound targets fp64/fp32 -- lower precisions
    (bf16/fp16) have too short a mantissa for ANY within-chunk product to stay exact (one heterogeneous
    channel already exceeds bf16's ~8-bit headroom), so they are forced to C=1 (exact but sequential)
    rather than silently reusing the fp32 budget. The bound uses the GLOBAL alpha-min (one C for all
    chunks): safe by construction (never too large for the worst chunk) but conservative -- a single
    tiny alpha shortens every chunk. Also used (conservatively) by `_memory_chunk_scan`."""
    budgets = {torch.float64: (11.0, 8), torch.float32: (5.0, 4)}   # (log-budget, cap)
    log_budget, cap = budgets.get(compute_dtype, (0.0, 1))          # bf16/fp16/unknown -> C=1 (exact)
    if cap <= 1:
        return 1
    a = float(alpha.detach().abs().amin().clamp_min(torch.finfo(torch.float32).tiny).item())
    if a >= 1.0:
        return cap
    C = int(log_budget / -math.log(min(a, 1.0 - 1e-9))) + 1
    return max(1, min(C, cap))


def _token_blocks(alpha, omega, k, r_eff):
    """Build predict block U_t and token map M_t = W_t @ U_t.  All inputs [B,H,T,K]; r_eff [B,H,T]."""
    B, H, T, K = alpha.shape
    dev, dt = alpha.device, alpha.dtype
    inv_a = 1.0 / alpha
    idx = torch.arange(K, device=dev)

    U = alpha.new_zeros(B, H, T, 2 * K, 2 * K)
    U[..., idx, idx] = alpha                    # top-left   diag(alpha)
    U[..., idx, K + idx] = omega * inv_a        # top-right  diag(omega/alpha)
    U[..., K + idx, K + idx] = inv_a            # bot-right  diag(1/alpha)

    W = alpha.new_zeros(B, H, T, 2 * K, 2 * K)
    W[..., idx, idx] = 1.0
    W[..., K + idx, K + idx] = 1.0
    W[..., K:, :K] = (k[..., :, None] * k[..., None, :]) / r_eff[..., None, None]  # (1/r_eff) k k^T
    return U, W @ U


def _lft_apply(G, P, K):
    """Apply LFT block G=[[A,B],[C,D]] to a covariance P -> (A P + B)(C P + D)^-1 (symmetrized).
    G ``[...,2K,2K]``, P ``[...,K,K]`` (broadcastable)."""
    A, Bb = G[..., :K, :K], G[..., :K, K:]
    C, D = G[..., K:, :K], G[..., K:, K:]
    N = A @ P + Bb
    M = C @ P + D
    X = torch.linalg.solve(M.transpose(-1, -2), N.transpose(-1, -2)).transpose(-1, -2)
    return 0.5 * (X + X.transpose(-1, -2))       # re-symmetrize (P is symmetric)


def _within_chunk_prefix(Mc, C):
    """Inclusive prefix product of the token maps within one chunk (Hillis-Steele over the C axis),
    renormalized each step (LFT scale-invariance). ``Mc`` ``[B,H,C,2K,2K]`` -> ``[B,H,C,2K,2K]``."""
    Gc = _renorm(Mc)
    d = 1
    while d < C:
        upd = _renorm(Gc[:, :, d:] @ Gc[:, :, :-d])
        Gc = torch.cat([Gc[:, :, :d], upd], dim=2)
        d *= 2
    return Gc


def _chunk_gains(aa_c, oo_c, kk_c, r_eff_c, P, K):
    """One chunk of the covariance scan: build its 2K x 2K token blocks, within-chunk prefix-scan them,
    and return this chunk's gains ``kappa_c`` ``[B,H,C,K]`` plus the carried covariance ``P`` ``[B,H,K,K]``
    for the next chunk. Factored out so each chunk can be gradient-checkpointed independently (its blocks
    are recomputed in backward instead of retained) -- the full-T block tensor was the memory hog."""
    B, H, C = aa_c.shape[:3]
    U, M = _token_blocks(aa_c, oo_c, kk_c, r_eff_c)                                    # [B,H,C,2K,2K]
    Gc = _within_chunk_prefix(M, C)                                                    # inclusive prefix
    I2 = torch.eye(2 * K, device=U.device, dtype=U.dtype).view(1, 1, 1, 2 * K, 2 * K)
    Gprev = torch.cat([I2.expand(B, H, 1, 2 * K, 2 * K), Gc[:, :, :-1]], dim=2)        # G_{j-1} (G_{-1}=I)
    Ghat = _renorm(U @ Gprev)                                                          # predict map [B,H,C,2K,2K]
    chunk_map = Gc[:, :, -1]                                                           # full-chunk map [B,H,2K,2K]
    Phat = _lft_apply(Ghat, P[:, :, None], K)                                          # [B,H,C,K,K]
    Pk = (Phat * kk_c[..., None, :]).sum(-1)                                           # [B,H,C,K]
    denom = r_eff_c + (kk_c * Pk).sum(-1)                                              # [B,H,C]
    kap_c = Pk / denom[..., None]
    P_next = _lft_apply(chunk_map, P, K)                                               # carry to next chunk
    return kap_c, P_next


def exact_kla_gains_scan(k, alpha, omega, r, mu=1.0, dk_calibration=True, chunk_size=None,
                         compute_dtype=torch.float64, checkpoint=False):
    r"""Exact Kalman gains kappa_t via a CHUNKED block-matrix scan.

    The raw LFT prefix product is ill-conditioned (Riccati instability): its condition number grows
    ~ (1/alpha_min)^(C-1) with the chunk length C, so we scan the 2K x 2K blocks only *within* short
    chunks and carry the actual covariance P across chunks sequentially -- T/C sequential steps.

    ``chunk_size=None`` (default) auto-picks the largest numerically SAFE C from alpha_min + dtype
    (:func:`_safe_cov_chunk`); an explicit chunk_size is honored but WARNS if it exceeds that bound
    (the composed LFT then silently loses its small entries -> wrong gains -- see the calibration in
    scripts/analyses/_calib_scan_chunk.py). ``compute_dtype`` fp64 = exact (default), fp32 = fast.

    k, alpha, omega: ``[B,T,H,K]``; r: float / ``[H]`` / ``[B,T,H]``; mu: float / ``[H]``.
    Returns kappa ``[B,T,H,K]`` matching :func:`naive_recurrent_exact_kla`'s gains to fp precision.
    """
    # Resolve the within-chunk length. None -> auto-safe; explicit -> honor + warn if past the bound
    # (the max-abs renorm cannot rescue a chunk whose LFT condition number exceeds fp precision).
    safe_C = _safe_cov_chunk(alpha, compute_dtype)
    if chunk_size is None:
        chunk_size = safe_C
    elif int(chunk_size) > safe_C:
        key = (str(compute_dtype), int(chunk_size), safe_C)      # dedup: warn ONCE per config, not per batch
        if key not in _WARNED_UNSAFE_CHUNK:
            _WARNED_UNSAFE_CHUNK.add(key)
            warnings.warn(
                f"exact_kla_gains_scan: chunk_size={int(chunk_size)} exceeds the numerically safe chunk "
                f"{safe_C} for this alpha/dtype ({compute_dtype}). The within-chunk LFT prefix product "
                f"accumulates 1/alpha; past the dtype's dynamic range the max-abs renorm drops the map's "
                f"small entries -> WRONG gains (verified ~7e4 abs error vs naive_recurrent_exact_kla). "
                f"Pass chunk_size<={safe_C} or chunk_size=None (auto). [warned once per config]",
                stacklevel=2,
            )
    chunk_size = max(1, int(chunk_size))
    B, T, H, K = k.shape
    out_dtype = k.dtype
    cdt = compute_dtype        # fp64 = exact (default); fp32 = fast/training (needs a smaller chunk)
    dk = float(K) if dk_calibration else 1.0
    kk, aa, oo = (x.to(cdt).permute(0, 2, 1, 3).contiguous() for x in (k, alpha, omega))  # [B,H,T,K]

    if torch.is_tensor(r):
        r = r.to(cdt)
        if r.dim() == 0:
            r = r.expand(B, T, H)
        elif r.dim() == 1:
            r = r.view(1, 1, H).expand(B, T, H)
    else:
        r = kk.new_full((B, T, H), float(r))
    r_eff = (r / dk).permute(0, 2, 1).contiguous()                                    # [B,H,T]

    if torch.is_tensor(mu):
        inv_mu = (1.0 / mu.to(cdt)).view(1, H, 1, 1)
    else:
        inv_mu = 1.0 / float(mu)
    eye_k = torch.eye(K, device=kk.device, dtype=kk.dtype)
    P = (eye_k.view(1, 1, K, K) * inv_mu).expand(B, H, K, K).contiguous()             # P0 = mu^-1 I

    # pad T up to a multiple of chunk_size; padded tokens (alpha=1, omega=0, k=0) produce identity
    # LFT maps (U=W=I) so they neither perturb the prefix product nor the carried P, and their gains
    # are discarded by the [:T] slice below. We pad the small per-token arrays, NOT the 2K x 2K blocks.
    C = min(int(chunk_size), T)
    pad = (-T) % C
    if pad:
        aa = torch.cat([aa, aa.new_ones(B, H, pad, K)], dim=2)
        oo = torch.cat([oo, oo.new_zeros(B, H, pad, K)], dim=2)
        kk = torch.cat([kk, kk.new_zeros(B, H, pad, K)], dim=2)
        r_eff = torch.cat([r_eff, r_eff.new_ones(B, H, pad)], dim=2)
    Tp = T + pad
    nc = Tp // C
    aac, ooc, kkc = (x.view(B, H, nc, C, K) for x in (aa, oo, kk))
    r_effc = r_eff.view(B, H, nc, C)

    # Sequential over chunks, building each chunk's 2K x 2K blocks LAZILY (only one chunk resident) --
    # the full-T block tensor [B,H,T,2K,2K] was the forward-memory hog, and it was alive all at once, so
    # checkpointing the whole scan couldn't lower the peak. checkpoint=True additionally recomputes each
    # chunk's blocks in backward (freed after fwd) -> peak memory ~O(1) in the number of chunks.
    do_ckpt = bool(checkpoint) and torch.is_grad_enabled()
    if do_ckpt:
        from torch.utils.checkpoint import checkpoint as _ckpt
    kappas = []
    for c in range(nc):
        args = (aac[:, :, c], ooc[:, :, c], kkc[:, :, c], r_effc[:, :, c], P, K)
        if do_ckpt:
            kap_c, P = _ckpt(_chunk_gains, *args, use_reentrant=False)
        else:
            kap_c, P = _chunk_gains(*args)
        kappas.append(kap_c)
    kappa = torch.cat(kappas, dim=2)[:, :, :T]                                        # [B,H,T,K]
    return kappa.permute(0, 2, 1, 3).to(out_dtype).contiguous()                       # [B,T,H,K]


def _memory_read(q, k, v, alpha, kappa, scale, initial_state=None):
    """Given the gains kappa_t, run the (cheap) delta-rule memory + read: same recurrence as the naive
    reference, S_t = diag(a) S_{t-1} + kappa_t (v_t - (diag(a)S_{t-1})^T k_t)^T,  o_t = S_t^T q_t."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, alpha, kappa = (_hp(x) for x in (q, k, v, alpha, kappa))
    q = q * scale
    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state[0].to(k)
    o = q.new_zeros(B, T, H, V)
    for t in range(T):
        a, kt, vt, qt, kap = alpha[:, t], k[:, t], v[:, t], q[:, t], kappa[:, t]
        S = a[..., None] * S
        kTS = (kt[..., None] * S).sum(-2)
        S = S + kap[..., None] * (vt - kTS)[..., None, :]
        o[:, t] = (qt[..., None] * S).sum(-2)
    return o, S


def _memory_chunk_scan(q, k, v, alpha, kappa, scale, chunk_size=None, initial_state=None):
    r"""Chunked delta-rule memory + read with INDEPENDENT write (kappa) and read (k) keys.

    The exact gain kappa is NOT parallel to k, so GDN-2's kernel (which ties the read key to the write
    key by a per-channel gate) cannot express this memory. Here we chunk it directly: within a chunk,
    normalize out the per-channel decay (S~_t = diag(1/A_t) S_t with A_t = prod of alpha), which turns
    the recurrence into an undecayed delta rule with write ``kappa~=kappa/A_t`` and read ``k~=k*A_t``:
        u_t = v_t - k~_t^T S~_{t-1},   (I + tril(k~ . kappa~, -1)) U = V - k~ S~_0
        o_t = q~_t^T S~_0 + tril_incl(q~ . kappa~) U,   S~_end = S~_0 + kappa~^T U,  S_end = diag(A_C) S~_end.
    Parallel within a chunk (one triangular solve); T/chunk_size sequential chunk carries. The kappa/A_t
    term blows up like 1/prod(alpha), so ``chunk_size=None`` (default) auto-picks a numerically safe C
    from alpha_min + dtype (:func:`_safe_cov_chunk`, conservative here). Matches :func:`_memory_read`
    to fp precision.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    dt = torch.promote_types(q.dtype, torch.float32)
    if chunk_size is None:
        chunk_size = _safe_cov_chunk(alpha, dt)
    q, k, v, al, kap = (x.to(dt).permute(0, 2, 1, 3).contiguous() for x in (q, k, v, alpha, kappa))  # [B,H,T,*]
    q = q * scale

    C = min(int(chunk_size), T)
    pad = (-T) % C
    if pad:
        zK = q.new_zeros(B, H, pad, K)
        q = torch.cat([q, zK], 2); k = torch.cat([k, zK], 2); kap = torch.cat([kap, zK], 2)
        al = torch.cat([al, al.new_ones(B, H, pad, K)], 2)                 # pad decay=1 (no-op)
        v = torch.cat([v, v.new_zeros(B, H, pad, V)], 2)
    Tp = T + pad
    nc = Tp // C
    qc, kc, vc, ac, kpc = (x.view(B, H, nc, C, -1) for x in (q, k, v, al, kap))

    A = ac.clamp_min(1e-9).log().cumsum(dim=3).exp()                       # within-chunk cum decay [B,H,nc,C,K]
    k_t = kc * A                                                           # read key  k~ = k * A_t
    kap_t = kpc / A                                                        # write key kappa~ = kappa / A_t
    q_t = qc * A                                                           # query     q~ = q * A_t
    tril_s = torch.tril(q.new_ones(C, C), -1)                             # strictly lower
    tril_i = torch.tril(q.new_ones(C, C))                                 # lower incl diag
    M = torch.einsum("bhnck,bhnsk->bhncs", k_t, kap_t) * tril_s           # [B,H,nc,C,C]
    Pqk = torch.einsum("bhnck,bhnsk->bhncs", q_t, kap_t) * tril_i
    IpM = torch.eye(C, device=q.device, dtype=dt) + M
    A_end = A[:, :, :, -1]                                                 # [B,H,nc,K] cum decay at chunk end

    S = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state[0].to(dt)
    outs = []
    for c in range(nc):
        Bc = vc[:, :, c] - torch.einsum("bhck,bhkv->bhcv", k_t[:, :, c], S)         # v - k~ S~_0
        U = torch.linalg.solve_triangular(IpM[:, :, c], Bc, upper=False, unitriangular=True)
        Oc = torch.einsum("bhck,bhkv->bhcv", q_t[:, :, c], S) + torch.einsum("bhcs,bhsv->bhcv", Pqk[:, :, c], U)
        outs.append(Oc)
        S = A_end[:, :, c][..., None] * (S + torch.einsum("bhsk,bhsv->bhkv", kap_t[:, :, c], U))
    o = torch.cat(outs, dim=2)[:, :, :T].permute(0, 2, 1, 3).contiguous()          # [B,T,H,V]
    return o, S


def naive_parallel_exact_kla(q, k, v, alpha, omega, r=1.0, mu=1.0, scale=None,
                             dk_calibration=True, initial_state=None, output_final_state=False,
                             memory_backend="chunk", compute_dtype=torch.float64, cov_chunk_size=None,
                             checkpoint=False):
    """Drop-in faster twin of :func:`naive_recurrent_exact_kla`: gains via the parallel block-matrix
    covariance scan, then the chunked delta-rule memory read. ``compute_dtype`` = torch.float64 (EXACT,
    default -- matches the recurrent oracle to ~1e-9 even for heterogeneous alpha) or torch.float32
    (fast, ~2x, but APPROXIMATE: ~1e-4 typical / ~1e-3 for heterogeneous alpha, since the single-scalar
    max-abs renorm cannot serve all channels at 23-bit mantissa; use fp64 when exactness matters).
    ``cov_chunk_size=None`` (default) auto-picks the largest numerically SAFE covariance-scan chunk from
    alpha_min + dtype (:func:`_safe_cov_chunk`) -- small alpha needs a short chunk or the LFT prefix
    product loses precision; an explicit value is honored (warns once if unsafe). ``memory_backend`` =
    "chunk" (parallel, default; also auto-chunked, conservatively) or "loop" (sequential ref).
    ``checkpoint=True`` recomputes the covariance scan in backward instead of saving it (bit-identical).
    NOTE: this path restarts P0=mu^-1 I and returns (S, None) under output_final_state -- it cannot
    resume/return the dense covariance (raises on an initial covariance); use mode='recurrent' for that."""
    dtype = v.dtype
    K = q.shape[-1]
    if scale is None:
        scale = K ** -0.5
    # The parallel gains scan always restarts P0 = mu^-1 I and does NOT return the final covariance;
    # fail loud rather than silently diverge from naive_recurrent_exact_kla (which resumes/returns P).
    if initial_state is not None and len(initial_state) > 1 and initial_state[1] is not None:
        raise NotImplementedError(
            "naive_parallel_exact_kla cannot resume an initial covariance (initial_state[1] is not None); "
            "the parallel gains scan restarts P0=mu^-1 I each call. Use mode='recurrent' "
            "(naive_recurrent_exact_kla) for stateful covariance carry. NOTE: output_final_state here "
            "returns (S, None) -- the final covariance P is not produced by this path.")
    # cov_chunk_size=None -> exact_kla_gains_scan auto-resolves a numerically safe chunk (per alpha+dtype).
    kappa = exact_kla_gains_scan(k, alpha, omega, r=r, mu=mu, dk_calibration=dk_calibration,
                                 chunk_size=cov_chunk_size, compute_dtype=compute_dtype,
                                 checkpoint=checkpoint)
    mem = _memory_chunk_scan if memory_backend == "chunk" else _memory_read
    o, S = mem(q, k, v, alpha, kappa, scale, initial_state=initial_state)
    state = (S, None) if output_final_state else None
    return o.to(dtype), state
