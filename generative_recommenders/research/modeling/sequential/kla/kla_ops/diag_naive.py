# Naive (pure-PyTorch) Kalman Linear Attention (KLA).
#
# Reference implementation of KLA from the paper's Proposition 3.1 (Kalman Linear
# Attention). Mirrors the structure of lit_gpt/kda_ops/naive.py: a token recurrence
# (`naive_recurrent_kla`) and a chunk-parallel form (`naive_chunk_kla`). All plain torch ops
# (no Triton / tilelang): runs on CPU, autograd-differentiable, a readable correctness
# reference. The production diagonal-KLA token-mixer LAYER lives in lit_gpt/diag_kla.py
# (`DiagonalKalmanLinearAttention`).
#
# KLA is the delta-rule family with an *adaptive, anisotropic* diagonal-information
# gain in place of the fixed write direction beta_t * k_t used by DeltaNet/GDN/KDA.
# Per token, with token repr x_t, key/value (k_t, v_t), query q_t:
#
#   alpha_t = exp(-exp(A_log) * softplus(f_proj(x_t) + dt_bias))  # transition D_t=diag(alpha_t) (KDA gate)
#   omega_t = q_min + softplus(qn_proj(x_t))  # ADDITIVE process noise; paper notation omega_t
#   r_t     = r_min + softplus(r_proj(x_t))     # token-dependent observation noise (scalar per head)
#   p_hat_t = alpha_t^2 * p_{t-1} + omega_t     # predicted diagonal uncertainty (covariance predict)
#   kappa_t = (p_hat_t * k_t) / (r_t + sum_i p_hat_t,i k_t,i^2)   # adaptive Kalman gain
#   c_t     = 1/p_hat_t + d_k * (k_t * k_t) / r_t  # posterior info (c_0 = mu*1; mu is init-only)
#   S_t     = (I - kappa_t k_t^T) diag(alpha_t) S_{t-1} + kappa_t v_t^T
#   o_t     = S_t^T q_t
#
# The predicted uncertainty is the diagonal covariance predict p_hat_t = alpha_t^2 p_{t-1} + omega_t
# (paper eq app-predict). In information variables c = 1/p this is a linear-fractional (MOBIUS)
# recurrence in c_{t-1}:
#
#   c_t = c_{t-1} / (alpha_t^2 + omega_t c_{t-1}) + d_k k_t^2/r_t
#       = ((1 + u_t omega_t) c_{t-1} + u_t alpha_t^2) / (omega_t c_{t-1} + alpha_t^2),   u_t = d_k k_t^2/r_t,
#
# so each token defines a per-channel Mobius map M_t = [[1+u omega, u alpha^2],[omega, alpha^2]]. Mobius
# maps compose by 2x2 matrix multiplication, hence the prefix information states are computed by
# an ASSOCIATIVE SCAN over these matrices: c_t = n_t/d_t with [n_t,d_t]^T = M_t...M_1 [mu,1]^T
# (paper Prop 3.1 / appendix "Diagonal information: from the covariance recursion to the Mobius
# scan"). All M_t entries are positive, so the scan needs only scale renormalization (Mobius maps
# are scale-invariant: c = n/d) and stays numerically bounded -- unlike the earlier
# multiplicative-lambda affine scan (lam = (alpha^2 + rho)^-1 clamped <= 1), which dropped the
# exact alpha^2-contraction and needed the "information cannot grow" clamp as an approximation.
#
# This is the diagonal (per key channel) construction; the isotropic scalar specialization
# ISO-KLA lives in lit_gpt/kla_ops/iso_naive.py (one Mobius scan per head instead of one per channel).
# Because kappa_t depends only on the information scan (not on S_t), it is precomputed once, and
# the memory update is a standard delta-family affine scan with the vector gain kappa_t -- which
# is what makes the chunk form possible.
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange

_KLA_CFG_PRINTED = False  # print the KLA r_min/q_min config once per process


def _hp(x: torch.Tensor) -> torch.Tensor:
    """Upcast to fp32 minimum while preserving fp64.

    The recurrence runs in fp32 during training (the layer disables autocast); passing fp64
    inputs (the CPU reference) keeps fp64 so the Mobius scan can be validated to ~1e-12.
    """
    return x.to(torch.promote_types(x.dtype, torch.float32))


def _kla_kappa_seq(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: float | torch.Tensor = 1.0,
    mu: float | torch.Tensor = 1.0,
    initial_info: torch.Tensor | None = None,
    info_scale: float | None = None,
):
    """Sequential information scan -> adaptive gain kappa_t (reference).

    Exact diagonal Kalman with an ADDITIVE process noise (paper Prop 3.1)::

        p_hat_t = alpha_t^2 * p_{t-1} + omega_t = alpha_t^2 / c_{t-1} + omega_t
        kappa_t = (p_hat_t * k_t) / (r_t + sum_i p_hat_t,i k_t,i^2)       (anisotropic gain)
        c_t     = 1/p_hat_t + d_k k_t^2 / r_t                               (posterior info; c_0 = mu)

    Args (>=fp32 internally): k, alpha, omega ``[B,T,H,K]``; ``mu`` a float or per-head ``[H]``
    tensor; ``r`` a float, per-head ``[H]``, or token-dependent ``[B,T,H]`` tensor (observation
    noise). Returns ``(kappa [B,T,H,K], c_final [B,H,K])``. A length-T python loop; the recurrent
    reference and the validator for the fast chunk-parallel :func:`_kla_kappa`.
    """
    B, T, H, K = k.shape
    k, alpha, omega = _hp(k), _hp(alpha), _hp(omega)
    a2 = alpha * alpha
    r_tok = torch.is_tensor(r) and r.dim() == 3                    # [B,T,H] token-dependent obs noise
    if torch.is_tensor(r):
        r = _hp(r)
    r_fix = r.view(1, H, 1) if (torch.is_tensor(r) and not r_tok) else r  # [1,H,1] per-head or scalar
    mh = mu.to(k).view(1, H, 1) if torch.is_tensor(mu) else mu     # per-head prior broadcast over [B,H,K]
    # c_0 = mu*1: the diffuse prior enters ONLY through initialization, not as a per-step ridge
    c = initial_info.to(k) if initial_info is not None else (k.new_zeros(B, H, K) + mh)
    # info-increment scale s in the posterior-info update c_t = 1/p_hat + s*k^2/r
    # (paper: standardization scale; default s = d_k = K). Ablatable via info_scale.
    info_scale = float(K) if info_scale is None else float(info_scale)
    kappa = torch.zeros_like(k)
    for t in range(T):
        kt, a2t, omegat = k[:, t], a2[:, t], omega[:, t]          # [B,H,K]
        rh = r[:, t].unsqueeze(-1) if r_tok else r_fix            # [B,H,1] (token) or [1,H,1]/scalar
        p_hat = a2t / c + omegat                                  # predicted uncertainty (covariance predict)
        denom = rh + (p_hat * kt * kt).sum(-1, keepdim=True)      # [B,H,1]
        kappa[:, t] = p_hat * kt / denom
        c = 1.0 / p_hat + info_scale * (kt * kt) / rh             # posterior info: c_hat + d_k k^2/r
    return kappa, c


def _kla_kappa(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: float | torch.Tensor = 1.0,
    mu: float | torch.Tensor = 1.0,
    initial_info: torch.Tensor | None = None,
    chunk_size: int = 64,
    return_beta_ch: bool = False,
    info_scale: float | None = None,
):
    """Parallel information scan -> gain kappa (fast; == :func:`_kla_kappa_seq`).

    The additive process noise makes the per-channel information recursion a Mobius
    (linear-fractional) map, so the prefix information states are obtained by an ASSOCIATIVE
    SCAN over the 2x2 matrices ``M_t = [[1+u_t omega_t, u_t alpha_t^2],[omega_t, alpha_t^2]]`` with
    ``u_t = d_k k_t^2 / r_t`` (one map per key channel). We run an inclusive Hillis-Steele prefix
    product along T (``O(T log T)``, all-parallel), renormalizing each partial product by its max
    entry -- Mobius maps are scale-invariant (``c = n/d``) so this is exact and keeps the
    (all-positive) entries bounded. The gain at token t uses the EXCLUSIVE prefix ``c_{t-1}``.
    ``chunk_size`` is accepted for API symmetry with :func:`naive_chunk_kla` and ignored (the
    scan is global). Same arg conventions as :func:`_kla_kappa_seq`.

    Returns ``(kappa [B,T,H,K], c_final [B,H,K])``, or ``(kappa, beta_ch, c_final)`` when
    ``return_beta_ch`` -- ``beta_ch = p_hat / denom`` is the per-channel write strength with
    ``kappa = beta_ch * k`` (used by the GDN-2-kernel memory path: erase gate ``b = 1/beta_ch``).
    """
    B, T, H, K = k.shape
    info_scale = float(K) if info_scale is None else float(info_scale)  # s in u_t = s*k^2/r (default d_k=K)
    k, alpha, omega = _hp(k), _hp(alpha), _hp(omega)
    a2 = alpha * alpha                                            # [B,T,H,K]
    ksq = k * k                                                   # [B,T,H,K]  raw k^2 (gain denom)
    r_tok = torch.is_tensor(r) and r.dim() == 3                   # [B,T,H] token-dependent obs noise
    if torch.is_tensor(r):
        r = _hp(r)
    if r_tok:
        r_b = r[..., None]                                        # [B,T,H,1]
    elif torch.is_tensor(r):
        r_b = r.view(1, 1, H, 1)                                  # [H] per-head
    else:
        r_b = r                                                   # scalar
    u = (info_scale * ksq) / r_b                                  # u_t = s * k^2 / r_t  [B,T,H,K] (s default d_k)

    # per-channel Mobius matrices M_t = [[1 + u omega, u a2],[omega, a2]] (all entries > 0)
    row0 = torch.stack([1.0 + u * omega, u * a2], dim=-1)
    row1 = torch.stack([omega, a2], dim=-1)
    M = torch.stack([row0, row1], dim=-2)                         # [B,T,H,K,2,2]

    # inclusive prefix product along T:  P[t] = M_t @ M_{t-1} @ ... @ M_0  (Hillis-Steele)
    eye = torch.eye(2, dtype=M.dtype, device=M.device).view(1, 1, 1, 1, 2, 2)
    P = M / M.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    d = 1
    while d < T:
        pad = eye.expand(B, d, H, K, 2, 2)
        P_prev = torch.cat([pad, P[:, :T - d]], dim=1)            # P[t-d], identity for t < d
        P = P @ P_prev
        P = P / P.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        d *= 2

    # apply each prefix map to the prior [n_0, d_0] = [c_seed, 1]:  c_incl[t] = posterior info c_t
    if initial_info is not None:
        c0 = initial_info.to(k)[:, None]                         # [B,1,H,K]
    else:
        mh = mu.to(k).view(1, 1, H, 1) if torch.is_tensor(mu) else float(mu)
        c0 = a2.new_zeros(B, 1, H, K) + mh                       # c_0 = mu*1  [B,1,H,K]
    n = P[..., 0, 0] * c0 + P[..., 0, 1]                         # [B,T,H,K]
    den = P[..., 1, 0] * c0 + P[..., 1, 1]                       # [B,T,H,K]
    c_incl = n / den                                             # inclusive posterior info c_t
    # exclusive prefix c_{t-1}: shift right, seed token 0 with the prior c_{-1} = c_seed
    c_excl = torch.cat([c0, c_incl[:, :-1]], dim=1)              # [B,T,H,K]  c_{t-1}

    p_hat = a2 / c_excl + omega                                  # predicted uncertainty = alpha^2 p_{t-1} + omega
    denom = r_b + (p_hat * ksq).sum(-1, keepdim=True)            # [B,T,H,1]
    kappa = p_hat * k / denom
    if return_beta_ch:
        beta_ch = p_hat / denom                                 # per-channel write strength (kappa = beta_ch * k)
        return kappa, beta_ch, c_incl[:, -1]
    return kappa, c_incl[:, -1]                                  # c_final = inclusive c_T  [B,H,K]


def _kla_kappa_bwd_ref(
    k: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: float | torch.Tensor = 1.0,
    mu: float | torch.Tensor = 1.0,
    dkappa: torch.Tensor | None = None,
    dbeta_ch: torch.Tensor | None = None,
    info_scale: float | None = None,
):
    r"""Analytic (pure-PyTorch, fp64-capable) VJP of the per-channel Kalman gain scan.

    Reverse-mode gradient of :func:`_kla_kappa` (``return_beta_ch=True``): given the output
    cotangents ``dkappa`` and ``dbeta_ch`` (both ``[B,T,H,K]``, either may be ``None`` = 0),
    returns ``(dk, dalpha, domega, dr, dmu)`` with shapes matching the *canonical* inputs
    ``k,alpha,omega`` ``[B,T,H,K]``, ``r`` ``[B,T,H]``, ``mu`` ``[H]``. (``r``/``mu`` may be
    passed as float/``[H]``/``[B,T,H]`` scalars/tensors -- they are broadcast internally to the
    per-token/per-head canonical form; the returned ``dr``/``dmu`` are always ``[B,T,H]``/``[H]``,
    the natural per-token/per-head grads. Callers wanting the grad w.r.t. a broadcasted input
    sum over the broadcast dims, exactly as autograd does.)

    Implements ``docs/plans/2026-07-25-kla-kappa-backward-derivation.md`` EXACTLY:
      * §3 per-token gain VJP with the two per-token scalars ``D`` and ``G``;
      * §4 serial reverse affine scan (``t = T-1 -> 0``, per-channel carry ``s = c_bar``),
        the two-path ``dk``/``dr`` (gain path + info-update ``u = K k^2/r`` path), and the
        ``dmu`` batch+channel reduction.

    Runs internally at >= fp32 (fp64 in fp64), mirroring :func:`_hp`. A length-T python loop
    (reference / correctness anchor); vectorized over channels.
    """
    B, T, H, K = k.shape
    k, alpha, omega = _hp(k), _hp(alpha), _hp(omega)
    a2 = alpha * alpha                                            # [B,T,H,K]
    ksq = k * k                                                   # [B,T,H,K]
    Kf = float(K) if info_scale is None else float(info_scale)   # d_k = K (or ablation info-scale s)

    # --- observation noise r -> canonical per-token [B,T,H] (fp>=32) ---
    r_tok = torch.is_tensor(r) and r.dim() == 3
    if torch.is_tensor(r):
        r = _hp(r)
    if r_tok:
        r_bt = r                                                  # [B,T,H]
    elif torch.is_tensor(r):
        r_bt = r.view(1, 1, H).expand(B, T, H)                    # [H] per-head
    else:
        r_bt = k.new_full((B, T, H), float(r))                   # scalar
    r_b = r_bt[..., None]                                         # [B,T,H,1]

    # --- info prior mu -> canonical per-head [H] ---
    if torch.is_tensor(mu):
        mu_h = _hp(mu).view(H).to(k)                              # [H]
    else:
        mu_h = k.new_full((H,), float(mu))                       # scalar

    # --- cotangents (either may be absent = 0) ---
    dk_out = k.new_zeros(B, T, H, K) if dkappa is None else _hp(dkappa)
    db_out = k.new_zeros(B, T, H, K) if dbeta_ch is None else _hp(dbeta_ch)

    # --- recompute the forward EXCLUSIVE-prefix c_{t-1} (c_excl[t] = c_{i,t-1}, = mu at t=0)
    #     via the per-channel Mobius [n,d] scan (c = n/d), exactly as the forward does (§6). Uses the
    #     OVERFLOW-SAFE [n,d] form (NOT a raw scalar c, which blows fp32/fp64 to inf at tiny-omega for
    #     large T): n,d stay bounded (max-entry renorm; Mobius scale-invariant) so c_excl = n/d is
    #     finite even when legitimately huge (~5e14). Required for this ref to be the T=16384 /
    #     tiny-omega stress oracle (BG3). ---
    c_excl = k.new_empty(B, T, H, K)                             # c_{i,t-1}
    cn = mu_h.view(1, H, 1).expand(B, H, K).clone()             # [n,d] = [mu,1] => c_{i,-1} = mu
    cd = torch.ones_like(cn)
    for t in range(T):
        c_excl[:, t] = cn / cd                                   # c_{i,t-1} = n/d
        a2t, omt = a2[:, t], omega[:, t]                         # [B,H,K]
        u = Kf * ksq[:, t] / r_b[:, t]                           # u_t = d_k k^2/r  [B,H,K]
        # Mobius advance [n,d] <- M_t.[n,d], M_t = [[1+u om, u a2],[om, a2]]; then max-entry renorm.
        n_new = (1.0 + u * omt) * cn + (u * a2t) * cd
        d_new = omt * cn + a2t * cd
        mrn = torch.maximum(n_new.abs(), d_new.abs()).clamp_min(1e-30)
        cn = n_new / mrn
        cd = d_new / mrn

    # --- outputs ---
    dk_g = torch.zeros_like(k)
    dalpha = torch.zeros_like(k)
    domega = torch.zeros_like(k)
    dr = k.new_zeros(B, T, H)
    s = k.new_zeros(B, H, K)                                     # c_bar_{i,t}, init 0 (final state not an output)

    # --- §4 serial reverse affine scan (t = T-1 -> 0) ---
    for t in range(T - 1, -1, -1):
        cprev = c_excl[:, t]                                     # c_{i,t-1}  [B,H,K]
        rt = r_b[:, t]                                           # [B,H,1]
        p_hat = a2[:, t] / cprev + omega[:, t]                   # [B,H,K]
        kt = k[:, t]                                             # [B,H,K]
        ksqt = ksq[:, t]                                         # [B,H,K]
        # §3 per-token reductions D and G (over channels), both [B,H,1]
        D = rt + (p_hat * ksqt).sum(-1, keepdim=True)            # r_t + sum_j p_hat_j k_j^2
        kap = p_hat * kt / D                                     # kappa_i (recomputed)
        beta = p_hat / D                                         # beta_i (recomputed)
        dki_t = dk_out[:, t]                                     # dkappa_i
        dbi_t = db_out[:, t]                                     # dbeta_i
        G = (dki_t * kap + dbi_t * beta).sum(-1, keepdim=True)   # [B,H,1]
        # §3 gain-path adjoints
        Pbar_gain = (dki_t * kt + dbi_t) / D - ksqt * G / D      # adjoint of p_hat_m (gain)  [B,H,K]
        kbar_gain = dki_t * beta - 2.0 * p_hat * kt * G / D      # gain-path grad of k_m
        rbar_gain = -G / D                                       # gain-path grad of r_t  [B,H,1]
        # §4 total adjoint of p_hat (gain consumer + info-update consumer c_t = 1/p_hat + u)
        Pbar_tot = Pbar_gain + s * (-1.0 / (p_hat * p_hat))      # [B,H,K]
        # local input grads at (m,t)
        domega[:, t] = Pbar_tot
        da2_t = Pbar_tot / cprev                                 # da2_m = Pbar_tot / cprev
        dalpha[:, t] = 2.0 * alpha[:, t] * da2_t                 # dalpha = 2 alpha da2
        dk_g[:, t] = kbar_gain + s * (2.0 * Kf * kt / rt)        # gain path + info-update(u) path
        # dr: gain path (per-token scalar) + info-update scan-sum (K-reduction)
        dr_scan = -(Kf / (rt * rt)) * (s * ksqt).sum(-1, keepdim=True)  # -(K/r^2) sum_m s_m k_m^2
        dr[:, t] = (rbar_gain + dr_scan).squeeze(-1)            # [B,H]
        # push adjoint to the previous state c_{m,t-1}
        s = Pbar_tot * (-a2[:, t] / (cprev * cprev))            # c_bar_{i,t-1}
    # after t=0, s = c_bar_{i,-1} (adjoint of mu via channel m). mu[h] seeds every channel of
    # every batch element -> dmu[h] = sum_b sum_m c_bar_{m,-1}.
    dmu = s.sum(dim=(0, 2))                                      # [H]

    return dk_g, dalpha, domega, dr, dmu


def naive_recurrent_kla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: float | torch.Tensor = 1.0,
    mu: float | torch.Tensor = 1.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_info: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    r"""Token-by-token KLA reference (Proposition 3.1).

    Shapes: q,k,alpha,omega ``[B,T,H,K]``; v ``[B,T,H,V]`` (no GVA: value heads ==
    key heads). Returns ``(o, state)`` where ``state=(S [B,H,K,V], c [B,H,K])`` if
    ``output_final_state`` else ``None``.
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, alpha = _hp(q), _hp(k), _hp(v), _hp(alpha)
    q = q * scale

    kappa, c_final = _kla_kappa_seq(k, alpha, omega, r, mu, initial_info)

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state.to(k)
    o = torch.zeros_like(v)
    for t in range(T):
        a, kt, vt, qt, kap = alpha[:, t], k[:, t], v[:, t], q[:, t], kappa[:, t]
        S = a[..., None] * S                                  # diag(alpha_t) S_{t-1}
        kTS = (kt[..., None] * S).sum(-2)                     # k_t^T (D_t S_{t-1})  [B,H,V]
        S = S + kap[..., None] * (vt - kTS)[..., None, :]     # + kappa_t (v_t - kTS)^T
        o[:, t] = (qt[..., None] * S).sum(-2)                 # S_t^T q_t  [B,H,V]
    state = (S, c_final) if output_final_state else None
    return o.to(dtype), state


def naive_chunk_kla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: float | torch.Tensor = 1.0,
    mu: float | torch.Tensor = 1.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_info: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    info_scale: float | None = None,
):
    r"""Chunk-parallel KLA reference (WY transform of the memory scan).

    kappa_t is precomputed from the (parallel) Mobius information scan. With cumulative
    log-decay g = cumsum(log alpha) and gamma_i = exp(g_i), the memory update
    S_t = (I - kappa_t k_t^T) diag(alpha_t) S_{t-1} + kappa_t v_t^T is solved via the
    WY transform. The S-INDEPENDENT intra-chunk quantities are built batched over
    ALL NT chunks at once (as in KDA's naive_chunk_kda): the strictly-lower matrix
    M_{ij} = sum_d kappa_j k_i exp(g_i-g_j) (j<i), the readout N_{ij} with q_i (j<=i),
    and u = (I+M)^-1 v, w = (I+M)^-1 (gamma*k) via a batched unit-lower
    ``solve_triangular``. Only the S-dependent carry runs per chunk:
    U = u - w S0, o = (gamma*q) S0 + N U, S <- diag(gamma_C) S0 + (exp(g_C-g)*kappa)^T U.
    Same shapes as :func:`naive_recurrent_kla`; ``chunk_size`` must divide T.
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    assert T % BT == 0
    if scale is None:
        scale = K ** -0.5
    q, k, v, alpha = _hp(q), _hp(k), _hp(v), _hp(alpha)
    q = q * scale

    kappa, c_final = _kla_kappa(k, alpha, omega, r, mu, initial_info, chunk_size=chunk_size,
                                info_scale=info_scale)

    # -> [B, H, NT, BT, .]
    q, k, v, kap = (rearrange(x, 'b (n c) h d -> b h n c d', c=BT) for x in (q, k, v, kappa))
    g = rearrange(alpha, 'b (n c) h d -> b h n c d', c=BT).clamp_min(1e-6).log().cumsum(-2)  # cum log-decay
    gam = g.exp()                                                # gamma_i = decay from chunk start

    # --- intra-chunk quantities (M, N, u, w): all S-INDEPENDENT, so build them
    #     batched over ALL NT chunks at once (as KDA's naive_chunk_kda does), not
    #     inside the per-chunk carry loop. ---
    tri_excl = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)  # keep j<i
    tri_incl = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)  # keep j<=i
    M = torch.zeros(B, H, NT, BT, BT, dtype=q.dtype, device=q.device)
    N = torch.zeros(B, H, NT, BT, BT, dtype=q.dtype, device=q.device)
    for j in range(BT):                                          # BT iters, each batched over NT
        decay = (g - g[:, :, :, j:j+1, :]).clamp(max=0).exp()   # exp(g_i-g_j) [B,H,NT,BT,K]
        kj = kap[:, :, :, j, :]                                  # [B,H,NT,K]
        M[..., j] = torch.einsum('b h n c d, b h n d -> b h n c', k * decay, kj)
        N[..., j] = torch.einsum('b h n c d, b h n d -> b h n c', q * decay, kj)
    M = M.masked_fill(tri_excl, 0)                              # strictly lower (j<i)
    N = N.masked_fill(tri_incl, 0)                              # lower incl diag (j<=i)

    # (I + M) is unit lower-triangular -> u = (I+M)^-1 v, w = (I+M)^-1 (gamma*k), batched over NT.
    u = torch.linalg.solve_triangular(M, v, upper=False, unitriangular=True)         # [B,H,NT,BT,V]
    w = torch.linalg.solve_triangular(M, gam * k, upper=False, unitriangular=True)   # [B,H,NT,BT,K]
    gq = gam * q                                                # [B,H,NT,BT,K]
    gC = g[:, :, :, -1]                                         # [B,H,NT,K]  cum decay over the full chunk
    P = (gC[:, :, :, None, :] - g).clamp(max=0).exp() * kap     # [B,H,NT,BT,K]  = exp(g_C-g_i) * kappa_i

    # --- per-chunk carry: only the S-DEPENDENT part (a few batched matmuls per chunk) ---
    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state.to(k)
    o = torch.zeros(B, H, NT, BT, V, dtype=q.dtype, device=q.device)
    for n in range(NT):
        U = u[:, :, n] - torch.einsum('b h c k, b h k v -> b h c v', w[:, :, n], S)   # (I+M)^-1 (v - (gam k) S0)
        o[:, :, n] = torch.einsum('b h c k, b h k v -> b h c v', gq[:, :, n], S) \
            + torch.einsum('b h i j, b h j v -> b h i v', N[:, :, n], U)
        S = gC[:, :, n].exp()[..., None] * S + torch.einsum('b h c k, b h c v -> b h k v', P[:, :, n], U)

    o = rearrange(o, 'b h n c v -> b (n c) h v')
    state = (S, c_final) if output_final_state else None
    return o.to(dtype), state
