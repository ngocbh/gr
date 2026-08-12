# Naive (pure-PyTorch) Isotropic Kalman Linear Attention (ISO-KLA).
#
# Scalar-uncertainty specialization of KLA (paper Appendix A.3, "Isotropic Kalman
# Linear Attention", eq iso-kla-*). The covariance is isotropic P_t = b_t I, so the
# uncertainty is ONE scalar per head and the gain has the fixed delta-rule direction
# kappa_t = beta_t k_t (like KDA/GDN) -- but beta_t is derived from an EXACT scalar
# Kalman scan with an independently-parameterized ADDITIVE process noise:
#
#   alpha_t = exp(-exp(A_log) softplus(f_proj(x)+dt_bias))   # per-channel decay (KDA gate)
#   a_t     = mean_i(alpha_{t,i}^2)                          # avg contraction (SCALAR per head)
#   q_t     = q_min + softplus(q_proj(x))                    # ADDITIVE process noise (scalar per head)
#   r_t     = r_min + softplus(r_proj(x))                    # token-dependent obs noise (per head)
#   b_hat_t = a_t b_{t-1} + q_t                              # covariance predict (additive Kalman)
#   beta_t  = b_hat_t / (r_t + b_hat_t ||k_t||^2)            # scalar write strength
#   c_t     = 1/b_hat_t + ||k_t||^2 / r_t                    # info measurement update (trace); c_0 = mu
#   S_t     = (I - beta_t k_t k_t^T) diag(alpha_t) S_{t-1} + beta_t k_t v_t^T   # == KDA memory
#   o_t     = S_t^T q_t
#
# With an INDEPENDENT additive q_t the information recursion is no longer affine in c_t: it is
# a MOBIUS (linear-fractional) map c_t = (A_t c_{t-1}+B_t)/(C_t c_{t-1}+D_t). Mobius maps
# compose by 2x2 matrix multiplication, so the prefix info states are computed by an
# ASSOCIATIVE SCAN over the matrices M_t = [[1+u_t q_t, u_t a_t],[q_t, a_t]] (u_t=||k_t||^2/r_t):
# c_t = n_t/d_t with [n_t,d_t]^T = M_t...M_1 [mu,1]^T. This keeps EXACT Kalman covariance
# semantics -- covariance contraction from alpha is preserved (a_t<1 shrinks it) with NO
# lambda<=1 clamp -- and stays numerically bounded (additive noise => c bounded), unlike the
# earlier multiplicative lambda-scan which had to clamp lambda<=1 ("information cannot grow
# during prediction") as an approximation. All M_t entries are positive, so the scan needs
# only scale renormalization (Mobius maps are scale-invariant: c=n/d).
#
# The memory update is still exactly the KDA fixed-gain residual form, so it reuses
# lit_gpt/kda_ops/naive.py (naive_recurrent_kda / naive_chunk_kda) with g = log(alpha) and the
# ISO-derived scalar beta_t. ISO-KLA sits between fixed-gain delta rules (KDA: free beta) and
# diagonal KLA (per-channel gain): it adapts the scalar write strength from a Kalman
# uncertainty, but cannot assign per-channel write strengths -- so it has no per-channel
# gain-freeze failure mode.
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange

from generative_recommenders.research.modeling.sequential.kla.kda_ops.naive import naive_recurrent_kda, naive_chunk_kda

_ISO_CFG_PRINTED = False  # print the ISO-KLA config once per process
_ISO_BETA_VERBOSE = False  # gate _iso_beta_seq's per-head beta debug print (spams + forces .item() GPU syncs)

# chunk_kernel-mode beta backend selector (parity oracle switch):
#   "triton"  -> differentiable Triton iso_beta_chunk (fwd v1 + Triton parallel two-pass bwd) [DEFAULT]
#   "pytorch" -> pure-PyTorch _iso_beta_chunk (Hillis-Steele Mobius, autograd through torch)
# Env override ISO_BETA_BACKEND for the G4 grad-parity oracle; also settable per-layer via
# ``beta_backend=`` on the production ISO-KLA layer (lit_gpt/iso_kla.py).
_ISO_BETA_BACKEND_DEFAULT = os.environ.get("ISO_BETA_BACKEND", "triton").lower()


def _bcast_bth(x, like):
    """Broadcast ``x`` (scalar / ``[H]`` / ``[B,T,H]``) to ``[B,T,H]`` matching ``like``."""
    B, T, H = like.shape
    if not torch.is_tensor(x):
        return like.new_full((B, T, H), float(x))
    x = x.float()
    if x.dim() == 3:
        return x
    if x.dim() == 1:
        return x.view(1, 1, H).expand(B, T, H)
    if x.dim() == 0:
        return x.view(1, 1, 1).expand(B, T, H)
    raise ValueError(f"expected scalar / [H] / [B,T,H], got shape {tuple(x.shape)}")


def _iso_beta_seq(k, alpha, q_noise, r=1.0, mu=1.0, info_scale=None):
    """Sequential (token-by-token) additive-Kalman info scan -> write gate beta_t ``[B,T,H]``.

    Reference / ``recurrent`` mode. Exact scalar Kalman with an ADDITIVE process noise q_t:
      predict  b_hat_t = a_t b_{t-1} + q_t          (a_t = mean_i alpha_{t,i}^2, b = 1/c)
      gain     beta_t  = b_hat_t / (r_t + s b_hat_t ||k_t||^2)
      update   c_t     = 1/b_hat_t + s ||k_t||^2 / r_t    (c_0 = mu)
    beta_t uses the EXCLUSIVE prefix c_{t-1}. Args (fp32 internally): k, alpha ``[B,T,H,K]``;
    q_noise, r scalar / per-head ``[H]`` / token-dependent ``[B,T,H]``; mu float or ``[H]``.

    ``info_scale`` = the info-increment scale ``s`` (effective standardized ``||k~||^2`` from
    ``k~=sqrt(s)*k``). It multiplies BOTH the info-update term ``s*||k||^2/r`` AND the
    gain-denominator term ``s*b_hat*||k||^2``. Default ``None`` -> ``s = 1.0`` (bit-identical to
    the pre-ablation IsoKLA, which used ``||k||^2 = 1`` with NO d_k standardization).
    """
    B, T, H, K = k.shape
    # info-increment scale s in beta = b_hat/(r + s b_hat ||k||^2), c = 1/b_hat + s ||k||^2/r.
    # Default s = 1.0 (bit-identical to pre-ablation IsoKLA; ablation sweeps s in {1, sqrt(K), K}).
    info_scale = 1.0 if info_scale is None else float(info_scale)
    k, alpha = k.float(), alpha.float()
    a = (alpha * alpha).mean(-1)                              # [B,T,H]  a_t = mean_i alpha_i^2
    ksq = (k * k).sum(-1)                                     # [B,T,H]  ||k||^2
    q = _bcast_bth(q_noise, a)                                # [B,T,H]  additive process noise
    r = _bcast_bth(r, a)                                      # [B,T,H]  obs noise
    mh = mu.float().view(1, H) if torch.is_tensor(mu) else mu
    c = k.new_zeros(B, H) + mh                                # c_0 = mu (info, scalar per head)
    beta = torch.zeros(B, T, H, dtype=torch.float, device=k.device)
    for t in range(T):
        at, qt, rt, kt2 = a[:, t], q[:, t], r[:, t], ksq[:, t]   # [B,H]
        b_hat = at / c + qt                                   # b_hat_t = a_t b_{t-1} + q_t
        beta[:, t] = b_hat / (rt + b_hat * info_scale * kt2)  # beta_t (s scales ||k||^2)
        c = 1.0 / b_hat + info_scale * kt2 / rt              # posterior info c_t (c_hat = 1/b_hat)

    # per-head beta distribution (debug probe; recurrent path only). Gated behind
    # _ISO_BETA_VERBOSE (default off) -- the print spams AND forces .item() GPU
    # syncs; re-enable for training-debug.
    if _ISO_BETA_VERBOSE:
        for h in range(H):
            print(f"[ISO-KLA] head {h} beta_t mean/std: ", beta[:, :, h].mean().item(), beta[:, :, h].std().item())
    return beta


def _iso_beta_chunk(k, alpha, q_noise, r=1.0, mu=1.0, chunk_size=None, info_scale=None):
    """Parallel additive-Kalman info scan -> beta_t ``[B,T,H]`` (== :func:`_iso_beta_seq`).

    The additive process noise makes the information recursion a M\"obius (linear-fractional)
    map, so the prefix info states are obtained by an ASSOCIATIVE SCAN over the 2x2 matrices
    ``M_t = [[1+u_t q_t, u_t a_t],[q_t, a_t]]`` with ``u_t = s ||k_t||^2 / r_t``. We run an
    inclusive Hillis-Steele prefix product along T (``O(T log T)``, all-parallel), renormalizing
    each partial product by its max entry -- M\"obius maps are scale-invariant (``c = n/d``) so
    this is exact and keeps the (all-positive) entries bounded. The gain at token t uses the
    EXCLUSIVE prefix ``c_{t-1}``. ``chunk_size`` is accepted for API symmetry and ignored.

    ``info_scale`` = the info-increment scale ``s`` (default ``None`` -> ``1.0``, bit-identical to
    the pre-ablation IsoKLA). It multiplies both ``u_t`` and the gain-denom ``||k||^2`` term.
    """
    B, T, H, K = k.shape
    info_scale = 1.0 if info_scale is None else float(info_scale)   # s in u_t = s ||k||^2/r (default 1)
    k, alpha = k.float(), alpha.float()
    a = (alpha * alpha).mean(-1)                              # [B,T,H]  a_t
    ksq = (k * k).sum(-1)                                     # [B,T,H]  ||k||^2
    q = _bcast_bth(q_noise, a)                                # [B,T,H]  additive process noise
    r = _bcast_bth(r, a)                                      # [B,T,H]  obs noise
    u = info_scale * ksq / r                                  # [B,T,H]  u_t = s ||k||^2 / r_t
    mu_v = mu.float().view(1, 1, H) if torch.is_tensor(mu) else float(mu)

    # M_t = [[1 + u q, u a], [q, a]]   (all entries > 0)  -> [B,T,H,2,2]
    row0 = torch.stack([1.0 + u * q, u * a], dim=-1)
    row1 = torch.stack([q, a], dim=-1)
    M = torch.stack([row0, row1], dim=-2)                     # [B,T,H,2,2]

    # inclusive prefix product along T:  P[t] = M_t @ M_{t-1} @ ... @ M_1  (Hillis-Steele)
    eye = torch.eye(2, dtype=M.dtype, device=M.device).view(1, 1, 1, 2, 2)
    P = M / M.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    d = 1
    while d < T:
        pad = eye.expand(B, d, H, 2, 2)
        P_prev = torch.cat([pad, P[:, :T - d]], dim=1)        # P[t-d], identity for t < d
        P = P @ P_prev
        P = P / P.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        d *= 2

    # apply each prefix map to the prior [n_0, d_0] = [mu, 1]:  c_incl[t] = n_t / d_t
    n = P[..., 0, 0] * mu_v + P[..., 0, 1]                    # [B,T,H]
    den = P[..., 1, 0] * mu_v + P[..., 1, 1]                  # [B,T,H]
    c_incl = n / den                                          # inclusive info state
    # exclusive prefix c_{t-1}: shift right, seed token 0 with the prior c_{-1} = mu
    c0 = a.new_zeros(B, 1, H) + mu_v
    c_excl = torch.cat([c0, c_incl[:, :-1]], dim=1)           # [B,T,H]  c_{t-1}

    b_hat = a / c_excl + q                                    # b_hat_t = a_t b_{t-1} + q_t
    beta = b_hat / (r + b_hat * info_scale * ksq)            # beta_t (s scales ||k||^2)
    return beta


def naive_recurrent_iso_kla(q, k, v, alpha, q_noise, r=1.0, mu=1.0,
                            scale=None, initial_state=None, output_final_state=False,
                            info_scale=None):
    """Token-by-token ISO-KLA reference: additive-Kalman beta + KDA memory recurrence."""
    beta = _iso_beta_seq(k, alpha, q_noise, r, mu, info_scale=info_scale)   # [B,T,H]
    g = alpha.float().clamp_min(1e-6).log()                   # log decay [B,T,H,K]
    return naive_recurrent_kda(q, k, v, g, beta, scale=scale,
                               initial_state=initial_state, output_final_state=output_final_state)


def naive_chunk_iso_kla(q, k, v, alpha, q_noise, r=1.0, mu=1.0,
                        scale=None, initial_state=None, output_final_state=False, chunk_size=64,
                        info_scale=None):
    """Chunk-parallel ISO-KLA: additive-Kalman (Mobius scan) beta + KDA memory (WY) scan."""
    beta = _iso_beta_chunk(k, alpha, q_noise, r, mu, info_scale=info_scale)  # [B,T,H]
    g = alpha.float().clamp_min(1e-6).log()                   # log decay [B,T,H,K]
    return naive_chunk_kda(q, k, v, g, beta, scale=scale, initial_state=initial_state,
                           output_final_state=output_final_state, chunk_size=chunk_size)
