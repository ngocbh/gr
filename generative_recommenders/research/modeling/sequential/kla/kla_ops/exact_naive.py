# Naive (pure-PyTorch) EXACT Kalman Linear Attention (Exact KLA) -- recurrence only.
#
# Reference implementation of the paper's Proposition A.1 (Exact Kalman Linear Attention):
# the exact Kalman optimal update, with a diagonal transition D_t=diag(alpha_t)
# and an additive DIAGONAL process noise Omega_t=diag(omega_t), but keeping the covariance
# P_t DENSE (d_k x d_k). The gain is therefore the exact anisotropic Kalman gain
# kappa_t = P_hat_t k_t / (r_t + k_t^T P_hat_t k_t), NOT the diagonal surrogate that diagonal KLA
# (lit_gpt/kla_ops/diag_naive.py) or isotropic KLA (iso_naive.py) use. Because P_t is dense and
# the gain depends on the accumulated posterior covariance, the recurrence is O(d_k^2) per token
# and does NOT collapse to a parallel scan -- it is the ground-truth filter the scan-friendly
# approximations aim at, and (per the paper) "the naive recurrent version we implement first".
# The token-mixer LAYER that runs this recurrence lives in lit_gpt/exact_kla.py
# (`ExactKalmanLinearAttention`); this module holds only the reference function.
#
# Per token t (paper eq exact-kla-box-recurrence), with S_0 = 0 and P_0 = mu^-1 I:
#   predict:  S_hat_t = diag(alpha_t) S_{t-1}
#             P_hat_t = diag(alpha_t) P_{t-1} diag(alpha_t) + diag(omega_t)
#   gain:     kappa_t = P_hat_t k_t / (r_eff + k_t^T P_hat_t k_t)
#   update:   S_t = S_hat_t + kappa_t (v_t - S_hat_t^T k_t)^T
#             P_t = (I - kappa_t k_t^T) P_hat_t     (Joseph form for PSD stability, eq exact-kla-joseph)
#   read:     o_t = S_t^T q_t
#
# alpha_t = KDA gate (per key channel), omega_t = q_min + softplus(qn_proj(x)) (ADDITIVE process
# noise, per channel), r_t = r_min + softplus(r_proj(x)) (obs noise, per head), mu = info prior.
# `r_eff = r_t / d_k` applies the paper's calibrated measurement precision d_k/r_t (equivalently
# standardized keys k~=sqrt(d_k)k) so the dense reference aligns with the diagonal/isotropic KLA
# variants for l2-normalized keys (dk_calibration=True; r is learnable so the d_k factor is
# absorbable, it only sets the init scale). Equivalence checks (paper Implementation notes):
# projecting P_t to its diagonal after each step recovers diagonal KLA's p_hat=alpha^2 p+omega;
# averaging that diagonal recovers isotropic KLA's scalar b_hat.
from __future__ import annotations

import torch


def _hp(x: torch.Tensor) -> torch.Tensor:
    """Upcast to fp32 minimum while preserving fp64 (so the reference validates to ~1e-12)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def naive_recurrent_exact_kla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    omega: torch.Tensor,
    r: float | torch.Tensor = 1.0,
    mu: float | torch.Tensor = 1.0,
    scale: float | None = None,
    dk_calibration: bool = True,
    joseph: bool = True,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
):
    r"""Token-by-token EXACT Kalman Linear Attention (Proposition A.1) -- DENSE covariance.

    Shapes: q, k, alpha, omega ``[B,T,H,K]``; v ``[B,T,H,V]`` (no GVA); ``r`` a float, per-head
    ``[H]``, or token-dependent ``[B,T,H]``; ``mu`` a float or per-head ``[H]``. Returns
    ``(o [B,T,H,V], state)`` where ``state = (S [B,H,K,V], P [B,H,K,K])`` if ``output_final_state``.
    O(K^2) state and work per token; a slow but exact ground-truth reference (CPU-runnable, no
    Triton). ``dk_calibration`` scales the observation noise to ``r/d_k`` (precision d_k/r);
    ``joseph`` uses the symmetric Joseph covariance downdate (numerically robust, stays PSD).
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, alpha, omega = (_hp(x) for x in (q, k, v, alpha, omega))
    q = q * scale
    dk = float(K) if dk_calibration else 1.0

    # r -> [B,T,H]
    if torch.is_tensor(r):
        r = _hp(r)
        if r.dim() == 1:
            r = r.view(1, 1, H).expand(B, T, H)
        elif r.dim() != 3:
            raise ValueError(f"r must be scalar / [H] / [B,T,H], got {tuple(r.shape)}")
    else:
        r = q.new_full((B, T, H), float(r))

    # P_0 = mu^-1 I (per head); c_0 = mu
    if torch.is_tensor(mu):
        inv_mu = (1.0 / _hp(mu)).view(1, H, 1, 1)
    else:
        inv_mu = 1.0 / float(mu)
    eye_k = torch.eye(K, dtype=q.dtype, device=q.device)

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state[0].to(k)
        P = initial_state[1].to(k).clone()
    else:
        P = (eye_k.view(1, 1, K, K) * (inv_mu if not torch.is_tensor(inv_mu) else inv_mu)).expand(B, H, K, K).clone()

    o = torch.zeros(B, T, H, V, dtype=q.dtype, device=q.device)
    for t in range(T):
        a, om, kt, vt, qt = alpha[:, t], omega[:, t], k[:, t], v[:, t], q[:, t]   # [B,H,K] / [B,H,V]
        r_eff = r[:, t] / dk                                          # [B,H]

        # --- predict ---
        S = a[..., None] * S                                         # diag(alpha) S_{t-1}   [B,H,K,V]
        P = a[..., :, None] * P * a[..., None, :]                    # diag(alpha) P diag(alpha)  [B,H,K,K]
        P = P + torch.diag_embed(om)                                # + diag(omega)

        # --- gain (exact anisotropic Kalman gain; NOT proportional to k) ---
        Pk = (P * kt[..., None, :]).sum(-1)                          # P_hat k   [B,H,K]
        denom = r_eff + (kt * Pk).sum(-1)                            # r_eff + k^T P_hat k   [B,H]
        kappa = Pk / denom[..., None]                               # [B,H,K]

        # --- update memory ---
        kTS = (kt[..., None] * S).sum(-2)                            # S_hat^T k   [B,H,V]
        S = S + kappa[..., None] * (vt - kTS)[..., None, :]         # + kappa (v - S_hat^T k)^T

        # --- update covariance ---
        if joseph:
            J = eye_k - kappa[..., :, None] * kt[..., None, :]      # I - kappa k^T   [B,H,K,K]
            JP = torch.einsum("bhij,bhjl->bhil", J, P)
            P = torch.einsum("bhil,bhml->bhim", JP, J) + r_eff[..., None, None] * (kappa[..., :, None] * kappa[..., None, :])
        else:
            P = P - kappa[..., :, None] * Pk[..., None, :]          # (I - kappa k^T) P_hat  (P_hat symmetric)
            P = 0.5 * (P + P.transpose(-1, -2))                    # re-symmetrize

        # --- read ---
        o[:, t] = (qt[..., None] * S).sum(-2)                       # S_t^T q_t   [B,H,V]

    state = (S, P) if output_final_state else None
    return o.to(dtype), state
