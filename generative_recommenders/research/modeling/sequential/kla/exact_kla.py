# Exact Kalman Linear Attention (Exact KLA) -- dense-covariance token mixer.
#
# `ExactKalmanLinearAttention` is the ground-truth Exact KLA layer (paper Proposition A.1): the
# exact Kalman optimal update run as a layer, with a diagonal transition D_t=diag(alpha_t) and an
# additive DIAGONAL process noise Omega_t=diag(omega_t), but keeping the covariance P_t DENSE
# (d_k x d_k). The gain is the exact anisotropic Kalman gain
# kappa_t = P_hat_t k_t / (r_t + k_t^T P_hat_t k_t) -- NOT the scan-friendly diagonal (diag_kla) or
# isotropic (iso_kla) surrogate.
#
# It subclasses fla's `KimiDeltaAttention` for KDA's exact peripherals + config (q/k/v proj, short
# conv, low-rank f_proj decay gate, low-rank g_proj output gate, A_log/dt_bias, FusedRMSNormGated
# o_norm, o_proj) and replaces KDA's scalar-beta b_proj with the KLA params (qn_proj -> omega,
# r_proj -> r, mu_param -> mu). The recurrence itself lives in lit_gpt/kla_ops/exact_naive.py
# (`naive_recurrent_exact_kla`, the O(d_k^2)/token sequential ground truth) and
# lit_gpt/kla_ops/exact_scan.py (`naive_parallel_exact_kla`, the chunked parallel-covariance scan);
# the fully-fused Triton forward rides `gain_recurrent` (kla_ops/gain_recurrent.py) + `chunk_kalman`
# (kla_ops/kalman_chunk.py). This file only holds the layer.
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.layers import KimiDeltaAttention

from generative_recommenders.research.modeling.sequential.kla.kla_ops.exact_naive import naive_recurrent_exact_kla
from generative_recommenders.research.modeling.sequential.kla.kla_ops.exact_scan import naive_parallel_exact_kla

_EXACT_KLA_CFG_PRINTED = False


class ExactKalmanLinearAttention(KimiDeltaAttention):
    """Exact Kalman Linear Attention token mixer (pure-PyTorch reference, Proposition A.1).

    Subclasses fla's :class:`KimiDeltaAttention` for the EXACT KDA peripherals + config (q/k/v
    proj, short conv, low-rank f_proj decay gate, low-rank g_proj output gate, FusedRMSNormGated,
    o_proj) -- identical to :class:`~lit_gpt.diag_kla.DiagonalKalmanLinearAttention`, so it is an
    iso-parameter twin. It differs ONLY in the recurrence: instead of the scan-friendly
    diagonal/isotropic covariance surrogate, it runs the EXACT dense-covariance Kalman filter
    (:func:`~lit_gpt.kla_ops.exact_naive.naive_recurrent_exact_kla`). O(d_k^2) per token,
    sequential -- a slow ground-truth reference, not a fast trained path; use it at small scale
    (short seq, small head_dim).

    KLA-specific params (qn_proj -> omega, r_proj -> r, mu_param -> mu) are added on top of the
    inherited KDA params; KDA's scalar-beta b_proj is removed. Training forward only (no cache).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_heads: int = 16,
        expand_v: float = 1.0,
        num_v_heads: int | None = None,
        mode: str = "recurrent",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        r_min: float = 0.05,  # observation-noise floor (r_t = r_min + softplus(r_proj))
        mu: float = 1.0,
        q_min: float = 0.05,  # additive process-noise floor (q_t = q_min + softplus(qn_proj))
        dk_calibration: bool = True,
        checkpoint_scan: bool = True,  # recompute the (fp64, memory-heavy) covariance scan in backward
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
        **kwargs,
    ):
        # fla only allows mode in {chunk, fused_recurrent}; pass "chunk" to build the peripherals,
        # then run our own exact recurrence in forward (any incoming mode kwarg is absorbed).
        super().__init__(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            expand_v=expand_v,
            mode="chunk",
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
        )
        # mode="triton_fused" runs the FULLY-FUSED Triton path (gain_recurrent kappa kernel +
        # chunk_kalman memory kernel), K=64 only -- the fast forward, differentiable via a
        # Phase-A recompute backward (slow at long T until a Phase-B Triton backward lands).
        # mode="scan" runs the chunked parallel-covariance gains (exact, ~2x faster fwd+bwd at
        # small head_dim); mode="recurrent" is the sequential ground-truth. All three give
        # identical outputs (scan/recurrent verified ~1e-12; fused ~1e-2 bf16); "recurrent"
        # is the oracle/fallback.
        _mode = str(mode)
        self.mode = _mode if _mode in ("scan", "triton_fused") else "recurrent"
        self.r_min = r_min
        self.q_min = q_min
        self.dk_calibration = dk_calibration
        self.checkpoint_scan = checkpoint_scan

        del self.b_proj  # Exact KLA has no scalar write gate; the dense Kalman gain replaces it.
        # ADDITIVE process noise omega_t = q_min + softplus(qn_proj(x)) (>= 0, per key channel).
        # Same low-rank shape + NATURAL init as DiagonalKalmanLinearAttention (iso-param twin).
        self.qn_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=True),
        )
        # observation noise r_t = r_min + softplus(r_proj(x)) (token-dependent, per head).
        self.r_proj = nn.Linear(hidden_size, self.num_v_heads, bias=True)
        # information prior mu = softplus(mu_param) + 0.1 (learnable per head); P_0 = mu^-1 I.
        inv_mu = math.log(math.expm1(max(float(mu) - 0.1, 1e-3)))
        self.mu_param = nn.Parameter(torch.full((self.num_v_heads,), inv_mu, dtype=torch.float32))
        self.mu_param._no_weight_decay = True

        global _EXACT_KLA_CFG_PRINTED
        if not _EXACT_KLA_CFG_PRINTED:
            print(f"[ExactKalmanLinearAttention] r_min={r_min} q_min={q_min} (mu_init={mu}) "
                  f"dk_calibration={dk_calibration}; DENSE-covariance exact Kalman filter (Prop A.1), "
                  f"O(d_k^2)/token naive recurrence; mirrors fla KimiDeltaAttention peripherals.")
            _EXACT_KLA_CFG_PRINTED = True

    def _decay(self, x, f_proj, A_log, dt_bias):
        # KDA/GDN decay gate: exp(-exp(A_log) * softplus(f_proj(x) + dt_bias)), fp32.
        return (
            -A_log.float().exp().repeat_interleave(self.head_k_dim)
            * F.softplus(f_proj(x).float() + dt_bias)
        ).exp()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        assert attention_mask is None and not use_cache, (
            "ExactKalmanLinearAttention supports only the training forward path."
        )
        assert self.num_v_heads == self.num_heads, "naive Exact KLA assumes num_v_heads == num_heads (no GVA)."
        if self.use_short_conv:
            q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False)
            k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cache=None, output_final_state=False)
            v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        alpha = self._decay(hidden_states, self.f_proj, self.A_log, self.dt_bias)          # [B,T,gate_dim]
        omega = self.q_min + F.softplus(self.qn_proj(hidden_states).float())              # [B,T,gate_dim]
        q, k, alpha, omega = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim)
                              for x in (q, k, alpha, omega))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        r = self.r_min + F.softplus(self.r_proj(hidden_states).float())                   # [B,T,num_heads]
        mu = F.softplus(self.mu_param) + 0.1                                              # [num_heads]

        # Run the exact dense-covariance filter in fp32 (autocast off). mode="scan" uses the chunked
        # parallel-covariance gains (O(T/C) sequential depth); "triton_fused" uses the fully-fused
        # Triton path (gain_recurrent kappa kernel + chunk_kalman memory kernel, K=64 only);
        # "recurrent" the O(T) reference.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            if self.mode == "triton_fused":
                from generative_recommenders.research.modeling.sequential.kla.kla_ops.gain_recurrent import gain_recurrent
                from generative_recommenders.research.modeling.sequential.kla.kla_ops.kalman_chunk import chunk_kalman
                # q, k already F.normalized above; kappa is scale-free. Pass RAW alpha to the gain
                # kernel; pass g=log(alpha) to the memory kernel (its decay convention). scale is
                # baked into the memory kernel (do NOT pre-scale q). use_qk_l2norm_in_kernel=False
                # to avoid double-norm.
                scale = self.head_k_dim ** -0.5
                kappa = gain_recurrent(k, alpha, omega, r, mu=mu,
                                       dk_calibration=self.dk_calibration)
                g_log = alpha.clamp_min(1e-30).log()
                o, _ = chunk_kalman(q, k, kappa, v, g=g_log, scale=scale,
                                    use_qk_l2norm_in_kernel=False, backend="triton")
            elif self.mode == "scan":
                o, _ = naive_parallel_exact_kla(q, k, v, alpha, omega, r=r, mu=mu,
                                                dk_calibration=self.dk_calibration,
                                                checkpoint=self.checkpoint_scan)
            else:
                o, _ = naive_recurrent_exact_kla(q, k, v, alpha, omega, r=r, mu=mu,
                                                 dk_calibration=self.dk_calibration)

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o, None, past_key_values
