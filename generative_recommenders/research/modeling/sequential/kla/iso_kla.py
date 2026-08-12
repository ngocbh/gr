# Isotropic Kalman Linear Attention (ISO-KLA) -- production token mixer.
#
# `IsoKalmanLinearAttention` is the efficient, training-ready ISO-KLA layer. It subclasses
# fla's `KimiDeltaAttention`, inheriting KDA's EXACT peripherals + config (q/k/v proj, short
# conv, low-rank f_proj decay gate, low-rank g_proj output gate, A_log/dt_bias, FusedRMSNormGated
# o_norm, o_proj), and replaces KDA's free `b_proj` write gate with the additive-Kalman scalar
# write strength beta_t (paper appendix A.3).
#
# The two heavy pieces both run on Triton:
#   * beta_t  -- the scalar Kalman scan -- via `iso_beta_chunk` (lit_gpt/kla_ops/iso_chunk.py:
#     forward = chunked two-pass Mobius scan; backward = parallel affine two-pass reverse VJP;
#     a differentiable autograd.Function). It takes the precomputed scalar a_t = mean_i(alpha_i^2)
#     (Option A): a_t is formed here in PyTorch so its grad `da` flows to `alpha` by autograd.
#   * memory  -- being exactly the KDA fixed-gain residual form S_t = (I - beta k k^T) diag(alpha)
#     S_{t-1} + beta k v^T -- via fla's Triton `chunk_kda` (l2-norms q/k and rebuilds the alpha
#     gate in-kernel, so the beta-here / kernel-memory use identical k/alpha).
#
# This is the PRODUCTION path (goal: IsoKLA layer <=1.3x KDA @16k, met). The pure-PyTorch
# reference recurrences (the correctness oracles: `naive_recurrent_iso_kla` /
# `naive_chunk_iso_kla`, and the beta scans `_iso_beta_seq` / `_iso_beta_chunk`) live in
# lit_gpt/kla_ops/iso_naive.py.
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.layers import KimiDeltaAttention

from generative_recommenders.research.modeling.sequential.kla.kla_ops.iso_chunk import iso_beta_chunk, assert_normalized

_ISO_CFG_PRINTED = False  # print the ISO-KLA config once per process
# beta backend: "triton" (differentiable Triton iso_beta_chunk, fwd+bwd) [DEFAULT] or
# "pytorch" (pure-PyTorch Mobius oracle in iso_naive, for parity/debug). Env ISO_BETA_BACKEND.
_ISO_BETA_BACKEND_DEFAULT = os.environ.get("ISO_BETA_BACKEND", "triton").lower()
# Research diagnostics: stash per-forward gain summaries (effective write strength beta_eff=beta,
# gain anisotropy == 0 for the scalar/isotropic gain, omega, r) for the training loop to log to
# wandb. Default ON; set KLA_DIAG_LOG=0 to disable.
_KLA_DIAG_LOG = os.environ.get("KLA_DIAG_LOG", "1") == "1"


class IsoKalmanLinearAttention(KimiDeltaAttention):
    """Isotropic Kalman Linear Attention (production; Triton chunk kernel).

    Subclasses fla :class:`KimiDeltaAttention` for KDA's exact peripherals. The write gate
    ``beta_t`` is the additive-Kalman scalar Kalman gain (paper appendix A.3), computed by the
    Triton :func:`~lit_gpt.kla_ops.iso_chunk.iso_beta_chunk` scan; the memory update runs on fla's
    Triton ``chunk_kda`` (the KDA fixed-gain kernel). KLA-specific params (``qn_proj``,
    ``r_proj``, ``mu_param``) are added on top of the inherited KDA params; KDA's ``b_proj`` is
    removed. Training forward only (``attention_mask=None``, ``num_v_heads == num_heads``).

    ``beta_backend="triton"`` (default) uses the differentiable Triton beta kernel;
    ``"pytorch"`` uses the pure-PyTorch Mobius oracle (parity/debug), overridable via the
    ``ISO_BETA_BACKEND`` env var.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_heads: int = 16,
        expand_v: float = 1.0,
        num_v_heads: int | None = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        r_min: float = 0.05,  # observation-noise floor (r_t = r_min + softplus(r_proj))
        mu: float = 1.0,
        q_min: float = 0.05,  # additive process-noise floor (q_t = q_min + softplus(qn_proj))
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
        beta_backend: str | None = None,
        omega_coupling: bool = False,  # IsoKLA1: q_t = q_min + softplus(qn)*(1 - a_t)
        info_scale_mode: str = "one",  # info-increment scale s in u=s*k^2/r: "dk"->d_k, "sqrt"->sqrt(d_k), "one"->1
        **kwargs,
    ):
        # fla KimiDeltaAttention builds the shared peripherals + config; it only accepts
        # mode in {chunk, fused_recurrent}, so pass "chunk" (we override forward entirely).
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
        self.r_min = r_min
        self.q_min = q_min
        self.omega_coupling = bool(omega_coupling)
        # info-increment scale s in the posterior-info update c_t = 1/b_hat + s*||k||^2/r
        # (u = s*||k||^2/r) AND the gain denom r + s*b_hat*||k||^2. Ablation knob
        # (paper: standardization scale for l2-normed keys). "one" (DEFAULT) = s = 1.0 ->
        # bit-identical to the pre-ablation IsoKLA (||k||^2=1, NO d_k standardization; the
        # production path passes nothing); "sqrt" = sqrt(d_k); "dk" = d_k = K.
        assert info_scale_mode in ("dk", "sqrt", "one"), (
            f"info_scale_mode must be 'dk', 'sqrt', or 'one'; got {info_scale_mode!r}"
        )
        self.info_scale_mode = info_scale_mode
        _K = self.head_k_dim
        self.info_scale = {"dk": float(_K), "sqrt": float(_K) ** 0.5, "one": 1.0}[info_scale_mode]
        self.beta_backend = (beta_backend or _ISO_BETA_BACKEND_DEFAULT).lower()
        assert self.beta_backend in ("triton", "pytorch"), (
            f"beta_backend must be 'triton' or 'pytorch'; got {self.beta_backend!r}"
        )

        # --- the thing that belongs to us: scalar-Kalman write gate (beta_t * k_t) ---
        del self.b_proj  # ISO-KLA's beta comes from the additive-Kalman Mobius scan, not b_proj
        # observation noise r_t = r_min + softplus(r_proj(x)) and ADDITIVE process noise
        # q_t = q_min + softplus(qn_proj(x)) -- token-dependent scalars per head; the per-head
        # average contraction a_t = mean_i(alpha_{t,i}^2) is read off the KDA gate at runtime.
        self.r_proj = nn.Linear(hidden_size, self.num_v_heads, bias=True)
        self.qn_proj = nn.Linear(hidden_size, self.num_v_heads, bias=True)
        # information prior mu = softplus(mu_param) + 0.1 (learnable per head)
        inv_mu = math.log(math.expm1(max(float(mu) - 0.1, 1e-3)))
        self.mu_param = nn.Parameter(torch.full((self.num_v_heads,), inv_mu, dtype=torch.float32))
        self.mu_param._no_weight_decay = True

        global _ISO_CFG_PRINTED
        if not _ISO_CFG_PRINTED:
            print(f"[IsoKalmanLinearAttention] r_min={r_min} q_min={q_min} (mu_init={mu}); "
                  f"Triton chunk kernel (iso_beta_chunk beta + fla chunk_kda memory); "
                  f"beta_backend={self.beta_backend}; omega_coupling={self.omega_coupling}; "
                  f"info_scale_mode={self.info_scale_mode} (s={self.info_scale:g} in u=s*k^2/r; one=1 default); "
                  f"additive-Kalman scan (appendix A.3)")
            _ISO_CFG_PRINTED = True

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        assert attention_mask is None and not use_cache, (
            "IsoKalmanLinearAttention supports only the training forward path."
        )
        assert self.num_v_heads == self.num_heads, "ISO-KLA assumes num_v_heads == num_heads (no GVA)."
        from fla.ops.kda import chunk_kda

        if self.use_short_conv:
            q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False)
            k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cache=None, output_final_state=False)
            v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # alpha (memory transition, KDA gate); additive process noise q_t; obs noise r_t; prior mu
        g_raw = self.f_proj(hidden_states)                                  # raw f_proj (pre-gate)
        alpha = (
            -self.A_log.float().exp().repeat_interleave(self.head_k_dim)
            * F.softplus(g_raw.float() + self.dt_bias)
        ).exp()
        q_base = F.softplus(self.qn_proj(hidden_states).float())                 # [B,T,H]
        r = self.r_min + F.softplus(self.r_proj(hidden_states).float())          # [B,T,H]
        mu = F.softplus(self.mu_param) + 0.1                                     # [H]

        q, k, alpha = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, alpha))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # per-head average contraction a_t = mean_i(alpha_{t,i}^2); additive process noise q_t.
        # omega_coupling (IsoKLA1): q_t = q_min + softplus(qn)*(1 - a_t) -- transition-coupled
        # uncertainty (strongly-forgotten heads gain more process noise). Otherwise q_t is the
        # free additive noise q_min + softplus(qn) (IsoKLA).
        a = (alpha * alpha).mean(-1)                                     # [B,T,H] a_t (autograd-tracked)
        q_noise = self.q_min + (q_base * (1.0 - a) if self.omega_coupling else q_base)  # [B,T,H]

        # --- write gate beta_t from the additive-Kalman scan ---
        # ||k||^2 == 1 precondition: chunk_kda l2-norms k in-kernel (use_qk_l2norm_in_kernel), so
        # the k that drives the memory is unit-norm -- consistent with beta's ||k||^2=1 assumption.
        if self.beta_backend == "triton":
            if os.environ.get("ISO_ASSERT_NORM", "0") == "1":
                assert_normalized(F.normalize(k, p=2, dim=-1))          # debug guard (off by default)
            beta = iso_beta_chunk(a, q_noise, r, mu, out_dtype=torch.float32,
                                  info_scale=self.info_scale)           # [B,T,H]
        else:
            from generative_recommenders.research.modeling.sequential.kla.kla_ops.iso_naive import _iso_beta_chunk
            k_norm = F.normalize(k, p=2, dim=-1)
            beta = _iso_beta_chunk(k_norm, alpha, q_noise, r=r, mu=mu,
                                   info_scale=self.info_scale)          # pure-PyTorch Mobius oracle

        # research diagnostics: for the scalar/isotropic gain kappa=beta*k (||k||=1), the effective
        # write strength beta_eff=<kappa,k>=beta and the gain anisotropy is identically 0 (writes
        # along k) -- the ISO baseline the DiagKLA anisotropy is measured against.
        if _KLA_DIAG_LOG:
            with torch.no_grad():
                bt = beta.float()
                self._kla_diag = {
                    "beta_eff": bt.mean(), "beta_eff_std": bt.std(),
                    "aniso": torch.zeros((), device=bt.device),
                    "omega": q_noise.float().mean(), "omega_std": q_noise.float().std(),
                    "r": r.float().mean(), "r_std": r.float().std(),
                }

        # --- memory update = KDA fixed-gain residual form, on fla's Triton chunk_kda ---
        g_raw_h = rearrange(g_raw, "... (h d) -> ... h d", d=self.head_k_dim)
        o, _ = chunk_kda(
            q=q, k=k, v=v, g=g_raw_h, beta=beta,
            A_log=self.A_log, dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            output_final_state=False,
        )
        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o, None, past_key_values
