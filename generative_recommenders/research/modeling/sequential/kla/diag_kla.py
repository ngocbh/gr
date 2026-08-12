# Kalman Linear Attention (KLA) -- production token mixer (diagonal / per-channel gain).
#
# `DiagonalKalmanLinearAttention` is the efficient, training-ready diagonal KLA layer. It subclasses fla's
# `KimiDeltaAttention` (inheriting KDA's exact peripherals: q/k/v proj, short conv, low-rank
# f_proj decay gate, low-rank g_proj output gate, A_log/dt_bias, FusedRMSNormGated o_norm, o_proj)
# and replaces KDA's fixed scalar write gate with the ANISOTROPIC (per-key-channel) Kalman gain
# kappa_t = beta_ch_t * k_t (paper Prop 3.1), computed by the exact additive-process-noise Mobius
# information scan `_kla_kappa` (lit_gpt/kla_ops/naive.py; fp32 parallel per-channel scan).
#
# The MEMORY update -- KLA's generalized delta rule S_t = (I - kappa k^T) diag(alpha) S_{t-1}
# + kappa v^T -- runs (default, memory_backend="chunk_kalman") on the general Kalman memory
# kernel `chunk_kalman` (lit_gpt/kla_ops/kalman_chunk.py), which feeds the write key kappa and
# read key k INDEPENDENTLY (no 1/beta_ch reciprocal, so it stays finite even when a cold gain
# channel drives beta_ch -> 0). The legacy path (memory_backend="chunk_gdn2") rides the GDN-2
# Triton kernel `chunk_gdn2` (lit_gpt/gdn2_ops) via anchor k_G = kappa, erase gate b_G = 1/beta_ch
# (so b_G * kappa = k), write gate w_G = 1 -- kept for A/B + fallback; both express the same
# recurrence and agree wherever 1/beta_ch is finite. The pure-PyTorch reference recurrences
# (chunk / recurrent modes, the correctness oracles) live in lit_gpt/kla_ops/diag_naive.py
# (`naive_recurrent_kla` / `naive_chunk_kla`).
#
# The per-channel gain kappa is computed by the DIFFERENTIABLE Triton `kla_kappa_chunk` kernel
# (lit_gpt/kla_ops/chunk.py): forward = chunked two-pass per-channel Mobius scan; backward =
# parallel affine two-pass reverse VJP (a differentiable autograd.Function `_KlaKappaChunkFn`).
# This mirrors ISO-KLA's scalar-beta Triton kernel (`iso_beta_chunk`, lit_gpt/kla_ops/iso_chunk.py).
# The memory update rides the GDN-2 kernel, mirroring how ISO-KLA drives fla chunk_kda. This layer
# parallels `lit_gpt/gdn2.py` (layer) + `lit_gpt/gdn2_ops/` (kernels).
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.layers import KimiDeltaAttention

from generative_recommenders.research.modeling.sequential.kla.kla_ops.diag_naive import _kla_kappa

_KLA_CFG_PRINTED = False  # print the KLA config once per process
# kappa backend: "triton" (the DIFFERENTIABLE Triton kla_kappa_chunk kernel -- forward chunked
# two-pass + parallel affine two-pass reverse VJP, wrapped in _KlaKappaChunkFn; fwd+bwd) [DEFAULT]
# or "pytorch" (pure-PyTorch parallel Mobius scan _kla_kappa oracle, for parity/debug/fallback).
# Env KLA_KAPPA_BACKEND. Both are differentiable and drive the training path (grads flow to
# qn_proj/r_proj/mu_param/alpha/k); triton is the fast one (kills the ~8.5x slower pytorch-kappa).
_KLA_KAPPA_BACKEND_DEFAULT = os.environ.get("KLA_KAPPA_BACKEND", "triton").lower()
# memory backend: "chunk_kalman" (the general Kalman memory kernel -- INDEPENDENT write-key
# kappa / read-key k, no 1/beta_ch reciprocal, robust when a gain channel beta_ch -> 0) [DEFAULT]
# or "chunk_gdn2" (the legacy GDN-2 b_G=1/beta_ch trick, kept for A/B + fallback; blows up when
# a cold channel drives beta_ch -> 0). Both express the SAME recurrence
# S=(I - kappa k^T) diag(alpha) S + kappa v^T; they agree wherever 1/beta_ch is finite.
# Env KLA_MEMORY_BACKEND; overridable per-layer via the memory_backend ctor arg.
_KLA_MEMORY_BACKEND_DEFAULT = os.environ.get("KLA_MEMORY_BACKEND", "chunk_kalman").lower()
# Research diagnostics: stash per-forward Kalman-gain summaries (effective write strength
# beta_eff=<kappa,k>, gain anisotropy ||kappa-<kappa,k>k||/||kappa||, per-channel beta_ch, omega, r)
# on the layer for the training loop to log to wandb. **Default OFF** (opt in with KLA_DIAG_LOG=1):
# the block is NOT cheap -- its beta_ch_cold_frac does a per-forward .median() over the full
# [B,T,H,K] tensor, which profiling showed costs 36-43% of the DiagKLA forward at T>=2046 (and drives
# its worse-than-linear T-scaling). Consumers read it via getattr(..., None) so absence is safe.
_KLA_DIAG_LOG = os.environ.get("KLA_DIAG_LOG", "0") == "1"


class DiagonalKalmanLinearAttention(KimiDeltaAttention):
    """Kalman Linear Attention (production; diagonal per-channel gain, GDN-2 kernel memory).

    Subclasses fla :class:`KimiDeltaAttention` for KDA's exact peripherals. The write gain is the
    anisotropic Kalman gain ``kappa_t = beta_ch_t * k_t`` (paper Prop 3.1) from the additive-Kalman
    per-channel Mobius scan (:func:`~lit_gpt.kla_ops.diag_naive._kla_kappa`); the memory update runs
    (default ``memory_backend="chunk_kalman"``) on the general Kalman memory kernel
    :func:`~lit_gpt.kla_ops.kalman_chunk.chunk_kalman`, which takes the write key ``kappa`` and read
    key ``k`` INDEPENDENTLY (no ``1/beta_ch`` reciprocal -- robust when ``beta_ch -> 0``). The legacy
    ``memory_backend="chunk_gdn2"`` path rides the GDN-2 Triton kernel ``chunk_gdn2`` (anchor
    ``k=kappa``, erase ``b=1/beta_ch``, write ``w=1``); both express the same recurrence and agree
    wherever ``1/beta_ch`` is finite. KLA-specific params (``qn_proj``, ``r_proj``, ``mu_param``)
    replace KDA's ``b_proj``.
    Training forward only (``attention_mask=None``, ``num_v_heads == num_heads``). The gain +
    kernel run in fp32 (autocast off) -- matching the pure-PyTorch reference, and because the GDN-2
    kernel builds its WY A-matrix in fp32.

    ``kappa_backend="triton"`` (default) computes ``kappa`` with the DIFFERENTIABLE Triton
    :func:`~lit_gpt.kla_ops.diag_chunk.kla_kappa_chunk` kernel (forward chunked two-pass + parallel
    affine two-pass reverse VJP, wrapped in ``_KlaKappaChunkFn``); grads flow to ``qn_proj``/
    ``r_proj``/``mu_param``/``alpha``/``k`` -- this is the fast TRAINING path. ``kappa_backend=
    "pytorch"`` uses the pure-PyTorch Mobius scan (:func:`~lit_gpt.kla_ops.diag_naive._kla_kappa`)
    as the parity/debug oracle/fallback. Both are differentiable end-to-end; overridable via the
    ``KLA_KAPPA_BACKEND`` env var.
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
        kappa_backend: str | None = None,
        memory_backend: str | None = None,
        omega_coupling: bool = False,  # DiagKLA1: omega_t = q_min + softplus(qn)*(1 - alpha_t^2)
        info_scale_mode: str = "dk",  # info-increment scale s in u=s*k^2/r: "dk"->d_k, "sqrt"->sqrt(d_k), "one"->1
        **kwargs,
    ):
        # fla KimiDeltaAttention builds the shared peripherals + config; it only accepts mode in
        # {chunk, fused_recurrent}, so pass "chunk" (we override forward entirely).
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
        # info-increment scale s in the posterior-info update c_t = 1/p_hat + s*k^2/r (u=s*k^2/r).
        # Ablation knob (paper: standardization scale for l2-normed keys). "dk" (default) = d_k = K
        # -> bit-identical to production (model.py passes nothing); "sqrt" = sqrt(d_k); "one" = 1.
        assert info_scale_mode in ("dk", "sqrt", "one"), (
            f"info_scale_mode must be 'dk', 'sqrt', or 'one'; got {info_scale_mode!r}"
        )
        self.info_scale_mode = info_scale_mode
        _K = self.head_k_dim
        self.info_scale = {"dk": float(_K), "sqrt": float(_K) ** 0.5, "one": 1.0}[info_scale_mode]
        self.kappa_backend = (kappa_backend or _KLA_KAPPA_BACKEND_DEFAULT).lower()
        assert self.kappa_backend in ("triton", "pytorch"), (
            f"kappa_backend must be 'triton' or 'pytorch'; got {self.kappa_backend!r}"
        )
        self.memory_backend = (memory_backend or _KLA_MEMORY_BACKEND_DEFAULT).lower()
        assert self.memory_backend in ("chunk_kalman", "chunk_gdn2"), (
            f"memory_backend must be 'chunk_kalman' or 'chunk_gdn2'; got {self.memory_backend!r}"
        )

        # --- the thing that belongs to us: the anisotropic Kalman gain kappa = beta_ch * k ---
        del self.b_proj  # KLA has no scalar write gate beta
        # additive process noise omega_t (paper notation). The legacy qn_proj/q_min names are kept
        # for checkpoint and CLI compatibility.
        self.qn_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=True),
        )
        # ISOTROPIC gain init (baked in): zero qn_proj's output map so omega_t = q_min + softplus(0) is
        # UNIFORM across key channels at init -> the per-channel Kalman gain starts isotropic and LEARNS
        # anisotropy (fixes DiagKLA's init-seed grokking collapses; syn_expt Phase 5). The `_kla_gain_zero`
        # marker tells the model-wide weight init (SynthLM._init / GPT._init_weights) to KEEP it zero
        # instead of re-randomizing it after construction.
        nn.init.zeros_(self.qn_proj[1].weight)
        nn.init.zeros_(self.qn_proj[1].bias)
        self.qn_proj[1]._kla_gain_zero = True
        # observation noise r_t = r_min + softplus(r_proj(x)) (token-dependent, per head)
        self.r_proj = nn.Linear(hidden_size, self.num_v_heads, bias=True)
        # information prior mu = softplus(mu_param) + 0.1 (learnable per head)
        inv_mu = math.log(math.expm1(max(float(mu) - 0.1, 1e-3)))
        self.mu_param = nn.Parameter(torch.full((self.num_v_heads,), inv_mu, dtype=torch.float32))
        self.mu_param._no_weight_decay = True

        global _KLA_CFG_PRINTED
        if not _KLA_CFG_PRINTED:
            print(f"[DiagonalKalmanLinearAttention] r_min={r_min} q_min={q_min} (mu_init={mu}); "
                  f"per-channel Kalman gain (Mobius scan, Prop 3.1); memory_backend={self.memory_backend} "
                  f"(chunk_kalman = general Kalman memory, independent write-kappa/read-k, no 1/beta_ch); "
                  f"kappa_backend={self.kappa_backend} (triton = differentiable Triton "
                  f"kla_kappa_chunk, the default fwd+bwd path); omega_coupling={self.omega_coupling}; "
                  f"info_scale_mode={self.info_scale_mode} (s={self.info_scale:g} in u=s*k^2/r; dk=d_k default); "
                  f"isotropic gain init (qn_proj zeroed); "
                  f"mirrors fla KimiDeltaAttention peripherals (short conv, low-rank f_proj/g_proj)")
            _KLA_CFG_PRINTED = True

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        assert attention_mask is None and not use_cache, (
            "DiagonalKalmanLinearAttention supports only the training forward path."
        )
        assert self.num_v_heads == self.num_heads, "KLA assumes num_v_heads == num_heads (no GVA)."

        if self.use_short_conv:
            q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False)
            k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cache=None, output_final_state=False)
            v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # alpha (memory transition, KDA gate); additive per-channel process noise omega_t
        alpha = (
            -self.A_log.float().exp().repeat_interleave(self.head_k_dim)
            * F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        ).exp()
        # additive per-channel process noise omega_t. omega_coupling (DiagKLA1): omega_t =
        # q_min + softplus(qn)*(1 - alpha_t^2) -- transition-coupled uncertainty (strongly-forgotten
        # channels gain more process noise). Otherwise omega_t = q_min + softplus(qn) (DiagKLA).
        omega_base = F.softplus(self.qn_proj(hidden_states).float())   # [B,T,gate_dim]
        omega = self.q_min + (omega_base * (1.0 - alpha * alpha) if self.omega_coupling else omega_base)

        q, k, alpha, omega = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim)
                              for x in (q, k, alpha, omega))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        q = F.normalize(q, p=2, dim=-1)                                  # l2-norm keeps gain scale sane
        k = F.normalize(k, p=2, dim=-1)
        r = self.r_min + F.softplus(self.r_proj(hidden_states).float())  # [B,T,num_heads]
        mu = F.softplus(self.mu_param) + 0.1                             # [num_heads]

        # anisotropic Kalman gain kappa = beta_ch * k (per-channel Mobius scan); then the memory
        # update via self.memory_backend (chunk_kalman default: independent write-kappa/read-k;
        # legacy chunk_gdn2: anchor k=kappa, erase b=1/beta_ch, write w=1). fp32 / no autocast.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            if self.kappa_backend == "triton":
                # DIFFERENTIABLE Triton kappa: kla_kappa_chunk routes through _KlaKappaChunkFn
                # (autograd.Function) under grad, so grads flow to qn_proj/r_proj/mu_param/alpha/k;
                # under no_grad it calls the forward kernel directly. This is the fast training path.
                from generative_recommenders.research.modeling.sequential.kla.kla_ops.diag_chunk import kla_kappa_chunk
                kappa, beta_ch = kla_kappa_chunk(k.float(), alpha, omega, r=r, mu=mu,
                                                 info_scale=self.info_scale)
            else:
                kappa, beta_ch, _ = _kla_kappa(k.float(), alpha, omega, r=r, mu=mu,
                                               return_beta_ch=True, info_scale=self.info_scale)
            # research diagnostics (matches scripts/analyses/probe_kla_vs_iso_gain.py definitions):
            # effective write strength beta_eff=<kappa,k> and gain anisotropy (fraction of kappa
            # pointing off the unit key). k is already l2-normalized above.
            if _KLA_DIAG_LOG:
                with torch.no_grad():
                    kf, kap, bc = k.float(), kappa.float(), beta_ch.float()
                    beta_eff = (kap * kf).sum(-1)                                       # [B,T,H]
                    aniso = (kap - beta_eff[..., None] * kf).norm(dim=-1) / kap.norm(dim=-1).clamp_min(1e-9)
                    self._kla_diag = {
                        "beta_eff": beta_eff.mean(), "beta_eff_std": beta_eff.std(),
                        "aniso": aniso.mean(), "aniso_std": aniso.std(),
                        "beta_ch": bc.mean(), "beta_ch_std": bc.std(),
                        "beta_ch_cold_frac": (bc < 0.1 * bc.median()).float().mean(),
                        "omega": omega.float().mean(), "omega_std": omega.float().std(),
                        "r": r.float().mean(), "r_std": r.float().std(),
                    }
            g = alpha.clamp_min(1e-6).log()                             # log-decay [B,T,H,K]
            if self.memory_backend == "chunk_kalman":
                # General Kalman memory: feed the write key (kappa) and read key (k)
                # INDEPENDENTLY -- no 1/beta_ch reciprocal, so it stays finite/exact even
                # when a gain channel beta_ch -> 0 (the cold channels the GDN-2 trick blows
                # up on). Same recurrence S=(I - kappa k^T) diag(alpha) S + kappa v^T,
                # GDN-2-class speed. scale = K^-0.5 (matches chunk_gdn2 / naive_recurrent_kla).
                from generative_recommenders.research.modeling.sequential.kla.kla_ops.kalman_chunk import chunk_kalman
                o, _ = chunk_kalman(
                    q=q.float(), k=k.float(), kappa=kappa.float(), v=v.float(),
                    g=g, scale=self.head_k_dim ** -0.5,
                    use_qk_l2norm_in_kernel=False, output_final_state=False,
                )
            else:
                # legacy GDN-2 b_G=1/beta_ch trick (kept for A/B + fallback): anchor
                # k_G=kappa, erase b_G=1/beta_ch (so b_G*kappa=k), write w_G=1. The
                # reciprocal blows up as beta_ch -> 0.
                from generative_recommenders.research.modeling.sequential.kla.gdn2_ops.chunk_gdn2 import chunk_gdn2
                b_gate = 1.0 / beta_ch                                  # GDN-2 channel-wise erase gate
                o, _ = chunk_gdn2(
                    q=q.float(), k=kappa.float(), v=v.float(), g=g,
                    b=b_gate.float(), w=torch.ones_like(v, dtype=torch.float32),
                    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                    output_final_state=False,
                )
        o = o.to(v.dtype)
        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o, None, past_key_values
