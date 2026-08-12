# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Implements HSTU (Hierarchical Sequential Transduction Unit) in
Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
(https://arxiv.org/abs/2402.17152, ICML'24).
"""

import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.research.modeling.sequential.utils import (
    get_current_embeddings,
)
from generative_recommenders.research.modeling.similarity_module import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from generative_recommenders.research.rails.similarities.module import SimilarityModule


TIMESTAMPS_KEY = "timestamps"


class RelativeAttentionBiasModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias


HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

_SIGNED_ADDITIVE_NORMALIZATIONS = (
    "signed_additive_identity",
    "signed_additive_tanh",
    "signed_additive_abs_tanh",
    "signed_additive_abs_coefficient_oracle",
)


def _forgetting_log_survival(log_forget: torch.Tensor) -> torch.Tensor:
    """Build pathwise log-survival factors from per-token log forget gates.

    Args:
        log_forget: [B, N, H] log gates, with values at most zero.
    Returns:
        [B, H, N, N] where entry (i, j) is the sum of log gates j+1..i.
        Entries above the causal diagonal are zero and are removed by the
        attention mask at the call site.
    """
    accumulation_dtype = (
        torch.float64 if log_forget.dtype == torch.float64 else torch.float32
    )
    prefix = torch.cumsum(log_forget.to(accumulation_dtype), dim=1).transpose(1, 2)
    log_survival = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
    return torch.clamp_max(log_survival, 0.0)


def _forgetting_survival(log_forget: torch.Tensor) -> torch.Tensor:
    """Build pathwise survival factors from per-token log forget gates."""
    return torch.exp(_forgetting_log_survival(log_forget)).to(log_forget.dtype)


def _per_head_softmax_attention(
    padded_q: torch.Tensor,
    padded_k: torch.Tensor,
    padded_v: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    relative_attention_bias: Optional[torch.Tensor],
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    temperature: Optional[float] = None,
    log_forget: Optional[torch.Tensor] = None,
    valid_lengths: Optional[torch.Tensor] = None,
    scale_relative_attention_bias: bool = True,
) -> torch.Tensor:
    """Reference per-head softmax attention for the research HSTU shell."""
    B, n, _ = padded_q.shape
    q = padded_q.view(B, n, num_heads, attention_dim)
    k = padded_k.view(B, n, num_heads, attention_dim)
    v = padded_v.view(B, n, num_heads, linear_dim)
    logits = torch.einsum("bnhd,bmhd->bhnm", q, k)
    if relative_attention_bias is not None and scale_relative_attention_bias:
        logits = logits + relative_attention_bias.unsqueeze(1)
    logits = logits / (temperature or math.sqrt(attention_dim))
    if relative_attention_bias is not None and not scale_relative_attention_bias:
        logits = logits + relative_attention_bias.unsqueeze(1)
    if log_forget is not None:
        expected_shape = (B, n, num_heads)
        if tuple(log_forget.shape) != expected_shape:
            raise ValueError(
                f"log_forget must have shape {expected_shape}, "
                f"got {tuple(log_forget.shape)}"
            )
        log_survival = _forgetting_log_survival(log_forget)
        logits = logits.to(log_survival.dtype) + log_survival

    if invalid_attn_mask.dim() == 2:
        keep_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    elif invalid_attn_mask.dim() == 3:
        keep_mask = invalid_attn_mask.unsqueeze(1)
    else:
        raise ValueError(
            f"invalid_attn_mask must have rank 2 or 3, got {invalid_attn_mask.dim()}"
        )
    keep_mask = keep_mask != 0
    if valid_lengths is not None:
        if tuple(valid_lengths.shape) != (B,):
            raise ValueError(
                f"valid_lengths must have shape {(B,)}, "
                f"got {tuple(valid_lengths.shape)}"
            )
        valid_keys = torch.arange(n, device=logits.device).unsqueeze(0) < (
            valid_lengths.to(logits.device).unsqueeze(1)
        )
        keep_mask = keep_mask & valid_keys.unsqueeze(1).unsqueeze(1)

    has_valid_key = keep_mask.any(dim=-1, keepdim=True)
    masked_logits = logits.masked_fill(~keep_mask, -torch.inf)
    safe_logits = torch.where(has_valid_key, masked_logits, torch.zeros_like(logits))
    weights = F.softmax(safe_logits, dim=-1).masked_fill(~keep_mask, 0.0)
    value_dtype = v.dtype
    if weights.dtype != value_dtype:
        v = v.to(weights.dtype)
    output = torch.einsum("bhnm,bmhd->bnhd", weights, v)
    return output.reshape(B, n, num_heads * linear_dim).to(value_dtype)


def _apply_hstu_score_kernel(scores: torch.Tensor, kernel: str) -> torch.Tensor:
    """Apply a pointwise HSTU score kernel without changing head layout."""
    if kernel == "silu":
        return F.silu(scores)
    if kernel == "tanh":
        return torch.tanh(scores)
    if kernel == "taylor1":
        return scores * 0.5
    if kernel == "taylor2":
        compute_scores = (
            scores.float()
            if scores.dtype in (torch.float16, torch.bfloat16)
            else scores
        )
        output = 0.5 * compute_scores + 0.25 * compute_scores.square()
        return output.to(scores.dtype)
    raise ValueError(f"Unknown HSTU score kernel {kernel}")


def _per_head_hstu_weights(
    padded_q: torch.Tensor,
    padded_k: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    relative_attention_bias: Optional[torch.Tensor],
    score_kernel: str = "silu",
) -> torch.Tensor:
    """Build unnormalized per-head HSTU weights with fixed-length scaling."""
    n = padded_q.size(1)
    scores = torch.einsum("bnhd,bmhd->bhnm", padded_q, padded_k)
    if relative_attention_bias is not None:
        scores = scores + relative_attention_bias.unsqueeze(1)
    weights = _apply_hstu_score_kernel(scores, score_kernel) / n

    if invalid_attn_mask.dim() == 2:
        keep_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    elif invalid_attn_mask.dim() == 3:
        keep_mask = invalid_attn_mask.unsqueeze(1)
    else:
        raise ValueError(
            f"invalid_attn_mask must have rank 2 or 3, got {invalid_attn_mask.dim()}"
        )
    return weights * keep_mask


def _per_head_additive_dot_attention(
    padded_q: torch.Tensor,
    padded_k: torch.Tensor,
    padded_v: torch.Tensor,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
) -> torch.Tensor:
    """Reference additive first-moment attention with an inclusive causal scan."""
    B, n, _ = padded_q.shape
    accumulation_dtype = (
        torch.float64 if padded_q.dtype == torch.float64 else torch.float32
    )
    q = padded_q.reshape(B, n, num_heads, attention_dim).to(accumulation_dtype)
    k = padded_k.reshape(B, n, num_heads, attention_dim).to(accumulation_dtype)
    v = padded_v.reshape(B, n, num_heads, linear_dim).to(accumulation_dtype)

    updates = torch.einsum("bnhd,bnhe->bnhde", k, v)
    states = torch.cumsum(updates, dim=1)
    output = torch.einsum("bnhd,bnhde->bnhe", q, states) * (0.5 / n)
    return output.reshape(B, n, num_heads * linear_dim).to(padded_q.dtype)


def _validate_signed_feature_gamma(gamma: float) -> float:
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value <= 0.0:
        raise ValueError(
            f"signed feature gamma must be finite and positive, got {gamma}"
        )
    return gamma_value


def _apply_signed_additive_feature_map(
    x: torch.Tensor,
    feature_map: str,
    gamma: float,
) -> torch.Tensor:
    """Apply a fixed-width feature map used by signed additive attention.

    Tanh is applied coordinatewise before the feature inner product; this does
    not approximate a post-dot tanh or SiLU score kernel.
    """
    gamma_value = _validate_signed_feature_gamma(gamma)
    if feature_map == "identity":
        return x
    if feature_map == "tanh":
        return torch.tanh(gamma_value * x)
    if feature_map == "abs_tanh":
        return torch.abs(torch.tanh(gamma_value * x))
    raise ValueError(f"Unknown signed additive feature map {feature_map}")


def _per_head_signed_additive_feature_attention(
    padded_q: torch.Tensor,
    padded_k: torch.Tensor,
    padded_v: torch.Tensor,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    feature_map: str,
    gamma: float = 1.0,
    valid_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inclusive causal feature-memory scan with fixed ``0.5 / N`` scaling."""
    B, n, q_width = padded_q.shape
    if n <= 0:
        raise ValueError("signed additive attention requires a non-empty sequence")
    if q_width != num_heads * attention_dim or tuple(padded_k.shape) != (
        B,
        n,
        q_width,
    ):
        raise ValueError("padded_q and padded_k must match the configured head shape")
    if tuple(padded_v.shape) != (B, n, num_heads * linear_dim):
        raise ValueError("padded_v must match the configured value head shape")

    accumulation_dtype = (
        torch.float64 if padded_q.dtype == torch.float64 else torch.float32
    )
    q = padded_q.reshape(B, n, num_heads, attention_dim).to(accumulation_dtype)
    k = padded_k.reshape(B, n, num_heads, attention_dim).to(accumulation_dtype)
    v = padded_v.reshape(B, n, num_heads, linear_dim).to(accumulation_dtype)
    if valid_lengths is not None:
        if tuple(valid_lengths.shape) != (B,):
            raise ValueError(
                f"valid_lengths must have shape {(B,)}, "
                f"got {tuple(valid_lengths.shape)}"
            )
        valid = torch.arange(n, device=q.device).unsqueeze(0) < valid_lengths.to(
            q.device
        ).unsqueeze(1)
        q = torch.where(valid.view(B, n, 1, 1), q, torch.zeros_like(q))
        k = torch.where(valid.view(B, n, 1, 1), k, torch.zeros_like(k))
        v = torch.where(valid.view(B, n, 1, 1), v, torch.zeros_like(v))

    q_features = _apply_signed_additive_feature_map(q, feature_map, gamma)
    k_features = _apply_signed_additive_feature_map(k, feature_map, gamma)
    updates = torch.einsum("bnhd,bnhe->bnhde", k_features, v)
    states = torch.cumsum(updates, dim=1)
    output = torch.einsum("bnhd,bnhde->bnhe", q_features, states) * (0.5 / n)
    return output.reshape(B, n, num_heads * linear_dim).to(padded_q.dtype)


def _per_head_signed_additive_abs_coefficient_oracle(
    padded_q: torch.Tensor,
    padded_k: torch.Tensor,
    padded_v: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    gamma: float = 1.0,
    valid_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quadratic diagnostic using the magnitude of each signed pair coefficient."""
    B, n, q_width = padded_q.shape
    if n <= 0:
        raise ValueError("signed additive attention requires a non-empty sequence")
    if q_width != num_heads * attention_dim or tuple(padded_k.shape) != (
        B,
        n,
        q_width,
    ):
        raise ValueError("padded_q and padded_k must match the configured head shape")
    if tuple(padded_v.shape) != (B, n, num_heads * linear_dim):
        raise ValueError("padded_v must match the configured value head shape")

    accumulation_dtype = (
        torch.float64 if padded_q.dtype == torch.float64 else torch.float32
    )
    q = padded_q.reshape(B, n, num_heads, attention_dim).to(accumulation_dtype)
    k = padded_k.reshape(B, n, num_heads, attention_dim).to(accumulation_dtype)
    v = padded_v.reshape(B, n, num_heads, linear_dim).to(accumulation_dtype)
    q_features = _apply_signed_additive_feature_map(q, "tanh", gamma)
    k_features = _apply_signed_additive_feature_map(k, "tanh", gamma)
    coefficients = torch.einsum("bnhd,bmhd->bhnm", q_features, k_features).abs()

    if invalid_attn_mask.dim() == 2:
        keep_mask = invalid_attn_mask.view(1, 1, n, n)
    elif invalid_attn_mask.dim() == 3:
        if tuple(invalid_attn_mask.shape) != (B, n, n):
            raise ValueError(
                f"rank-three invalid_attn_mask must have shape {(B, n, n)}, "
                f"got {tuple(invalid_attn_mask.shape)}"
            )
        keep_mask = invalid_attn_mask.unsqueeze(1)
    else:
        raise ValueError(
            f"invalid_attn_mask must have rank 2 or 3, got {invalid_attn_mask.dim()}"
        )
    keep_mask = keep_mask.to(device=coefficients.device, dtype=torch.bool)
    if valid_lengths is not None:
        if tuple(valid_lengths.shape) != (B,):
            raise ValueError(
                f"valid_lengths must have shape {(B,)}, "
                f"got {tuple(valid_lengths.shape)}"
            )
        valid = torch.arange(n, device=coefficients.device).unsqueeze(
            0
        ) < valid_lengths.to(coefficients.device).unsqueeze(1)
        keep_mask = keep_mask & valid.view(B, 1, n, 1)
        keep_mask = keep_mask & valid.view(B, 1, 1, n)
    weights = torch.where(keep_mask, coefficients, torch.zeros_like(coefficients))
    output = torch.einsum("bhnm,bmhe->bnhe", weights * (0.5 / n), v)
    return output.reshape(B, n, num_heads * linear_dim).to(padded_q.dtype)


def _attach_zero_relative_bias_dependency(
    output: torch.Tensor,
    relative_attention_bias: Optional[RelativeAttentionBiasModule],
) -> torch.Tensor:
    """Keep dormant RAB parameters in DDP's graph without building pairwise bias."""
    if relative_attention_bias is None:
        return output
    zero = torch.zeros((), dtype=output.dtype, device=output.device)
    for parameter in relative_attention_bias.parameters():
        zero = zero + parameter.sum().to(output.dtype) * 0.0
    return output + zero


def _per_head_forgetting_tail_attention(
    padded_q: torch.Tensor,
    padded_k: torch.Tensor,
    padded_v: torch.Tensor,
    survival: torch.Tensor,
    old_mask: torch.Tensor,
    feature_map: str = "identity",
    gamma: float = 1.0,
) -> torch.Tensor:
    """Quadratic quality oracle for the old-key forgotten feature moment.

    The arithmetic is FP32 (FP64 for tests). A fused delayed-state recurrence
    can replace this pairwise reference after the mechanism is validated.
    """
    n = padded_q.size(1)
    accumulation_dtype = (
        torch.float64 if padded_q.dtype == torch.float64 else torch.float32
    )
    q_features = _apply_signed_additive_feature_map(
        padded_q.to(accumulation_dtype), feature_map, gamma
    )
    k_features = _apply_signed_additive_feature_map(
        padded_k.to(accumulation_dtype), feature_map, gamma
    )
    tail_scores = torch.einsum(
        "bnhd,bmhd->bhnm",
        q_features,
        k_features,
    )
    tail_weights = (
        tail_scores
        * survival.to(accumulation_dtype)
        * old_mask.to(accumulation_dtype)
        * (0.5 / n)
    )
    return torch.einsum(
        "bhnm,bmhe->bnhe", tail_weights, padded_v.to(accumulation_dtype)
    )


def _per_head_recurrent_forgetting_tail_attention(
    padded_q: torch.Tensor,
    padded_k: torch.Tensor,
    padded_v: torch.Tensor,
    log_forget: torch.Tensor,
    window_size: int,
    feature_map: str = "identity",
    gamma: float = 1.0,
    valid_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Delayed feature-state reference for a causal, right-padded old tail.

    For ``t = i - window_size``, the state contains keys through ``t`` and the
    lag survival ``F[i, t]`` multiplies the already-mapped query feature. This
    helper intentionally does not accept arbitrary pair-mask holes.
    """
    B, n, num_heads, attention_dim = padded_q.shape
    if n <= 0:
        raise ValueError("recurrent forgetting tail requires a non-empty sequence")
    if tuple(padded_k.shape) != (B, n, num_heads, attention_dim):
        raise ValueError("padded_q and padded_k must have the same shape")
    if padded_v.shape[:3] != (B, n, num_heads):
        raise ValueError("padded_v must match the batch, sequence, and head dimensions")
    if tuple(log_forget.shape) != (B, n, num_heads):
        raise ValueError(
            f"log_forget must have shape {(B, n, num_heads)}, "
            f"got {tuple(log_forget.shape)}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if valid_lengths is not None and tuple(valid_lengths.shape) != (B,):
        raise ValueError(
            f"valid_lengths must have shape {(B,)}, "
            f"got {tuple(valid_lengths.shape)}"
        )

    accumulation_dtype = (
        torch.float64 if padded_q.dtype == torch.float64 else torch.float32
    )
    q = padded_q.to(accumulation_dtype)
    k = padded_k.to(accumulation_dtype)
    v = padded_v.to(accumulation_dtype)
    log_f = log_forget.to(accumulation_dtype)
    if valid_lengths is None:
        valid = torch.ones(B, n, dtype=torch.bool, device=padded_q.device)
    else:
        valid = torch.arange(n, device=padded_q.device).unsqueeze(0) < (
            valid_lengths.to(padded_q.device).unsqueeze(1)
        )
    q = torch.where(valid.view(B, n, 1, 1), q, torch.zeros_like(q))
    k = torch.where(valid.view(B, n, 1, 1), k, torch.zeros_like(k))
    v = torch.where(valid.view(B, n, 1, 1), v, torch.zeros_like(v))
    q_features = _apply_signed_additive_feature_map(q, feature_map, gamma)
    k_features = _apply_signed_additive_feature_map(k, feature_map, gamma)

    if window_size >= n:
        dependency = (q_features.sum() + k_features.sum() + v.sum() + log_f.sum()) * 0.0
        return torch.zeros_like(v) + dependency

    linear_dim = padded_v.size(-1)
    state = torch.zeros(
        B,
        num_heads,
        attention_dim,
        linear_dim,
        dtype=accumulation_dtype,
        device=padded_q.device,
    )
    tail_rows = []
    for key_idx in range(n - window_size):
        key_is_valid = valid[:, key_idx].view(B, 1)
        forget = torch.where(
            key_is_valid,
            torch.exp(log_f[:, key_idx]),
            torch.ones_like(log_f[:, key_idx]),
        )
        update = torch.einsum("bhd,bhe->bhde", k_features[:, key_idx], v[:, key_idx])
        state = forget.view(B, num_heads, 1, 1) * state + update

        query_idx = key_idx + window_size
        query_is_valid = valid[:, query_idx].view(B, 1)
        lag_log_survival = log_f[:, key_idx + 1 : query_idx + 1].sum(dim=1)
        lag_log_survival = torch.where(
            query_is_valid,
            torch.clamp_max(lag_log_survival, 0.0),
            torch.zeros_like(lag_log_survival),
        )
        survived_query = q_features[:, query_idx] * torch.exp(
            lag_log_survival
        ).unsqueeze(-1)
        tail_row = torch.einsum("bhd,bhde->bhe", survived_query, state)
        tail_rows.append(tail_row * query_is_valid.to(accumulation_dtype).unsqueeze(-1))

    prefix = torch.zeros_like(v[:, :window_size])
    tail = torch.stack(tail_rows, dim=1) * (0.5 / n)
    return torch.cat((prefix, tail), dim=1)


def _per_head_local_forgetting_attention(
    padded_q: torch.Tensor,
    padded_k: torch.Tensor,
    padded_v: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    relative_attention_bias: Optional[torch.Tensor],
    log_forget: torch.Tensor,
    window_size: int,
    valid_lengths: Optional[torch.Tensor] = None,
    tail_gain: Optional[torch.Tensor] = None,
    tail_feature_map: str = "identity",
    signed_feature_gamma: float = 1.0,
) -> torch.Tensor:
    """Reference local FoHSTU with an optional disjoint old-key moment tail.

    The exact local term covers ``i - window_size < j <= i`` and keeps HSTU's
    pairwise relative bias. The tail covers ``j <= i - window_size`` with a
    fixed additive feature moment and no relative bias. Both terms use the same
    content-dependent path survival and fixed ``1 / N`` scale.

    Args:
        padded_q: [B, N, H, Dq].
        padded_k: [B, N, H, Dq].
        padded_v: [B, N, H, Dv].
        invalid_attn_mask: [N, N] or [B, N, N] keep mask.
        relative_attention_bias: optional [B, N, N] pairwise bias.
        log_forget: [B, N, H] log forget gates.
        window_size: positive number of recent keys in the exact local term.
        valid_lengths: optional [B] unpadded sequence lengths.
        tail_gain: optional [H] gain. If omitted, the tail is not evaluated.
        tail_feature_map: fixed feature map for the optional old tail.
        signed_feature_gamma: fixed positive scale used by bounded maps.
    Returns:
        [B, N, H, Dv] attention output.
    """
    B, n, num_heads, attention_dim = padded_q.shape
    if tuple(padded_k.shape) != (B, n, num_heads, attention_dim):
        raise ValueError("padded_q and padded_k must have the same shape")
    if padded_v.shape[:3] != (B, n, num_heads):
        raise ValueError("padded_v must match the batch, sequence, and head dimensions")
    if tuple(log_forget.shape) != (B, n, num_heads):
        raise ValueError(
            f"log_forget must have shape {(B, n, num_heads)}, "
            f"got {tuple(log_forget.shape)}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if tail_gain is not None and tuple(tail_gain.shape) != (num_heads,):
        raise ValueError(
            f"tail_gain must have shape {(num_heads,)}, got {tuple(tail_gain.shape)}"
        )

    if invalid_attn_mask.dim() == 2:
        keep_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    elif invalid_attn_mask.dim() == 3:
        keep_mask = invalid_attn_mask.unsqueeze(1)
    else:
        raise ValueError(
            f"invalid_attn_mask must have rank 2 or 3, got {invalid_attn_mask.dim()}"
        )
    keep_mask = keep_mask.to(device=padded_q.device, dtype=torch.bool)

    positions = torch.arange(n, device=padded_q.device)
    distances = positions.unsqueeze(1) - positions.unsqueeze(0)
    local_mask = (
        keep_mask
        & (distances >= 0).view(1, 1, n, n)
        & (distances < window_size).view(1, 1, n, n)
    )
    old_mask = keep_mask & (distances >= window_size).view(1, 1, n, n)
    valid_queries: Optional[torch.Tensor] = None
    if valid_lengths is not None:
        if tuple(valid_lengths.shape) != (B,):
            raise ValueError(
                f"valid_lengths must have shape {(B,)}, "
                f"got {tuple(valid_lengths.shape)}"
            )
        valid_positions = positions.unsqueeze(0) < valid_lengths.to(
            padded_q.device
        ).unsqueeze(1)
        valid_queries = valid_positions.view(B, 1, n, 1)
        valid_pairs = valid_queries & valid_positions.view(B, 1, 1, n)
        local_mask = local_mask & valid_pairs
        old_mask = old_mask & valid_pairs

    survival = _forgetting_survival(log_forget)
    local_scores = torch.einsum("bnhd,bmhd->bhnm", padded_q, padded_k)
    if relative_attention_bias is not None:
        local_scores = local_scores + relative_attention_bias.unsqueeze(1)
    local_weights = (
        F.silu(local_scores)
        * survival.to(local_scores.dtype)
        * local_mask.to(local_scores.dtype)
        / n
    )
    local_output = torch.einsum("bhnm,bmhe->bnhe", local_weights, padded_v)

    if tail_gain is None:
        return local_output

    tail_output = _per_head_forgetting_tail_attention(
        padded_q=padded_q,
        padded_k=padded_k,
        padded_v=padded_v,
        survival=survival,
        old_mask=old_mask,
        feature_map=tail_feature_map,
        gamma=signed_feature_gamma,
    )
    accumulation_dtype = tail_output.dtype
    combined = local_output.to(accumulation_dtype) + tail_output * tail_gain.to(
        accumulation_dtype
    ).view(1, 1, num_heads, 1)
    if valid_queries is not None:
        combined = combined * valid_queries.transpose(1, 2).to(combined.dtype)
    return combined.to(local_output.dtype)


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cached_q: Optional[torch.Tensor],
    cached_k: Optional[torch.Tensor],
    delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
    x_offsets: torch.Tensor,
    all_timestamps: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    rel_attn_bias: Optional[RelativeAttentionBiasModule],
    forget_weight: Optional[torch.Tensor] = None,
    forget_bias: Optional[torch.Tensor] = None,
    score_kernel: str = "silu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B: int = x_offsets.size(0) - 1
    n: int = invalid_attn_mask.size(-1)
    if delta_x_offsets is not None:
        padded_q, padded_k = cached_q, cached_k
        flattened_offsets = delta_x_offsets[1] + torch.arange(
            start=0,
            end=B * n,
            step=n,
            device=delta_x_offsets[1].device,
            dtype=delta_x_offsets[1].dtype,
        )
        assert isinstance(padded_q, torch.Tensor)
        assert isinstance(padded_k, torch.Tensor)
        padded_q = (
            padded_q.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=q,
            )
            .view(B, n, -1)
        )
        padded_k = (
            padded_k.view(B * n, -1)
            .index_copy_(
                dim=0,
                index=flattened_offsets,
                source=k,
            )
            .view(B, n, -1)
        )
    else:
        padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
            values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )
        padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
            values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        )

    padded_q_heads = padded_q.view(B, n, num_heads, attention_dim)
    padded_k_heads = padded_k.view(B, n, num_heads, attention_dim)
    relative_attention_bias = (
        rel_attn_bias(all_timestamps)
        if all_timestamps is not None and rel_attn_bias is not None
        else None
    )
    qk_attn = _per_head_hstu_weights(
        padded_q=padded_q_heads,
        padded_k=padded_k_heads,
        invalid_attn_mask=invalid_attn_mask,
        relative_attention_bias=relative_attention_bias,
        score_kernel=score_kernel,
    )
    if forget_weight is not None:
        assert forget_bias is not None
        forget_logits = torch.einsum(
            "bnhd,hd->bnh", padded_k_heads, forget_weight
        ) + forget_bias.view(1, 1, num_heads)
        qk_attn = qk_attn * _forgetting_survival(F.logsigmoid(forget_logits)).to(
            qk_attn.dtype
        )
    attn_output = torch.ops.fbgemm.dense_to_jagged(
        torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
                B, n, num_heads, linear_dim
            ),
        ).reshape(B, n, num_heads * linear_dim),
        [x_offsets],
    )[0]
    return attn_output, padded_q, padded_k


class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
        kda_time_gate: str = "continuous",
        kda_gate_rank: int = 0,
        kda_o_rank: int = 0,
        kla_omega_coupling: bool = False,
        forgetting_min_period: float = 8.0,
        forgetting_max_period: float = 256.0,
        hybrid_window_size: int = 64,
        softmax_temperature: float = 0.0,
        signed_feature_gamma: float = 1.0,
        hybrid_tail_feature_map: str = "identity",
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        if self._normalization == "additive_dot" and self._rel_attn_bias is not None:
            raise ValueError(
                "additive_dot cannot represent pairwise relative attention bias; "
                "disable relative attention bias"
            )
        if not math.isfinite(softmax_temperature) or softmax_temperature < 0:
            raise ValueError(
                "softmax_temperature must be finite and non-negative, "
                f"got {softmax_temperature}"
            )
        self._softmax_temperature: float = softmax_temperature
        self._signed_feature_gamma: float = _validate_signed_feature_gamma(
            signed_feature_gamma
        )
        if (
            self._normalization
            in ("local_forgetting_rel_bias", "hybrid_forgetting_rel_bias")
            and hybrid_window_size <= 0
        ):
            raise ValueError(
                f"hybrid_window_size must be positive, got {hybrid_window_size}"
            )
        self._hybrid_window_size: int = hybrid_window_size
        if self._normalization in (
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        ) and hybrid_tail_feature_map not in ("identity", "tanh", "abs_tanh"):
            raise ValueError(
                "hybrid_tail_feature_map must be identity, tanh, or abs_tanh, "
                f"got {hybrid_tail_feature_map}"
            )
        self._hybrid_tail_feature_map: str = hybrid_tail_feature_map
        self._kda_time_gate: str = kda_time_gate
        self._kla_omega_coupling: bool = kla_omega_coupling
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 2 * num_heads
                        + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        _o_in: int = linear_hidden_dim * num_heads * (3 if concat_ua else 1)
        # Output projection. Low-rank (in->r->D) when kda_o_rank>0, so we can
        # match HSTU params by mildly compressing _o AND the KDA gate together
        # (rather than crushing one), while leaving uvqk full.
        if kda_o_rank and kda_o_rank > 0:
            self._o = torch.nn.Sequential(
                torch.nn.Linear(_o_in, kda_o_rank, bias=False),
                torch.nn.Linear(kda_o_rank, embedding_dim, bias=True),
            )
            torch.nn.init.xavier_uniform_(self._o[0].weight)
            torch.nn.init.xavier_uniform_(self._o[1].weight)
        else:
            self._o = torch.nn.Linear(in_features=_o_in, out_features=embedding_dim)
            torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon

        if self._normalization in (
            "forgetting_rel_bias",
            "fixed_forgetting_rel_bias",
            "forgetting_softmax_rel_bias",
            "fixed_forgetting_softmax_rel_bias",
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        ):
            if (
                not math.isfinite(forgetting_min_period)
                or not math.isfinite(forgetting_max_period)
                or forgetting_min_period <= 0
                or forgetting_max_period < forgetting_min_period
            ):
                raise ValueError(
                    "forgetting periods must be finite and satisfy "
                    f"0 < min <= max, got {forgetting_min_period}, "
                    f"{forgetting_max_period}"
                )
            periods = torch.logspace(
                math.log10(forgetting_min_period),
                math.log10(forgetting_max_period),
                steps=num_heads,
                dtype=torch.float32,
            )
            initial_forget = torch.exp(-periods.reciprocal())
            initial_bias = torch.logit(initial_forget)
            forget_weight = torch.zeros(num_heads, attention_dim, dtype=torch.float32)
            if self._normalization in (
                "forgetting_rel_bias",
                "forgetting_softmax_rel_bias",
                "local_forgetting_rel_bias",
                "hybrid_forgetting_rel_bias",
            ):
                self._forget_weight = torch.nn.Parameter(forget_weight)
                self._forget_bias = torch.nn.Parameter(initial_bias)
            else:
                self.register_buffer("_forget_weight", forget_weight)
                self.register_buffer("_forget_bias", initial_bias)

        if self._normalization in (
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        ):
            # Zero construction consumes no RNG, keeping the two arms matched.
            self._hybrid_tail_rho = torch.nn.Parameter(
                torch.zeros(num_heads, dtype=torch.float32)
            )

        # KDA / IsoKLA linear-attention core: keep HSTU's uvqk / U-gate / _o /
        # residual and swap ONLY the `SiLU(QKᵀ)/N·V + rel-bias` operator for a
        # gated delta-rule scan over HSTU's own Q, K, V (fla chunk_kda memory).
        # The delta rule needs a per-head write strength β and a per-channel
        # forget gate g (+ A_log, dt_bias); rel-bias is dropped. "kda" makes β a
        # free b_proj logit; "iso_kla" (vendored hyper-delta-net IsoKLA) makes β
        # the additive-Kalman gain from the Triton iso_beta_chunk scan (adds
        # r_proj/qn_proj/mu instead of b_proj). fla/kla imported here (not at
        # module top) so this file stays importable on GPU-less nodes.
        if self._normalization in ("kda", "iso_kla", "diag_kla", "exact_kla"):
            import math as _math

            H, dk = num_heads, attention_dim
            if self._normalization in ("kda", "iso_kla"):
                from fla.ops.kda import chunk_kda as _chunk_kda

                self._chunk_kda = _chunk_kda
            # Forget-gate projection. Low-rank (D->r->H*dk) when kda_gate_rank>0
            # (as in KDA's real f_proj), else a full Linear. Low rank lets us
            # match HSTU's param count while keeping dqk/dv EXACTLY (the +16% of
            # a full gate comes almost entirely from this D*H*dk matrix).
            if kda_gate_rank and kda_gate_rank > 0:
                self._kda_f_proj = torch.nn.Sequential(
                    torch.nn.Linear(embedding_dim, kda_gate_rank, bias=False),
                    torch.nn.Linear(kda_gate_rank, H * dk, bias=False),
                )
            else:
                self._kda_f_proj = torch.nn.Linear(embedding_dim, H * dk, bias=False)
            self._kda_A_log = torch.nn.Parameter(
                torch.log(torch.empty(H, dtype=torch.float32).uniform_(1, 16))
            )
            _dt = torch.exp(
                torch.rand(H * dk, dtype=torch.float32)
                * (_math.log(0.1) - _math.log(0.001))
                + _math.log(0.001)
            ).clamp(min=1e-4)
            self._kda_dt_bias = torch.nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))
            # Time-aware forget gate: fold the inter-event gap Δt into α_t in
            # place of HSTU's rab_time. A per-channel weight on log1p(Δt) is added
            # to the gate pre-activation; 0-init => starts as vanilla (content-only)
            # KDA and learns to use time. A per-step decay α_t=f(Δt_t) makes the
            # state decay between positions j->i ≈ exp(-λ·(ts_i-ts_j)), the
            # continuous-time analogue of rab_time's timespan bucket bias.
            if self._kda_time_gate == "continuous":
                self._kda_time_w = torch.nn.Parameter(
                    torch.zeros(H, dk, dtype=torch.float32)
                )
            # Write gate. KDA: free logit b_proj (sigmoid in-kernel). IsoKLA:
            # additive-Kalman scalar gain β_t from the Triton iso_beta_chunk scan
            # (needs obs-noise r_t, process-noise q_t, prior μ).
            if self._normalization == "kda":
                self._kda_b_proj = torch.nn.Linear(embedding_dim, H, bias=False)
            elif self._normalization == "iso_kla":
                from generative_recommenders.research.modeling.sequential.kla.kla_ops.iso_chunk import (
                    iso_beta_chunk,
                )

                self._iso_beta_chunk = iso_beta_chunk
                self._kla_r_min: float = 0.05
                self._kla_q_min: float = 0.05
                self._kla_r_proj = torch.nn.Linear(embedding_dim, H, bias=True)
                self._kla_qn_proj = torch.nn.Linear(embedding_dim, H, bias=True)
                _inv_mu = _math.log(_math.expm1(max(1.0 - 0.1, 1e-3)))
                self._kla_mu_param = torch.nn.Parameter(
                    torch.full((H,), _inv_mu, dtype=torch.float32)
                )
            else:  # diag_kla / exact_kla: per-channel gain + chunk_kalman memory
                # (Gated DeltaNet-2 core). diag_kla = diagonal gain (kla_kappa_chunk);
                # exact_kla = dense-covariance gain (gain_recurrent, K=64 only).
                from generative_recommenders.research.modeling.sequential.kla.kla_ops.kalman_chunk import (
                    chunk_kalman,
                )

                self._chunk_kalman = chunk_kalman
                if self._normalization == "diag_kla":
                    from generative_recommenders.research.modeling.sequential.kla.kla_ops.diag_chunk import (
                        kla_kappa_chunk,
                    )

                    self._kla_kappa_chunk = kla_kappa_chunk
                else:  # exact_kla
                    from generative_recommenders.research.modeling.sequential.kla.kla_ops.gain_recurrent import (
                        gain_recurrent,
                    )

                    self._gain_recurrent = gain_recurrent
                self._kla_r_min: float = 0.05
                self._kla_q_min: float = 0.05
                self._kla_r_proj = torch.nn.Linear(embedding_dim, H, bias=True)
                # per-channel process noise omega_t (gate_dim = H*dk).
                self._kla_qn_proj = torch.nn.Linear(embedding_dim, H * dk, bias=True)
                _inv_mu = _math.log(_math.expm1(max(1.0 - 0.1, 1e-3)))
                self._kla_mu_param = torch.nn.Parameter(
                    torch.full((H,), _inv_mu, dtype=torch.float32)
                )

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(  # pyre-ignore [3]
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[HSTUCacheState] = None,
        return_cache_states: bool = False,
    ):
        """
        Args:
            x: (\sum_i N_i, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: optional (B, N) x int64.
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
            delta_x_offsets: optional 2-tuple ((B,) x int32, (B,) x int32).
                For the 1st element in the tuple, each element is in [0, x_offsets[-1]). For the
                2nd element in the tuple, each element is in [0, N).
            cache: Optional 4-tuple of (v, padded_q, padded_k, output) from prior runs,
                where all except padded_q, padded_k are jagged.
        Returns:
            x' = f(x), (\sum_i N_i, D) x float.
        """
        if delta_x_offsets is not None and self._normalization in (
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        ):
            raise NotImplementedError(
                "Local/hybrid forgetting attention does not support "
                "incremental (delta) decoding."
            )
        if (
            delta_x_offsets is not None
            and self._normalization in _SIGNED_ADDITIVE_NORMALIZATIONS
        ):
            raise NotImplementedError(
                "Signed additive attention does not support incremental "
                "(delta) decoding."
            )
        n: int = invalid_attn_mask.size(-1)
        cached_q = None
        cached_k = None
        if delta_x_offsets is not None:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # [delta_x_offsets[0], :].
            assert cache is not None
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache

        normed_x = self._norm_input(x)

        if self._linear_config == "uvqk":
            batched_mm_output = torch.mm(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        if delta_x_offsets is not None:
            # pyrefly: ignore [unbound-name]
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        B: int = x_offsets.size(0) - 1
        if self._normalization in (
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        ):
            padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
                values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
                values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_v = torch.ops.fbgemm.jagged_to_padded_dense(
                values=v, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_q_heads = padded_q.view(B, n, self._num_heads, self._attention_dim)
            padded_k_heads = padded_k.view(B, n, self._num_heads, self._attention_dim)
            padded_v_heads = padded_v.view(B, n, self._num_heads, self._linear_dim)
            forget_logits = torch.einsum(
                "bnhd,hd->bnh", padded_k_heads, self._forget_weight
            ) + self._forget_bias.view(1, 1, self._num_heads)
            relative_attention_bias = (
                self._rel_attn_bias(all_timestamps)
                if self._rel_attn_bias is not None and all_timestamps is not None
                else None
            )
            tail_gain = 2.0 * torch.tanh(self._hybrid_tail_rho / 2.0)
            attention_heads = _per_head_local_forgetting_attention(
                padded_q=padded_q_heads,
                padded_k=padded_k_heads,
                padded_v=padded_v_heads,
                invalid_attn_mask=invalid_attn_mask,
                relative_attention_bias=relative_attention_bias,
                log_forget=F.logsigmoid(forget_logits),
                window_size=self._hybrid_window_size,
                valid_lengths=x_offsets[1:] - x_offsets[:-1],
                tail_gain=(
                    tail_gain
                    if self._normalization == "hybrid_forgetting_rel_bias"
                    else None
                ),
                tail_feature_map=self._hybrid_tail_feature_map,
                signed_feature_gamma=self._signed_feature_gamma,
            )
            if self._normalization == "local_forgetting_rel_bias":
                # Keep the matched gain parameter in the DDP graph while the
                # local-only arm remains exactly tail-free.
                attention_heads = attention_heads + (
                    tail_gain.to(attention_heads.dtype).view(1, 1, self._num_heads, 1)
                    * 0.0
                )
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                attention_heads.reshape(B, n, self._num_heads * self._linear_dim),
                [x_offsets],
            )[0]
        elif self._normalization in (
            "rel_bias",
            "hstu_rel_bias",
            "tanh_rel_bias",
            "forgetting_rel_bias",
            "fixed_forgetting_rel_bias",
            "taylor1_rel_bias",
            "taylor2_rel_bias",
        ):
            attn_output, padded_q, padded_k = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                cached_q=cached_q,
                cached_k=cached_k,
                delta_x_offsets=delta_x_offsets,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                rel_attn_bias=self._rel_attn_bias,
                forget_weight=getattr(self, "_forget_weight", None),
                forget_bias=getattr(self, "_forget_bias", None),
                score_kernel={
                    "tanh_rel_bias": "tanh",
                    "taylor1_rel_bias": "taylor1",
                    "taylor2_rel_bias": "taylor2",
                }.get(self._normalization, "silu"),
            )
        elif self._normalization in (
            "softmax_rel_bias",
            "softmax_canonical_rel_bias",
            "forgetting_softmax_rel_bias",
            "fixed_forgetting_softmax_rel_bias",
        ):
            if delta_x_offsets is not None:
                B = x_offsets.size(0) - 1
                padded_q, padded_k = cached_q, cached_k
                flattened_offsets = delta_x_offsets[1] + torch.arange(
                    start=0,
                    end=B * n,
                    step=n,
                    device=delta_x_offsets[1].device,
                    dtype=delta_x_offsets[1].dtype,
                )
                assert padded_q is not None
                assert padded_k is not None
                padded_q = (
                    padded_q.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=q,
                    )
                    .view(B, n, -1)
                )
                padded_k = (
                    padded_k.view(B * n, -1)
                    .index_copy_(
                        dim=0,
                        index=flattened_offsets,
                        source=k,
                    )
                    .view(B, n, -1)
                )
            else:
                padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
                padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )

            padded_v = torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n])
            relative_attention_bias = (
                self._rel_attn_bias(all_timestamps)
                if self._rel_attn_bias is not None and all_timestamps is not None
                else None
            )
            log_forget = None
            if self._normalization in (
                "forgetting_softmax_rel_bias",
                "fixed_forgetting_softmax_rel_bias",
            ):
                padded_k_heads = padded_k.view(
                    B, n, self._num_heads, self._attention_dim
                )
                forget_logits = torch.einsum(
                    "bnhd,hd->bnh", padded_k_heads, self._forget_weight
                ) + self._forget_bias.view(1, 1, self._num_heads)
                log_forget = F.logsigmoid(forget_logits)
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                _per_head_softmax_attention(
                    padded_q=padded_q,
                    padded_k=padded_k,
                    padded_v=padded_v,
                    invalid_attn_mask=invalid_attn_mask,
                    relative_attention_bias=relative_attention_bias,
                    num_heads=self._num_heads,
                    attention_dim=self._attention_dim,
                    linear_dim=self._linear_dim,
                    temperature=self._softmax_temperature or None,
                    log_forget=log_forget,
                    valid_lengths=x_offsets[1:] - x_offsets[:-1],
                    scale_relative_attention_bias=(
                        self._normalization != "softmax_canonical_rel_bias"
                    ),
                ),
                [x_offsets],
            )[0]
        elif self._normalization == "additive_dot":
            if delta_x_offsets is not None:
                raise NotImplementedError(
                    "Additive dot attention does not support incremental (delta) decoding."
                )
            padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
                values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
                values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_v = torch.ops.fbgemm.jagged_to_padded_dense(
                values=v, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                _per_head_additive_dot_attention(
                    padded_q=padded_q,
                    padded_k=padded_k,
                    padded_v=padded_v,
                    num_heads=self._num_heads,
                    attention_dim=self._attention_dim,
                    linear_dim=self._linear_dim,
                ),
                [x_offsets],
            )[0]
        elif self._normalization in _SIGNED_ADDITIVE_NORMALIZATIONS:
            padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
                values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
                values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            padded_v = torch.ops.fbgemm.jagged_to_padded_dense(
                values=v, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
            )
            valid_lengths = x_offsets[1:] - x_offsets[:-1]
            if self._normalization == "signed_additive_abs_coefficient_oracle":
                dense_attention = _per_head_signed_additive_abs_coefficient_oracle(
                    padded_q=padded_q,
                    padded_k=padded_k,
                    padded_v=padded_v,
                    invalid_attn_mask=invalid_attn_mask,
                    num_heads=self._num_heads,
                    attention_dim=self._attention_dim,
                    linear_dim=self._linear_dim,
                    gamma=self._signed_feature_gamma,
                    valid_lengths=valid_lengths,
                )
            else:
                feature_map = {
                    "signed_additive_identity": "identity",
                    "signed_additive_tanh": "tanh",
                    "signed_additive_abs_tanh": "abs_tanh",
                }[self._normalization]
                dense_attention = _per_head_signed_additive_feature_attention(
                    padded_q=padded_q,
                    padded_k=padded_k,
                    padded_v=padded_v,
                    num_heads=self._num_heads,
                    attention_dim=self._attention_dim,
                    linear_dim=self._linear_dim,
                    feature_map=feature_map,
                    gamma=self._signed_feature_gamma,
                    valid_lengths=valid_lengths,
                )
            dense_attention = _attach_zero_relative_bias_dependency(
                dense_attention, self._rel_attn_bias
            )
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                dense_attention, [x_offsets]
            )[0]
        elif self._normalization in ("kda", "iso_kla", "diag_kla", "exact_kla"):
            assert (
                delta_x_offsets is None
            ), "KDA/KLA core does not support incremental (delta) decoding."
            # HSTU's own Q, K, V -> padded dense [B, n, H, d], fed to a gated
            # delta-rule / Kalman linear-attention op. Fixed [B, n] shape => the
            # triton kernel compiles once. Intrinsically causal (no attn mask).
            H, dk, dv = self._num_heads, self._attention_dim, self._linear_dim

            def _pad(values: torch.Tensor) -> torch.Tensor:
                return torch.ops.fbgemm.jagged_to_padded_dense(
                    values=values,
                    offsets=[x_offsets],
                    max_lengths=[n],
                    padding_value=0.0,
                )

            padded_q = _pad(q).view(B, n, H, dk)
            padded_k = _pad(k).view(B, n, H, dk)
            padded_v = _pad(v).view(B, n, H, dv)
            g = _pad(self._kda_f_proj(normed_x)).view(
                B, n, H, dk
            )  # forget-gate pre-act
            if self._kda_time_gate == "continuous" and all_timestamps is not None:
                # Fold inter-event gap Δt into the forget gate (replaces rab_time):
                # Δt_t = |ts_t - ts_{t-1}|, per-channel weight on log1p(Δt).
                ts = all_timestamps.to(torch.float32)  # [B, N]
                dt = torch.zeros_like(ts)
                dt[:, 1:] = (ts[:, 1:] - ts[:, :-1]).abs()
                tf = torch.log1p(dt)  # [B, N]
                if tf.size(1) < n:
                    tf = F.pad(tf, (0, n - tf.size(1)))
                elif tf.size(1) > n:
                    tf = tf[:, :n]
                g = g + tf.view(B, n, 1, 1) * self._kda_time_w.view(1, 1, H, dk)

            if self._normalization in ("kda", "iso_kla"):
                if self._normalization == "kda":
                    beta = _pad(self._kda_b_proj(normed_x)).view(B, n, H)  # logit
                    use_beta_sigmoid = True
                else:  # iso_kla: additive-Kalman scalar write gain over Q/K/V
                    A = (
                        self._kda_A_log.float()
                        .exp()
                        .repeat_interleave(dk)
                        .view(1, 1, H, dk)
                    )
                    dtb = self._kda_dt_bias.view(1, 1, H, dk)
                    alpha = torch.exp(-A * F.softplus(g.float() + dtb))  # [B,n,H,dk]
                    a_t = (alpha * alpha).mean(-1)  # [B,n,H]
                    q_base = F.softplus(
                        _pad(self._kla_qn_proj(normed_x)).view(B, n, H).float()
                    )
                    q_noise = self._kla_q_min + (
                        q_base * (1.0 - a_t) if self._kla_omega_coupling else q_base
                    )
                    r_t = self._kla_r_min + F.softplus(
                        _pad(self._kla_r_proj(normed_x)).view(B, n, H).float()
                    )
                    mu = F.softplus(self._kla_mu_param) + 0.1  # [H]
                    beta = self._iso_beta_chunk(
                        a_t, q_noise, r_t, mu, out_dtype=torch.float32, info_scale=1.0
                    )  # [B,n,H] Kalman gain (a value, not a logit)
                    use_beta_sigmoid = False
                o, _ = self._chunk_kda(
                    q=padded_q.bfloat16(),
                    k=padded_k.bfloat16(),
                    v=padded_v.bfloat16(),
                    g=g.bfloat16(),
                    beta=beta.bfloat16(),
                    A_log=self._kda_A_log,
                    dt_bias=self._kda_dt_bias,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=use_beta_sigmoid,
                )  # [B, n, H, dv]
            else:  # diag_kla / exact_kla: per-channel gain + chunk_kalman (fp32)
                A = (
                    self._kda_A_log.float()
                    .exp()
                    .repeat_interleave(dk)
                    .view(1, 1, H, dk)
                )
                dtb = self._kda_dt_bias.view(1, 1, H, dk)
                alpha = torch.exp(-A * F.softplus(g.float() + dtb))  # [B,n,H,dk] decay
                omega_base = F.softplus(
                    _pad(self._kla_qn_proj(normed_x)).view(B, n, H, dk).float()
                )
                omega = self._kla_q_min + (
                    omega_base * (1.0 - alpha * alpha)
                    if self._kla_omega_coupling
                    else omega_base
                )  # per-channel process noise (coupling: scale by 1-alpha^2)
                r_t = self._kla_r_min + F.softplus(
                    _pad(self._kla_r_proj(normed_x)).view(B, n, H).float()
                )
                mu = F.softplus(self._kla_mu_param) + 0.1  # [H]
                qn = F.normalize(padded_q.float(), p=2, dim=-1)
                kn = F.normalize(padded_k.float(), p=2, dim=-1)
                with torch.autocast(device_type="cuda", enabled=False):
                    if self._normalization == "diag_kla":
                        kappa, _ = self._kla_kappa_chunk(
                            kn, alpha, omega, r=r_t, mu=mu, info_scale=1.0
                        )  # diagonal gain
                    else:  # exact_kla: dense-covariance gain (K=64 kernel)
                        kappa = self._gain_recurrent(
                            kn, alpha, omega, r_t, mu=mu, dk_calibration=True
                        )
                    o, _ = self._chunk_kalman(
                        q=qn,
                        k=kn,
                        kappa=kappa.float(),
                        v=padded_v.float(),
                        g=alpha.clamp_min(1e-6).log(),
                        scale=dk**-0.5,
                        use_qk_l2norm_in_kernel=False,
                        output_final_state=False,
                    )  # [B, n, H, dv]
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                o.reshape(B, n, H * dv).to(normed_x.dtype), [x_offsets]
            )[0]
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        attn_output = (
            attn_output
            if delta_x_offsets is None
            else attn_output[delta_x_offsets[0], :]
        )
        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + x
        )

        if delta_x_offsets is not None:
            # pyrefly: ignore [unbound-name]
            new_outputs = cached_outputs.index_copy_(
                dim=0, index=delta_x_offsets[0], source=new_outputs
            )

        if return_cache_states and delta_x_offsets is None:
            v = v.contiguous()

        return new_outputs, (v, padded_q, padded_k, new_outputs)


class HSTUJagged(torch.nn.Module):
    def __init__(
        self,
        modules: List[SequentialTransductionUnitJagged],
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (\sum_i N_i, D) x float
            x_offsets: (B + 1) x int32
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}
            return_cache_states: bool. True if we should return cache states.

        Returns:
            x' = f(x), (\sum_i N_i, D) x float
        """
        cache_states: List[HSTUCacheState] = []

        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                x, cache_states_i = layer(
                    x=x,
                    x_offsets=x_offsets,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    delta_x_offsets=delta_x_offsets,
                    cache=cache[i] if cache is not None else None,
                    return_cache_states=return_cache_states,
                )
                if return_cache_states:
                    cache_states.append(cache_states_i)

        return x, cache_states

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
        Returns:
            x' = f(x), (B, N, D) x float
        """
        if len(x.size()) == 3:
            x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]

        jagged_x, cache_states = self.jagged_forward(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        y = torch.ops.fbgemm.jagged_to_padded_dense(
            values=jagged_x,
            offsets=[x_offsets],
            max_lengths=[invalid_attn_mask.size(1)],
            padding_value=0.0,
        )
        return y, cache_states


class HSTU(SequentialEncoderWithLearnedSimilarityModule):
    """
    Implements HSTU (Hierarchical Sequential Transduction Unit) in
    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,
    https://arxiv.org/abs/2402.17152.

    Note that this implementation is intended for reproducing experiments in
    the traditional sequential recommender setting (Section 4.1.1), and does
    not yet use optimized kernels discussed in the paper.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_config: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        kda_time_gate: str = "continuous",
        kda_gate_rank: int = 0,
        kda_o_rank: int = 0,
        kla_omega_coupling: bool = False,
        forgetting_min_period: float = 8.0,
        forgetting_max_period: float = 256.0,
        hybrid_window_size: int = 64,
        softmax_temperature: float = 0.0,
        signed_feature_gamma: float = 1.0,
        hybrid_tail_feature_map: str = "identity",
        verbose: bool = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias
        self._normalization: str = normalization
        self._kda_time_gate: str = kda_time_gate
        self._kla_omega_coupling: bool = kla_omega_coupling
        self._kda_gate_rank: int = kda_gate_rank
        self._kda_o_rank: int = kda_o_rank
        self._forgetting_min_period: float = forgetting_min_period
        self._forgetting_max_period: float = forgetting_max_period
        self._hybrid_window_size: int = hybrid_window_size
        self._softmax_temperature: float = softmax_temperature
        self._signed_feature_gamma: float = _validate_signed_feature_gamma(
            signed_feature_gamma
        )
        self._hybrid_tail_feature_map: str = hybrid_tail_feature_map
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    # TODO: change to lambda x.
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_sequence_len
                            + max_output_len,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    concat_ua=concat_ua,
                    kda_time_gate=kda_time_gate,
                    kda_gate_rank=kda_gate_rank,
                    kda_o_rank=kda_o_rank,
                    kla_omega_coupling=kla_omega_coupling,
                    forgetting_min_period=forgetting_min_period,
                    forgetting_max_period=forgetting_max_period,
                    hybrid_window_size=hybrid_window_size,
                    softmax_temperature=softmax_temperature,
                    signed_feature_gamma=self._signed_feature_gamma,
                    hybrid_tail_feature_map=hybrid_tail_feature_map,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length + max_output_len,
                        self._max_sequence_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = verbose
        self.reset_params()

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if ("_hstu" in name) or ("_embedding_module" in name):
                if self._verbose:
                    print(f"Skipping init for {name}")
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(
                        f"Initialize {name} as xavier normal: {params.data.size()} params"
                    )
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        debug_str = (
            f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        if self._normalization == "softmax_rel_bias":
            debug_str += "-softmax"
            if self._softmax_temperature > 0:
                debug_str += f"t{self._softmax_temperature:g}"
        if self._normalization == "softmax_canonical_rel_bias":
            debug_str += "-softmax-canonical"
            if self._softmax_temperature > 0:
                debug_str += f"t{self._softmax_temperature:g}"
        if self._normalization in (
            "forgetting_softmax_rel_bias",
            "fixed_forgetting_softmax_rel_bias",
        ):
            tag = (
                "fosoftmax"
                if self._normalization == "forgetting_softmax_rel_bias"
                else "fixedfosoftmax"
            )
            debug_str += f"-{tag}"
            if self._softmax_temperature > 0:
                debug_str += f"-temp{self._softmax_temperature:g}"
            debug_str += (
                f"-h{self._forgetting_min_period:g}" f"-{self._forgetting_max_period:g}"
            )
        if self._normalization == "taylor1_rel_bias":
            debug_str += "-taylor1"
        if self._normalization == "taylor2_rel_bias":
            debug_str += "-taylor2"
        if self._normalization == "tanh_rel_bias":
            debug_str += "-tanh-attn"
        if self._normalization == "additive_dot":
            debug_str += "-additive-dot"
        if self._normalization in _SIGNED_ADDITIVE_NORMALIZATIONS:
            tag = {
                "signed_additive_identity": "safa-identity",
                "signed_additive_tanh": "safa-tanh",
                "signed_additive_abs_tanh": "safa-abs-tanh",
                "signed_additive_abs_coefficient_oracle": "safa-abscoef-oracle",
            }[self._normalization]
            debug_str += f"-{tag}-g{self._signed_feature_gamma:g}"
        if self._normalization in (
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        ):
            tag = (
                "localfohstu"
                if self._normalization == "local_forgetting_rel_bias"
                else "hybridfohstu"
            )
            debug_str += (
                f"-{tag}-w{self._hybrid_window_size}"
                f"-t{self._forgetting_min_period:g}"
                f"-{self._forgetting_max_period:g}"
            )
            if self._normalization == "hybrid_forgetting_rel_bias":
                debug_str += (
                    f"-tail-{self._hybrid_tail_feature_map}"
                    f"-g{self._signed_feature_gamma:g}"
                )
        if self._normalization in (
            "forgetting_rel_bias",
            "fixed_forgetting_rel_bias",
        ):
            tag = (
                "forget"
                if self._normalization == "forgetting_rel_bias"
                else "fixedforget"
            )
            debug_str += (
                f"-{tag}-t{self._forgetting_min_period:g}"
                f"-{self._forgetting_max_period:g}"
            )
        if self._normalization in ("kda", "iso_kla", "diag_kla", "exact_kla"):
            tag = {
                "kda": "kda",
                "iso_kla": "isokla",
                "diag_kla": "diagkla",
                "exact_kla": "exactkla",
            }[self._normalization]
            debug_str += f"-{tag}" + ("-t" if self._kda_time_gate != "none" else "")
            if self._kla_omega_coupling:
                debug_str += "-oc"
            if self._kda_gate_rank and self._kda_gate_rank > 0:
                debug_str += f"-r{self._kda_gate_rank}"
            if self._kda_o_rank and self._kda_o_rank > 0:
                debug_str += f"-or{self._kda_o_rank}"
        return debug_str

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        [B, N] -> [B, N, D].
        """
        device = past_lengths.device
        float_dtype = past_embeddings.dtype
        B, N, _ = past_embeddings.size()

        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        float_dtype = user_embeddings.dtype
        user_embeddings, cached_states = self._hstu(
            x=user_embeddings,
            x_offsets=torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths),
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if TIMESTAMPS_KEY in past_payloads
                else None
            ),
            invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        return self._output_postproc(user_embeddings), cached_states

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Runs the main encoder.

        Args:
            past_lengths: (B,) x int64
            past_ids: (B, N,) x int64 where the latest engaged ids come first. In
                particular, past_ids[i, past_lengths[i] - 1] should correspond to
                the latest engaged values.
            past_embeddings: (B, N, D) x float or (\sum_b N_b, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            encoded_embeddings of [B, N, D].
        """
        encoded_embeddings, _ = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        return encoded_embeddings

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache: Optional[List[HSTUCacheState]],
        return_cache_states: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Args:
            past_lengths: (B,) x int64.
            past_ids: (B, N,) x int64.
            past_embeddings: (B, N, D,) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).
            return_cache_states: bool.

        Returns:
            (B, D) x float, representing embeddings for the current state.
        """
        encoded_seq_embeddings, cache_states = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )  # [B, N, D]
        current_embeddings = get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )
        if return_cache_states:
            return current_embeddings, cache_states
        else:
            return current_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Runs encoder to obtain the current hidden states.

        Args:
            past_lengths: (B,) x int.
            past_ids: (B, N,) x int.
            past_embeddings: (B, N, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            (B, D,) x float, representing encoded states at the most recent time step.
        """
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
