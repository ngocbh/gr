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

#!/usr/bin/env python3

# pyre-strict
import abc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from generative_recommenders.common import fx_unwrap_optional_tensor, HammerModule
from generative_recommenders.modules.mhc import (
    expand_streams,
    HyperConnection,
    reduce_streams,
)
from generative_recommenders.ops.hstu_attention import delta_hstu_mha, hstu_mha
from generative_recommenders.ops.hstu_compute import (
    hstu_compute_output,
    hstu_compute_uqvk,
    hstu_preprocess_and_attention,
)
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from torch.autograd.profiler import record_function


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


class STU(HammerModule, abc.ABC):
    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass


@dataclass
class STULayerConfig:
    embedding_dim: int
    num_heads: int
    hidden_dim: int
    attention_dim: int
    output_dropout_ratio: float = 0.3
    causal: bool = True
    target_aware: bool = True
    max_attn_len: Optional[int] = None
    attn_alpha: Optional[float] = None
    use_group_norm: bool = False
    recompute_normed_x: bool = True
    recompute_uvqk: bool = True
    recompute_y: bool = True
    sort_by_length: bool = True
    contextual_seq_len: int = 0
    neutreno_lambda: float = 0.0
    neutreno_after_norm: bool = False
    mhc_num_streams: int = 4
    mhc_num_iters: int = 20
    mhc_tau: float = 0.05


@torch.fx.wrap
def _update_kv_cache(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    max_kv_caching_len: int,
    kv_caching_lengths: Optional[torch.Tensor],
    orig_k_cache: Optional[torch.Tensor],
    orig_v_cache: Optional[torch.Tensor],
    orig_max_kv_caching_len: int,
    orig_kv_caching_offsets: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, Optional[torch.Tensor]]:
    if kv_caching_lengths is not None:
        kv_caching_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            kv_caching_lengths
        )
        delta_offsets = seq_offsets - kv_caching_offsets
        k_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=fx_unwrap_optional_tensor(k).flatten(1, 2),
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        v_cache, _ = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=fx_unwrap_optional_tensor(v).flatten(1, 2),
            max_len_left=None,
            max_len_right=None,
            offsets_left=kv_caching_offsets,
            offsets_right=delta_offsets,
        )
        if max_kv_caching_len == 0:
            max_kv_caching_len = int(kv_caching_lengths.max().item())
        return (
            k_cache,
            v_cache,
            max_kv_caching_len,
            kv_caching_offsets,
        )
    else:
        return (
            orig_k_cache,
            orig_v_cache,
            orig_max_kv_caching_len,
            orig_kv_caching_offsets,
        )


@torch.fx.wrap
def _construct_full_kv(
    delta_k: torch.Tensor,
    delta_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    max_kv_caching_len: int,
    kv_caching_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    L, _ = delta_k.shape
    B = kv_caching_offsets.shape[0] - 1
    delta_size = L // B
    full_k = concat_2D_jagged(
        max_seq_len=max_kv_caching_len + delta_size,
        values_left=k_cache,
        values_right=delta_k,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_v = concat_2D_jagged(
        max_seq_len=max_kv_caching_len + delta_size,
        values_left=v_cache,
        values_right=delta_v,
        max_len_left=max_kv_caching_len,
        max_len_right=delta_size,
        offsets_left=kv_caching_offsets,
        offsets_right=None,
    )
    full_kv_caching_offsets = kv_caching_offsets + delta_size * torch.arange(
        B + 1, device=delta_k.device
    )
    return (
        full_k,
        full_v,
        max_kv_caching_len + delta_size,
        full_kv_caching_offsets,
    )


class STULayer(STU):
    max_kv_caching_len: int
    k_cache: Optional[torch.Tensor]
    v_cache: Optional[torch.Tensor]
    kv_caching_offsets: Optional[torch.Tensor]

    def __init__(
        self,
        config: STULayerConfig,
        is_inference: bool = False,
    ) -> None:
        super().__init__(
            is_inference=is_inference,
        )
        self.reset_kv_cache()
        self._num_heads: int = config.num_heads
        self._embedding_dim: int = config.embedding_dim
        self._hidden_dim: int = config.hidden_dim
        self._attention_dim: int = config.attention_dim
        self._output_dropout_ratio: float = config.output_dropout_ratio
        self._target_aware: bool = config.target_aware
        self._causal: bool = config.causal
        self._max_attn_len: int = config.max_attn_len or 0
        self._attn_alpha: float = config.attn_alpha or 1.0 / (self._attention_dim**0.5)
        self._use_group_norm: bool = config.use_group_norm
        self._recompute_normed_x: bool = config.recompute_normed_x
        self._recompute_uvqk: bool = config.recompute_uvqk
        self._recompute_y: bool = config.recompute_y
        self._sort_by_length: bool = config.sort_by_length
        self._contextual_seq_len: int = config.contextual_seq_len

        self._uvqk_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    self._embedding_dim,
                    (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._uvqk_weight)
        self._uvqk_beta: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(
                (self._hidden_dim * 2 + self._attention_dim * 2) * self._num_heads,
            ),
        )
        self._input_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((self._embedding_dim,)),
        )
        self._input_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((self._embedding_dim,)),
        )
        self._output_weight = torch.nn.Parameter(
            torch.empty(
                (
                    self._hidden_dim * self._num_heads * 3,
                    self._embedding_dim,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._output_weight)
        output_norm_shape: int = (
            self._hidden_dim * self._num_heads
            if not self._use_group_norm
            else self._num_heads
        )
        self._output_norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((output_norm_shape,)),
        )
        self._output_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((output_norm_shape,)),
        )

    def reset_kv_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.kv_caching_offsets = None
        self.max_kv_caching_len = 0

    def update_kv_cache(
        self,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        max_kv_caching_len: int,
        kv_caching_lengths: Optional[torch.Tensor],
    ) -> None:
        self.k_cache, self.v_cache, self.max_kv_caching_len, self.kv_caching_offsets = (
            _update_kv_cache(
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                k=k,
                v=v,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
                orig_k_cache=self.k_cache,
                orig_v_cache=self.v_cache,
                orig_max_kv_caching_len=self.max_kv_caching_len,
                orig_kv_caching_offsets=self.kv_caching_offsets,
            )
        )

    def construct_full_kv(
        self,
        delta_k: torch.Tensor,
        delta_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        return _construct_full_kv(
            delta_k=delta_k,
            delta_v=delta_v,
            k_cache=fx_unwrap_optional_tensor(self.k_cache),
            v_cache=fx_unwrap_optional_tensor(self.v_cache),
            max_kv_caching_len=self.max_kv_caching_len,
            # pyrefly: ignore [bad-argument-type]
            kv_caching_offsets=self.kv_caching_offsets,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with record_function("## stu_preprocess_and_attention ##"):
            u, attn_output, k, v = hstu_preprocess_and_attention(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(x.dtype),
                uvqk_bias=self._uvqk_beta.to(x.dtype),
                max_seq_len=max_seq_len,
                seq_offsets=x_offsets,
                attn_alpha=self._attn_alpha,
                causal=self._causal,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                recompute_uvqk_in_backward=self._recompute_uvqk,
                recompute_normed_x_in_backward=self._recompute_normed_x,
                sort_by_length=self._sort_by_length,
                prefill=kv_caching_lengths is not None,
                kernel=self.hammer_kernel(),
            )

        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=x_offsets,
            k=k,
            v=v,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )

        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=attn_output,
                u=u,
                x=x,
                norm_weight=self._output_norm_weight.to(x.dtype),
                norm_bias=self._output_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="none",
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
            )

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with record_function("## stu_compute_uqvk ##"):
            delta_u, delta_q, delta_k, delta_v = hstu_compute_uqvk(
                x=delta_x,
                norm_weight=self._input_norm_weight.to(delta_x.dtype),
                norm_bias=self._input_norm_bias.to(delta_x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(delta_x.dtype),
                uvqk_bias=self._uvqk_beta.to(delta_x.dtype),
                kernel=self.hammer_kernel(),
            )
        k, v, max_seq_len, seq_offsets = self.construct_full_kv(
            delta_k=delta_k.flatten(1, 2),
            delta_v=delta_v.flatten(1, 2),
        )
        self.update_kv_cache(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            k=k,
            v=v,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )
        k = k.view(-1, self._num_heads, self._attention_dim)
        v = v.view(-1, self._num_heads, self._hidden_dim)
        with record_function("## delta_hstu_mha ##"):
            delta_attn_output = delta_hstu_mha(
                max_seq_len=max_seq_len,
                alpha=self._attn_alpha,
                delta_q=delta_q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                kernel=self.hammer_kernel(),
            ).view(-1, self._hidden_dim * self._num_heads)
        with record_function("## stu_compute_output ##"):
            return hstu_compute_output(
                attn=delta_attn_output,
                u=delta_u,
                x=delta_x,
                norm_weight=self._output_norm_weight.to(delta_x.dtype),
                norm_bias=self._output_norm_bias.to(delta_x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(delta_x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="none",
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
            )


class STUStack(STU):
    def __init__(
        self,
        stu_list: List[STU],
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=stu_list)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self._stu_layers:
            x = layer(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
        return x

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self._stu_layers:
            delta_x = layer.cached_forward(  # pyre-ignore [29]
                delta_x=delta_x,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
        return delta_x


class STULayerNeuTRENO(STULayer):
    """STU layer with the NeuTRENO anti-oversmoothing term.

    Each layer adds ``neutreno_lambda * (v_0 - v_l)`` to its attention output, where
    ``v_0`` is the first layer's value and ``v_l`` this layer's value. Because the
    fully-fused preprocess+attention path does not expose ``v``, this uses the unfused
    uvqk + attention path (attention itself still runs the same kernel). The term is
    threaded by :class:`STUStackNeuTRENO`. ``neutreno_lambda = 0`` recovers vanilla STU.

    The term is injected at one of two points, controlled by ``neutreno_after_norm``:

    * ``False`` (default) -- BEFORE the output norm: ``norm(attn + lambda*(v_0 - v))``.
      This keeps the fused ``hstu_compute_output`` kernel.
    * ``True`` -- AFTER the output norm: ``norm(attn) + lambda*(v_0 - v)`` feeds the
      ``u``-gate. The fused kernel never exposes the normalized tensor, so this path
      unfuses the output stage (norm -> gate -> concat -> dropout -> addmm) in eager
      torch, mirroring ``pytorch_hstu_compute_output`` with the term added post-norm.
    """

    def __init__(self, config: STULayerConfig, is_inference: bool = False) -> None:
        super().__init__(config=config, is_inference=is_inference)
        self._neutreno_lambda: float = config.neutreno_lambda
        self._neutreno_after_norm: bool = config.neutreno_after_norm

    def _compute_output_after_norm(
        self,
        attn: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        post_norm_term: torch.Tensor,
    ) -> torch.Tensor:
        """Unfused ``hstu_compute_output`` with ``post_norm_term`` added after the norm.

        Mirrors ``pytorch_norm_mul_dropout`` (fp32 norm, ``concat_u=concat_x=True``,
        ``mul_u_activation_type='none'``) + the final ``addmm`` residual, except the
        NeuTRENO term is added to the normalized attention output before the ``u`` gate.
        The raw-attn concat slot is left untouched so only the gated branch carries the
        term -- the post-norm analogue of the research ``neutreno_after_norm`` path.
        """
        dtype = x.dtype
        attn_f = attn.to(torch.float32)
        u_f = u.to(torch.float32)
        norm_weight = self._output_norm_weight.to(torch.float32)
        norm_bias = self._output_norm_bias.to(torch.float32)
        eps = 1e-6
        if self._use_group_norm:
            normed = F.group_norm(
                attn_f.view(-1, self._num_heads, self._hidden_dim),
                num_groups=self._num_heads,
                weight=norm_weight,
                bias=norm_bias,
                eps=eps,
            ).view(-1, self._num_heads * self._hidden_dim)
        else:
            normed = F.layer_norm(
                attn_f,
                normalized_shape=(attn_f.shape[-1],),
                weight=norm_weight,
                bias=norm_bias,
                eps=eps,
            )
        a = normed + post_norm_term.to(torch.float32)
        y = torch.cat([u_f, attn_f, u_f * a], dim=1)
        y = F.dropout(y, p=self._output_dropout_ratio, training=self.training)
        return torch.addmm(x, y.to(dtype), self._output_weight.to(dtype)).to(dtype)

    def forward(  # pyre-ignore [14]
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
        v_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with record_function("## stu_neutreno_uqvk ##"):
            u, q, k, v = hstu_compute_uqvk(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(x.dtype),
                uvqk_bias=self._uvqk_beta.to(x.dtype),
                kernel=self.hammer_kernel(),
            )
        with record_function("## stu_neutreno_mha ##"):
            attn_output = hstu_mha(
                max_seq_len=max_seq_len,
                alpha=self._attn_alpha,
                q=q,
                k=k,
                v=v,
                seq_offsets=x_offsets,
                causal=self._causal,
                dropout_pr=0.0,
                training=False,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                sort_by_length=self._sort_by_length,
                kernel=self.hammer_kernel(),
            ).view(-1, self._hidden_dim * self._num_heads)

        # NeuTRENO term ``lambda * (v_0 - v)``. It is None on the first layer (where
        # ``v_0`` is None) and when ``lambda == 0`` -- in both cases the two branches
        # below reduce to the vanilla STU output.
        term: Optional[torch.Tensor] = None
        if self._neutreno_lambda > 0.0 and v_0 is not None:
            term = self._neutreno_lambda * (v_0 - v).reshape(
                -1, self._num_heads * self._hidden_dim
            )

        if self._neutreno_after_norm and term is not None:
            # === AFTER-NORM ===  output is built from ``norm(attn) + term``.
            # The fused kernel never exposes ``norm(attn)``, so run the unfused
            # output stage to inject the term between the norm and the u-gate.
            with record_function("## stu_compute_output_after_norm ##"):
                out = self._compute_output_after_norm(
                    attn=attn_output, u=u, x=x, post_norm_term=term
                )
        else:
            # === BEFORE-NORM (default) ===  output is built from ``norm(attn + term)``.
            # Folding the term into ``attn`` first lets the fused output kernel absorb
            # it for free (no term -> this is exactly the vanilla STU output).
            if term is not None:
                attn_output = attn_output + term
            with record_function("## stu_compute_output ##"):
                out = hstu_compute_output(
                    attn=attn_output,
                    u=u,
                    x=x,
                    norm_weight=self._output_norm_weight.to(x.dtype),
                    norm_bias=self._output_norm_bias.to(x.dtype),
                    norm_eps=1e-6,
                    dropout_ratio=self._output_dropout_ratio,
                    output_weight=self._output_weight.to(x.dtype),
                    group_norm=self._use_group_norm,
                    num_heads=self._num_heads,
                    linear_dim=self._hidden_dim,
                    concat_u=True,
                    concat_x=True,
                    mul_u_activation_type="none",
                    training=self.training,
                    kernel=self.hammer_kernel(),
                    recompute_y_in_backward=self._recompute_y,
                )
        return out, v


class STUStackNeuTRENO(STU):
    """STU stack that threads ``v_0`` (first layer value) to every NeuTRENO layer."""

    def __init__(
        self,
        stu_list: List[STULayerNeuTRENO],
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=stu_list)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v_0: Optional[torch.Tensor] = None
        for i, layer in enumerate(self._stu_layers):
            x, v = layer(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
                v_0=v_0,
            )
            if i == 0:
                v_0 = v
        return x

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "STUStackNeuTRENO does not support incremental/cached decoding."
        )


class _RMSNorm(torch.nn.Module):
    """Minimal RMSNorm (computed in fp32 for stability); used by AttnRes keys."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self._eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self._eps)
        return (xf * rms).to(dtype) * self.weight


class _AttnResidual(torch.nn.Module):
    """Attention Residual: per-token softmax mix over previous layer/block reps.

    Given reps ``[v_0, ..., v_{m-1}]`` (each a jagged ``[L, D]`` tensor), returns
    ``h = sum_i softmax_i( w . RMSNorm(v_i) ) * v_i`` where ``w`` is this call site's
    learned pseudo-query (``proj.weight``), zero-init so the mix starts uniform.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self._norm = _RMSNorm(embedding_dim, eps)
        self._proj = torch.nn.Linear(embedding_dim, 1, bias=False)
        torch.nn.init.zeros_(self._proj.weight)

    def forward(self, reps: List[torch.Tensor]) -> torch.Tensor:
        v = torch.stack(reps, dim=0)  # [M, L, D]
        k = self._norm(v)  # [M, L, D]
        logits = torch.einsum("d,mld->ml", self._proj.weight.squeeze(0), k)  # [M, L]
        weights = torch.softmax(logits, dim=0)  # [M, L]
        return torch.einsum("ml,mld->ld", weights, v)  # [L, D]


class STULayerAttnRes(STULayer):
    """STU layer that returns its delta only (no residual add), for AttnRes stacking.

    ``hstu_compute_output`` always adds the input ``x`` back via ``addmm`` (``out = x +
    y @ W``); the fixed residual is removed here by subtracting ``x`` so the residual
    stream can be reconstructed by :class:`STUStackAttnRes` via a learned softmax mix.
    ``concat_x`` stays True so the output projection keeps its native 3x input width.
    """

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with record_function("## stu_preprocess_and_attention ##"):
            u, attn_output, k, v = hstu_preprocess_and_attention(
                x=x,
                norm_weight=self._input_norm_weight.to(x.dtype),
                norm_bias=self._input_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                num_heads=self._num_heads,
                attn_dim=self._attention_dim,
                hidden_dim=self._hidden_dim,
                uvqk_weight=self._uvqk_weight.to(x.dtype),
                uvqk_bias=self._uvqk_beta.to(x.dtype),
                max_seq_len=max_seq_len,
                seq_offsets=x_offsets,
                attn_alpha=self._attn_alpha,
                causal=self._causal,
                num_targets=num_targets if self._target_aware else None,
                max_attn_len=self._max_attn_len,
                contextual_seq_len=self._contextual_seq_len,
                recompute_uvqk_in_backward=self._recompute_uvqk,
                recompute_normed_x_in_backward=self._recompute_normed_x,
                sort_by_length=self._sort_by_length,
                prefill=kv_caching_lengths is not None,
                kernel=self.hammer_kernel(),
            )

        with record_function("## stu_compute_output ##"):
            out = hstu_compute_output(
                attn=attn_output,
                u=u,
                x=x,
                norm_weight=self._output_norm_weight.to(x.dtype),
                norm_bias=self._output_norm_bias.to(x.dtype),
                norm_eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                output_weight=self._output_weight.to(x.dtype),
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="none",
                training=self.training,
                kernel=self.hammer_kernel(),
                recompute_y_in_backward=self._recompute_y,
            )
        # hstu_compute_output returns ``x + delta``; strip the fixed residual so the
        # AttnRes mix in STUStackAttnRes owns the residual stream.
        return out - x


class STUStackAttnRes(STU):
    """STU stack with Attention Residuals (input-dependent softmax residual mix).

    ``attnres_block_size`` HSTU layers form one AttnRes block; 1 (default) is Full
    AttnRes (each layer mixes over all previous layer outputs).
    """

    def __init__(
        self,
        stu_list: List[STULayerAttnRes],
        embedding_dim: int,
        attnres_block_size: int = 1,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=stu_list)
        self._attnres_block_size: int = attnres_block_size
        self._attn_res: torch.nn.ModuleList = torch.nn.ModuleList(
            [_AttnResidual(embedding_dim) for _ in stu_list]
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        blocks: List[torch.Tensor] = [x]
        partial: Optional[torch.Tensor] = None
        for i, layer in enumerate(self._stu_layers):
            pool = blocks if partial is None else blocks + [partial]
            h = self._attn_res[i](pool)
            delta = layer(
                x=h,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
            partial = delta if partial is None else partial + delta
            if (i + 1) % self._attnres_block_size == 0:
                blocks.append(partial)
                partial = None
        return partial if partial is not None else blocks[-1]

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "STUStackAttnRes does not support incremental/cached decoding."
        )


class STULayermHC(STULayerAttnRes):
    """STU layer for manifold-constrained hyper-connection (mHC) stacking.

    The layer must return its branch output only -- the residual is owned by the
    per-layer :class:`HyperConnection` in :class:`STUStackmHC`. This is exactly the
    delta-only behaviour of :class:`STULayerAttnRes` (``hstu_compute_output``
    returns ``x + delta`` and the ``forward`` strips ``x``), so the forward is
    inherited unchanged.
    """


class STUStackmHC(STU):
    """STU stack wired with static manifold-constrained hyper-connections.

    The hidden state is widened into ``mhc_num_streams`` parallel residual streams.
    Each layer reads a single input via ``H_pre``, the streams are mixed by a
    doubly-stochastic ``H_res`` (Sinkhorn-projected), the layer's branch output is
    written back onto every stream via ``H_post``, and the streams are summed at the
    end. ``mhc_num_streams = 1`` recovers (approximately) the vanilla residual.
    """

    def __init__(
        self,
        stu_list: List[STULayermHC],
        num_streams: int = 4,
        num_iters: int = 20,
        tau: float = 0.05,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=stu_list)
        self._num_streams: int = num_streams
        self._hyper_connections: torch.nn.ModuleList = torch.nn.ModuleList(
            [
                HyperConnection(
                    num_streams=num_streams,
                    layer_index=i,
                    num_iters=num_iters,
                    tau=tau,
                )
                for i in range(len(stu_list))
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        streams = expand_streams(x, self._num_streams)
        for layer, hc in zip(self._stu_layers, self._hyper_connections):
            branch_input, mixed = hc.width_connection(streams)
            branch_out = layer(
                x=branch_input,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
                max_kv_caching_len=max_kv_caching_len,
                kv_caching_lengths=kv_caching_lengths,
            )
            streams = hc.depth_connection(branch_out, mixed)
        return reduce_streams(streams)

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "STUStackmHC does not support incremental/cached decoding."
        )
