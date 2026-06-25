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
"""Vanilla-PyTorch HSTU (``STU_PYTORCH``).

A hackable reference implementation of one HSTU block where every stage is plain
eager PyTorch **except the attention itself**. The ``A.V`` attention is the only
O(N^2) step; a naive dense PyTorch implementation materializes the full
``[B, H, N, N]`` score matrix (32+ GiB at seq_len 16384 / batch 16 / bf16) and
OOMs, while the fused Triton kernel never materializes it. So the attention runs
through the Triton ``hstu_mha`` kernel and everything before/after it -- the
``uvqk`` projection and the gated output projection (norm, ``u``-gate, concat,
dropout, residual) -- runs in vanilla eager PyTorch.

This makes the attention call the single swappable slot: drop in an alternative
long-term-memory mechanism (e.g. gated DeltaNet) there, leaving the readable
eager PyTorch pre/post stages untouched. Parameters are inherited from
``STULayer``, so a ``STU_PYTORCH`` run matches the fused ``STU`` baseline up to
kernel numerics.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.stu import STU, STULayer
from generative_recommenders.ops.hstu_attention import hstu_mha
from generative_recommenders.ops.pytorch.pt_hstu_linear import (
    pytorch_hstu_compute_output,
)
from torch.autograd.profiler import record_function


class STULayerPyTorch(STULayer):
    """One HSTU block: Triton attention, vanilla eager-PyTorch everything else.

    Inherits the parameter construction (uvqk projection, input/output norms,
    output projection) from :class:`STULayer`; only the forward is overridden:

      1. ``layer_norm(x)`` then a single ``uvqk`` projection, split into
         ``u, v, q, k`` with ``u = silu(u)`` -- vanilla eager PyTorch;
      2. HSTU attention ``A.V`` via the Triton :func:`hstu_mha` kernel
         (SiLU-pointwise, ``/N`` normalized, causal + target-aware masked). This
         is the only O(N^2) step; the kernel keeps it from materializing the full
         score matrix and OOMing at long sequence lengths;
      3. gated output via :func:`pytorch_hstu_compute_output`
         (``norm(attn)`` gated by ``u``, concat ``[u, x, gated]``, dropout, and
         the ``x + y @ W`` residual) -- vanilla eager PyTorch.
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
        dtype = x.dtype
        with record_function("## stu_pytorch_uvqk ##"):
            normed_x = F.layer_norm(
                x,
                normalized_shape=(x.shape[-1],),
                weight=self._input_norm_weight.to(dtype),
                bias=self._input_norm_bias.to(dtype),
                eps=1e-6,
            )
            uvqk = torch.addmm(
                self._uvqk_beta.to(dtype), normed_x, self._uvqk_weight.to(dtype)
            )
            u, v, q, k = torch.split(
                uvqk,
                [
                    self._hidden_dim * self._num_heads,
                    self._hidden_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=1,
            )
            u = F.silu(u)
            q = q.view(-1, self._num_heads, self._attention_dim)
            k = k.view(-1, self._num_heads, self._attention_dim)
            v = v.view(-1, self._num_heads, self._hidden_dim)

        # Attention is the only O(N^2) step -- run it through the Triton kernel so
        # the [B, H, N, N] score matrix is never materialized (a naive dense
        # PyTorch einsum OOMs at seq_len 16384). This is the swappable slot for
        # alternative long-term-memory mechanisms.
        with record_function("## stu_pytorch_mha (triton) ##"):
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
                kernel=HammerKernel.TRITON,
            ).view(-1, self._hidden_dim * self._num_heads)

        with record_function("## stu_pytorch_compute_output ##"):
            return pytorch_hstu_compute_output(
                attn=attn_output,
                u=u,
                x=x,
                norm_weight=self._output_norm_weight.to(dtype),
                norm_bias=self._output_norm_bias.to(dtype),
                output_weight=self._output_weight.to(dtype),
                eps=1e-6,
                dropout_ratio=self._output_dropout_ratio,
                training=self.training,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="none",
                group_norm=self._use_group_norm,
                num_heads=self._num_heads,
                linear_dim=self._hidden_dim,
            )


class STUStackPyTorch(STU):
    """Sequential stack of :class:`STULayerPyTorch` blocks (vanilla residual)."""

    def __init__(
        self,
        stu_list: List[STULayerPyTorch],
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
        raise NotImplementedError(
            "STUStackPyTorch does not support incremental/cached decoding."
        )
