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
KDA (Kimi Delta Attention) encoder for the research sequential-retrieval setting.

This is a drop-in replacement for :class:`HSTU` (see ``hstu.py``): it mirrors the
same public surface (``generate_user_embeddings`` / ``forward`` / ``encode`` /
``get_item_embeddings`` / ``debug_str``) so it plugs into
``get_sequential_encoder`` and ``train_fn`` with no changes anywhere else — only
the sequence mixer differs. Embedding, input preprocessing, output
postprocessing and the similarity head are all reused unchanged, so a KDA-vs-HSTU
comparison isolates the token-mixing block.

The mixer is a stack of pre-norm residual Kimi Delta Attention blocks
(``fla.layers.KimiDeltaAttention``), a linear-attention (gated delta-rule)
architecture from *Kimi Linear: An Expressive, Efficient Attention Architecture*
(https://arxiv.org/abs/2510.26692). Unlike HSTU's softmax-free quadratic
attention with an explicit relative-position/time bias, KDA is a causal linear
recurrence: causality is intrinsic to the left-to-right scan (no ``[N, N]`` mask)
and temporal structure is carried by KDA's per-channel forget gate rather than an
additive attention bias. Blocks run on a fixed-shape dense ``(B, N', D)`` batch
(real tokens left-aligned, zero-padded tail) so the KDA triton kernels compile
once; the variable-length packed (``cu_seqlens``) layout was ~100x slower because
the changing total token count forces per-batch kernel recompilation.

The KDA triton kernels require bf16 inputs, so each block runs under a bf16
autocast region while the surrounding model (embeddings, similarity, etc.) stays
in fp32 — matching how HSTU is trained here.
"""

from typing import Dict, List, Optional, Tuple, Union

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


class KDAJagged(torch.nn.Module):
    """A stack of pre-norm residual Kimi Delta Attention blocks.

    Each block computes ``x <- x + dropout(KDA(layernorm(x)))``. The
    ``KimiDeltaAttention`` layer is itself a complete token mixer (q/k/v/gate
    projections, optional short conv, gated RMSNorm and output projection), so a
    block is a standard pre-norm residual unit. LayerNorm is parameter-free
    (``F.layer_norm`` without affine), mirroring HSTU's use of ``_norm_input``.

    Blocks run on a fixed-shape dense ``(B, N', D)`` batch (see :meth:`forward`
    for why the variable-length ``cu_seqlens`` layout is avoided).
    """

    def __init__(
        self,
        embedding_dim: int,
        layers: List[torch.nn.Module],
        dropout_ratio: float = 0.0,
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._layers: torch.nn.ModuleList = torch.nn.ModuleList(layers)
        self._dropout_ratio: float = dropout_ratio
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype
        self._eps: float = epsilon

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) or (\\sum_i N_i, D) x float.
            x_offsets: (B + 1) x int.
            max_seq_len: padded length N' for the dense output.
        Returns:
            (B, N', D) x float.

        The KDA blocks run on a FIXED-shape dense (B, N', D) batch rather than
        the variable-length packed (cu_seqlens) layout. With cu_seqlens the total
        token count changes every batch, which makes the KDA triton kernels
        re-autotune/recompile on every step (~100x slowdown observed on ml-1m).
        A fixed (B, N') shape compiles once. KDA is strictly causal (left-to-right
        scan) and real tokens are left-aligned with zero padding at the tail, so
        the padding never leaks into the valid prefix -- on real positions the
        (B, N', D) result is identical to the packed path.
        """
        if len(x.size()) == 3:
            x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]
        # Repad to the fixed window N' (padding_value=0 -> zero tail).
        h = torch.ops.fbgemm.jagged_to_padded_dense(
            values=x,
            offsets=[x_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )  # [B, N', D]
        for layer in self._layers:
            residual = h
            normed = self._norm(h)
            with torch.autocast(
                "cuda",
                enabled=self._autocast_dtype is not None,
                dtype=self._autocast_dtype or torch.bfloat16,
            ):
                o, _, _ = layer(hidden_states=normed)  # batched causal, fixed shape
            o = o.to(residual.dtype)
            o = F.dropout(o, p=self._dropout_ratio, training=self.training)
            h = residual + o
        # Re-zero padding positions so output semantics match HSTU exactly.
        jagged_out = torch.ops.fbgemm.dense_to_jagged(h, [x_offsets])[0]
        return torch.ops.fbgemm.jagged_to_padded_dense(
            values=jagged_out,
            offsets=[x_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )


class KDA(SequentialEncoderWithLearnedSimilarityModule):
    """
    Kimi Delta Attention encoder, a linear-attention drop-in replacement for HSTU
    in the traditional sequential recommender setting (Section 4.1.1 of the HSTU
    paper). See the module docstring for the design rationale.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        head_dim: int,
        expand_v: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        num_v_heads: Optional[int] = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        kda_dropout_rate: float = 0.2,
        autocast_bf16: bool = True,
        kla_variant: str = "kda",
        verbose: bool = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._max_output_len: int = max_output_len
        # Dense output length N', identical to HSTU's attn-mask window, so the
        # encoded tensor shape matches HSTU's exactly.
        self._max_padded_length: int = max_sequence_len + max_output_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module

        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._head_dim: int = head_dim
        self._expand_v: float = expand_v
        self._num_v_heads: int = num_v_heads if num_v_heads is not None else num_heads
        self._use_short_conv: bool = use_short_conv
        self._conv_size: int = conv_size
        self._kda_dropout_rate: float = kda_dropout_rate
        self._kla_variant: str = kla_variant.lower()

        self._kda = KDAJagged(
            embedding_dim=embedding_dim,
            layers=[self._make_mixer_layer(i) for i in range(num_blocks)],
            dropout_ratio=kda_dropout_rate,
            autocast_dtype=torch.bfloat16 if autocast_bf16 else None,
        )
        self._verbose: bool = verbose
        self.reset_params()

    def _make_mixer_layer(self, layer_idx: int) -> torch.nn.Module:
        """Build one token-mixer block for the selected KLA variant.

        All variants subclass fla's ``KimiDeltaAttention`` (same dense
        ``forward(hidden_states)`` -> ``(o, None, ...)``), so ``KDAJagged``
        drives them uniformly. Imported lazily so this file stays importable on
        GPU-less login nodes (fla/triton import only at layer build).
        """
        common = dict(
            hidden_size=self._embedding_dim,
            head_dim=self._head_dim,
            num_heads=self._num_heads,
            num_v_heads=self._num_v_heads,
            expand_v=self._expand_v,
            use_short_conv=self._use_short_conv,
            conv_size=self._conv_size,
            layer_idx=layer_idx,
        )
        if self._kla_variant == "kda":
            from fla.layers import KimiDeltaAttention

            return KimiDeltaAttention(mode="chunk", **common)
        elif self._kla_variant == "iso":
            # Vendored from hyper-delta-net (Gated DeltaNet-2 / Kalman Linear
            # Attention). Fast Triton beta scan for training; naive recurrent
            # reference lives in kla/kla_ops/iso_naive.py for validation.
            from generative_recommenders.research.modeling.sequential.kla.iso_kla import (
                IsoKalmanLinearAttention,
            )

            return IsoKalmanLinearAttention(**common)
        else:
            raise ValueError(f"Unknown kla_variant {self._kla_variant!r}")

    def reset_params(self) -> None:
        # Keep fla's own KDA init and the embedding-table init; xavier-init the
        # rest (input preproc, similarity head), matching HSTU.reset_params so
        # both models share identical non-mixer initialization.
        for name, params in self.named_parameters():
            if ("_kda" in name) or ("_embedding_module" in name):
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
        sc = f"-sc{self._conv_size}" if self._use_short_conv else "-nosc"
        tag = self._kla_variant.upper()
        return (
            f"{tag}-b{self._num_blocks}-h{self._num_heads}-dk{self._head_dim}"
            + f"-ev{self._expand_v}-vh{self._num_v_heads}{sc}"
            + f"-d{self._kda_dropout_rate}"
        )

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        [B, N] -> [B, N', D].
        """
        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        user_embeddings = self._kda(
            x=user_embeddings,
            x_offsets=torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths),
            max_seq_len=self._max_padded_length,
        )
        return self._output_postproc(user_embeddings)

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
            past_ids: (B, N,) x int64 where the latest engaged ids come first.
            past_embeddings: (B, N, D) x float or (\\sum_b N_b, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            encoded_embeddings of [B, N', D].
        """
        encoded_embeddings = self.generate_user_embeddings(
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
    ) -> torch.Tensor:
        """
        Returns:
            (B, D) x float, representing embeddings for the current state.
        """
        encoded_seq_embeddings = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )  # [B, N', D]
        return get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[torch.Tensor]] = None,
        return_cache_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Runs encoder to obtain the current hidden states.

        ``delta_x_offsets`` / ``cache`` / ``return_cache_states`` are accepted for
        interface parity with HSTU (M-FALCON incremental decoding) but are not
        used by the KDA path; the full-sequence encode is cheap here.

        Returns:
            (B, D,) x float, encoded states at the most recent time step.
        """
        current_embeddings = self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        if return_cache_states:
            return current_embeddings, []
        return current_embeddings
