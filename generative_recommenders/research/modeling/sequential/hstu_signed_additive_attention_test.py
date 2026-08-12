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

import contextlib
import io
import math
import unittest
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.hstu import (
    HSTU,
    RelativeBucketedTimeAndPositionBasedBias,
    SequentialTransductionUnitJagged,
    _apply_signed_additive_feature_map,
    _per_head_additive_dot_attention,
    _per_head_hstu_weights,
    _per_head_signed_additive_abs_coefficient_oracle,
    _per_head_signed_additive_feature_attention,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
)
from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import (
    DotProductSimilarity,
)


class _FailIfEvaluatedRelativeBias(RelativeBucketedTimeAndPositionBasedBias):
    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        raise AssertionError("signed additive modes must not evaluate pairwise RAB")


def _feature_map(x: torch.Tensor, feature_map: str, gamma: float) -> torch.Tensor:
    if feature_map == "identity":
        return x
    features = torch.tanh(gamma * x)
    if feature_map == "tanh":
        return features
    if feature_map == "abs_tanh":
        return features.abs()
    raise AssertionError(f"unexpected test feature map {feature_map}")


def _dense_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    feature_map: str,
    gamma: float,
    valid_lengths: Optional[torch.Tensor] = None,
    absolute_coefficient: bool = False,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, _ = q.shape
    q_features = _feature_map(q, feature_map, gamma)
    k_features = _feature_map(k, feature_map, gamma)
    coefficients = torch.einsum(
        "bnhd,bmhd->bhnm", q_features, k_features
    )
    if absolute_coefficient:
        coefficients = coefficients.abs()
    mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device)
    ).view(1, 1, seq_len, seq_len)
    if valid_lengths is not None:
        valid = torch.arange(seq_len, device=q.device).unsqueeze(0) < valid_lengths.to(
            q.device
        ).unsqueeze(1)
        mask = mask & valid.view(batch_size, 1, seq_len, 1)
        mask = mask & valid.view(batch_size, 1, 1, seq_len)
    weights = coefficients * mask.to(coefficients.dtype) * (0.5 / seq_len)
    return torch.einsum("bhnm,bmhe->bnhe", weights, v).reshape(
        batch_size, seq_len, num_heads * v.size(-1)
    )


@contextlib.contextmanager
def _cpu_jagged_ops() -> Iterator[None]:
    namespace = torch.ops.fbgemm
    missing = object()
    saved = {
        name: getattr(namespace, name, missing)
        for name in ("jagged_to_padded_dense", "dense_to_jagged")
    }

    def jagged_to_padded_dense(
        values: torch.Tensor,
        offsets: list[torch.Tensor],
        max_lengths: list[int],
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        seq_offsets = offsets[0]
        max_length = max_lengths[0]
        batch_size = seq_offsets.numel() - 1
        output = torch.full(
            (batch_size, max_length, *values.shape[1:]),
            padding_value,
            dtype=values.dtype,
            device=values.device,
        )
        for batch_idx in range(batch_size):
            start = int(seq_offsets[batch_idx].item())
            end = int(seq_offsets[batch_idx + 1].item())
            output[batch_idx, : end - start] = values[start:end]
        return output

    def dense_to_jagged(
        values: torch.Tensor, offsets: list[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        seq_offsets = offsets[0]
        pieces = []
        for batch_idx in range(seq_offsets.numel() - 1):
            length = int(
                (seq_offsets[batch_idx + 1] - seq_offsets[batch_idx]).item()
            )
            pieces.append(values[batch_idx, :length])
        return (torch.cat(pieces, dim=0),)

    namespace.jagged_to_padded_dense = jagged_to_padded_dense
    namespace.dense_to_jagged = dense_to_jagged
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is missing:
                delattr(namespace, name)
            else:
                setattr(namespace, name, previous)
class SignedAdditiveFeatureAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(250302130)
        self.batch_size = 2
        self.seq_len = 5
        self.num_heads = 2
        self.attention_dim = 3
        self.linear_dim = 2
        self.q = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.attention_dim,
            generator=generator,
            dtype=torch.float64,
        )
        self.k = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.attention_dim,
            generator=generator,
            dtype=torch.float64,
        )
        self.v = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.linear_dim,
            generator=generator,
            dtype=torch.float64,
        )
        self.valid_lengths = torch.tensor([self.seq_len, self.seq_len - 2])

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        feature_map: str,
        gamma: float,
        valid_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _per_head_signed_additive_feature_attention(
            padded_q=q.reshape(self.batch_size, self.seq_len, -1),
            padded_k=k.reshape(self.batch_size, self.seq_len, -1),
            padded_v=v.reshape(self.batch_size, self.seq_len, -1),
            num_heads=self.num_heads,
            attention_dim=self.attention_dim,
            linear_dim=self.linear_dim,
            feature_map=feature_map,
            gamma=gamma,
            valid_lengths=valid_lengths,
        )

    def _assert_outputs_and_gradients_match_dense(self, feature_map: str) -> None:
        gamma = 0.7
        upstream = torch.randn_like(
            self.v.reshape(self.batch_size, self.seq_len, -1)
        )
        scan_inputs = tuple(
            tensor.clone().requires_grad_() for tensor in (self.q, self.k, self.v)
        )
        dense_inputs = tuple(
            tensor.clone().requires_grad_() for tensor in (self.q, self.k, self.v)
        )

        scan_output = self._scan(
            *scan_inputs,
            feature_map=feature_map,
            gamma=gamma,
            valid_lengths=self.valid_lengths,
        )
        dense_output = _dense_oracle(
            *dense_inputs,
            feature_map=feature_map,
            gamma=gamma,
            valid_lengths=self.valid_lengths,
        )
        torch.testing.assert_close(scan_output, dense_output)

        scan_gradients = torch.autograd.grad(
            (scan_output * upstream).sum(), scan_inputs
        )
        dense_gradients = torch.autograd.grad(
            (dense_output * upstream).sum(), dense_inputs
        )
        for scan_gradient, dense_gradient in zip(scan_gradients, dense_gradients):
            torch.testing.assert_close(scan_gradient, dense_gradient)

    def test_signed_tanh_scan_matches_dense_outputs_and_gradients(self) -> None:
        self._assert_outputs_and_gradients_match_dense("tanh")

    def test_abs_tanh_scan_matches_factorized_dense_outputs_and_gradients(
        self,
    ) -> None:
        self._assert_outputs_and_gradients_match_dense("abs_tanh")

    def test_identity_is_exactly_legacy_additive_dot(self) -> None:
        actual = self._scan(
            self.q,
            self.k,
            self.v,
            feature_map="identity",
            gamma=1.0,
        )
        expected = _per_head_additive_dot_attention(
            padded_q=self.q.reshape(self.batch_size, self.seq_len, -1),
            padded_k=self.k.reshape(self.batch_size, self.seq_len, -1),
            padded_v=self.v.reshape(self.batch_size, self.seq_len, -1),
            num_heads=self.num_heads,
            attention_dim=self.attention_dim,
            linear_dim=self.linear_dim,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_repeated_evidence_adds_exactly_at_fixed_sequence_length(self) -> None:
        q = torch.tensor([[[[0.75, -0.5]], [[0.75, -0.5]]]], dtype=torch.float64)
        repeated_k = torch.tensor(
            [[[[0.25, -1.0]], [[0.25, -1.0]]]], dtype=torch.float64
        )
        repeated_v = torch.tensor(
            [[[[1.5, -0.25]], [[1.5, -0.25]]]], dtype=torch.float64
        )
        single_k = repeated_k.clone()
        single_v = repeated_v.clone()
        single_k[:, 0] = 0.0
        single_v[:, 0] = 0.0

        def attention(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return _per_head_signed_additive_feature_attention(
                padded_q=q.reshape(1, 2, 2),
                padded_k=k.reshape(1, 2, 2),
                padded_v=v.reshape(1, 2, 2),
                num_heads=1,
                attention_dim=2,
                linear_dim=2,
                feature_map="tanh",
                gamma=0.7,
            )

        doubled = attention(repeated_k, repeated_v)
        single = attention(single_k, single_v)
        torch.testing.assert_close(
            doubled[:, 1], 2.0 * single[:, 1], rtol=0.0, atol=0.0
        )

    def test_variable_padding_ignores_poisoned_tail(self) -> None:
        clean = self._scan(
            self.q,
            self.k,
            self.v,
            feature_map="tanh",
            gamma=1.0,
            valid_lengths=self.valid_lengths,
        )
        poisoned = [tensor.clone() for tensor in (self.q, self.k, self.v)]
        for tensor in poisoned:
            tensor[1, self.valid_lengths[1] :] = 1e6
        actual = self._scan(
            *poisoned,
            feature_map="tanh",
            gamma=1.0,
            valid_lengths=self.valid_lengths,
        )
        torch.testing.assert_close(actual, clean)
        torch.testing.assert_close(
            actual[1, self.valid_lengths[1] :],
            torch.zeros_like(actual[1, self.valid_lengths[1] :]),
        )

    def test_single_token_and_negative_coefficient(self) -> None:
        q = torch.tensor([[[[2.0, 1.0]]]], dtype=torch.float64)
        k = torch.tensor([[[[-1.0, -0.5]]]], dtype=torch.float64)
        v = torch.tensor([[[[3.0, -2.0]]]], dtype=torch.float64)
        actual = _per_head_signed_additive_feature_attention(
            q.reshape(1, 1, 2),
            k.reshape(1, 1, 2),
            v.reshape(1, 1, 2),
            num_heads=1,
            attention_dim=2,
            linear_dim=2,
            feature_map="tanh",
            gamma=0.5,
        ).reshape(1, 1, 1, 2)
        coefficient = torch.dot(torch.tanh(0.5 * q[0, 0, 0]), torch.tanh(0.5 * k[0, 0, 0]))
        self.assertLess(coefficient.item(), 0.0)
        torch.testing.assert_close(actual[0, 0, 0], 0.5 * coefficient * v[0, 0, 0])

    def test_absolute_coefficient_oracle_is_not_abs_feature_kernel(self) -> None:
        q = torch.tensor([[[1.0, 1.0]]], dtype=torch.float64)
        k = torch.tensor([[[1.0, -1.0]]], dtype=torch.float64)
        v = torch.tensor([[[2.0]]], dtype=torch.float64)
        causal_mask = torch.ones(1, 1, dtype=torch.bool)
        signed_coefficient = torch.dot(torch.tanh(q[0, 0]), torch.tanh(k[0, 0]))
        self.assertEqual(signed_coefficient.item(), 0.0)

        absolute_coefficient = _per_head_signed_additive_abs_coefficient_oracle(
            padded_q=q,
            padded_k=k,
            padded_v=v,
            invalid_attn_mask=causal_mask,
            num_heads=1,
            attention_dim=2,
            linear_dim=1,
            gamma=1.0,
        )
        abs_feature = _per_head_signed_additive_feature_attention(
            padded_q=q,
            padded_k=k,
            padded_v=v,
            num_heads=1,
            attention_dim=2,
            linear_dim=1,
            feature_map="abs_tanh",
            gamma=1.0,
        )
        torch.testing.assert_close(absolute_coefficient, torch.zeros_like(absolute_coefficient))
        self.assertGreater(abs_feature.abs().max().item(), 0.0)

        random_oracle = _per_head_signed_additive_abs_coefficient_oracle(
            padded_q=self.q.reshape(self.batch_size, self.seq_len, -1),
            padded_k=self.k.reshape(self.batch_size, self.seq_len, -1),
            padded_v=self.v.reshape(self.batch_size, self.seq_len, -1),
            invalid_attn_mask=torch.tril(
                torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)
            ),
            num_heads=self.num_heads,
            attention_dim=self.attention_dim,
            linear_dim=self.linear_dim,
            gamma=0.7,
            valid_lengths=self.valid_lengths,
        )
        random_dense = _dense_oracle(
            self.q,
            self.k,
            self.v,
            feature_map="tanh",
            gamma=0.7,
            valid_lengths=self.valid_lengths,
            absolute_coefficient=True,
        )
        torch.testing.assert_close(random_oracle, random_dense)

    def test_dtype_is_restored_after_fp32_accumulation(self) -> None:
        q = self.q.float().to(torch.float16)
        k = self.k.float().to(torch.float16)
        v = self.v.float().to(torch.float16)
        actual = self._scan(q, k, v, feature_map="tanh", gamma=0.7)
        expected = _dense_oracle(
            q.float(), k.float(), v.float(), feature_map="tanh", gamma=0.7
        )
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual.float(), expected, rtol=2e-3, atol=2e-3)

    def test_invalid_gamma_and_unknown_feature_map_are_rejected(self) -> None:
        for gamma in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with self.subTest(gamma=gamma), self.assertRaisesRegex(
                ValueError, "gamma must be finite and positive"
            ):
                _apply_signed_additive_feature_map(
                    torch.ones(1), feature_map="tanh", gamma=gamma
                )
        with self.assertRaisesRegex(ValueError, "Unknown signed additive feature map"):
            _apply_signed_additive_feature_map(
                torch.ones(1), feature_map="tanh_typo", gamma=1.0
            )
        with self.assertRaisesRegex(ValueError, "gamma must be finite and positive"):
            SequentialTransductionUnitJagged(
                embedding_dim=4,
                linear_hidden_dim=2,
                attention_dim=2,
                dropout_ratio=0.0,
                attn_dropout_ratio=0.0,
                num_heads=2,
                linear_activation="silu",
                normalization="signed_additive_tanh",
                signed_feature_gamma=0.0,
            )

    def test_new_modes_reject_incremental_decoding_before_using_cache(self) -> None:
        for normalization in (
            "signed_additive_identity",
            "signed_additive_tanh",
            "signed_additive_abs_tanh",
            "signed_additive_abs_coefficient_oracle",
        ):
            with self.subTest(normalization=normalization):
                module = SequentialTransductionUnitJagged(
                    embedding_dim=4,
                    linear_hidden_dim=2,
                    attention_dim=2,
                    dropout_ratio=0.0,
                    attn_dropout_ratio=0.0,
                    num_heads=2,
                    linear_activation="silu",
                    normalization=normalization,
                )
                with self.assertRaisesRegex(
                    NotImplementedError, "Signed additive attention"
                ):
                    module(
                        x=torch.zeros(1, 4),
                        x_offsets=torch.tensor([0, 1]),
                        all_timestamps=None,
                        invalid_attn_mask=torch.ones(1, 1),
                        delta_x_offsets=(torch.tensor([0]), torch.tensor([0])),
                    )

    def test_dormant_relative_bias_is_zero_connected_for_ddp(self) -> None:
        generator = torch.Generator().manual_seed(19)
        modes = (
            "signed_additive_identity",
            "signed_additive_tanh",
            "signed_additive_abs_tanh",
            "signed_additive_abs_coefficient_oracle",
        )
        with _cpu_jagged_ops():
            for mode in modes:
                with self.subTest(mode=mode):
                    relative_bias = _FailIfEvaluatedRelativeBias(
                        max_seq_len=3,
                        num_buckets=4,
                        bucketization_fn=lambda x: x.abs().clamp(max=4).long(),
                    )
                    module = SequentialTransductionUnitJagged(
                        embedding_dim=4,
                        linear_hidden_dim=2,
                        attention_dim=2,
                        dropout_ratio=0.0,
                        attn_dropout_ratio=0.0,
                        num_heads=2,
                        linear_activation="silu",
                        relative_attention_bias_module=relative_bias,
                        normalization=mode,
                        signed_feature_gamma=0.7,
                    ).eval()
                    x = torch.randn(3, 4, generator=generator)
                    offsets = torch.tensor([0, 3])
                    timestamps = torch.tensor([[1, 2, 4]])
                    causal_mask = torch.tril(torch.ones(3, 3))

                    reference, _ = module(
                        x=x,
                        x_offsets=offsets,
                        all_timestamps=timestamps,
                        invalid_attn_mask=causal_mask,
                    )
                    with torch.no_grad():
                        relative_bias._ts_w.copy_(
                            torch.randn(
                                relative_bias._ts_w.shape, generator=generator
                            )
                            * 1000.0
                        )
                        relative_bias._pos_w.copy_(
                            torch.randn(
                                relative_bias._pos_w.shape, generator=generator
                            )
                            * 1000.0
                        )
                    changed, _ = module(
                        x=x,
                        x_offsets=offsets,
                        all_timestamps=timestamps,
                        invalid_attn_mask=causal_mask,
                    )
                    torch.testing.assert_close(changed, reference, rtol=0.0, atol=0.0)

                    module.zero_grad(set_to_none=True)
                    changed.square().sum().backward()
                    for name, parameter in module.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(
                        relative_bias._ts_w.grad,
                        torch.zeros_like(relative_bias._ts_w),
                    )
                    torch.testing.assert_close(
                        relative_bias._pos_w.grad,
                        torch.zeros_like(relative_bias._pos_w),
                    )

    def test_default_hstu_score_path_remains_silu(self) -> None:
        q = self.q[:, :3]
        k = self.k[:, :3]
        bias = torch.randn(2, 3, 3, dtype=torch.float64)
        mask = torch.tril(torch.ones(3, 3, dtype=torch.float64))
        scores = torch.einsum("bnhd,bmhd->bhnm", q, k) + bias.unsqueeze(1)
        actual = _per_head_hstu_weights(q, k, mask, bias)
        torch.testing.assert_close(
            actual, F.silu(scores) * mask.view(1, 1, 3, 3) / 3
        )


class SignedAdditiveInventoryTest(unittest.TestCase):
    @staticmethod
    def _build_model(dataset: str, normalization: str) -> HSTU:
        if dataset == "ml-1m":
            embedding_dim, num_items = 50, 3952
            num_blocks, num_heads, dqk, dv = 8, 2, 25, 25
        elif dataset == "ml-20m":
            embedding_dim, num_items = 256, 131262
            num_blocks, num_heads, dqk, dv = 16, 8, 32, 32
        else:
            raise AssertionError(f"unexpected dataset {dataset}")
        with contextlib.redirect_stdout(io.StringIO()), torch.device("meta"):
            return HSTU(
                max_sequence_len=200,
                max_output_len=11,
                embedding_dim=embedding_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                linear_dim=dv,
                attention_dim=dqk,
                normalization=normalization,
                linear_config="uvqk",
                linear_activation="silu",
                linear_dropout_rate=0.2,
                attn_dropout_rate=0.0,
                embedding_module=LocalEmbeddingModule(
                    num_items=num_items, item_embedding_dim=embedding_dim
                ),
                similarity_module=DotProductSimilarity(),
                input_features_preproc_module=(
                    LearnablePositionalEmbeddingInputFeaturesPreprocessor(
                        max_sequence_len=211,
                        embedding_dim=embedding_dim,
                        dropout_rate=0.2,
                    )
                ),
                output_postproc_module=L2NormEmbeddingPostprocessor(
                    embedding_dim=embedding_dim, eps=1e-6
                ),
                signed_feature_gamma=0.7,
                verbose=False,
            )

    @staticmethod
    def _parameter_inventory(
        model: torch.nn.Module,
    ) -> Dict[str, Tuple[Tuple[int, ...], int, torch.dtype, bool]]:
        return {
            name: (
                tuple(parameter.shape),
                parameter.numel(),
                parameter.dtype,
                parameter.requires_grad,
            )
            for name, parameter in model.named_parameters()
        }

    @staticmethod
    def _buffer_inventory(
        model: torch.nn.Module,
    ) -> Dict[str, Tuple[Tuple[int, ...], int, torch.dtype]]:
        return {
            name: (tuple(buffer.shape), buffer.numel(), buffer.dtype)
            for name, buffer in model.named_buffers()
        }

    def test_ml1_and_ml20_named_inventories_are_exactly_hstu_matched(self) -> None:
        expected_totals = {"ml-1m": 313_000, "ml-20m": 38_913_120}
        expected_blocks = {"ml-1m": 8, "ml-20m": 16}
        modes = (
            "signed_additive_identity",
            "signed_additive_tanh",
            "signed_additive_abs_tanh",
            "signed_additive_abs_coefficient_oracle",
        )
        for dataset in ("ml-1m", "ml-20m"):
            baseline = self._build_model(dataset, "rel_bias")
            baseline_parameters = self._parameter_inventory(baseline)
            baseline_buffers = self._buffer_inventory(baseline)
            self.assertEqual(
                sum(entry[1] for entry in baseline_parameters.values()),
                expected_totals[dataset],
            )
            for mode in modes:
                with self.subTest(dataset=dataset, mode=mode):
                    model = self._build_model(dataset, mode)
                    parameters = self._parameter_inventory(model)
                    buffers = self._buffer_inventory(model)
                    self.assertEqual(parameters, baseline_parameters)
                    self.assertEqual(buffers, baseline_buffers)
                    self.assertFalse(any("gamma" in name for name in parameters))
                    self.assertFalse(any("gamma" in name for name in buffers))
                    self.assertEqual(model._signed_feature_gamma, 0.7)
                    self.assertEqual(
                        len(
                            [
                                name
                                for name in parameters
                                if name.endswith("._rel_attn_bias._ts_w")
                            ]
                        ),
                        expected_blocks[dataset],
                    )
                    self.assertEqual(
                        len(
                            [
                                name
                                for name in parameters
                                if name.endswith("._rel_attn_bias._pos_w")
                            ]
                        ),
                        expected_blocks[dataset],
                    )
                    if mode == "signed_additive_abs_coefficient_oracle":
                        self.assertIn("oracle", model.debug_str())


if __name__ == "__main__":
    unittest.main()
