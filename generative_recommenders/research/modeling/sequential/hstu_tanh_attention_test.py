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
import unittest
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.hstu import (
    HSTU,
    SequentialTransductionUnitJagged,
    _apply_hstu_score_kernel,
    _per_head_hstu_weights,
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


class HSTUTanhAttentionTest(unittest.TestCase):
    @staticmethod
    def _attention_output(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        relative_bias: torch.Tensor,
    ) -> torch.Tensor:
        weights = _per_head_hstu_weights(
            padded_q=q,
            padded_k=k,
            invalid_attn_mask=mask,
            relative_attention_bias=relative_bias,
            score_kernel="tanh",
        )
        return torch.einsum("bhnm,bmhe->bnhe", weights, v)

    def test_matches_hand_computed_per_head_formula(self) -> None:
        q = torch.tensor(
            [
                [
                    [[1.0, -0.5], [0.25, 0.75]],
                    [[-0.5, 1.0], [1.5, -0.25]],
                    [[0.75, 0.5], [-1.0, 0.5]],
                ]
            ],
            dtype=torch.float64,
        )
        k = torch.tensor(
            [
                [
                    [[0.5, 1.0], [-0.25, 1.0]],
                    [[1.0, -1.0], [0.5, 0.5]],
                    [[-0.5, 0.25], [1.0, -0.75]],
                ]
            ],
            dtype=torch.float64,
        )
        v = torch.tensor(
            [
                [
                    [[1.0, 2.0], [-1.0, 0.5]],
                    [[0.5, -1.0], [2.0, 1.0]],
                    [[-2.0, 0.25], [0.75, -0.5]],
                ]
            ],
            dtype=torch.float64,
        )
        relative_bias = torch.tensor(
            [[[0.1, 0.0, 0.0], [-0.2, 0.3, 0.0], [0.4, -0.1, 0.2]]],
            dtype=torch.float64,
        )
        mask = torch.tril(torch.ones(3, 3, dtype=torch.float64))

        expected = torch.zeros_like(v)
        for query_idx in range(3):
            for head_idx in range(2):
                for key_idx in range(3):
                    if not mask[query_idx, key_idx]:
                        continue
                    score = (
                        torch.dot(
                            q[0, query_idx, head_idx], k[0, key_idx, head_idx]
                        )
                        + relative_bias[0, query_idx, key_idx]
                    )
                    expected[0, query_idx, head_idx] += (
                        torch.tanh(score) * v[0, key_idx, head_idx] / 3
                    )

        actual = self._attention_output(q, k, v, mask, relative_bias)
        torch.testing.assert_close(actual, expected)

    def test_heads_are_isolated(self) -> None:
        generator = torch.Generator().manual_seed(250302130)
        q = torch.randn(2, 5, 2, 3, generator=generator, dtype=torch.float64)
        k = torch.randn(2, 5, 2, 3, generator=generator, dtype=torch.float64)
        v = torch.randn(2, 5, 2, 2, generator=generator, dtype=torch.float64)
        relative_bias = torch.randn(2, 5, 5, generator=generator, dtype=torch.float64)
        mask = torch.tril(torch.ones(5, 5, dtype=torch.float64))

        original = self._attention_output(q, k, v, mask, relative_bias)
        changed_k = k.clone()
        changed_v = v.clone()
        changed_k[:, :, 1] = changed_k[:, :, 1] * 7.0 + 3.0
        changed_v[:, :, 1] = changed_v[:, :, 1] * -5.0 + 2.0
        changed = self._attention_output(
            q, changed_k, changed_v, mask, relative_bias
        )

        torch.testing.assert_close(original[:, :, 0], changed[:, :, 0])
        self.assertFalse(torch.allclose(original[:, :, 1], changed[:, :, 1]))

    def test_gradients_match_direct_reference(self) -> None:
        generator = torch.Generator().manual_seed(42)
        q = torch.randn(2, 4, 3, 2, generator=generator, dtype=torch.float64)
        k = torch.randn(2, 4, 3, 2, generator=generator, dtype=torch.float64)
        v = torch.randn(2, 4, 3, 2, generator=generator, dtype=torch.float64)
        bias = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64)
        mask = torch.tril(torch.ones(2, 4, 4, dtype=torch.float64))
        mask[1, 3, :] = 0.0
        upstream = torch.randn(
            2, 4, 3, 2, generator=generator, dtype=torch.float64
        )

        actual_inputs = tuple(
            tensor.clone().requires_grad_() for tensor in (q, k, v, bias)
        )
        actual_output = self._attention_output(
            actual_inputs[0],
            actual_inputs[1],
            actual_inputs[2],
            mask,
            actual_inputs[3],
        )
        actual_gradients = torch.autograd.grad(
            (actual_output * upstream).sum(), actual_inputs
        )

        reference_inputs = tuple(
            tensor.clone().requires_grad_() for tensor in (q, k, v, bias)
        )
        reference_scores = torch.einsum(
            "bnhd,bmhd->bhnm", reference_inputs[0], reference_inputs[1]
        ) + reference_inputs[3].unsqueeze(1)
        reference_weights = (
            torch.tanh(reference_scores) * mask.unsqueeze(1) / q.size(1)
        )
        reference_output = torch.einsum(
            "bhnm,bmhe->bnhe", reference_weights, reference_inputs[2]
        )
        reference_gradients = torch.autograd.grad(
            (reference_output * upstream).sum(), reference_inputs
        )

        for actual, reference in zip(actual_gradients, reference_gradients):
            torch.testing.assert_close(actual, reference)

    def test_rank_three_mask_blocks_invalid_and_padded_positions(self) -> None:
        generator = torch.Generator().manual_seed(17)
        q = torch.randn(2, 5, 2, 2, generator=generator, dtype=torch.float64)
        k = torch.randn(2, 5, 2, 2, generator=generator, dtype=torch.float64)
        v = torch.randn(2, 5, 2, 3, generator=generator, dtype=torch.float64)
        bias = torch.randn(2, 5, 5, generator=generator, dtype=torch.float64)
        lengths = torch.tensor([5, 3])
        positions = torch.arange(5)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        mask = (
            torch.tril(torch.ones(5, 5, dtype=torch.bool)).unsqueeze(0)
            & valid.unsqueeze(2)
            & valid.unsqueeze(1)
        )

        clean = self._attention_output(q, k, v, mask, bias)
        poisoned_q = q.clone()
        poisoned_k = k.clone()
        poisoned_v = v.clone()
        poisoned_bias = bias.clone()
        poisoned_q[1, 3:] = 1e6
        poisoned_k[1, 3:] = -1e6
        poisoned_v[1, 3:] = 1e6
        poisoned_bias[1] = torch.where(
            mask[1], poisoned_bias[1], torch.full_like(poisoned_bias[1], 1e6)
        )
        poisoned = self._attention_output(
            poisoned_q, poisoned_k, poisoned_v, mask, poisoned_bias
        )

        torch.testing.assert_close(poisoned[0], clean[0])
        torch.testing.assert_close(poisoned[1, :3], clean[1, :3])
        torch.testing.assert_close(
            poisoned[1, 3:], torch.zeros_like(poisoned[1, 3:])
        )

    def test_default_score_kernel_remains_exact_silu(self) -> None:
        generator = torch.Generator().manual_seed(7)
        q = torch.randn(1, 4, 2, 3, generator=generator, dtype=torch.float64)
        k = torch.randn(1, 4, 2, 3, generator=generator, dtype=torch.float64)
        bias = torch.randn(1, 4, 4, generator=generator, dtype=torch.float64)
        mask = torch.tril(torch.ones(4, 4, dtype=torch.float64))
        scores = torch.einsum("bnhd,bmhd->bhnm", q, k) + bias.unsqueeze(1)
        expected = F.silu(scores) * mask.view(1, 1, 4, 4) / 4

        default = _per_head_hstu_weights(q, k, mask, bias)
        explicit = _per_head_hstu_weights(q, k, mask, bias, score_kernel="silu")
        torch.testing.assert_close(default, expected)
        torch.testing.assert_close(default, explicit)

    def test_fixed_positive_attention_scale_is_removed_by_output_norm(self) -> None:
        module = SequentialTransductionUnitJagged(
            embedding_dim=6,
            linear_hidden_dim=3,
            attention_dim=2,
            dropout_ratio=0.0,
            attn_dropout_ratio=0.0,
            num_heads=2,
            linear_activation="silu",
        )
        attention_output = torch.tensor(
            [
                [-3.0, -2.0, -0.5, 0.75, 2.0, 4.0],
                [2.5, -1.5, 3.0, -2.0, 0.25, 1.0],
            ],
            dtype=torch.float64,
        )

        unscaled = module._norm_attn_output(attention_output)
        half_scaled = module._norm_attn_output(0.5 * attention_output)
        epsilon_adjusted = F.layer_norm(
            attention_output,
            normalized_shape=[6],
            eps=1e-6 / (0.5**2),
        )

        # LN(c*x; eps) = LN(x; eps/c^2) for c > 0. Only epsilon remains.
        torch.testing.assert_close(half_scaled, epsilon_adjusted)
        self.assertLess((half_scaled - unscaled).abs().max().item(), 2e-6)

    def test_score_kernel_validation(self) -> None:
        scores = torch.tensor([-2.0, -0.25, 0.0, 0.5, 3.0], dtype=torch.float64)
        torch.testing.assert_close(
            _apply_hstu_score_kernel(scores, "tanh"), torch.tanh(scores)
        )
        with self.assertRaisesRegex(ValueError, "Unknown HSTU score kernel"):
            _apply_hstu_score_kernel(scores, "tanh_typo")


class HSTUTanhParameterInventoryTest(unittest.TestCase):
    @staticmethod
    def _build_ml20_model(normalization: str) -> HSTU:
        with contextlib.redirect_stdout(io.StringIO()), torch.device("meta"):
            return HSTU(
                max_sequence_len=200,
                max_output_len=11,
                embedding_dim=256,
                num_blocks=16,
                num_heads=8,
                linear_dim=32,
                attention_dim=32,
                normalization=normalization,
                linear_config="uvqk",
                linear_activation="silu",
                linear_dropout_rate=0.2,
                attn_dropout_rate=0.0,
                embedding_module=LocalEmbeddingModule(
                    num_items=131262, item_embedding_dim=256
                ),
                similarity_module=DotProductSimilarity(),
                input_features_preproc_module=(
                    LearnablePositionalEmbeddingInputFeaturesPreprocessor(
                        max_sequence_len=211,
                        embedding_dim=256,
                        dropout_rate=0.2,
                    )
                ),
                output_postproc_module=L2NormEmbeddingPostprocessor(
                    embedding_dim=256, eps=1e-6
                ),
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

    def test_ml20_named_inventory_is_exactly_parameter_matched(self) -> None:
        baseline = self._build_ml20_model("rel_bias")
        tanh = self._build_ml20_model("tanh_rel_bias")
        baseline_parameters = self._parameter_inventory(baseline)
        tanh_parameters = self._parameter_inventory(tanh)

        self.assertEqual(tanh_parameters, baseline_parameters)
        self.assertTrue(all(entry[3] for entry in tanh_parameters.values()))
        self.assertEqual(
            sum(entry[1] for entry in tanh_parameters.values()), 38_913_120
        )
        self.assertEqual(
            sum(
                entry[1]
                for name, entry in tanh_parameters.items()
                if name.startswith("_hstu.")
            ),
            5_255_776,
        )
        self.assertEqual(
            len(
                [
                    name
                    for name in tanh_parameters
                    if name.endswith("._rel_attn_bias._ts_w")
                ]
            ),
            16,
        )
        self.assertEqual(
            len(
                [
                    name
                    for name in tanh_parameters
                    if name.endswith("._rel_attn_bias._pos_w")
                ]
            ),
            16,
        )

        baseline_buffers = self._buffer_inventory(baseline)
        tanh_buffers = self._buffer_inventory(tanh)
        self.assertEqual(tanh_buffers, baseline_buffers)
        self.assertEqual(
            tanh_buffers,
            {"_attn_mask": ((211, 211), 44_521, torch.bool)},
        )
        self.assertTrue(set(tanh_parameters).isdisjoint(tanh_buffers))
        self.assertNotIn("tanh", baseline.debug_str())
        self.assertIn("-tanh-attn", tanh.debug_str())


if __name__ == "__main__":
    unittest.main()
