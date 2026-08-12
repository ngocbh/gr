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
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.hstu import (
    HSTU,
    RelativePositionalBias,
    SequentialTransductionUnitJagged,
    _forgetting_survival,
    _per_head_forgetting_tail_attention,
    _per_head_local_forgetting_attention,
    _per_head_recurrent_forgetting_tail_attention,
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


def _feature_map(x: torch.Tensor, feature_map: str, gamma: float) -> torch.Tensor:
    if feature_map == "identity":
        return x
    features = torch.tanh(gamma * x)
    if feature_map == "tanh":
        return features
    if feature_map == "abs_tanh":
        return features.abs()
    raise AssertionError(f"unexpected test feature map {feature_map}")


def _old_mask(
    seq_len: int,
    window_size: int,
    valid_lengths: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(seq_len, device=valid_lengths.device)
    distances = positions.unsqueeze(1) - positions.unsqueeze(0)
    valid = positions.unsqueeze(0) < valid_lengths.unsqueeze(1)
    return (
        (distances >= window_size).view(1, 1, seq_len, seq_len)
        & valid.view(valid_lengths.numel(), 1, seq_len, 1)
        & valid.view(valid_lengths.numel(), 1, 1, seq_len)
    )


def _literal_tail(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_forget: torch.Tensor,
    window_size: int,
    feature_map: str,
    gamma: float,
    valid_lengths: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, _ = q.shape
    q_features = _feature_map(q, feature_map, gamma)
    k_features = _feature_map(k, feature_map, gamma)
    batches = []
    for batch_idx in range(batch_size):
        queries = []
        for query_idx in range(seq_len):
            heads = []
            for head_idx in range(num_heads):
                terms = []
                if query_idx < valid_lengths[batch_idx]:
                    for key_idx in range(query_idx - window_size + 1):
                        survival = torch.exp(
                            log_forget[
                                batch_idx,
                                key_idx + 1 : query_idx + 1,
                                head_idx,
                            ].sum()
                        )
                        terms.append(
                            torch.dot(
                                q_features[batch_idx, query_idx, head_idx],
                                k_features[batch_idx, key_idx, head_idx],
                            )
                            * survival
                            * v[batch_idx, key_idx, head_idx]
                            * (0.5 / seq_len)
                        )
                if terms:
                    heads.append(torch.stack(terms).sum(dim=0))
                else:
                    heads.append(
                        v[batch_idx, query_idx, head_idx] * 0.0
                        + q[batch_idx, query_idx, head_idx].sum() * 0.0
                        + k[batch_idx, query_idx, head_idx].sum() * 0.0
                        + log_forget[batch_idx, query_idx, head_idx] * 0.0
                    )
            queries.append(torch.stack(heads))
        batches.append(torch.stack(queries))
    return torch.stack(batches)


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
        output = torch.full(
            (seq_offsets.numel() - 1, max_length, *values.shape[1:]),
            padding_value,
            dtype=values.dtype,
            device=values.device,
        )
        for batch_idx in range(seq_offsets.numel() - 1):
            start = int(seq_offsets[batch_idx])
            end = int(seq_offsets[batch_idx + 1])
            output[batch_idx, : end - start] = values[start:end]
        return output

    def dense_to_jagged(
        values: torch.Tensor, offsets: list[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        seq_offsets = offsets[0]
        pieces = []
        for batch_idx in range(seq_offsets.numel() - 1):
            length = int(seq_offsets[batch_idx + 1] - seq_offsets[batch_idx])
            pieces.append(values[batch_idx, :length])
        return (torch.cat(pieces),)

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


class SignedLIFTTailTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(250302130)
        self.batch_size = 2
        self.seq_len = 6
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
        self.log_forget = F.logsigmoid(
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                generator=generator,
                dtype=torch.float64,
            )
        )
        self.valid_lengths = torch.tensor([self.seq_len, self.seq_len - 2])

    def _dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_forget: torch.Tensor,
        window_size: int,
        feature_map: str,
        gamma: float = 0.7,
    ) -> torch.Tensor:
        return _per_head_forgetting_tail_attention(
            padded_q=q,
            padded_k=k,
            padded_v=v,
            survival=_forgetting_survival(log_forget),
            old_mask=_old_mask(self.seq_len, window_size, self.valid_lengths),
            feature_map=feature_map,
            gamma=gamma,
        )

    def test_dense_tanh_and_abs_tanh_match_literal_outputs_and_gradients(
        self,
    ) -> None:
        for feature_map in ("tanh", "abs_tanh"):
            with self.subTest(feature_map=feature_map):
                actual_inputs = tuple(
                    tensor.clone().requires_grad_()
                    for tensor in (self.q, self.k, self.v, self.log_forget)
                )
                expected_inputs = tuple(
                    tensor.clone().requires_grad_()
                    for tensor in (self.q, self.k, self.v, self.log_forget)
                )
                actual = self._dense(*actual_inputs, 3, feature_map)
                expected = _literal_tail(
                    *expected_inputs,
                    window_size=3,
                    feature_map=feature_map,
                    gamma=0.7,
                    valid_lengths=self.valid_lengths,
                )
                torch.testing.assert_close(actual, expected)
                upstream = torch.randn_like(actual)
                actual_grads = torch.autograd.grad(
                    (actual * upstream).sum(), actual_inputs
                )
                expected_grads = torch.autograd.grad(
                    (expected * upstream).sum(), expected_inputs
                )
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_recurrence_matches_dense_outputs_and_input_gate_gradients(self) -> None:
        for feature_map in ("identity", "tanh", "abs_tanh"):
            for window_size in (1, self.seq_len, self.seq_len + 1):
                with self.subTest(feature_map=feature_map, window_size=window_size):
                    recurrent_inputs = tuple(
                        tensor.clone().requires_grad_()
                        for tensor in (self.q, self.k, self.v, self.log_forget)
                    )
                    dense_inputs = tuple(
                        tensor.clone().requires_grad_()
                        for tensor in (self.q, self.k, self.v, self.log_forget)
                    )
                    recurrent = _per_head_recurrent_forgetting_tail_attention(
                        *recurrent_inputs,
                        window_size=window_size,
                        feature_map=feature_map,
                        gamma=0.7,
                        valid_lengths=self.valid_lengths,
                    )
                    dense = self._dense(
                        *dense_inputs, window_size, feature_map, gamma=0.7
                    )
                    torch.testing.assert_close(recurrent, dense)
                    upstream = torch.randn_like(recurrent)
                    recurrent_grads = torch.autograd.grad(
                        (recurrent * upstream).sum(), recurrent_inputs
                    )
                    dense_grads = torch.autograd.grad(
                        (dense * upstream).sum(), dense_inputs
                    )
                    for recurrent_grad, dense_grad in zip(recurrent_grads, dense_grads):
                        torch.testing.assert_close(recurrent_grad, dense_grad)

    def test_padding_extreme_gates_and_heads_are_isolated(self) -> None:
        clean = _per_head_recurrent_forgetting_tail_attention(
            self.q,
            self.k,
            self.v,
            self.log_forget,
            window_size=2,
            feature_map="tanh",
            gamma=0.7,
            valid_lengths=self.valid_lengths,
        )
        poisoned = [tensor.clone() for tensor in (self.q, self.k, self.v)]
        poisoned_log_forget = self.log_forget.clone()
        valid_length = int(self.valid_lengths[1])
        poisoned[0][1, valid_length:] = 1e6
        poisoned[1][1, valid_length:] = -1e6
        poisoned[2][1, valid_length:] = 1e6
        poisoned_log_forget[1, valid_length:] = -1e6
        padded = _per_head_recurrent_forgetting_tail_attention(
            *poisoned,
            poisoned_log_forget,
            window_size=2,
            feature_map="tanh",
            gamma=0.7,
            valid_lengths=self.valid_lengths,
        )
        torch.testing.assert_close(padded[1, :valid_length], clean[1, :valid_length])
        torch.testing.assert_close(
            padded[1, valid_length:], torch.zeros_like(padded[1, valid_length:])
        )

        changed_k = self.k.clone()
        changed_v = self.v.clone()
        changed_log_forget = self.log_forget.clone()
        changed_k[:, :, 1] = changed_k[:, :, 1] * 7.0 + 2.0
        changed_v[:, :, 1] = changed_v[:, :, 1] * -5.0
        changed_log_forget[:, :, 1] = -1000.0
        changed = _per_head_recurrent_forgetting_tail_attention(
            self.q,
            changed_k,
            changed_v,
            changed_log_forget,
            window_size=2,
            feature_map="tanh",
            gamma=0.7,
            valid_lengths=self.valid_lengths,
        )
        torch.testing.assert_close(changed[:, :, 0], clean[:, :, 0])
        self.assertFalse(torch.allclose(changed[:, :, 1], clean[:, :, 1]))

    def test_fixed_half_over_n_scaling(self) -> None:
        seq_len = 3
        q = torch.ones(1, seq_len, 1, 2, dtype=torch.float64)
        k = torch.ones_like(q)
        v = torch.ones(1, seq_len, 1, 1, dtype=torch.float64)
        log_forget = torch.zeros(1, seq_len, 1, dtype=torch.float64)
        output = _per_head_recurrent_forgetting_tail_attention(
            q,
            k,
            v,
            log_forget,
            window_size=1,
            feature_map="tanh",
            gamma=1.0,
        )
        coefficient = 2 * torch.tanh(torch.tensor(1.0, dtype=torch.float64)).square()
        self.assertEqual(output[0, 0, 0, 0].item(), 0.0)
        torch.testing.assert_close(output[0, 2, 0, 0], coefficient / seq_len)

    def test_identity_default_and_combined_local_tail_regressions(self) -> None:
        survival = _forgetting_survival(self.log_forget)
        mask = _old_mask(self.seq_len, 2, self.valid_lengths)
        default = _per_head_forgetting_tail_attention(
            self.q, self.k, self.v, survival, mask
        )
        identity = _per_head_forgetting_tail_attention(
            self.q,
            self.k,
            self.v,
            survival,
            mask,
            feature_map="identity",
            gamma=1.0,
        )
        torch.testing.assert_close(default, identity, rtol=0.0, atol=0.0)
        legacy_scores = torch.einsum("bnhd,bmhd->bhnm", self.q, self.k)
        legacy_weights = (
            legacy_scores
            * survival
            * mask.to(legacy_scores.dtype)
            * (0.5 / self.seq_len)
        )
        legacy = torch.einsum("bhnm,bmhe->bnhe", legacy_weights, self.v)
        torch.testing.assert_close(default, legacy, rtol=0.0, atol=0.0)

        common = dict(
            padded_q=self.q,
            padded_k=self.k,
            padded_v=self.v,
            invalid_attn_mask=torch.tril(torch.ones(self.seq_len, self.seq_len)),
            relative_attention_bias=torch.randn(
                self.batch_size,
                self.seq_len,
                self.seq_len,
                dtype=torch.float64,
            ),
            log_forget=self.log_forget,
            window_size=2,
            valid_lengths=self.valid_lengths,
        )
        local = _per_head_local_forgetting_attention(**common)
        rho = torch.tensor([0.4, -0.7], dtype=torch.float64, requires_grad=True)
        gain = 2.0 * torch.tanh(rho / 2.0)
        combined = _per_head_local_forgetting_attention(
            **common,
            tail_gain=gain,
            tail_feature_map="tanh",
            signed_feature_gamma=0.7,
        )
        tail = self._dense(
            self.q, self.k, self.v, self.log_forget, 2, "tanh", gamma=0.7
        )
        expected = local + tail * gain.view(1, 1, self.num_heads, 1)
        torch.testing.assert_close(combined, expected)
        combined.square().sum().backward()
        self.assertIsNotNone(rho.grad)
        assert rho.grad is not None
        self.assertGreater(rho.grad.abs().sum().item(), 0.0)

    def test_selector_validation_scope_and_debug_label(self) -> None:
        for normalization in (
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        ):
            with self.subTest(normalization=normalization), self.assertRaisesRegex(
                ValueError, "hybrid_tail_feature_map"
            ):
                SequentialTransductionUnitJagged(
                    embedding_dim=4,
                    linear_hidden_dim=2,
                    attention_dim=2,
                    dropout_ratio=0.0,
                    attn_dropout_ratio=0.0,
                    num_heads=2,
                    linear_activation="silu",
                    normalization=normalization,
                    hybrid_tail_feature_map="typo",
                )
        unrelated = SequentialTransductionUnitJagged(
            embedding_dim=4,
            linear_hidden_dim=2,
            attention_dim=2,
            dropout_ratio=0.0,
            attn_dropout_ratio=0.0,
            num_heads=2,
            linear_activation="silu",
            normalization="rel_bias",
            hybrid_tail_feature_map="unused",
        )
        self.assertEqual(unrelated._hybrid_tail_feature_map, "unused")

    def test_hybrid_forward_connects_every_parameter_gradient(self) -> None:
        module = SequentialTransductionUnitJagged(
            embedding_dim=4,
            linear_hidden_dim=2,
            attention_dim=2,
            dropout_ratio=0.0,
            attn_dropout_ratio=0.0,
            num_heads=2,
            linear_activation="silu",
            relative_attention_bias_module=RelativePositionalBias(max_seq_len=4),
            normalization="hybrid_forgetting_rel_bias",
            hybrid_window_size=2,
            hybrid_tail_feature_map="tanh",
            signed_feature_gamma=0.7,
        )
        with _cpu_jagged_ops():
            output, _ = module(
                x=torch.randn(7, 4),
                x_offsets=torch.tensor([0, 4, 7]),
                all_timestamps=torch.arange(8).reshape(2, 4),
                invalid_attn_mask=torch.tril(torch.ones(4, 4)),
            )
            (output * torch.randn_like(output)).sum().backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class SignedLIFTInventoryTest(unittest.TestCase):
    @staticmethod
    def _build_model(
        dataset: str,
        normalization: str,
        feature_map: str,
    ) -> HSTU:
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
                hybrid_window_size=32,
                hybrid_tail_feature_map=feature_map,
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

    def test_ml1_and_ml20_feature_arms_match_local_named_inventories(self) -> None:
        expected_totals = {"ml-1m": 313_432, "ml-20m": 38_917_472}
        for dataset in ("ml-1m", "ml-20m"):
            local = self._build_model(dataset, "local_forgetting_rel_bias", "identity")
            expected_parameters = self._parameter_inventory(local)
            expected_buffers = self._buffer_inventory(local)
            self.assertEqual(
                sum(entry[1] for entry in expected_parameters.values()),
                expected_totals[dataset],
            )
            for feature_map in ("identity", "tanh", "abs_tanh"):
                with self.subTest(dataset=dataset, feature_map=feature_map):
                    model = self._build_model(
                        dataset, "hybrid_forgetting_rel_bias", feature_map
                    )
                    self.assertEqual(
                        self._parameter_inventory(model), expected_parameters
                    )
                    self.assertEqual(self._buffer_inventory(model), expected_buffers)
                    self.assertFalse(
                        any(
                            "hybrid_tail_feature_map" in name
                            or "signed_feature_gamma" in name
                            for name in expected_parameters
                        )
                    )
                    self.assertIn(f"tail-{feature_map}-g0.7", model.debug_str())
                    for layer in model._hstu._attention_layers:
                        self.assertEqual(layer._hybrid_tail_feature_map, feature_map)


if __name__ == "__main__":
    unittest.main()
