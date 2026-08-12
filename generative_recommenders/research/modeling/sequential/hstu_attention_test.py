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

# pyre-strict

import math
import unittest
from typing import Optional

import torch
from generative_recommenders.research.modeling.sequential.hstu import (
    RelativePositionalBias,
    SequentialTransductionUnitJagged,
    _apply_hstu_score_kernel,
    _forgetting_log_survival,
    _forgetting_survival,
    _per_head_additive_dot_attention,
    _per_head_forgetting_tail_attention,
    _per_head_hstu_weights,
    _per_head_local_forgetting_attention,
    _per_head_softmax_attention,
)


def _manual_per_head_softmax(
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
    batch_size, seq_len, _ = padded_q.shape
    q = padded_q.view(batch_size, seq_len, num_heads, attention_dim)
    k = padded_k.view(batch_size, seq_len, num_heads, attention_dim)
    v = padded_v.view(batch_size, seq_len, num_heads, linear_dim)
    output = torch.zeros(
        batch_size,
        seq_len,
        num_heads,
        linear_dim,
        dtype=padded_q.dtype,
    )
    lengths = (
        valid_lengths
        if valid_lengths is not None
        else torch.full((batch_size,), seq_len, dtype=torch.long)
    )

    def _base_logit(
        batch_idx: int, query_idx: int, head_idx: int, key_idx: int
    ) -> torch.Tensor:
        dot_product = torch.dot(
            q[batch_idx, query_idx, head_idx], k[batch_idx, key_idx, head_idx]
        )
        relative_bias = (
            relative_attention_bias[batch_idx, query_idx, key_idx]
            if relative_attention_bias is not None
            else 0.0
        )
        scale = temperature or math.sqrt(attention_dim)
        if scale_relative_attention_bias:
            return (dot_product + relative_bias) / scale
        return dot_product / scale + relative_bias

    for batch_idx in range(batch_size):
        for query_idx in range(seq_len):
            mask_row = (
                invalid_attn_mask[query_idx]
                if invalid_attn_mask.dim() == 2
                else invalid_attn_mask[batch_idx, query_idx]
            )
            valid_keys = torch.nonzero(
                mask_row
                * (torch.arange(seq_len) < lengths[batch_idx]).to(mask_row.dtype),
                as_tuple=False,
            ).flatten()
            if valid_keys.numel() == 0:
                continue
            for head_idx in range(num_heads):
                logits = torch.stack(
                    [
                        _base_logit(batch_idx, query_idx, head_idx, key_idx)
                        + (
                            log_forget[
                                batch_idx,
                                key_idx + 1 : query_idx + 1,
                                head_idx,
                            ].sum()
                            if log_forget is not None
                            else 0.0
                        )
                        for key_idx in valid_keys
                    ]
                )
                probabilities = torch.softmax(logits, dim=0)
                selected_values = torch.stack(
                    [v[batch_idx, key_idx, head_idx] for key_idx in valid_keys]
                )
                output[batch_idx, query_idx, head_idx] = torch.sum(
                    probabilities.unsqueeze(-1) * selected_values,
                    dim=0,
                )
    return output.reshape(batch_size, seq_len, num_heads * linear_dim)


def _explicit_pairwise_taylor1(
    padded_q: torch.Tensor,
    padded_k: torch.Tensor,
    padded_v: torch.Tensor,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
) -> torch.Tensor:
    batch_size, seq_len, _ = padded_q.shape
    q = padded_q.reshape(batch_size, seq_len, num_heads, attention_dim)
    k = padded_k.reshape(batch_size, seq_len, num_heads, attention_dim)
    v = padded_v.reshape(batch_size, seq_len, num_heads, linear_dim)
    scores = torch.einsum("bnhd,bmhd->bhnm", q, k)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=scores.device, dtype=scores.dtype)
    )
    weights = 0.5 / seq_len * scores * causal_mask.view(1, 1, seq_len, seq_len)
    return torch.einsum("bhnm,bmhe->bnhe", weights, v).reshape(
        batch_size, seq_len, num_heads * linear_dim
    )


class HSTUScoreKernelTest(unittest.TestCase):
    def test_taylor_kernels_match_exact_polynomials(self) -> None:
        scores = torch.tensor([-4.0, -1.0, -0.25, 0.0, 0.5, 2.0], dtype=torch.float64)

        torch.testing.assert_close(
            _apply_hstu_score_kernel(scores, "silu"), torch.nn.functional.silu(scores)
        )
        torch.testing.assert_close(
            _apply_hstu_score_kernel(scores, "taylor1"), 0.5 * scores
        )
        torch.testing.assert_close(
            _apply_hstu_score_kernel(scores, "taylor2"),
            0.5 * scores + 0.25 * scores.square(),
        )

    def test_unknown_kernel_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown HSTU score kernel"):
            _apply_hstu_score_kernel(torch.ones(1), "unknown")

    def test_pairwise_weights_include_bias_mask_and_fixed_length_scale(self) -> None:
        generator = torch.Generator().manual_seed(17)
        q = torch.randn(2, 4, 3, 2, generator=generator, dtype=torch.float64)
        k = torch.randn(2, 4, 3, 2, generator=generator, dtype=torch.float64)
        relative_bias = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64)
        causal_mask = torch.tril(torch.ones(4, 4, dtype=torch.float64))
        raw_scores = torch.einsum("bnhd,bmhd->bhnm", q, k)
        raw_scores = raw_scores + relative_bias.unsqueeze(1)

        for kernel in ("taylor1", "taylor2"):
            with self.subTest(kernel=kernel):
                expected = (
                    _apply_hstu_score_kernel(raw_scores, kernel)
                    / 4
                    * causal_mask.view(1, 1, 4, 4)
                )
                actual = _per_head_hstu_weights(
                    padded_q=q,
                    padded_k=k,
                    invalid_attn_mask=causal_mask,
                    relative_attention_bias=relative_bias,
                    score_kernel=kernel,
                )
                torch.testing.assert_close(actual, expected)


class AdditiveDotAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(20260812)
        self.num_heads = 3
        self.attention_dim = 2
        self.linear_dim = 2
        self.q = torch.randn(2, 5, 6, generator=generator, dtype=torch.float64)
        self.k = torch.randn(2, 5, 6, generator=generator, dtype=torch.float64)
        self.v = torch.randn(2, 5, 6, generator=generator, dtype=torch.float64)

    def _additive(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return _per_head_additive_dot_attention(
            padded_q=q,
            padded_k=k,
            padded_v=v,
            num_heads=self.num_heads,
            attention_dim=self.attention_dim,
            linear_dim=self.linear_dim,
        )

    def _pairwise(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return _explicit_pairwise_taylor1(
            padded_q=q,
            padded_k=k,
            padded_v=v,
            num_heads=self.num_heads,
            attention_dim=self.attention_dim,
            linear_dim=self.linear_dim,
        )

    def test_rejects_pairwise_relative_bias(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot represent pairwise"):
            SequentialTransductionUnitJagged(
                embedding_dim=4,
                linear_hidden_dim=2,
                attention_dim=2,
                dropout_ratio=0.0,
                attn_dropout_ratio=0.0,
                num_heads=2,
                linear_activation="silu",
                relative_attention_bias_module=RelativePositionalBias(max_seq_len=5),
                normalization="additive_dot",
            )

    def test_matches_explicit_causal_pairwise_taylor1(self) -> None:
        torch.testing.assert_close(
            self._additive(self.q, self.k, self.v),
            self._pairwise(self.q, self.k, self.v),
        )

    def test_gradients_match_explicit_pairwise_taylor1(self) -> None:
        q_scan = self.q.clone().requires_grad_()
        k_scan = self.k.clone().requires_grad_()
        v_scan = self.v.clone().requires_grad_()
        q_pair = self.q.clone().requires_grad_()
        k_pair = self.k.clone().requires_grad_()
        v_pair = self.v.clone().requires_grad_()
        upstream = torch.randn_like(self.q)

        scan_grads = torch.autograd.grad(
            (self._additive(q_scan, k_scan, v_scan) * upstream).sum(),
            (q_scan, k_scan, v_scan),
        )
        pair_grads = torch.autograd.grad(
            (self._pairwise(q_pair, k_pair, v_pair) * upstream).sum(),
            (q_pair, k_pair, v_pair),
        )

        for scan_grad, pair_grad in zip(scan_grads, pair_grads):
            torch.testing.assert_close(scan_grad, pair_grad)

    def test_heads_are_isolated(self) -> None:
        original = self._additive(self.q, self.k, self.v).reshape(
            2, 5, self.num_heads, self.linear_dim
        )
        changed_k = self.k.reshape(2, 5, self.num_heads, self.attention_dim).clone()
        changed_v = self.v.reshape(2, 5, self.num_heads, self.linear_dim).clone()
        changed_k[:, :, 1, :] = changed_k[:, :, 1, :] * 7.0 + 3.0
        changed_v[:, :, 1, :] = changed_v[:, :, 1, :] * -5.0 + 2.0
        changed = self._additive(
            self.q,
            changed_k.reshape_as(self.k),
            changed_v.reshape_as(self.v),
        ).reshape(2, 5, self.num_heads, self.linear_dim)

        torch.testing.assert_close(original[:, :, 0, :], changed[:, :, 0, :])
        self.assertFalse(torch.allclose(original[:, :, 1, :], changed[:, :, 1, :]))

    def test_variable_right_padding_has_no_state_updates(self) -> None:
        valid_length = 3
        padded_q = self.q.clone()
        padded_k = self.k.clone()
        padded_v = self.v.clone()
        padded_q[0, valid_length:] = 0.0
        padded_k[0, valid_length:] = 0.0
        padded_v[0, valid_length:] = 0.0

        actual = self._additive(padded_q, padded_k, padded_v)
        expected = self._pairwise(padded_q, padded_k, padded_v)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual[0, valid_length:], torch.zeros_like(actual[0, valid_length:])
        )

        probe_q = padded_q.clone().reshape(2, 5, self.num_heads, self.attention_dim)
        probe_q[0, valid_length:] = probe_q[0, valid_length - 1]
        probe_output = self._additive(
            probe_q.reshape_as(padded_q), padded_k, padded_v
        ).reshape(2, 5, self.num_heads, self.linear_dim)
        torch.testing.assert_close(
            probe_output[0, valid_length], probe_output[0, valid_length + 1]
        )


class LocalForgettingMomentAttentionTest(unittest.TestCase):
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
        self.log_forget = torch.nn.functional.logsigmoid(
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                generator=generator,
                dtype=torch.float64,
            )
            + 2.0
        )
        self.causal_mask = torch.tril(
            torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)
        )
        self.valid_lengths = torch.tensor([self.seq_len, self.seq_len - 2])

    def _old_mask(self, window_size: int) -> torch.Tensor:
        positions = torch.arange(self.seq_len)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)
        valid = positions.unsqueeze(0) < self.valid_lengths.unsqueeze(1)
        return (
            self.causal_mask.view(1, 1, self.seq_len, self.seq_len)
            & (distances >= window_size).view(1, 1, self.seq_len, self.seq_len)
            & valid.view(self.batch_size, 1, self.seq_len, 1)
            & valid.view(self.batch_size, 1, 1, self.seq_len)
        )

    def _explicit_tail(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_forget: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        batch_outputs = []
        for batch_idx in range(self.batch_size):
            query_outputs = []
            for query_idx in range(self.seq_len):
                head_outputs = []
                for head_idx in range(self.num_heads):
                    terms = []
                    if query_idx < self.valid_lengths[batch_idx]:
                        for key_idx in range(query_idx - window_size + 1):
                            if key_idx >= self.valid_lengths[batch_idx]:
                                continue
                            survival = torch.exp(
                                log_forget[
                                    batch_idx,
                                    key_idx + 1 : query_idx + 1,
                                    head_idx,
                                ].sum()
                            )
                            terms.append(
                                (0.5 / self.seq_len)
                                * torch.dot(
                                    q[batch_idx, query_idx, head_idx],
                                    k[batch_idx, key_idx, head_idx],
                                )
                                * survival
                                * v[batch_idx, key_idx, head_idx]
                            )
                    if terms:
                        head_outputs.append(torch.stack(terms).sum(dim=0))
                    else:
                        # Keep all inputs connected for gradient comparison.
                        head_outputs.append(
                            q[batch_idx, query_idx, head_idx].sum() * 0.0
                            + k[batch_idx, query_idx, head_idx].sum() * 0.0
                            + v[batch_idx, query_idx, head_idx] * 0.0
                            + log_forget[batch_idx, query_idx, head_idx] * 0.0
                        )
                query_outputs.append(torch.stack(head_outputs))
            batch_outputs.append(torch.stack(query_outputs))
        return torch.stack(batch_outputs)

    def test_tail_matches_explicit_old_pair_oracle_and_gradients(self) -> None:
        window_size = 3
        q_actual = self.q.clone().requires_grad_()
        k_actual = self.k.clone().requires_grad_()
        v_actual = self.v.clone().requires_grad_()
        log_forget_actual = self.log_forget.clone().requires_grad_()
        q_expected = self.q.clone().requires_grad_()
        k_expected = self.k.clone().requires_grad_()
        v_expected = self.v.clone().requires_grad_()
        log_forget_expected = self.log_forget.clone().requires_grad_()

        actual = _per_head_forgetting_tail_attention(
            padded_q=q_actual,
            padded_k=k_actual,
            padded_v=v_actual,
            survival=_forgetting_survival(log_forget_actual),
            old_mask=self._old_mask(window_size),
        )
        expected = self._explicit_tail(
            q_expected,
            k_expected,
            v_expected,
            log_forget_expected,
            window_size,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual[:, :window_size], torch.zeros_like(actual[:, :window_size])
        )

        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            (actual * upstream).sum(),
            (q_actual, k_actual, v_actual, log_forget_actual),
        )
        expected_grads = torch.autograd.grad(
            (expected * upstream).sum(),
            (q_expected, k_expected, v_expected, log_forget_expected),
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_zero_gain_is_exact_local_identity(self) -> None:
        common = dict(
            padded_q=self.q,
            padded_k=self.k,
            padded_v=self.v,
            invalid_attn_mask=self.causal_mask,
            relative_attention_bias=torch.randn(
                self.batch_size,
                self.seq_len,
                self.seq_len,
                dtype=torch.float64,
            ),
            log_forget=self.log_forget,
            window_size=3,
            valid_lengths=self.valid_lengths,
        )
        local = _per_head_local_forgetting_attention(**common)
        hybrid_at_initialization = _per_head_local_forgetting_attention(
            **common, tail_gain=torch.zeros(self.num_heads, dtype=torch.float64)
        )
        torch.testing.assert_close(hybrid_at_initialization, local, rtol=0.0, atol=0.0)

        rho = torch.zeros(self.num_heads, dtype=torch.float64, requires_grad=True)
        output = _per_head_local_forgetting_attention(
            **common, tail_gain=2.0 * torch.tanh(rho / 2.0)
        )
        (output * torch.randn_like(output)).sum().backward()
        self.assertIsNotNone(rho.grad)
        assert rho.grad is not None
        self.assertTrue(torch.isfinite(rho.grad).all())
        self.assertGreater(rho.grad.abs().sum().item(), 0.0)

    def test_full_window_matches_full_fohstu(self) -> None:
        relative_bias = torch.randn(
            self.batch_size,
            self.seq_len,
            self.seq_len,
            dtype=torch.float64,
        )
        actual = _per_head_local_forgetting_attention(
            padded_q=self.q,
            padded_k=self.k,
            padded_v=self.v,
            invalid_attn_mask=self.causal_mask,
            relative_attention_bias=relative_bias,
            log_forget=self.log_forget,
            window_size=self.seq_len,
            valid_lengths=self.valid_lengths,
            tail_gain=torch.tensor([1.5, -0.75], dtype=torch.float64),
        )
        full_weights = _per_head_hstu_weights(
            padded_q=self.q,
            padded_k=self.k,
            invalid_attn_mask=self.causal_mask,
            relative_attention_bias=relative_bias,
            score_kernel="silu",
        ) * _forgetting_survival(self.log_forget)
        valid = torch.arange(self.seq_len).unsqueeze(0) < self.valid_lengths.unsqueeze(
            1
        )
        full_weights = full_weights * (
            valid.view(self.batch_size, 1, self.seq_len, 1)
            & valid.view(self.batch_size, 1, 1, self.seq_len)
        )
        expected = torch.einsum("bhnm,bmhe->bnhe", full_weights, self.v)
        torch.testing.assert_close(actual, expected)

    def test_tail_excludes_relative_bias_and_padding_and_isolates_heads(self) -> None:
        window_size = 2
        rank_three_mask = self.causal_mask.unsqueeze(0).expand(self.batch_size, -1, -1)
        bias_a = torch.randn(
            self.batch_size, self.seq_len, self.seq_len, dtype=torch.float64
        )
        bias_b = torch.randn_like(bias_a)

        def _run(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            log_forget: torch.Tensor,
            bias: torch.Tensor,
            tail_gain: Optional[torch.Tensor],
        ) -> torch.Tensor:
            return _per_head_local_forgetting_attention(
                padded_q=q,
                padded_k=k,
                padded_v=v,
                invalid_attn_mask=rank_three_mask,
                relative_attention_bias=bias,
                log_forget=log_forget,
                window_size=window_size,
                valid_lengths=self.valid_lengths,
                tail_gain=tail_gain,
            )

        gain = torch.ones(self.num_heads, dtype=torch.float64)
        tail_a = _run(self.q, self.k, self.v, self.log_forget, bias_a, gain) - _run(
            self.q, self.k, self.v, self.log_forget, bias_a, None
        )
        tail_b = _run(self.q, self.k, self.v, self.log_forget, bias_b, gain) - _run(
            self.q, self.k, self.v, self.log_forget, bias_b, None
        )
        torch.testing.assert_close(tail_a, tail_b)

        poisoned_q = self.q.clone()
        poisoned_k = self.k.clone()
        poisoned_v = self.v.clone()
        poisoned_log_forget = self.log_forget.clone()
        valid_length = self.valid_lengths[1].item()
        poisoned_q[1, valid_length:] = 1e6
        poisoned_k[1, valid_length:] = -1e6
        poisoned_v[1, valid_length:] = 1e6
        poisoned_log_forget[1, valid_length:] = -1e6
        poisoned = _run(
            poisoned_q,
            poisoned_k,
            poisoned_v,
            poisoned_log_forget,
            bias_a,
            gain,
        )
        clean = _run(self.q, self.k, self.v, self.log_forget, bias_a, gain)
        torch.testing.assert_close(poisoned[1, :valid_length], clean[1, :valid_length])
        torch.testing.assert_close(
            poisoned[1, valid_length:], torch.zeros_like(poisoned[1, valid_length:])
        )

        changed_k = self.k.clone()
        changed_v = self.v.clone()
        changed_log_forget = self.log_forget.clone()
        changed_k[:, :, 1] = changed_k[:, :, 1] * 5.0 + 2.0
        changed_v[:, :, 1] = changed_v[:, :, 1] * -3.0
        changed_log_forget[:, :, 1] = -0.01
        changed = _run(self.q, changed_k, changed_v, changed_log_forget, bias_a, gain)
        torch.testing.assert_close(changed[:, :, 0], clean[:, :, 0])
        self.assertFalse(torch.allclose(changed[:, :, 1], clean[:, :, 1]))

    def test_modes_have_matched_zero_initialized_tail_gain(self) -> None:
        modules = []
        for normalization in (
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        ):
            modules.append(
                SequentialTransductionUnitJagged(
                    embedding_dim=4,
                    linear_hidden_dim=2,
                    attention_dim=2,
                    dropout_ratio=0.0,
                    attn_dropout_ratio=0.0,
                    num_heads=2,
                    linear_activation="silu",
                    normalization=normalization,
                    hybrid_window_size=3,
                )
            )
        for module in modules:
            torch.testing.assert_close(
                module._hybrid_tail_rho, torch.zeros(2, dtype=torch.float32)
            )
        self.assertEqual(
            sum(parameter.numel() for parameter in modules[0].parameters()),
            sum(parameter.numel() for parameter in modules[1].parameters()),
        )

    def test_delta_rejection_does_not_mutate_cache(self) -> None:
        for normalization in (
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
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
                    hybrid_window_size=2,
                )
                cached_v = torch.randn(2, 4)
                cached_v_before = cached_v.clone()
                cache = (
                    cached_v,
                    torch.randn(1, 3, 4),
                    torch.randn(1, 3, 4),
                    torch.randn(2, 4),
                )
                with self.assertRaisesRegex(NotImplementedError, "delta"):
                    module(
                        x=torch.randn(2, 4),
                        x_offsets=torch.tensor([0, 2]),
                        all_timestamps=None,
                        invalid_attn_mask=torch.tril(torch.ones(3, 3)),
                        delta_x_offsets=(torch.tensor([1]), torch.tensor([1])),
                        cache=cache,
                    )
                torch.testing.assert_close(cached_v, cached_v_before)


class PerHeadSoftmaxAttentionTest(unittest.TestCase):
    def test_rejects_nonfinite_forgetting_periods(self) -> None:
        for min_period, max_period in (
            (float("nan"), 256.0),
            (8.0, float("inf")),
        ):
            with self.subTest(
                min_period=min_period, max_period=max_period
            ), self.assertRaisesRegex(ValueError, "finite"):
                SequentialTransductionUnitJagged(
                    embedding_dim=4,
                    linear_hidden_dim=2,
                    attention_dim=2,
                    dropout_ratio=0.0,
                    attn_dropout_ratio=0.0,
                    num_heads=2,
                    linear_activation="silu",
                    normalization="forgetting_softmax_rel_bias",
                    forgetting_min_period=min_period,
                    forgetting_max_period=max_period,
                )

    def test_rejects_nonfinite_temperature(self) -> None:
        for temperature in (float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(
                ValueError, "finite and non-negative"
            ):
                SequentialTransductionUnitJagged(
                    embedding_dim=4,
                    linear_hidden_dim=2,
                    attention_dim=2,
                    dropout_ratio=0.0,
                    attn_dropout_ratio=0.0,
                    num_heads=2,
                    linear_activation="silu",
                    normalization="softmax_rel_bias",
                    softmax_temperature=temperature,
                )

    def test_matches_manual_oracle_and_heads_differ(self) -> None:
        padded_q = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]], dtype=torch.float64
        )
        padded_k = torch.tensor(
            [[[1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]]], dtype=torch.float64
        )
        # Both heads see the same values, so different outputs must come from
        # their independently computed attention distributions.
        padded_v = torch.tensor(
            [[[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]], dtype=torch.float64
        )
        invalid_attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.float64))
        relative_attention_bias = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.1, -0.2, 0.0], [0.2, 0.0, -0.1]]],
            dtype=torch.float64,
        )

        actual = _per_head_softmax_attention(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=invalid_attn_mask,
            relative_attention_bias=relative_attention_bias,
            num_heads=2,
            attention_dim=1,
            linear_dim=1,
        )
        expected = _manual_per_head_softmax(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=invalid_attn_mask,
            relative_attention_bias=relative_attention_bias,
            num_heads=2,
            attention_dim=1,
            linear_dim=1,
        )

        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual[:, -1, 0], actual[:, -1, 1]))

    def test_configurable_temperature_matches_manual_oracle(self) -> None:
        padded_q = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float64)
        padded_k = torch.tensor([[[2.0, 1.0], [1.0, -1.0]]], dtype=torch.float64)
        padded_v = torch.tensor([[[1.0, 10.0], [3.0, 20.0]]], dtype=torch.float64)
        invalid_attn_mask = torch.tril(torch.ones(2, 2, dtype=torch.float64))
        relative_attention_bias = torch.zeros(1, 2, 2, dtype=torch.float64)

        actual = _per_head_softmax_attention(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=invalid_attn_mask,
            relative_attention_bias=relative_attention_bias,
            num_heads=1,
            attention_dim=2,
            linear_dim=2,
            temperature=1.0,
        )
        expected = _manual_per_head_softmax(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=invalid_attn_mask,
            relative_attention_bias=relative_attention_bias,
            num_heads=1,
            attention_dim=2,
            linear_dim=2,
            temperature=1.0,
        )

        torch.testing.assert_close(actual, expected)

    def test_canonical_bias_scaling_matches_oracle_and_preserves_default(self) -> None:
        padded_q = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        padded_k = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        padded_v = torch.tensor([[[1.0], [5.0]]], dtype=torch.float64)
        relative_attention_bias = torch.tensor(
            [[[0.0, 0.0], [1.5, -0.5]]], dtype=torch.float64
        )
        causal_mask = torch.tril(torch.ones(2, 2, dtype=torch.float64))
        common = dict(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=causal_mask,
            relative_attention_bias=relative_attention_bias,
            num_heads=1,
            attention_dim=4,
            linear_dim=1,
        )

        legacy_default = _per_head_softmax_attention(**common)
        legacy_explicit = _per_head_softmax_attention(
            **common, scale_relative_attention_bias=True
        )
        canonical = _per_head_softmax_attention(
            **common, scale_relative_attention_bias=False
        )
        expected_canonical = _manual_per_head_softmax(
            **common, scale_relative_attention_bias=False
        )

        torch.testing.assert_close(legacy_default, legacy_explicit)
        torch.testing.assert_close(canonical, expected_canonical)
        self.assertFalse(torch.allclose(canonical, legacy_default))

    def test_causal_mask_blocks_future_values(self) -> None:
        padded_q = torch.zeros(1, 3, 1, dtype=torch.float64)
        padded_k = torch.zeros_like(padded_q)
        padded_v = torch.tensor([[[1.0], [3.0], [1000.0]]], dtype=torch.float64)
        invalid_attn_mask = torch.tril(torch.ones(3, 3, dtype=torch.float64))

        actual = _per_head_softmax_attention(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=invalid_attn_mask,
            relative_attention_bias=torch.zeros(1, 3, 3, dtype=torch.float64),
            num_heads=1,
            attention_dim=1,
            linear_dim=1,
        )

        torch.testing.assert_close(
            actual[0, 0], torch.tensor([1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            actual[0, 1], torch.tensor([2.0], dtype=torch.float64)
        )

    def test_fosoftmax_matches_explicit_survival_normalized_oracle(self) -> None:
        generator = torch.Generator().manual_seed(29)
        padded_q = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64)
        padded_k = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64)
        padded_v = torch.randn(2, 4, 6, generator=generator, dtype=torch.float64)
        relative_attention_bias = torch.randn(
            2, 4, 4, generator=generator, dtype=torch.float64
        )
        log_forget = torch.nn.functional.logsigmoid(
            torch.randn(2, 4, 2, generator=generator, dtype=torch.float64)
        )
        causal_mask = torch.tril(torch.ones(4, 4, dtype=torch.float64))
        valid_lengths = torch.tensor([4, 2])

        actual = _per_head_softmax_attention(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=causal_mask,
            relative_attention_bias=relative_attention_bias,
            num_heads=2,
            attention_dim=2,
            linear_dim=3,
            temperature=1.7,
            log_forget=log_forget,
            valid_lengths=valid_lengths,
        )
        expected = _manual_per_head_softmax(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=causal_mask,
            relative_attention_bias=relative_attention_bias,
            num_heads=2,
            attention_dim=2,
            linear_dim=3,
            temperature=1.7,
            log_forget=log_forget,
            valid_lengths=valid_lengths,
        )

        torch.testing.assert_close(actual, expected)

    def test_unit_forgetting_reduces_to_softmax(self) -> None:
        generator = torch.Generator().manual_seed(31)
        padded_q = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64)
        padded_k = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64)
        padded_v = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64)
        causal_mask = torch.tril(torch.ones(4, 4, dtype=torch.float64))
        valid_lengths = torch.tensor([4, 3])
        common = dict(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=causal_mask,
            relative_attention_bias=None,
            num_heads=2,
            attention_dim=2,
            linear_dim=2,
            temperature=1.0,
            valid_lengths=valid_lengths,
        )

        expected = _per_head_softmax_attention(**common)
        actual = _per_head_softmax_attention(
            **common,
            log_forget=torch.zeros(2, 4, 2, dtype=torch.float64),
        )

        torch.testing.assert_close(actual, expected)

    def test_padding_keys_do_not_change_real_outputs(self) -> None:
        padded_q = torch.tensor([[[1.0], [2.0], [30.0], [40.0]]], dtype=torch.float64)
        padded_k = torch.tensor([[[0.5], [1.0], [100.0], [200.0]]], dtype=torch.float64)
        padded_v = torch.tensor([[[1.0], [3.0], [1e8], [-1e8]]], dtype=torch.float64)
        log_forget = torch.tensor(
            [[[-0.1], [-0.2], [-20.0], [-30.0]]], dtype=torch.float64
        )
        causal_mask = torch.tril(torch.ones(4, 4, dtype=torch.float64))

        actual = _per_head_softmax_attention(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=causal_mask,
            relative_attention_bias=None,
            num_heads=1,
            attention_dim=1,
            linear_dim=1,
            temperature=1.0,
            log_forget=log_forget,
            valid_lengths=torch.tensor([2]),
        )
        expected = _per_head_softmax_attention(
            padded_q=padded_q[:, :2],
            padded_k=padded_k[:, :2],
            padded_v=padded_v[:, :2],
            invalid_attn_mask=causal_mask[:2, :2],
            relative_attention_bias=None,
            num_heads=1,
            attention_dim=1,
            linear_dim=1,
            temperature=1.0,
            log_forget=log_forget[:, :2],
            valid_lengths=torch.tensor([2]),
        )

        torch.testing.assert_close(actual[:, :2], expected)

    def test_empty_mask_row_is_zero_and_finite(self) -> None:
        causal_mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.float64))
        causal_mask[:, 1, :] = 0.0
        actual = _per_head_softmax_attention(
            padded_q=torch.ones(1, 3, 1, dtype=torch.float64),
            padded_k=torch.ones(1, 3, 1, dtype=torch.float64),
            padded_v=torch.arange(1, 4, dtype=torch.float64).view(1, 3, 1),
            invalid_attn_mask=causal_mask,
            relative_attention_bias=None,
            num_heads=1,
            attention_dim=1,
            linear_dim=1,
            log_forget=torch.full((1, 3, 1), -0.2, dtype=torch.float64),
            valid_lengths=torch.tensor([3]),
        )

        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual[0, 1], torch.zeros(1, dtype=torch.float64))

    def test_learned_forgetting_gate_has_finite_nonzero_gradients(self) -> None:
        generator = torch.Generator().manual_seed(37)
        padded_q = torch.randn(1, 4, 4, generator=generator, dtype=torch.float64)
        padded_k = torch.randn(1, 4, 4, generator=generator, dtype=torch.float64)
        padded_v = torch.randn(1, 4, 4, generator=generator, dtype=torch.float64)
        forget_weight = torch.randn(
            2, 2, generator=generator, dtype=torch.float64, requires_grad=True
        )
        forget_bias = torch.tensor([0.3, 1.1], dtype=torch.float64, requires_grad=True)
        log_forget = torch.nn.functional.logsigmoid(
            torch.einsum("bnhd,hd->bnh", padded_k.view(1, 4, 2, 2), forget_weight)
            + forget_bias.view(1, 1, 2)
        )
        output = _per_head_softmax_attention(
            padded_q=padded_q,
            padded_k=padded_k,
            padded_v=padded_v,
            invalid_attn_mask=torch.tril(torch.ones(4, 4, dtype=torch.float64)),
            relative_attention_bias=None,
            num_heads=2,
            attention_dim=2,
            linear_dim=2,
            temperature=1.0,
            log_forget=log_forget,
            valid_lengths=torch.tensor([4]),
        )
        upstream = torch.randn(output.shape, generator=generator, dtype=torch.float64)
        (output * upstream).sum().backward()

        for gradient in (forget_weight.grad, forget_bias.grad):
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)


class ForgettingSurvivalTest(unittest.TestCase):
    def test_fixed_horizon_is_linear_distance_bias(self) -> None:
        sequence_length = 5
        horizons = torch.tensor([2.0, 8.0], dtype=torch.float64)
        log_forget = -horizons.reciprocal().view(1, 1, 2).expand(1, sequence_length, 2)
        actual = _forgetting_log_survival(log_forget)
        positions = torch.arange(sequence_length, dtype=torch.float64)
        distances = positions.view(-1, 1) - positions.view(1, -1)
        expected = -distances.clamp_min(0.0).view(
            1, 1, sequence_length, sequence_length
        ) / horizons.view(1, 2, 1, 1)

        torch.testing.assert_close(actual, expected)

    def test_matches_explicit_products(self) -> None:
        forget = torch.tensor(
            [
                [
                    [0.95, 0.90],
                    [0.80, 0.70],
                    [0.60, 0.50],
                    [0.40, 0.30],
                ]
            ],
            dtype=torch.float64,
        )
        actual = _forgetting_survival(forget.log())

        expected = torch.ones(1, 2, 4, 4, dtype=torch.float64)
        for head_idx in range(2):
            for query_idx in range(4):
                for key_idx in range(query_idx + 1):
                    expected[0, head_idx, query_idx, key_idx] = torch.prod(
                        forget[0, key_idx + 1 : query_idx + 1, head_idx]
                    )

        self.assertEqual(actual.shape, (1, 2, 4, 4))
        torch.testing.assert_close(actual, expected)

    def test_zero_log_forget_is_all_ones(self) -> None:
        actual = _forgetting_survival(torch.zeros(2, 5, 3, dtype=torch.float64))

        torch.testing.assert_close(actual, torch.ones(2, 3, 5, 5, dtype=torch.float64))

    def test_boundary_gate_suppresses_only_earlier_keys(self) -> None:
        epsilon = 1e-9
        log_forget = torch.zeros(1, 5, 1, dtype=torch.float64)
        log_forget[0, 2, 0] = math.log(epsilon)

        survival = _forgetting_survival(log_forget)[0, 0]

        torch.testing.assert_close(
            survival[2:, :2], torch.full((3, 2), epsilon, dtype=torch.float64)
        )
        for query_idx in range(2):
            torch.testing.assert_close(
                survival[query_idx, : query_idx + 1],
                torch.ones(query_idx + 1, dtype=torch.float64),
            )
        for query_idx in range(2, 5):
            torch.testing.assert_close(
                survival[query_idx, 2 : query_idx + 1],
                torch.ones(query_idx - 1, dtype=torch.float64),
            )


if __name__ == "__main__":
    unittest.main()
