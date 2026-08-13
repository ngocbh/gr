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
import hashlib
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Iterator, List, Tuple
from unittest import mock

import torch
from generative_recommenders.research.modeling.sequential.hstu import (
    RelativeAttentionBiasModule,
    RelativeBucketedTimeAndPositionBasedBias,
    SequentialTransductionUnitJagged,
    _forgetting_survival,
    _signed_additive_attention_weights,
)


_PRISTINE_REFERENCE_PATH = (
    Path(__file__).with_name("testdata") / "pristine_hstu_f209228.py"
)
_PRISTINE_REFERENCE_BLOB_SHA = "78329791d6c87d7826bef07fc89f7021ad197a37"


def _git_blob_sha(path: Path) -> str:
    contents = path.read_bytes()
    header = f"blob {len(contents)}\0".encode()
    return hashlib.sha1(header + contents).hexdigest()


def _load_pristine_reference() -> ModuleType:
    actual_sha = _git_blob_sha(_PRISTINE_REFERENCE_PATH)
    if actual_sha != _PRISTINE_REFERENCE_BLOB_SHA:
        raise AssertionError(
            "Frozen pristine HSTU reference has changed: "
            f"expected {_PRISTINE_REFERENCE_BLOB_SHA}, found {actual_sha}"
        )
    module_name = "_safa_pristine_hstu_f209228_78329791"
    spec = importlib.util.spec_from_file_location(module_name, _PRISTINE_REFERENCE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_PRISTINE_HSTU = _load_pristine_reference()


class _ZeroRelativeAttentionBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self._max_seq_len = max_seq_len

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            all_timestamps.size(0),
            self._max_seq_len,
            self._max_seq_len,
            device=all_timestamps.device,
            dtype=torch.float32,
        )


def _jagged_to_padded_dense(*args, **kwargs) -> torch.Tensor:
    values = kwargs["values"] if "values" in kwargs else args[0]
    offsets_list = kwargs["offsets"] if "offsets" in kwargs else args[1]
    max_lengths = kwargs["max_lengths"] if "max_lengths" in kwargs else args[2]
    padding_value = kwargs.get("padding_value", 0.0)
    offsets = offsets_list[0]
    max_length = max_lengths[0]
    rows: List[torch.Tensor] = []
    for batch_index in range(offsets.numel() - 1):
        start = int(offsets[batch_index].item())
        end = int(offsets[batch_index + 1].item())
        row = values[start:end]
        padding = values.new_full(
            (max_length - (end - start), *values.shape[1:]), padding_value
        )
        rows.append(torch.cat((row, padding), dim=0))
    return torch.stack(rows)


def _dense_to_jagged(
    dense: torch.Tensor, offsets_list: List[torch.Tensor]
) -> Tuple[torch.Tensor]:
    offsets = offsets_list[0]
    rows: List[torch.Tensor] = []
    for batch_index in range(offsets.numel() - 1):
        length = int((offsets[batch_index + 1] - offsets[batch_index]).item())
        rows.append(dense[batch_index, :length])
    return (torch.cat(rows, dim=0),)


@contextlib.contextmanager
def _reference_jagged_ops() -> Iterator[None]:
    namespace = torch.ops.fbgemm
    with mock.patch.object(
        namespace,
        "jagged_to_padded_dense",
        _jagged_to_padded_dense,
        create=True,
    ), mock.patch.object(
        namespace,
        "dense_to_jagged",
        _dense_to_jagged,
        create=True,
    ):
        yield


def _make_layer(
    attention_mode: str, num_heads: int = 2
) -> SequentialTransductionUnitJagged:
    return SequentialTransductionUnitJagged(
        embedding_dim=8,
        linear_hidden_dim=3,
        attention_dim=2,
        dropout_ratio=0.0,
        attn_dropout_ratio=0.0,
        num_heads=num_heads,
        linear_activation="silu",
        relative_attention_bias_module=_ZeroRelativeAttentionBias(max_seq_len=4),
        normalization="rel_bias",
        linear_config="uvqk",
        attention_mode=attention_mode,
    )


def _bucketize_time(time_delta: torch.Tensor) -> torch.Tensor:
    return (torch.log(torch.abs(time_delta).clamp(min=1)) / 0.301).long()


def _make_frozen_reference_pair() -> Tuple[torch.nn.Module, torch.nn.Module]:
    common_kwargs = {
        "embedding_dim": 8,
        "linear_hidden_dim": 3,
        "attention_dim": 2,
        "dropout_ratio": 0.2,
        "attn_dropout_ratio": 0.0,
        "num_heads": 2,
        "linear_activation": "silu",
        "normalization": "rel_bias",
        "linear_config": "uvqk",
    }

    torch.manual_seed(11)
    actual = SequentialTransductionUnitJagged(
        **common_kwargs,
        relative_attention_bias_module=RelativeBucketedTimeAndPositionBasedBias(
            max_seq_len=4,
            num_buckets=8,
            bucketization_fn=_bucketize_time,
        ),
        attention_mode="hstu",
    )
    torch.manual_seed(11)
    reference = _PRISTINE_HSTU.SequentialTransductionUnitJagged(
        **common_kwargs,
        relative_attention_bias_module=(
            _PRISTINE_HSTU.RelativeBucketedTimeAndPositionBasedBias(
                max_seq_len=4,
                num_buckets=8,
                bucketization_fn=_bucketize_time,
            )
        ),
    )
    return actual, reference


class SAFATest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self._offsets = torch.tensor([0, 4, 7], dtype=torch.long)
        self._timestamps = torch.arange(8, dtype=torch.long).view(2, 4)
        self._mask = torch.tril(torch.ones(4, 4))

    def test_pristine_reference_has_expected_upstream_blob_sha(self) -> None:
        self.assertEqual(
            _git_blob_sha(_PRISTINE_REFERENCE_PATH),
            _PRISTINE_REFERENCE_BLOB_SHA,
        )

    def test_hstu_matches_frozen_upstream_through_adamw_step(self) -> None:
        actual, reference = _make_frozen_reference_pair()
        actual_parameters = dict(actual.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        gate_names = {"_forget_weight", "_forget_bias"}
        self.assertEqual(set(actual_parameters) - set(reference_parameters), gate_names)
        self.assertEqual(set(reference_parameters), set(actual_parameters) - gate_names)
        for name, parameter in reference_parameters.items():
            torch.testing.assert_close(
                actual_parameters[name], parameter, rtol=0, atol=0
            )

        nonzero_relative_bias = actual._rel_attn_bias(self._timestamps)
        self.assertGreater(torch.count_nonzero(nonzero_relative_bias).item(), 0)
        self.assertEqual(actual._dropout_ratio, 0.2)
        self.assertEqual(reference._dropout_ratio, 0.2)

        torch.manual_seed(37)
        x_actual = torch.randn(7, 8, requires_grad=True)
        x_reference = x_actual.detach().clone().requires_grad_(True)

        with _reference_jagged_ops():
            torch.manual_seed(101)
            actual_output, _ = actual(
                x=x_actual,
                x_offsets=self._offsets,
                all_timestamps=self._timestamps,
                invalid_attn_mask=self._mask,
            )
            torch.manual_seed(101)
            reference_output, _ = reference(
                x=x_reference,
                x_offsets=self._offsets,
                all_timestamps=self._timestamps,
                invalid_attn_mask=self._mask,
            )

        torch.testing.assert_close(actual_output, reference_output, rtol=0, atol=0)
        coefficient = torch.linspace(0.5, 1.5, actual_output.numel()).view_as(
            actual_output
        )
        actual_loss = (actual_output.square() * coefficient).sum()
        reference_loss = (reference_output.square() * coefficient).sum()
        torch.testing.assert_close(actual_loss, reference_loss, rtol=0, atol=0)
        actual_loss.backward()
        reference_loss.backward()

        self.assertIsNotNone(actual._forget_weight.grad)
        self.assertIsNotNone(actual._forget_bias.grad)
        self.assertEqual(torch.count_nonzero(actual._forget_weight.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(actual._forget_bias.grad).item(), 0)
        torch.testing.assert_close(x_actual.grad, x_reference.grad, rtol=0, atol=0)
        for name, reference_parameter in reference_parameters.items():
            self.assertIsNotNone(actual_parameters[name].grad)
            self.assertIsNotNone(reference_parameter.grad)
            torch.testing.assert_close(
                actual_parameters[name].grad,
                reference_parameter.grad,
                rtol=0,
                atol=0,
            )

        forget_weight_before = actual._forget_weight.detach().clone()
        forget_bias_before = actual._forget_bias.detach().clone()
        actual_optimizer = torch.optim.AdamW(
            actual.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.0
        )
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.0
        )
        actual_optimizer.step()
        reference_optimizer.step()
        for name, reference_parameter in reference_parameters.items():
            torch.testing.assert_close(
                actual_parameters[name],
                reference_parameter,
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            actual._forget_weight, forget_weight_before, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual._forget_bias, forget_bias_before, rtol=0, atol=0
        )

    def test_modes_have_identical_named_parameter_inventory(self) -> None:
        torch.manual_seed(17)
        hstu = _make_layer("hstu", num_heads=4)
        torch.manual_seed(17)
        safa = _make_layer("safa", num_heads=4)
        hstu_inventory = {
            name: (tuple(parameter.shape), parameter.numel())
            for name, parameter in hstu.named_parameters()
        }
        safa_inventory = {
            name: (tuple(parameter.shape), parameter.numel())
            for name, parameter in safa.named_parameters()
        }
        self.assertEqual(hstu_inventory, safa_inventory)
        self.assertEqual(hstu_inventory["_forget_weight"][0], (4, 2))
        self.assertEqual(hstu_inventory["_forget_bias"][0], (4,))
        for name, parameter in hstu.named_parameters():
            torch.testing.assert_close(
                parameter, dict(safa.named_parameters())[name], rtol=0, atol=0
            )

    def test_initial_forgetting_periods_are_log_spaced(self) -> None:
        layer = _make_layer("safa", num_heads=4)
        expected = torch.logspace(
            math.log10(8.0), math.log10(256.0), steps=4, dtype=torch.float64
        )
        actual = -torch.log(torch.sigmoid(layer._forget_bias.double())).reciprocal()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(torch.count_nonzero(layer._forget_weight).item(), 0)

        single_head = _make_layer("safa", num_heads=1)
        actual_single = -torch.log(
            torch.sigmoid(single_head._forget_bias.double())
        ).reciprocal()
        torch.testing.assert_close(
            actual_single,
            torch.tensor([8.0], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_survival_uses_exclusive_key_inclusive_query_path(self) -> None:
        gates = torch.tensor([[[0.8], [0.5], [0.25]]], dtype=torch.float64)
        survival = _forgetting_survival(torch.log(gates))
        diagonal = survival[0, 0].diagonal()
        torch.testing.assert_close(diagonal, torch.ones_like(diagonal))
        self.assertAlmostEqual(survival[0, 0, 1, 0].item(), 0.5)
        self.assertAlmostEqual(survival[0, 0, 2, 1].item(), 0.25)
        self.assertAlmostEqual(survival[0, 0, 2, 0].item(), 0.125)
        self.assertAlmostEqual(survival[0, 0, 0, 2].item(), 1.0)
        self.assertTrue(torch.all(survival > 0).item())
        self.assertTrue(torch.all(survival <= 1).item())

    def test_safa_preserves_negative_pair_scores(self) -> None:
        padded_q = torch.ones(1, 2, 1, 1)
        padded_k = -torch.ones(1, 2, 1, 1)
        mask = torch.tril(torch.ones(2, 2))
        forget_weight = torch.zeros(1, 1)
        forget_bias = torch.zeros(1)
        hstu = _signed_additive_attention_weights(
            padded_q,
            padded_k,
            mask,
            None,
            "hstu",
            forget_weight,
            forget_bias,
        )
        safa = _signed_additive_attention_weights(
            padded_q,
            padded_k,
            mask,
            None,
            "safa",
            forget_weight,
            forget_bias,
        )
        self.assertLess(hstu[0, 0, 1, 0].item(), 0.0)
        self.assertLess(safa[0, 0, 1, 0].item(), 0.0)
        torch.testing.assert_close(
            safa[0, 0, 1, 0], 0.5 * hstu[0, 0, 1, 0], rtol=1e-6, atol=1e-7
        )

    def test_safa_gate_parameters_receive_nonzero_gradients(self) -> None:
        padded_q = torch.tensor([[[[0.4, -0.2]], [[0.1, 0.7]], [[-0.6, 0.3]]]])
        padded_k = torch.tensor([[[[0.3, 0.5]], [[-0.4, 0.2]], [[0.8, -0.1]]]])
        forget_weight = torch.zeros(1, 2, requires_grad=True)
        forget_bias = torch.zeros(1, requires_grad=True)
        weights = _signed_additive_attention_weights(
            padded_q,
            padded_k,
            torch.tril(torch.ones(3, 3)),
            None,
            "safa",
            forget_weight,
            forget_bias,
        )
        coefficient = torch.arange(1, 10, dtype=weights.dtype).view(1, 1, 3, 3)
        (weights * coefficient).sum().backward()
        self.assertGreater(forget_weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(forget_bias.grad.abs().sum().item(), 0.0)

    def test_invalid_attention_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown attention_mode"):
            _make_layer("unknown")

    def test_delta_cache_matches_full_recomputation(self) -> None:
        for attention_mode in ("hstu", "safa"):
            with self.subTest(attention_mode=attention_mode):
                torch.manual_seed(29)
                layer = _make_layer(attention_mode)
                layer.eval()
                original = torch.randn(7, 8)
                updated = original.clone()
                delta_indices = torch.tensor([3, 6], dtype=torch.long)
                delta_positions = torch.tensor([3, 2], dtype=torch.long)
                updated[delta_indices] += torch.tensor(
                    [[0.3] * 8, [-0.2] * 8], dtype=updated.dtype
                )

                with torch.no_grad(), _reference_jagged_ops():
                    _, cache = layer(
                        x=original,
                        x_offsets=self._offsets,
                        all_timestamps=self._timestamps,
                        invalid_attn_mask=self._mask,
                        return_cache_states=True,
                    )
                    full, _ = layer(
                        x=updated,
                        x_offsets=self._offsets,
                        all_timestamps=self._timestamps,
                        invalid_attn_mask=self._mask,
                    )
                    delta, _ = layer(
                        x=updated,
                        x_offsets=self._offsets,
                        all_timestamps=self._timestamps,
                        invalid_attn_mask=self._mask,
                        delta_x_offsets=(delta_indices, delta_positions),
                        cache=cache,
                    )
                torch.testing.assert_close(delta, full, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
