# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

import importlib.util
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import gin
from generative_recommenders.research.modeling.sequential import encoder_utils


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts/audit_safa_ab.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("audit_safa_ab", AUDIT_PATH)
assert AUDIT_SPEC is not None
assert AUDIT_SPEC.loader is not None
audit_safa_ab = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = audit_safa_ab
AUDIT_SPEC.loader.exec_module(audit_safa_ab)


class _EmbeddingStub:
    item_embedding_dim = 32


class HSTUEncoderModeTest(unittest.TestCase):
    def setUp(self) -> None:
        gin.clear_config()

    def tearDown(self) -> None:
        gin.clear_config()

    def _required_arguments(self):
        return {
            "max_sequence_length": 8,
            "max_output_length": 2,
            "embedding_module": _EmbeddingStub(),
            "similarity_module": mock.sentinel.similarity,
            "input_preproc_module": mock.sentinel.input_preprocessor,
            "output_postproc_module": mock.sentinel.output_postprocessor,
            "activation_checkpoint": False,
            "verbose": False,
        }

    def test_forwards_safa_mode(self) -> None:
        with mock.patch.object(encoder_utils, "HSTU") as hstu_class:
            encoder_utils.hstu_encoder(
                **self._required_arguments(), attention_mode="safa"
            )

        self.assertEqual(hstu_class.call_args.kwargs["attention_mode"], "safa")

    def test_forwards_default_hstu_mode(self) -> None:
        with mock.patch.object(encoder_utils, "HSTU") as hstu_class:
            encoder_utils.hstu_encoder(**self._required_arguments())

        self.assertEqual(hstu_class.call_args.kwargs["attention_mode"], "hstu")

    def test_rejects_unknown_mode_before_construction(self) -> None:
        with mock.patch.object(encoder_utils, "HSTU") as hstu_class:
            with self.assertRaisesRegex(ValueError, "expected 'hstu' or 'safa'"):
                encoder_utils.hstu_encoder(
                    **self._required_arguments(), attention_mode="unknown"
                )

        hstu_class.assert_not_called()


class PairedConfigTest(unittest.TestCase):
    def tearDown(self) -> None:
        gin.clear_config()

    def test_only_attention_mode_differs(self) -> None:
        for spec in audit_safa_ab.PAIR_SPECS.values():
            with self.subTest(dataset=spec.dataset):
                audit_safa_ab.assert_config_pair(spec)

    def test_preserves_canonical_upstream_large_bindings(self) -> None:
        for spec in audit_safa_ab.PAIR_SPECS.values():
            with self.subTest(dataset=spec.dataset):
                audit_safa_ab.assert_upstream_fidelity(spec)

    def test_dataset_specs_cover_supported_datasets(self) -> None:
        expected = {
            "amzn-books": (
                695_762,
                44_865_440,
                1_152,
                44_866_592,
                "b70951a7c770dc52da5b9333ec30d86e2b84e7208a02e8824142685006c96de4",
                "hstu-matched-sampled-softmax-n512-large-final.gin",
                "safa-sampled-softmax-n512-large-final.gin",
            ),
            "kuairand-1k": (
                192_120,
                12_824_160,
                1_152,
                12_825_312,
                "e50dc8d5b5df6b383edce70d373ed1298f223c4c129107de106221640009c5de",
                "hstu-matched-sampled-softmax-n512-large-final.gin",
                "safa-sampled-softmax-n512-large-final.gin",
            ),
            "ml-1m": (
                3_952,
                313_000,
                416,
                313_416,
                "2ca8f1559267c3a1741b2343092f2d2c55bcf2aff00265fa0dca8d628e6cf6c8",
                "hstu-matched-sampled-softmax-n128-large-final.gin",
                "safa-sampled-softmax-n128-large-final.gin",
            ),
            "ml-20m": (
                131_262,
                38_913_120,
                4_224,
                38_917_344,
                "38636c03bbbbb842fd4a6fb81fa3f21e93ddf39d6509bb8d96bec42667c7f4d5",
                "hstu-matched-sampled-softmax-n128-large-final.gin",
                "safa-sampled-softmax-n128-large-final.gin",
            ),
        }
        self.assertEqual(set(audit_safa_ab.PAIR_SPECS), set(expected))
        for dataset, values in expected.items():
            with self.subTest(dataset=dataset):
                spec = audit_safa_ab.PAIR_SPECS[dataset]
                self.assertEqual(
                    (
                        spec.max_item_id,
                        spec.expected_backbone_parameters,
                        spec.expected_forget_parameters,
                        spec.expected_total_parameters,
                        spec.expected_inventory_sha256,
                        spec.hstu_config.name,
                        spec.safa_config.name,
                    ),
                    values,
                )

    def test_inventory_sha_drift_is_rejected_before_count_gate(self) -> None:
        spec = audit_safa_ab.PAIR_SPECS["ml-1m"]
        signature = {"weight": ((spec.expected_total_parameters,), True, 1)}
        with mock.patch.object(
            audit_safa_ab,
            "_inventory",
            side_effect=((signature, "0" * 64), (signature, "0" * 64)),
        ):
            with self.assertRaisesRegex(
                AssertionError, "parameter inventory SHA-256 mismatch"
            ):
                audit_safa_ab.audit_pair(spec)

    def test_shared_drift_from_upstream_is_rejected(self) -> None:
        spec = audit_safa_ab.PAIR_SPECS["ml-1m"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            drifted_config = Path(temporary_directory) / spec.hstu_config.name
            drifted_config.write_text(
                spec.hstu_config.read_text() + "\ntrain_fn.random_seed = 99\n"
            )
            drifted_spec = replace(spec, hstu_config=drifted_config)
            with self.assertRaisesRegex(
                AssertionError, "drift from canonical upstream LARGE"
            ):
                audit_safa_ab.assert_upstream_fidelity(drifted_spec)

    def test_kuairand_pair_uses_frozen_large_schedule(self) -> None:
        spec = audit_safa_ab.PAIR_SPECS["kuairand-1k"]
        for config_path in (spec.hstu_config, spec.safa_config):
            with self.subTest(config=config_path.name):
                gin.clear_config()
                gin.parse_config_file(str(config_path))
                self.assertEqual(
                    gin.query_parameter("train_fn.max_sequence_length"), 2048
                )
                self.assertEqual(gin.query_parameter("train_fn.local_batch_size"), 4)
                self.assertEqual(gin.query_parameter("train_fn.eval_batch_size"), 4)
                self.assertEqual(gin.query_parameter("train_fn.num_epochs"), 101)
                self.assertEqual(gin.query_parameter("train_fn.item_embedding_dim"), 64)
                self.assertEqual(gin.query_parameter("train_fn.num_negatives"), 512)
                self.assertEqual(gin.query_parameter("train_fn.full_eval_every_n"), 5)
                self.assertEqual(
                    gin.query_parameter("train_fn.partial_eval_num_iters"), 8
                )
                self.assertEqual(gin.query_parameter("hstu_encoder.num_blocks"), 16)
                self.assertEqual(gin.query_parameter("hstu_encoder.num_heads"), 8)
                self.assertEqual(gin.query_parameter("hstu_encoder.dqk"), 8)
                self.assertEqual(gin.query_parameter("hstu_encoder.dv"), 8)


if __name__ == "__main__":
    unittest.main()
