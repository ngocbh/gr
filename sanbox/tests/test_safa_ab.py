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

    def test_dataset_specs_cover_amazon_and_movielens(self) -> None:
        expected = {
            "amzn-books": (
                695_762,
                44_865_440,
                1_152,
                44_866_592,
                "hstu-matched-sampled-softmax-n512-large-final.gin",
                "safa-sampled-softmax-n512-large-final.gin",
            ),
            "ml-1m": (
                3_952,
                313_000,
                416,
                313_416,
                "hstu-matched-sampled-softmax-n128-large-final.gin",
                "safa-sampled-softmax-n128-large-final.gin",
            ),
            "ml-20m": (
                131_262,
                38_913_120,
                4_224,
                38_917_344,
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
                        spec.hstu_config.name,
                        spec.safa_config.name,
                    ),
                    values,
                )

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


if __name__ == "__main__":
    unittest.main()
