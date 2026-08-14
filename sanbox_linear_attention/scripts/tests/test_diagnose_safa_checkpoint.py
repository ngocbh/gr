#!/usr/bin/env python3

import hashlib
import tempfile
import types
import unittest
from pathlib import Path

import torch

from scripts.diagnose_safa_checkpoint import (
    AttentionDiagnostics,
    DiagnosticError,
    GAP_BINS,
    MetricStrata,
    _build_model_and_dataset,
    _gap_labels,
    data_fingerprint,
    deterministic_dataset_indices,
    require_unchanged_file,
    validate_checkpoint_bundle,
    validate_final_epoch,
    validate_state_dict_schema,
)


class CheckpointBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = "hstu_encoder.attention_mode = 'safa'\n"
        config_sha = hashlib.sha256(self.config.encode("utf-8")).hexdigest()
        self.checkpoint = {
            "epoch": 100,
            "dataset_name": "ml-1m",
            "model_state_dict": {},
            "attention_mode": "safa",
            "random_seed": 42,
            "parameter_count": 123,
            "parameter_inventory_sha256": "a" * 64,
            "resolved_gin_config": self.config,
            "resolved_gin_config_sha256": config_sha,
            "experiment_config_sha256": "b" * 64,
            "source_root": "/immutable/source",
            "source_commit": "commit-id",
            "source_tree": "tree-id",
            "source_manifest": "c" * 64,
            "slurm_array_job_id": "123456",
            "slurm_array_task_id": 1,
            "slurm_job_id": "123457",
            "slurm_job_qos": "h200_dev",
            "slurm_job_partition": "h200",
            "slurm_restart_count": 0,
        }
        self.metadata = {
            key: self.checkpoint[key]
            for key in (
                "dataset_name",
                "attention_mode",
                "random_seed",
                "parameter_count",
                "parameter_inventory_sha256",
                "resolved_gin_config_sha256",
                "experiment_config_sha256",
                "source_commit",
                "source_tree",
                "source_manifest",
                "slurm_array_job_id",
                "slurm_array_task_id",
                "slurm_job_id",
                "slurm_job_qos",
                "slurm_job_partition",
                "slurm_restart_count",
            )
        }
        self.metadata["source_root"] = self.checkpoint["source_root"]
        self.source = {
            key: self.checkpoint[key]
            for key in ("source_commit", "source_tree", "source_manifest")
        }

    def test_accepts_three_matching_artifacts(self) -> None:
        validate_checkpoint_bundle(
            self.checkpoint,
            self.config,
            self.metadata,
            self.source,
        )

    def test_accepts_ml20_high_qos(self) -> None:
        checkpoint = {
            **self.checkpoint,
            "dataset_name": "ml-20m",
            "slurm_job_qos": "h200_mrs_2_high",
        }
        metadata = {
            **self.metadata,
            "dataset_name": "ml-20m",
            "slurm_job_qos": "h200_mrs_2_high",
        }
        validate_checkpoint_bundle(
            checkpoint,
            self.config,
            metadata,
            self.source,
        )

    def test_reconstruction_rejects_gin_checkpoint_dataset_mismatch(self) -> None:
        values = {
            "hstu_encoder.attention_mode": "safa",
            "train_fn.dataset_name": "ml-20m",
            "train_fn.max_sequence_length": 200,
            "train_fn.positional_sampling_ratio": None,
            "train_fn.eval_batch_size": 128,
            "train_fn.eval_user_max_batch_size": None,
            "train_fn.main_module": "HSTU",
            "train_fn.main_module_bf16": False,
            "train_fn.dropout_rate": 0.2,
            "train_fn.user_embedding_norm": "l2_norm",
            "train_fn.sampling_strategy": "local",
            "train_fn.item_l2_norm": True,
            "train_fn.top_k_method": "MIPSBruteForceTopK",
            "train_fn.embedding_module_type": "local",
            "train_fn.item_embedding_dim": 256,
            "train_fn.interaction_module_type": "DotProduct",
            "train_fn.gr_output_length": 10,
            "train_fn.l2_norm_eps": 1e-6,
            "train_fn.enable_tf32": True,
            "train_fn.random_seed": 42,
            "train_fn.num_epochs": 101,
        }

        class FakeGin:
            @staticmethod
            def clear_config() -> None:
                pass

            @staticmethod
            def parse_config(config: str) -> None:
                del config

            @staticmethod
            def query_parameter(name: str):
                return values[name]

        bundle = types.SimpleNamespace(
            checkpoint=self.checkpoint,
            resolved_config=self.config,
        )
        train = types.SimpleNamespace(
            _config_identities=lambda *_args, **_kwargs: (
                self.checkpoint["resolved_gin_config_sha256"],
                self.checkpoint["experiment_config_sha256"],
            )
        )
        with self.assertRaisesRegex(DiagnosticError, "dataset_name mismatch"):
            _build_model_and_dataset(
                bundle,
                {"gin": FakeGin, "train": train},
                torch.device("cpu"),
            )

    def test_rejects_unknown_attention_mode(self) -> None:
        checkpoint = {**self.checkpoint, "attention_mode": "unknown"}
        with self.assertRaisesRegex(DiagnosticError, "hstu.*safa"):
            validate_checkpoint_bundle(
                checkpoint,
                self.config,
                self.metadata,
                self.source,
            )

    def test_rejects_config_content_mismatch(self) -> None:
        with self.assertRaisesRegex(DiagnosticError, "operative config differs"):
            validate_checkpoint_bundle(
                self.checkpoint,
                self.config + "# drift\n",
                self.metadata,
                self.source,
            )

    def test_rejects_run_metadata_mismatch(self) -> None:
        metadata = {**self.metadata, "random_seed": 43}
        with self.assertRaisesRegex(DiagnosticError, "random_seed"):
            validate_checkpoint_bundle(
                self.checkpoint,
                self.config,
                metadata,
                self.source,
            )

    def test_rejects_source_snapshot_mismatch(self) -> None:
        source = {**self.source, "source_tree": "different-tree"}
        with self.assertRaisesRegex(DiagnosticError, "source snapshot mismatch"):
            validate_checkpoint_bundle(
                self.checkpoint,
                self.config,
                self.metadata,
                source,
            )

    def test_rejects_scheduler_task_mapping_mismatch(self) -> None:
        checkpoint = {**self.checkpoint, "slurm_array_task_id": 0}
        metadata = {**self.metadata, "slurm_array_task_id": 0}
        with self.assertRaisesRegex(DiagnosticError, "does not match"):
            validate_checkpoint_bundle(
                checkpoint,
                self.config,
                metadata,
                self.source,
            )

    def test_rejects_scheduler_partition_and_restart_drift(self) -> None:
        for key, value, message in (
            ("slurm_job_partition", "h100", "partition"),
            ("slurm_restart_count", -1, "restart_count"),
        ):
            with self.subTest(key=key):
                checkpoint = {**self.checkpoint, key: value}
                metadata = {**self.metadata, key: value}
                with self.assertRaisesRegex(DiagnosticError, message):
                    validate_checkpoint_bundle(
                        checkpoint,
                        self.config,
                        metadata,
                        self.source,
                    )

    def test_requires_complete_scheduler_provenance_by_default(self) -> None:
        checkpoint = dict(self.checkpoint)
        metadata = dict(self.metadata)
        for key in (
            "slurm_array_job_id",
            "slurm_array_task_id",
            "slurm_job_id",
            "slurm_job_qos",
            "slurm_job_partition",
            "slurm_restart_count",
        ):
            del checkpoint[key]
            del metadata[key]
        with self.assertRaisesRegex(DiagnosticError, "incomplete SLURM"):
            validate_checkpoint_bundle(
                checkpoint,
                self.config,
                metadata,
                self.source,
            )
        validate_checkpoint_bundle(
            checkpoint,
            self.config,
            metadata,
            self.source,
            require_slurm_provenance=False,
        )

    def test_final_epoch_must_match_configured_training_horizon(self) -> None:
        validate_final_epoch(100, 101)
        with self.assertRaisesRegex(DiagnosticError, "not final"):
            validate_final_epoch(99, 101)

    def test_state_schema_rejects_dtype_shape_and_key_drift(self) -> None:
        expected = {"weight": torch.zeros(2, dtype=torch.float32)}
        validate_state_dict_schema(expected, {"weight": torch.ones(2)})
        with self.assertRaisesRegex(DiagnosticError, "dtype mismatch"):
            validate_state_dict_schema(
                expected,
                {"weight": torch.ones(2, dtype=torch.bfloat16)},
            )
        with self.assertRaisesRegex(DiagnosticError, "shape mismatch"):
            validate_state_dict_schema(expected, {"weight": torch.ones(3)})
        with self.assertRaisesRegex(DiagnosticError, "keys mismatch"):
            validate_state_dict_schema(expected, {"other": torch.ones(2)})

    def test_checkpoint_digest_is_pinned_against_later_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint"
            checkpoint.write_bytes(b"first")
            digest = hashlib.sha256(b"first").hexdigest()
            require_unchanged_file(checkpoint, digest, "checkpoint")
            checkpoint.write_bytes(b"second")
            with self.assertRaisesRegex(DiagnosticError, "changed"):
                require_unchanged_file(checkpoint, digest, "checkpoint")


class _Layer(torch.nn.Module):
    def __init__(self, num_heads: int = 1) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._forget_weight = torch.nn.Parameter(torch.zeros(num_heads, 1))
        self._forget_bias = torch.nn.Parameter(torch.zeros(num_heads))


class AttentionDiagnosticsTest(unittest.TestCase):
    def test_reports_gate_survival_half_life_and_signed_mass(self) -> None:
        layer = _Layer()
        collector = AttentionDiagnostics(
            [layer],
            histogram_bins=10000,
            pair_samples_per_batch=32,
            sample_seed=7,
        )
        collector.start_batch(torch.tensor([3]))
        coefficients = torch.tensor(
            [[[[1.0, 0.0, 0.0], [-1.0, 2.0, 0.0], [-3.0, 4.0, -5.0]]]]
        )
        collector.observe(
            padded_k=torch.zeros(1, 3, 1, 1),
            attention_mode="safa",
            forget_weight=layer._forget_weight,
            forget_bias=layer._forget_bias,
            attention_weights=coefficients,
        )
        collector.finish_batch()

        head = collector.result()[0]["heads"][0]
        forgetting = head["forgetting"]
        signed = head["signed_coefficients"]
        self.assertTrue(forgetting["available"])
        self.assertEqual(forgetting["gate_valid_position_count"], 3)
        self.assertAlmostEqual(forgetting["gate_quantiles"]["p50"], 0.5, places=3)
        self.assertEqual(forgetting["transition_gate_count"], 2)
        self.assertAlmostEqual(forgetting["effective_half_life_events"], 1.0, places=6)
        self.assertEqual(forgetting["survival_positive_lag_population_count"], 3)
        self.assertEqual(forgetting["survival_positive_lag_sample_count"], 3)
        self.assertAlmostEqual(forgetting["survival_quantiles"]["p50"], 0.5, places=3)
        self.assertEqual(signed["coefficient_count"], 6)
        self.assertAlmostEqual(signed["negative_count_fraction"], 0.5, places=6)
        self.assertAlmostEqual(signed["negative_absolute_mass_fraction"], 9 / 16)

    def test_rejects_attention_mode_mismatch(self) -> None:
        layer = _Layer()
        collector = AttentionDiagnostics(
            [layer],
            histogram_bins=100,
            pair_samples_per_batch=4,
            sample_seed=0,
        )
        collector.start_batch(torch.tensor([2]))
        with self.assertRaisesRegex(DiagnosticError, "disagrees"):
            collector.observe(
                padded_k=torch.zeros(1, 2, 1, 1),
                attention_mode="hstu",
                forget_weight=layer._forget_weight,
                forget_bias=layer._forget_bias,
                attention_weights=torch.zeros(1, 1, 2, 2),
            )

    def test_hstu_reports_forgetting_unavailable_but_signed_mass(self) -> None:
        layer = _Layer()
        collector = AttentionDiagnostics(
            [layer],
            histogram_bins=100,
            pair_samples_per_batch=4,
            sample_seed=0,
            attention_mode="hstu",
        )
        collector.start_batch(torch.tensor([2]))
        collector.observe(
            padded_k=torch.zeros(1, 2, 1, 1),
            attention_mode="hstu",
            forget_weight=layer._forget_weight,
            forget_bias=layer._forget_bias,
            attention_weights=torch.tensor([[[[1.0, 0.0], [-2.0, 3.0]]]]),
        )
        collector.finish_batch()
        head = collector.result()[0]["heads"][0]
        self.assertFalse(head["forgetting"]["available"])
        self.assertEqual(head["signed_coefficients"]["coefficient_count"], 3)
        self.assertAlmostEqual(
            head["signed_coefficients"]["negative_absolute_mass_fraction"],
            2 / 6,
        )

    def test_rejects_noncausal_coefficients(self) -> None:
        layer = _Layer()
        collector = AttentionDiagnostics(
            [layer],
            histogram_bins=100,
            pair_samples_per_batch=4,
            sample_seed=0,
        )
        collector.start_batch(torch.tensor([2]))
        with self.assertRaisesRegex(DiagnosticError, "causal mask"):
            collector.observe(
                padded_k=torch.zeros(1, 2, 1, 1),
                attention_mode="safa",
                forget_weight=layer._forget_weight,
                forget_bias=layer._forget_bias,
                attention_weights=torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]]),
            )

    def test_signed_mass_vectorization_masks_variable_length_padding(self) -> None:
        layer = _Layer(num_heads=2)
        collector = AttentionDiagnostics(
            [layer],
            histogram_bins=100,
            pair_samples_per_batch=4,
            sample_seed=0,
            attention_mode="hstu",
        )
        collector.start_batch(torch.tensor([2, 3]))
        head_zero = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [-2.0, 3.0, 0.0], [100.0, 100.0, 100.0]],
                [[-1.0, 0.0, 0.0], [2.0, -3.0, 0.0], [4.0, -5.0, 6.0]],
            ]
        )
        head_one = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [-100.0, -100.0, -100.0]],
                [[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]],
            ]
        )
        coefficients = torch.stack((head_zero, head_one), dim=1)
        collector.observe(
            padded_k=torch.zeros(2, 3, 2, 1),
            attention_mode="hstu",
            forget_weight=layer._forget_weight,
            forget_bias=layer._forget_bias,
            attention_weights=coefficients,
        )
        collector.finish_batch()
        heads = collector.result()[0]["heads"]
        self.assertEqual(heads[0]["signed_coefficients"]["coefficient_count"], 9)
        self.assertAlmostEqual(
            heads[0]["signed_coefficients"]["negative_absolute_mass"], 11.0
        )
        self.assertAlmostEqual(
            heads[0]["signed_coefficients"]["positive_absolute_mass"], 16.0
        )
        self.assertEqual(heads[1]["signed_coefficients"]["negative_absolute_mass"], 0.0)
        self.assertAlmostEqual(
            heads[1]["signed_coefficients"]["positive_absolute_mass"], 27.0
        )


class SamplingAndMetricsTest(unittest.TestCase):
    def test_dataset_sample_is_bounded_unique_and_reproducible(self) -> None:
        first = deterministic_dataset_indices(1000, 37, 11)
        second = deterministic_dataset_indices(1000, 37, 11)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 37)
        self.assertEqual(len(set(first)), 37)
        self.assertNotEqual(first, deterministic_dataset_indices(1000, 37, 12))

    def test_metric_strata_preserve_per_example_alignment(self) -> None:
        strata = MetricStrata(("short", "long"))
        metrics = {
            "ndcg@10": torch.tensor([0.0, 1.0, 0.5]),
            "ndcg@50": torch.tensor([0.2, 0.8, 0.4]),
            "hr@10": torch.tensor([False, True, True]),
            "hr@50": torch.tensor([True, True, True]),
            "mrr": torch.tensor([0.1, 0.5, 0.2]),
        }
        strata.update(["short", "long", "short"], metrics)
        result = {row["label"]: row for row in strata.result()}
        self.assertEqual(result["short"]["count"], 2)
        self.assertAlmostEqual(result["short"]["metrics"]["ndcg@10"], 0.25)
        self.assertEqual(result["long"]["metrics"]["hr@10"], 1.0)

    def test_gap_strata_use_last_valid_history_timestamp(self) -> None:
        row = {
            "historical_timestamps": torch.tensor(
                [[100, 200, 0], [100, 0, 0], [500, 600, 0]]
            ),
            "target_timestamps": torch.tensor([200 + 2 * 86400, 100, 599]),
        }
        labels, invalid, reason = _gap_labels(row, torch.tensor([2, 1, 2]))
        expected_label = next(label for upper, label in GAP_BINS if upper == 7 * 86400)
        self.assertEqual(labels[0], expected_label)
        self.assertEqual(labels[1], "<=1d")
        self.assertIsNone(labels[2])
        self.assertEqual(invalid, 1)
        self.assertIn("non-chronological", reason)

    def test_gap_strata_reject_non_chronological_history(self) -> None:
        row = {
            "historical_timestamps": torch.tensor([[100, 90, 0]]),
            "target_timestamps": torch.tensor([200]),
        }
        labels, invalid, reason = _gap_labels(row, torch.tensor([2]))
        self.assertEqual(labels, [None])
        self.assertEqual(invalid, 1)
        self.assertIn("non-chronological", reason)

    def test_data_fingerprint_covers_ratings_and_item_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ratings = root / "ml-1m/sasrec_format.csv"
            items = root / "processed/ml-1m/movies.csv"
            ratings.parent.mkdir(parents=True)
            items.parent.mkdir(parents=True)
            ratings.write_text("ratings\n", encoding="utf-8")
            items.write_text("items\n", encoding="utf-8")
            first = data_fingerprint(root, "ml-1m")
            items.write_text("changed\n", encoding="utf-8")
            second = data_fingerprint(root, "ml-1m")
        self.assertFalse(first["authenticated_by_training_artifact"])
        self.assertNotEqual(first["combined_sha256"], second["combined_sha256"])
        self.assertEqual(len(first["files"]), 2)


if __name__ == "__main__":
    unittest.main()
