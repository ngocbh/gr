#!/usr/bin/env python3
"""CPU-only tests for the Signed-LIFT feature-geometry diagnostic."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMITTED_REPORT = ROOT / "paper/diagnostics/signed_lift_feature_geometry.json"
EXPECTED_COMMITTED_REPORT_SHA256 = (
    "e08e4e9143e4b47b92e3594aa3e98da1b40cc29931c45944525c3660cf25d8e2"
)


_MODULE_NAME = "_gr_probe_signed_lift_geometry_under_test"
_MODULE_PATH = Path(__file__).with_name("probe_signed_lift_geometry.py")
if _MODULE_NAME in sys.modules:
    probe = sys.modules[_MODULE_NAME]
else:
    _SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"could not load geometry probe from {_MODULE_PATH}")
    probe = importlib.util.module_from_spec(_SPEC)
    sys.modules[_MODULE_NAME] = probe
    _SPEC.loader.exec_module(probe)


class SignedLIFTGeometryProbeTest(unittest.TestCase):
    def test_canonical_training_snapshot_tree_and_manifest(self) -> None:
        provenance = probe.verify_training_snapshot(
            probe.CANONICAL_TRAINING_SNAPSHOT
        )
        self.assertEqual(
            provenance["manifest_sha256"], probe.SNAPSHOT_MANIFEST_SHA256
        )
        self.assertEqual(provenance["manifest_entry_count"], 485)
        self.assertEqual(
            provenance["legacy_unmanifested_root_file_allowlist"],
            probe.LEGACY_UNMANIFESTED_ROOT_FILES,
        )
        self.assertEqual(
            provenance["file_count_including_manifest_and_legacy_metadata"], 489
        )
        self.assertGreater(provenance["research_python_file_count"], 0)
        self.assertFalse(provenance["symlinks_allowed"])
        self.assertFalse(provenance["special_nodes_allowed"])

    def test_completed_checkpoint_epoch_must_equal_100_exactly(self) -> None:
        self.assertEqual(probe.validate_checkpoint_epoch(100), 100)
        for invalid in (False, 0, 99, 101, "100", 100.0):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "exactly 100"):
                    probe.validate_checkpoint_epoch(invalid)

    def test_old_pair_mask_is_causal_w32_and_respects_lengths(self) -> None:
        lengths = torch.tensor([35, 33], dtype=torch.int64)
        mask = probe.old_pair_mask(lengths=lengths, sequence_length=36)
        self.assertEqual(tuple(mask.shape), (2, 36, 36))
        self.assertTrue(mask[0, 32, 0])
        self.assertTrue(mask[0, 34, 2])
        self.assertFalse(mask[0, 31, 0])
        self.assertFalse(mask[0, 34, 3])
        self.assertTrue(mask[1, 32, 0])
        self.assertFalse(mask[1, 33, 0])
        self.assertEqual(int(mask[0].sum()), 6)
        self.assertEqual(int(mask[1].sum()), 1)

    def test_survival_is_product_of_intervening_forget_gates(self) -> None:
        log_forget = torch.full((1, 4, 1), math.log(0.5), dtype=torch.float64)
        survival = probe.survival_from_log_forget(log_forget)
        self.assertEqual(tuple(survival.shape), (1, 1, 4, 4))
        self.assertAlmostEqual(float(survival[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(survival[0, 0, 2, 0]), 0.25)
        self.assertAlmostEqual(float(survival[0, 0, 3, 0]), 0.125)
        self.assertAlmostEqual(float(survival[0, 0, 3, 1]), 0.25)
        self.assertAlmostEqual(float(survival[0, 0, 0, 3]), 1.0)

    def test_coefficient_summary_and_correlation_formulas(self) -> None:
        values = torch.tensor([-2.0, -1.0, 0.0, 3.0], dtype=torch.float64)
        summary = probe.coefficient_summary(values)
        self.assertEqual(summary["pair_count"], 4)
        self.assertAlmostEqual(summary["negative_pair_fraction"], 0.5)
        self.assertAlmostEqual(summary["negative_l1_mass_fraction"], 0.5)
        self.assertAlmostEqual(summary["rms"], math.sqrt(14.0 / 4.0))
        self.assertAlmostEqual(probe.pearson_correlation(values, 2.0 * values), 1.0)
        self.assertAlmostEqual(probe.pearson_correlation(values, -values), -1.0)
        with self.assertRaisesRegex(ValueError, "zero variance"):
            probe.pearson_correlation(torch.ones(4), torch.arange(4.0))

    def test_feature_map_formulas(self) -> None:
        x = torch.tensor(
            [[[[-2.0, -1.0, 1.0, 2.0], [0.0, 1.0, 2.0, 4.0]]]],
            dtype=torch.float64,
        )
        standardized = probe.standardized_tanh_feature_map(x)
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        expected = torch.tanh((x - mean) / torch.sqrt(variance + 1e-6))
        torch.testing.assert_close(standardized, expected)

        signed_sqrt = probe.signed_sqrt_rms_feature_map(x)
        raw = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-6)
        expected_signed_sqrt = raw / torch.sqrt(
            raw.square().mean(dim=-1, keepdim=True)
        ).clamp_min(1e-6)
        torch.testing.assert_close(signed_sqrt, expected_signed_sqrt)
        rms = torch.sqrt(signed_sqrt.square().mean(dim=-1))
        torch.testing.assert_close(rms, torch.ones_like(rms))

        original_rms = torch.sqrt(x.square().mean(dim=-1))
        standardized_matched = probe.standardized_tanh_rms_matched_feature_map(x)
        signed_sqrt_matched = probe.signed_sqrt_rms_matched_feature_map(x)
        torch.testing.assert_close(
            torch.sqrt(standardized_matched.square().mean(dim=-1)),
            original_rms,
        )
        torch.testing.assert_close(
            torch.sqrt(signed_sqrt_matched.square().mean(dim=-1)),
            original_rms,
        )

    def test_head_metrics_include_required_and_exploratory_maps(self) -> None:
        q = torch.tensor(
            [[[1.0, -2.0], [2.0, 1.0], [-1.0, 3.0]]],
            dtype=torch.float64,
        )
        k = torch.tensor(
            [[[2.0, 1.0], [-1.0, 2.0], [3.0, -1.0]]],
            dtype=torch.float64,
        )
        survival = torch.ones((1, 3, 3), dtype=torch.float64)
        mask = torch.tensor(
            [[[False, False, False], [True, False, False], [True, True, False]]]
        )
        metrics = probe.head_feature_metrics(
            q=q,
            k=k,
            survival=survival,
            old_mask=mask,
            gammas=(0.5, 1.0, 2.0),
        )
        self.assertEqual([entry["gamma"] for entry in metrics["gamma_sweep"]], [0.5, 1.0, 2.0])
        for entry in metrics["gamma_sweep"]:
            self.assertIn("identity_tanh_correlation", entry)
            self.assertIn("tanh_identity_rms_ratio", entry)
            self.assertIn("abs_tanh_tanh_rms_ratio", entry)
            self.assertIn("negative_pair_fraction", entry["tanh_coefficients"])
        exploratory = metrics["exploratory_feature_maps"]
        self.assertEqual(
            set(exploratory),
            {
                "per_vector_standardized_tanh",
                "signed_sqrt_unit_rms",
                "per_vector_standardized_tanh_rms_matched",
                "signed_sqrt_rms_matched",
            },
        )
        for feature_metrics in exploratory.values():
            self.assertTrue(feature_metrics["exploratory_only"])
            self.assertIn("identity_correlation", feature_metrics)
            self.assertIn("rms_ratio_to_tanh_gamma_1", feature_metrics)

    def test_layout_shims_pack_and_restore_namespace_exactly(self) -> None:
        namespace = torch.ops.fbgemm
        names = probe.FBGEMM_LAYOUT_OPS
        before = {
            name: (name in namespace.__dict__, namespace.__dict__.get(name))
            for name in names
        }
        with probe.scoped_fbgemm_layout_shims() as installed:
            lengths = torch.tensor([2, 1], dtype=torch.int64)
            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
            dense = torch.arange(2 * 3 * 2).reshape(2, 3, 2)
            jagged = torch.ops.fbgemm.dense_to_jagged(dense, [offsets])[0]
            restored = torch.ops.fbgemm.jagged_to_padded_dense(
                values=jagged,
                offsets=[offsets],
                max_lengths=[3],
                padding_value=-1,
            )
            torch.testing.assert_close(offsets, torch.tensor([0, 2, 3]))
            torch.testing.assert_close(jagged, torch.cat((dense[0, :2], dense[1, :1])))
            torch.testing.assert_close(restored[0, :2], dense[0, :2])
            torch.testing.assert_close(restored[1, :1], dense[1, :1])
            self.assertTrue(set(installed).issubset(set(names)))
        after = {
            name: (name in namespace.__dict__, namespace.__dict__.get(name))
            for name in names
        }
        self.assertEqual(before.keys(), after.keys())
        for name in names:
            self.assertEqual(before[name][0], after[name][0])
            if before[name][0]:
                self.assertIs(before[name][1], after[name][1])

        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            with probe.scoped_fbgemm_layout_shims():
                with torch.inference_mode():
                    raise RuntimeError("injected failure")
        after_failure = {
            name: (name in namespace.__dict__, namespace.__dict__.get(name))
            for name in names
        }
        for name in names:
            self.assertEqual(before[name][0], after_failure[name][0])
            if before[name][0]:
                self.assertIs(before[name][1], after_failure[name][1])

    def test_cpu_thread_scope_restores_on_success_and_failure(self) -> None:
        original = torch.get_num_threads()
        requested = 1 if original != 1 else 2
        with probe.scoped_cpu_thread_count(requested):
            self.assertEqual(torch.get_num_threads(), requested)
        self.assertEqual(torch.get_num_threads(), original)
        with self.assertRaisesRegex(RuntimeError, "thread failure"):
            with probe.scoped_cpu_thread_count(requested):
                self.assertEqual(torch.get_num_threads(), requested)
                raise RuntimeError("thread failure")
        self.assertEqual(torch.get_num_threads(), original)

    def test_verified_file_bytes_toctou_and_symlink_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "payload.pt"
            buffer = __import__("io").BytesIO()
            torch.save({"value": torch.tensor([1, 2, 3])}, buffer)
            original_payload = buffer.getvalue()
            path.write_bytes(original_payload)
            expected_sha = hashlib.sha256(original_payload).hexdigest()
            payload, digest, identity = probe.read_verified_regular_file(
                path, expected_sha
            )
            self.assertEqual(payload, original_payload)
            self.assertEqual(digest, expected_sha)

            path.write_bytes(b"mutated")
            with self.assertRaises(ValueError):
                probe.revalidate_regular_file(path, expected_sha, identity)
            loaded = probe.load_checkpoint_from_verified_bytes(payload)
            torch.testing.assert_close(loaded["value"], torch.tensor([1, 2, 3]))

            target = root / "target"
            target.write_bytes(b"target")
            symlink = root / "symlink"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                probe.read_verified_regular_file(symlink)

    def test_atomic_output_fsync_and_rejects_symlink_or_nonregular(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            with mock.patch.object(probe.os, "fsync", wraps=os.fsync) as fsync:
                probe._write_atomic(output, "{\"ok\": true}\n")
                self.assertGreaterEqual(fsync.call_count, 2)
            self.assertEqual(output.read_text(), "{\"ok\": true}\n")

            target = root / "target.json"
            target.write_text("unchanged")
            symlink = root / "report-link.json"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink or nonregular"):
                probe._write_atomic(symlink, "changed")
            self.assertEqual(target.read_text(), "unchanged")

            directory_output = root / "directory-output"
            directory_output.mkdir()
            with self.assertRaisesRegex(ValueError, "symlink or nonregular"):
                probe._write_atomic(directory_output, "changed")

            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                probe._write_atomic(linked_parent / "child.json", "changed")

            collision_output = root / "collision-report.json"
            collision_temp = root / ".collision-report.json.tmp-collision"
            collision_temp.write_text("preexisting")
            with mock.patch.object(
                probe.secrets, "token_hex", return_value="collision"
            ):
                with self.assertRaises(FileExistsError):
                    probe._write_atomic(collision_output, "replacement")
            self.assertEqual(collision_temp.read_text(), "preexisting")
            self.assertFalse(collision_output.exists())

            collision_temp.unlink()
            collision_target = root / "collision-target"
            collision_target.write_text("target unchanged")
            collision_temp.symlink_to(collision_target)
            with mock.patch.object(
                probe.secrets, "token_hex", return_value="collision"
            ):
                with self.assertRaises(FileExistsError):
                    probe._write_atomic(collision_output, "replacement")
            self.assertTrue(collision_temp.is_symlink())
            self.assertEqual(collision_target.read_text(), "target unchanged")
            self.assertFalse(collision_output.exists())

    def test_machine_readable_interpretation_limits(self) -> None:
        global_limits = probe.global_interpretation_limits((0.5, 1.0, 2.0))
        self.assertFalse(global_limits["confirmatory_inference"])
        self.assertFalse(global_limits["accuracy_estimate"])
        self.assertEqual(
            global_limits["gamma_roles"]["1.0"], "current_setting_diagnostic"
        )
        self.assertEqual(
            global_limits["gamma_roles"]["0.5"], "post_hoc_exploratory"
        )
        local = probe.checkpoint_interpretation_limits("local")
        lift = probe.checkpoint_interpretation_limits("lift")
        self.assertFalse(local["tail_active_during_training"])
        self.assertTrue(lift["tail_active_during_training"])
        self.assertFalse(lift["post_hoc_transforms_are_trained_representations"])

    def test_sampling_and_canonical_json_are_deterministic(self) -> None:
        first = probe.select_user_indices(
            total_users=10, max_users=4, sampling="first", seed=123
        )
        self.assertEqual(first, [0, 1, 2, 3])
        seeded_a = probe.select_user_indices(
            total_users=10, max_users=4, sampling="seeded", seed=123
        )
        seeded_b = probe.select_user_indices(
            total_users=10, max_users=4, sampling="seeded", seed=123
        )
        self.assertEqual(seeded_a, seeded_b)
        self.assertEqual(len(set(seeded_a)), 4)

        report_a = {"z": [3, 2, 1], "a": {"value": 0.125}}
        report_b = {"a": {"value": 0.125}, "z": [3, 2, 1]}
        encoded_a = probe.canonical_json(report_a)
        encoded_b = probe.canonical_json(report_b)
        self.assertEqual(encoded_a, encoded_b)
        self.assertEqual(json.loads(encoded_a), report_a)

    def test_aggregate_summary_is_derived_from_head_records(self) -> None:
        def head(value: float) -> dict:
            exploratory = {}
            for name in probe.EXPLORATORY_MAP_NAMES:
                exploratory[name] = {
                    "identity_correlation": value + 0.1,
                    "rms_ratio_to_tanh_gamma_1": value + 1.0,
                    "coefficients": {
                        "negative_pair_fraction": value / 10.0,
                        "negative_l1_mass_fraction": value / 20.0,
                    },
                }
            return {
                "head": 0,
                "gamma_sweep": [
                    {
                        "gamma": 1.0,
                        "identity_tanh_correlation": value,
                        "tanh_identity_rms_ratio": value + 2.0,
                        "abs_tanh_tanh_rms_ratio": value + 3.0,
                        "tanh_coefficients": {
                            "negative_pair_fraction": value / 4.0,
                            "negative_l1_mass_fraction": value / 5.0,
                        },
                        "q_tanh_saturation": {
                            "abs_ge_0_95_fraction": value / 10.0,
                            "abs_ge_0_99_fraction": value / 20.0,
                        },
                        "k_tanh_saturation": {
                            "abs_ge_0_95_fraction": value / 8.0,
                            "abs_ge_0_99_fraction": value / 16.0,
                        },
                    }
                ],
                "exploratory_feature_maps": exploratory,
            }

        checkpoints = [
            {
                "path": f"/checkpoint-{index}",
                "sha256": str(index) * 64,
                "filename": f"checkpoint-{index}.pt",
                "kind": "lift" if index == 1 else "local",
                "seed": 41 + index,
                "job_id": 1671578,
                "task_id": index,
                "restart_count": 0,
                "layers": [{"layer": 0, "heads": [head(value)]}],
            }
            for index, value in ((1, 0.2), (2, 0.6))
        ]
        summary = probe.build_aggregate_summary(checkpoints, (1.0,))
        aggregate = summary["all_checkpoints"]
        self.assertEqual(aggregate["entry_count"], 2)
        correlation = aggregate["gamma_sweep"][0][
            "identity_tanh_correlation"
        ]
        self.assertEqual(correlation["count"], 2)
        self.assertAlmostEqual(correlation["min"], 0.2)
        self.assertAlmostEqual(correlation["mean"], 0.4)
        self.assertAlmostEqual(correlation["max"], 0.6)
        exploratory = aggregate["exploratory_feature_maps"][
            "signed_sqrt_rms_matched"
        ]
        self.assertAlmostEqual(
            exploratory["rms_ratio_to_tanh_gamma_1"]["mean"], 1.4
        )
        self.assertAlmostEqual(
            exploratory["negative_pair_fraction"]["mean"], 0.04
        )
        self.assertEqual(len(summary["per_checkpoint"]), 2)
        self.assertEqual(summary["per_checkpoint"][0]["entry_count"], 1)
        self.assertAlmostEqual(
            summary["per_checkpoint"][1]["gamma_sweep"][0][
                "identity_tanh_correlation"
            ]["mean"],
            0.6,
        )
        self.assertEqual(
            probe.canonical_json(summary), probe.canonical_json(summary)
        )

        broken = json.loads(json.dumps(checkpoints))
        broken[0]["layers"][0]["heads"][0]["gamma_sweep"][0]["gamma"] = 2.0
        with self.assertRaisesRegex(ValueError, "configured gammas"):
            probe.build_aggregate_summary(broken, (1.0,))

    def test_invalid_inputs_fail_closed_without_full_dataset(self) -> None:
        for gammas in ((), (0.0,), (-1.0,), (1.0, 1.0), (float("nan"),)):
            with self.subTest(gammas=gammas):
                with self.assertRaises(ValueError):
                    probe.validate_gammas(gammas)
        with self.assertRaises(ValueError):
            probe.select_user_indices(0, 1, "first", 0)
        with self.assertRaises(ValueError):
            probe.select_user_indices(3, 4, "first", 0)
        with self.assertRaises(ValueError):
            probe.select_user_indices(3, 2, "unknown", 0)

        with self.assertRaisesRegex(ValueError, "checkpoint filename"):
            probe.checkpoint_kind(Path("not-an-hstu-checkpoint.pt"))
        canonical_paths = [
            Path(name) for name in sorted(probe.EXPECTED_CHECKPOINTS)
        ]
        probe.validate_canonical_checkpoint_set(canonical_paths)
        with self.assertRaisesRegex(ValueError, "basename set mismatch"):
            probe.validate_canonical_checkpoint_set(canonical_paths[:-1])
        with self.assertRaisesRegex(ValueError, "basenames must be unique"):
            probe.validate_canonical_checkpoint_set(
                [*canonical_paths[:-1], canonical_paths[0]]
            )
        local_name = next(
            name
            for name, metadata in probe.EXPECTED_CHECKPOINTS.items()
            if metadata["kind"] == "local"
        )
        lift_name = next(
            name
            for name, metadata in probe.EXPECTED_CHECKPOINTS.items()
            if metadata["kind"] == "lift"
        )
        self.assertEqual(probe.checkpoint_kind(Path(local_name)), "local")
        self.assertEqual(probe.checkpoint_kind(Path(lift_name)), "lift")
        with self.assertRaises(ValueError):
            probe.checkpoint_kind(Path(local_name.replace("-t1-r0", "-t2-r0")))

        with self.assertRaisesRegex(ValueError, "mixed DDP prefix"):
            probe.strip_ddp_prefix(
                {"module.weight": torch.ones(1), "bias": torch.ones(1)}
            )

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.csv"
            data_path.write_text("user_id,sequence_item_ids\n1,1\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                probe.validate_data_csv(data_path)

        probe.validate_tail_gain_state("local", torch.zeros(16))
        probe.validate_tail_gain_state("lift", torch.ones(16))
        with self.assertRaises(ValueError):
            probe.validate_tail_gain_state("local", torch.ones(16))
        with self.assertRaises(ValueError):
            probe.validate_tail_gain_state("lift", torch.zeros(16))

    def test_state_inventory_rejects_key_shape_dtype_mask_and_nonfinite_drift(self) -> None:
        expected = {
            "weight": torch.zeros((2, 3), dtype=torch.float32),
            "_attn_mask": torch.tensor([[False, True], [False, False]]),
        }
        valid = {key: value.clone() for key, value in expected.items()}
        probe.validate_state_inventory(valid, expected)

        mutations = (
            {"weight": valid["weight"]},
            {**valid, "extra": torch.ones(1)},
            {**valid, "weight": torch.zeros((3, 2))},
            {**valid, "weight": torch.zeros((2, 3), dtype=torch.float64)},
            {**valid, "weight": torch.full((2, 3), float("nan"))},
            {**valid, "_attn_mask": ~valid["_attn_mask"]},
        )
        for mutation in mutations:
            with self.subTest(keys=tuple(mutation)):
                with self.assertRaises(ValueError):
                    probe.validate_state_inventory(mutation, expected)

    @unittest.skipUnless(
        os.environ.get("GR_RUN_SIGNED_LIFT_GEOMETRY_E2E") == "1",
        "set GR_RUN_SIGNED_LIFT_GEOMETRY_E2E=1 for frozen-source regeneration",
    )
    def test_committed_report_regenerates_from_frozen_training_source(self) -> None:
        checkpoint_root = Path(
            "/checkpoints/ngocbh/longhstu/checkpoints/ml-1m-l200"
        )
        checkpoints = [
            checkpoint_root / filename
            for filename in sorted(probe.EXPECTED_CHECKPOINTS)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            command = [
                sys.executable,
                str(_MODULE_PATH),
                *(str(path) for path in checkpoints),
                "--data-csv",
                "/checkpoints/ngocbh/longhstu/datasets/ml-1m/sasrec_format.csv",
                "--training-snapshot",
                str(probe.CANONICAL_TRAINING_SNAPSHOT),
                "--max-users",
                "128",
                "--gammas",
                "0.5",
                "1",
                "2",
                "4",
                "--sampling",
                "first",
                "--json-output",
                str(output),
            ]
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            generated = output.read_bytes()
        committed = COMMITTED_REPORT.read_bytes()
        self.assertEqual(generated, committed)
        self.assertEqual(
            hashlib.sha256(committed).hexdigest(),
            EXPECTED_COMMITTED_REPORT_SHA256,
        )
        report = json.loads(committed)
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(
            report["aggregate_summary"]["all_checkpoints"]["entry_count"], 96
        )
        self.assertTrue(all(item["epoch"] == 100 for item in report["checkpoints"]))
        provenance = report["runtime_source_provenance"]
        self.assertTrue(provenance["bootstrap_before_gr_imports"])
        self.assertEqual(
            provenance["training_snapshot"]["manifest_sha256"],
            probe.SNAPSHOT_MANIFEST_SHA256,
        )
        snapshot_prefix = str(probe.CANONICAL_TRAINING_SNAPSHOT) + "/"
        for module in provenance["loaded_gr_modules"].values():
            if module["kind"] == "source_module":
                self.assertTrue(module["path"].startswith(snapshot_prefix))
            else:
                self.assertTrue(
                    all(path.startswith(snapshot_prefix) for path in module["paths"])
                )
        selection = report["data"]["selection_disclosure"]
        self.assertFalse(selection["held_out_sample"])
        self.assertFalse(selection["predeclared_sample"])
        self.assertFalse(selection["representative_sample_claim"])


if __name__ == "__main__":
    unittest.main()
