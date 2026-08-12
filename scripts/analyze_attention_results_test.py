#!/usr/bin/env python3
"""Tests for strict attention-result validation and aggregation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from unittest import mock


MODULE_NAME = "_gr_analyze_attention_results_under_test"
MODULE_PATH = Path(__file__).with_name("analyze_attention_results.py")
MODULE_SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise ImportError(f"could not load analyzer from {MODULE_PATH}")
analyzer = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_NAME] = analyzer
MODULE_SPEC.loader.exec_module(analyzer)


DEFAULT_POLICIES: Dict[str, Dict[str, Any]] = {
    "tail": {
        "type": "fixed_tail_screen",
        "metric": "eval_epoch/ndcg@10.tail_mean_steps_96_100",
        "required_pairs": 3,
        "mean_delta_min": 0.002,
        "positive_fraction_min": 2.0 / 3.0,
        "min_delta_min": -0.001,
    },
    "geometry": {
        "type": "fixed_positive_fraction_screen",
        "metric": "eval_epoch/ndcg@10.tail_mean_steps_96_100",
        "required_pairs": 3,
        "positive_fraction_min": 2.0 / 3.0,
    },
    "softmax": {
        "type": "insufficient_seeds",
        "metric": "eval_epoch/ndcg@10.tail_mean_steps_96_100",
        "minimum_pairs": 3,
    },
    "none": {
        "type": "no_fixed_decision_rule",
        "metric": "eval_epoch/ndcg@10.tail_mean_steps_96_100",
    },
    "cross": {
        "type": "cross_snapshot_ineligible",
        "metric": "eval_epoch/ndcg@10.tail_mean_steps_96_100",
    },
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _scalar_task(
    variant: str = "candidate", seed: int = 42, run_name: str = "candidate-seed42"
) -> Dict[str, Any]:
    return {
        "scalar": True,
        "variant": variant,
        "seed": seed,
        "run_name": run_name,
    }


def _array_task(task_id: int, variant: str, seed: int, run_name: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "variant": variant,
        "seed": seed,
        "run_name": run_name,
    }


class AnalyzerFixture(unittest.TestCase):
    job_id = "900"

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.snapshot_data = self.snapshot / "source.txt"
        self.snapshot_data.write_text("sealed\n", encoding="utf-8")
        data_hash = hashlib.sha256(self.snapshot_data.read_bytes()).hexdigest()
        self.manifest = self.snapshot / "SOURCE_SHA256SUMS"
        self.manifest.write_text(f"{data_hash}  source.txt\n", encoding="utf-8")
        self.run_root = self.root / "exps" / "ml-1m-l200"
        self.run_root.mkdir(parents=True)
        self.spec_path = self.root / "spec.json"
        self.sacct_path = self.root / "sacct.txt"
        self.scalar_by_directory: Dict[Path, Mapping[str, Sequence[Any]]] = {}
        self.loader_calls: List[Path] = []
        self.experiment: Dict[str, Any] = {}
        self.write_spec([_scalar_task()])

    def write_spec(
        self,
        tasks: Sequence[Mapping[str, Any]],
        comparisons: Sequence[Mapping[str, Any]] = (),
        *,
        variant_metadata: Mapping[str, Any] = {},
        snapshot_root: Path | None = None,
        manifest_sha256: str | None = None,
        manifest_export_required: bool = False,
        policies: Mapping[str, Any] = DEFAULT_POLICIES,
    ) -> None:
        self.experiment = {
            "label": "fixture",
            "job_id": self.job_id,
            "snapshot": {
                "root": str(snapshot_root or self.snapshot),
                "manifest_sha256": manifest_sha256
                or hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                "legacy_unmanifested_files": {},
            },
            "run_root": str(self.run_root),
            "scheduler": {
                "job_name": "fixture-job",
                "qos": "h200_mrs_shared",
                "work_dir": str(self.root),
                "wrapper": {
                    "form": "relative_workdir",
                    "token": "scripts/fixture.sh",
                },
                "snapshot_export": "GR_CODE_SNAPSHOT",
                "manifest_export_required": manifest_export_required,
            },
            "overall_policy": "per_comparison_only",
            "screen_provenance": "no_fixed_screen",
            "expected_steps": {"start": 0, "end": 100},
            "tasks": [dict(task) for task in tasks],
            "comparisons": [dict(comparison) for comparison in comparisons],
        }
        if variant_metadata:
            self.experiment["variant_metadata"] = dict(variant_metadata)
        registry = {
            "schema_version": 1,
            "experiments": {self.job_id: self.experiment},
            "policies": dict(policies),
        }
        self.spec_path.write_text(
            json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def write_sacct(self, lines: Sequence[str]) -> None:
        expanded: List[str] = []
        scheduler = self.experiment["scheduler"]
        snapshot = self.experiment["snapshot"]
        exports = f"ALL,GR_CODE_SNAPSHOT={snapshot['root']}"
        if scheduler["manifest_export_required"]:
            exports += ",GR_SNAPSHOT_MANIFEST_SHA256=" + snapshot["manifest_sha256"]
        submit_line = (
            f"sbatch --parsable --export={exports} " f"{scheduler['wrapper']['token']}"
        )
        for line in lines:
            fields = line.split("|")
            if len(fields) == 4:
                job_id, state, exit_code, restarts = fields
                line = "|".join(
                    (
                        job_id,
                        scheduler["job_name"],
                        scheduler["qos"],
                        state,
                        exit_code,
                        restarts,
                        submit_line,
                        scheduler["work_dir"],
                    )
                )
            expanded.append(line)
        self.sacct_path.write_text("\n".join(expanded) + "\n", encoding="utf-8")

    def suffix(self, task: Mapping[str, Any], restart: int) -> str:
        task_suffix = "" if task.get("scalar") is True else f"-t{task['task_id']}"
        return f"{task['run_name']}-j{self.job_id}{task_suffix}-r{restart}"

    def make_scalars(
        self,
        ndcg: float = 0.2,
        *,
        throughput_by_step: bool = False,
    ) -> Dict[str, List[Any]]:
        result: Dict[str, List[Any]] = {}
        for tag, offset in (
            ("eval_epoch/ndcg@10", 0.0),
            ("eval_epoch/hr@10", 0.1),
            ("eval_epoch/mrr", -0.05),
        ):
            result[tag] = [
                analyzer.ScalarPoint(step, ndcg + offset) for step in range(101)
            ]
        result["performance/train_examples_per_second"] = [
            analyzer.ScalarPoint(
                step,
                float(step if step > 0 else 1) if throughput_by_step else 1000.0 + ndcg,
            )
            for step in range(101)
        ]
        return result

    def add_run(
        self,
        task: Mapping[str, Any],
        restart: int,
        *,
        scalars: Mapping[str, Sequence[Any]] | None = None,
        prefix: str = "model",
        event_content: bytes | None = None,
    ) -> Path:
        directory = self.run_root / f"{prefix}-{self.suffix(task, restart)}"
        directory.mkdir()
        event = directory / "events.out.tfevents.fixture"
        event.write_bytes(event_content or directory.name.encode("utf-8"))
        self.scalar_by_directory[directory] = scalars or self.make_scalars()
        return directory

    def scalar_loader(self, directory: Path) -> Mapping[str, Sequence[Any]]:
        self.loader_calls.append(directory)
        return self.scalar_by_directory[directory]

    def analyze(self, *, skip_full_snapshot_check: bool = True) -> Dict[str, Any]:
        return analyzer.analyze_experiment(
            self.spec_path,
            self.job_id,
            sacct_file=self.sacct_path,
            skip_full_snapshot_check=skip_full_snapshot_check,
            scalar_loader=self.scalar_loader,
        )


class SchedulerIsolationTest(AnalyzerFixture):
    def _completed_scalar_fields(self) -> List[str]:
        self.write_sacct([f"{self.job_id}|COMPLETED|0:0|0"])
        return self.sacct_path.read_text(encoding="utf-8").strip().split("|")

    def test_incomplete_r1_never_falls_back_to_complete_r0(self) -> None:
        task = self.experiment["tasks"][0]
        self.add_run(task, 0)
        self.write_sacct([f"{self.job_id}|RUNNING|0:0|1"])

        report = self.analyze()

        self.assertEqual(report["status"], "INCOMPLETE_TEST_ONLY")
        self.assertIsNone(report["decision"])
        self.assertFalse(report["evidence"]["scientific_evidence_eligible"])
        self.assertEqual(report["runs"], [])
        self.assertEqual(self.loader_calls, [])

    def test_complete_r1_uses_only_r1_and_its_provenance(self) -> None:
        task = self.experiment["tasks"][0]
        r0 = self.add_run(task, 0, event_content=b"stale-r0")
        r1 = self.add_run(task, 1, event_content=b"authoritative-r1")
        self.write_sacct([f"{self.job_id}|COMPLETED|0:0|1"])

        report = self.analyze()

        self.assertEqual(report["status"], "COMPLETE_TEST_ONLY")
        self.assertEqual(report["evidence"]["evidence_class"], "TEST_ONLY")
        self.assertEqual(Path(report["runs"][0]["run_directory"]), r1)
        self.assertEqual(self.loader_calls, [r1])
        rendered = analyzer.render_json(report)
        self.assertIn(_sha256_bytes(b"authoritative-r1"), rendered)
        self.assertNotIn(_sha256_bytes(b"stale-r0"), rendered)
        self.assertNotIn(str(r0), rendered)

    def test_each_array_task_uses_its_own_scheduler_restart(self) -> None:
        tasks = [
            _array_task(0, "a", 42, "a-seed42"),
            _array_task(1, "b", 42, "b-seed42"),
        ]
        self.write_spec(tasks)
        directories = {
            0: self.add_run(tasks[0], 1),
            1: self.add_run(tasks[1], 2),
        }
        self.write_sacct(
            [
                f"{self.job_id}_1|COMPLETED|0:0|2",
                f"{self.job_id}_0|COMPLETED|0:0|1",
            ]
        )

        report = self.analyze()

        self.assertEqual(
            [run["scheduler"]["restarts"] for run in report["runs"]], [1, 2]
        )
        self.assertEqual(
            [Path(run["run_directory"]) for run in report["runs"]],
            [directories[0], directories[1]],
        )

    def test_compressed_pending_array_record_is_incomplete(self) -> None:
        tasks = [
            _array_task(0, "a", 42, "a-seed42"),
            _array_task(1, "b", 42, "b-seed42"),
        ]
        self.write_spec(tasks)
        self.write_sacct([f"{self.job_id}_[0-1]|PENDING|0:0|0"])

        report = self.analyze()

        self.assertEqual(report["status"], "INCOMPLETE_TEST_ONLY")
        self.assertTrue(report["scheduler"]["compressed_or_parent_records"])
        self.assertTrue(
            all(
                record["state"] == "MISSING_EXACT_TASK_RECORD"
                for record in report["scheduler"]["records"]
            )
        )

    def test_compressed_range_stays_incomplete_even_with_exact_task_rows(
        self,
    ) -> None:
        tasks = [
            _array_task(0, "a", 42, "a-seed42"),
            _array_task(1, "b", 42, "b-seed42"),
        ]
        self.write_spec(tasks)
        self.write_sacct(
            [
                f"{self.job_id}_0|COMPLETED|0:0|0",
                f"{self.job_id}_1|COMPLETED|0:0|0",
                f"{self.job_id}_[0-1]|PENDING|0:0|0",
            ]
        )

        report = self.analyze()

        self.assertEqual(report["status"], "INCOMPLETE_TEST_ONLY")
        self.assertEqual(self.loader_calls, [])

    def test_failed_or_nonzero_task_is_incomplete_without_decision(self) -> None:
        self.write_sacct([f"{self.job_id}|FAILED|1:0|0"])
        report = self.analyze()
        self.assertEqual(report["status"], "INCOMPLETE_TEST_ONLY")
        self.assertIsNone(report["decision"])

    def test_unexpected_or_duplicate_jobid_binding_is_invalid(self) -> None:
        cases = (
            [f"{self.job_id}.batch|COMPLETED|0:0|0"],
            [
                f"{self.job_id}|COMPLETED|0:0|0",
                f"{self.job_id}|COMPLETED|0:0|0",
            ],
        )
        for lines in cases:
            with self.subTest(lines=lines):
                self.write_sacct(lines)
                with self.assertRaises(analyzer.InvalidEvidence):
                    self.analyze()

    def test_scheduler_execution_binding_rejects_each_forged_field(self) -> None:
        base = self._completed_scalar_fields()
        forged_rows: Dict[str, List[str]] = {}
        for name, index, value in (
            ("job_name", 1, "forged-job"),
            ("qos", 2, "h200_dev"),
            ("work_dir", 7, "/tmp/forged"),
        ):
            fields = list(base)
            fields[index] = value
            forged_rows[name] = fields
        wrapper = list(base)
        wrapper[6] = wrapper[6].replace("scripts/fixture.sh", "scripts/forged.sh")
        forged_rows["wrapper"] = wrapper
        snapshot = list(base)
        snapshot[6] = snapshot[6].replace(str(self.snapshot), "/tmp/forged-snapshot")
        forged_rows["snapshot_export"] = snapshot

        for name, fields in forged_rows.items():
            with self.subTest(name=name):
                self.sacct_path.write_text("|".join(fields) + "\n", encoding="utf-8")
                with self.assertRaises(analyzer.InvalidEvidence):
                    self.analyze()

    def test_wrapper_tokens_reject_relative_and_absolute_traversal(self) -> None:
        self.experiment["scheduler"]["wrapper"] = {
            "form": "relative_workdir",
            "token": "../outside.sh",
        }
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "without traversal"):
            analyzer._validate_experiment(
                self.job_id, self.experiment, DEFAULT_POLICIES
            )

        outside = self.root / "outside.sh"
        outside.write_text("#!/bin/sh\n", encoding="utf-8")
        self.experiment["scheduler"]["wrapper"] = {
            "form": "absolute_snapshot",
            "token": str(self.snapshot / ".." / "outside.sh"),
        }
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "without traversal"):
            analyzer._validate_experiment(
                self.job_id, self.experiment, DEFAULT_POLICIES
            )

    def test_required_manifest_export_is_exact(self) -> None:
        task = _scalar_task()
        self.write_spec([task], manifest_export_required=True)
        fields = self._completed_scalar_fields()
        fields[6] = fields[6].replace(
            self.experiment["snapshot"]["manifest_sha256"], "0" * 64
        )
        self.sacct_path.write_text("|".join(fields) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "manifest export"):
            self.analyze()

    def test_compressed_row_scheduler_metadata_is_also_validated(self) -> None:
        tasks = [
            _array_task(0, "a", 42, "a-seed42"),
            _array_task(1, "b", 42, "b-seed42"),
        ]
        self.write_spec(tasks)
        self.write_sacct([f"{self.job_id}_[0-1]|PENDING|0:0|0"])
        fields = self.sacct_path.read_text(encoding="utf-8").strip().split("|")
        fields[2] = "h200_dev"
        self.sacct_path.write_text("|".join(fields) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "QOS mismatch"):
            self.analyze()

    def test_scheduler_report_retains_execution_fields(self) -> None:
        self.write_sacct([f"{self.job_id}|PENDING|0:0|0"])
        report = self.analyze()
        record = report["scheduler"]["records"][0]
        self.assertEqual(record["job_name"], "fixture-job")
        self.assertEqual(record["qos"], "h200_mrs_shared")
        self.assertEqual(record["work_dir"], str(self.root))
        self.assertEqual(record["submit_binding"]["wrapper_form"], "relative_workdir")
        self.assertEqual(
            record["submit_binding"]["exports"]["GR_CODE_SNAPSHOT"],
            str(self.snapshot),
        )
        self.assertEqual(
            report["scheduler"]["source_sha256"],
            hashlib.sha256(self.sacct_path.read_bytes()).hexdigest(),
        )


class ScalarIntegrityTest(AnalyzerFixture):
    def setUp(self) -> None:
        super().setUp()
        self.task = self.experiment["tasks"][0]
        self.run_dir = self.add_run(
            self.task, 0, scalars=self.make_scalars(0.2, throughput_by_step=True)
        )
        self.write_sacct([f"{self.job_id}|COMPLETED|0:0|0"])

    def test_exact_tail_final_best_and_throughput_windows(self) -> None:
        scalars = self.scalar_by_directory[self.run_dir]
        scalars["eval_epoch/ndcg@10"] = [
            analyzer.ScalarPoint(step, step / 1000.0) for step in range(101)
        ]

        report = self.analyze()

        metrics = report["runs"][0]["metrics"]
        ndcg = metrics["eval_epoch/ndcg@10"]
        self.assertAlmostEqual(ndcg["tail_mean_steps_96_100"], 0.098)
        self.assertAlmostEqual(ndcg["final_step_100"], 0.1)
        self.assertAlmostEqual(ndcg["best_steps_0_100"], 0.1)
        self.assertEqual(ndcg["best_step"], 100)
        self.assertAlmostEqual(
            metrics["performance/train_examples_per_second"][
                "arithmetic_mean_of_epoch_rates_steps_1_100"
            ],
            50.5,
        )
        self.assertAlmostEqual(
            metrics["performance/train_examples_per_second"][
                "harmonic_mean_equal_work_steps_1_100"
            ],
            statistics.harmonic_mean(range(1, 101)),
        )

    def test_missing_duplicate_conflicting_nonfinite_and_extra_steps_fail(self) -> None:
        tag = "eval_epoch/ndcg@10"
        original = list(self.scalar_by_directory[self.run_dir][tag])
        mutations = {
            "missing": original[:-1],
            "duplicate": original + [analyzer.ScalarPoint(100, 0.2)],
            "conflicting": original + [analyzer.ScalarPoint(100, 0.3)],
            "nonfinite": original[:-1] + [analyzer.ScalarPoint(100, math.inf)],
            "extra": original + [analyzer.ScalarPoint(101, 0.2)],
        }
        for name, points in mutations.items():
            with self.subTest(name=name):
                self.scalar_by_directory[self.run_dir][tag] = points
                with self.assertRaises(analyzer.InvalidEvidence):
                    self.analyze()
                self.scalar_by_directory[self.run_dir][tag] = list(original)

    def test_missing_required_tag_fails(self) -> None:
        del self.scalar_by_directory[self.run_dir]["eval_epoch/hr@10"]
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "missing required"):
            self.analyze()

    def test_metric_domains_reject_invalid_rates(self) -> None:
        cases = (
            ("eval_epoch/ndcg@10", -0.001, "outside"),
            ("eval_epoch/hr@10", 1.001, "outside"),
            ("eval_epoch/mrr", -0.001, "outside"),
            ("performance/train_examples_per_second", 0.0, "non-positive"),
        )
        for tag, value, message in cases:
            with self.subTest(tag=tag, value=value):
                original = self.scalar_by_directory[self.run_dir][tag][0]
                self.scalar_by_directory[self.run_dir][tag][0] = analyzer.ScalarPoint(
                    0, value
                )
                with self.assertRaisesRegex(analyzer.InvalidEvidence, message):
                    self.analyze()
                self.scalar_by_directory[self.run_dir][tag][0] = original

    def test_multiple_matching_run_directories_fail(self) -> None:
        self.add_run(self.task, 0, prefix="second")
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "found 2"):
            self.analyze()

    def test_symlink_event_file_fails(self) -> None:
        event = next(self.run_dir.glob("events.out.tfevents*"))
        target = self.root / "event-target"
        target.write_bytes(event.read_bytes())
        event.unlink()
        event.symlink_to(target)
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "non-symlink regular"):
            self.analyze()

    def test_event_change_during_load_fails(self) -> None:
        event = next(self.run_dir.glob("events.out.tfevents*"))

        def mutating_loader(directory: Path) -> Mapping[str, Sequence[Any]]:
            event.write_bytes(b"changed")
            return self.scalar_by_directory[directory]

        with self.assertRaisesRegex(analyzer.InvalidEvidence, "changed while"):
            analyzer.analyze_experiment(
                self.spec_path,
                self.job_id,
                sacct_file=self.sacct_path,
                skip_full_snapshot_check=True,
                scalar_loader=mutating_loader,
            )


class AggregationAndPolicyTest(AnalyzerFixture):
    def _paired(self, deltas: Sequence[float]) -> Dict[str, Any]:
        seeds = list(range(42, 42 + len(deltas)))
        return {
            "n_pairs": len(deltas),
            "mean_delta": statistics.fmean(deltas),
            "sample_sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "min_delta": min(deltas),
            "count_positive": sum(delta > 0.0 for delta in deltas),
            "positive_fraction": sum(delta > 0.0 for delta in deltas) / len(deltas),
            "candidate_seeds": seeds,
            "baseline_seeds": seeds,
            "paired_seeds": seeds,
            "candidate_only_seeds": [],
            "baseline_only_seeds": [],
        }

    def test_seed_pairing_is_task_order_independent_and_uses_sample_sd(self) -> None:
        tasks = [
            _array_task(5, "candidate", 44, "candidate-seed44"),
            _array_task(0, "baseline", 42, "baseline-seed42"),
            _array_task(3, "candidate", 42, "candidate-seed42"),
            _array_task(2, "baseline", 44, "baseline-seed44"),
            _array_task(4, "candidate", 43, "candidate-seed43"),
            _array_task(1, "baseline", 43, "baseline-seed43"),
        ]
        comparison = {
            "id": "candidate_vs_baseline",
            "candidate": "candidate",
            "baseline": "baseline",
            "policy": "none",
        }
        self.write_spec(tasks, [comparison])
        candidate_values = {42: 0.101, 43: 0.103, 44: 0.105}
        for task in tasks:
            value = (
                candidate_values[task["seed"]]
                if task["variant"] == "candidate"
                else 0.1
            )
            self.add_run(task, 0, scalars=self.make_scalars(value))
        self.write_sacct(
            [
                f"{self.job_id}_{task['task_id']}|COMPLETED|0:0|0"
                for task in reversed(tasks)
            ]
        )

        report = self.analyze()

        paired = report["comparisons"][0]["paired"]
        self.assertEqual([pair["seed"] for pair in paired["pairs"]], [42, 43, 44])
        self.assertAlmostEqual(paired["mean_delta"], 0.003)
        self.assertAlmostEqual(paired["sample_sd"], 0.002)
        candidate_aggregate = report["aggregates"]["candidate"]["metrics"][
            "eval_epoch/ndcg@10"
        ]["tail_mean_steps_96_100"]
        self.assertAlmostEqual(candidate_aggregate["mean"], 0.103)
        self.assertAlmostEqual(candidate_aggregate["sample_sd"], 0.002)

    def test_tail_screen_inclusive_boundary_passes(self) -> None:
        decision = analyzer._apply_policy(
            DEFAULT_POLICIES["tail"],
            self._paired([-0.001, 0.003, 0.004]),
            [42, 43, 44],
        )
        self.assertEqual(decision["status"], "PASS")
        self.assertTrue(all(check["passed"] for check in decision["checks"].values()))

    def test_tail_screen_fails_each_gate_independently(self) -> None:
        cases = {
            "mean": [-0.001, 0.002, 0.0049],
            "positive": [-0.001, -0.001, 0.01],
            "minimum": [-0.0011, 0.0036, 0.0036],
            "pairs": [0.003, 0.003],
        }
        for gate, deltas in cases.items():
            with self.subTest(gate=gate):
                decision = analyzer._apply_policy(
                    DEFAULT_POLICIES["tail"],
                    self._paired(deltas),
                    [42, 43, 44],
                )
                self.assertEqual(decision["status"], "FAIL")

    def test_signed_geometry_requires_strict_tanh_win_in_two_of_three(self) -> None:
        passing = analyzer._apply_policy(
            DEFAULT_POLICIES["geometry"],
            self._paired([0.0, 0.001, 0.002]),
            [42, 43, 44],
        )
        failing = analyzer._apply_policy(
            DEFAULT_POLICIES["geometry"],
            self._paired([0.0, 0.0, 0.002]),
            [42, 43, 44],
        )
        self.assertEqual(passing["status"], "PASS")
        self.assertEqual(failing["status"], "FAIL")

    def test_fixed_screen_requires_exact_declared_seed_sets(self) -> None:
        paired = self._paired([0.003, 0.003, 0.003])
        paired["candidate_seeds"] = [42, 43, 45]
        paired["paired_seeds"] = [42, 43]
        paired["candidate_only_seeds"] = [45]
        paired["n_pairs"] = 2
        decision = analyzer._apply_policy(
            DEFAULT_POLICIES["tail"], paired, [42, 43, 44]
        )
        self.assertEqual(decision["status"], "FAIL")
        self.assertFalse(decision["checks"]["required_pairs"]["passed"])
        self.assertFalse(decision["checks"]["candidate_seed_set"]["passed"])
        self.assertFalse(decision["checks"]["paired_seed_set"]["passed"])
        self.assertFalse(decision["checks"]["no_unpaired_seeds"]["passed"])

    def test_one_seed_and_unspecified_rules_fail_closed(self) -> None:
        one_pair = self._paired([0.1])
        self.assertEqual(
            analyzer._apply_policy(DEFAULT_POLICIES["softmax"], one_pair)["status"],
            "INSUFFICIENT_SEEDS",
        )
        self.assertEqual(
            analyzer._apply_policy(DEFAULT_POLICIES["none"], one_pair)["status"],
            "NO_FIXED_DECISION_RULE",
        )
        self.assertEqual(
            analyzer._apply_policy(DEFAULT_POLICIES["cross"], None)["status"],
            "INELIGIBLE_CROSS_SNAPSHOT",
        )

    def test_quadratic_oracle_is_not_linear_throughput_claim_eligible(self) -> None:
        task = _scalar_task("oracle", 42, "oracle-seed42")
        self.write_spec(
            [task],
            variant_metadata={
                "oracle": {
                    "complexity": "quadratic_oracle",
                    "linear_throughput_claim_eligible": False,
                }
            },
        )
        self.add_run(task, 0)
        self.write_sacct([f"{self.job_id}|COMPLETED|0:0|0"])
        report = self.analyze()
        aggregate = report["aggregates"]["oracle"]
        self.assertEqual(aggregate["complexity"], "quadratic_oracle")
        self.assertFalse(aggregate["linear_throughput_claim_eligible"])


class ProvenanceAndOutputTest(AnalyzerFixture):
    def _install_legacy_snapshot_metadata(self) -> Dict[str, str]:
        contents = {
            "GIT_COMMIT": b"commit\n",
            "GIT_STATUS": b"status\n",
            "WORKTREE.patch": b"patch\n",
        }
        mapping: Dict[str, str] = {}
        for name, content in contents.items():
            (self.snapshot / name).write_bytes(content)
            mapping[name] = hashlib.sha256(content).hexdigest()
        self.experiment["snapshot"]["legacy_unmanifested_files"] = mapping
        return mapping

    def test_snapshot_manifest_hash_mismatch_fails(self) -> None:
        task = self.experiment["tasks"][0]
        self.write_spec([task], manifest_sha256="0" * 64)
        self.write_sacct([f"{self.job_id}|PENDING|0:0|0"])
        with self.assertRaisesRegex(
            analyzer.InvalidEvidence, "manifest SHA256 mismatch"
        ):
            self.analyze()

    def test_snapshot_root_and_manifest_symlinks_fail(self) -> None:
        symlink_root = self.root / "snapshot-link"
        symlink_root.symlink_to(self.snapshot, target_is_directory=True)
        task = self.experiment["tasks"][0]
        self.write_spec([task], snapshot_root=symlink_root)
        self.write_sacct([f"{self.job_id}|PENDING|0:0|0"])
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "snapshot root"):
            self.analyze()

        symlink_root.unlink()
        manifest_target = self.root / "manifest-target"
        manifest_target.write_bytes(self.manifest.read_bytes())
        self.manifest.unlink()
        self.manifest.symlink_to(manifest_target)
        self.write_spec(
            [task],
            manifest_sha256=hashlib.sha256(manifest_target.read_bytes()).hexdigest(),
        )
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "snapshot manifest"):
            self.analyze()

    def test_full_snapshot_check_detects_changed_source(self) -> None:
        task = self.experiment["tasks"][0]
        self.write_sacct([f"{self.job_id}|PENDING|0:0|0"])
        self.snapshot_data.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(
            analyzer.InvalidEvidence, "checksum validation failed"
        ):
            self.analyze(skip_full_snapshot_check=False)

    def test_full_snapshot_tree_rejects_extra_and_missing_nodes(self) -> None:
        snapshot = self.experiment["snapshot"]

        extra_file = self.snapshot / "extra.txt"
        extra_file.write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "file set mismatch"):
            analyzer._verify_snapshot(snapshot, False)
        extra_file.unlink()

        extra_directory = self.snapshot / "empty"
        extra_directory.mkdir()
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "directory set mismatch"):
            analyzer._verify_snapshot(snapshot, False)
        extra_directory.rmdir()

        original = self.snapshot_data.read_bytes()
        self.snapshot_data.unlink()
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "file set mismatch"):
            analyzer._verify_snapshot(snapshot, False)
        self.snapshot_data.write_bytes(original)

    def test_full_snapshot_tree_rejects_symlink_and_fifo(self) -> None:
        snapshot = self.experiment["snapshot"]
        symlink = self.snapshot / "linked"
        symlink.symlink_to(self.snapshot_data)
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "contains symlink"):
            analyzer._verify_snapshot(snapshot, False)
        symlink.unlink()

        fifo = self.snapshot / "pipe"
        os.mkfifo(fifo)
        try:
            with self.assertRaisesRegex(analyzer.InvalidEvidence, "special node"):
                analyzer._verify_snapshot(snapshot, False)
        finally:
            fifo.unlink()

    def test_legacy_unmanifested_exception_is_exact_hashed_and_visible(self) -> None:
        mapping = self._install_legacy_snapshot_metadata()
        provenance = analyzer._verify_snapshot(self.experiment["snapshot"], False)
        self.assertTrue(provenance["legacy_exception_applied"])
        self.assertEqual(provenance["legacy_unmanifested_files"], mapping)

        (self.snapshot / "GIT_STATUS").write_text("mutated\n", encoding="utf-8")
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "hash mismatch"):
            analyzer._verify_snapshot(self.experiment["snapshot"], False)

        (self.snapshot / "GIT_STATUS").write_text("status\n", encoding="utf-8")
        self.experiment["snapshot"]["legacy_unmanifested_files"]["GIT_STATUS"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "hash mismatch"):
            analyzer._verify_snapshot(self.experiment["snapshot"], False)

    def test_legacy_unmanifested_exception_rejects_missing_and_extra_key(self) -> None:
        mapping = self._install_legacy_snapshot_metadata()
        (self.snapshot / "WORKTREE.patch").unlink()
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "missing legacy"):
            analyzer._verify_snapshot(self.experiment["snapshot"], False)

        self.experiment["snapshot"]["legacy_unmanifested_files"] = {
            **mapping,
            "extra.txt": "0" * 64,
        }
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "must declare exactly"):
            analyzer._validate_experiment(
                self.job_id, self.experiment, DEFAULT_POLICIES
            )

    def test_each_injection_marks_evidence_test_only(self) -> None:
        production = analyzer._evidence_descriptor(None, False, None)
        self.assertEqual(production["evidence_class"], "PRODUCTION")
        self.assertTrue(production["scientific_evidence_eligible"])
        for kwargs in (
            {"sacct_file": self.sacct_path, "skip": False, "loader": None},
            {"sacct_file": None, "skip": True, "loader": None},
            {"sacct_file": None, "skip": False, "loader": self.scalar_loader},
        ):
            with self.subTest(kwargs=kwargs):
                evidence = analyzer._evidence_descriptor(
                    kwargs["sacct_file"], kwargs["skip"], kwargs["loader"]
                )
                self.assertEqual(evidence["evidence_class"], "TEST_ONLY")
                self.assertFalse(evidence["scientific_evidence_eligible"])

    def test_atomic_json_replaces_regular_file(self) -> None:
        output = self.root / "atomic.json"
        output.write_text("old\n", encoding="utf-8")
        analyzer._write_json(output, {"value": 1})
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"value": 1})
        self.assertEqual(list(self.root.glob(".atomic.json.tmp.*")), [])

    def test_atomic_json_rejects_symlink_destination(self) -> None:
        target = self.root / "target.json"
        target.write_text("unchanged\n", encoding="utf-8")
        output = self.root / "linked.json"
        output.symlink_to(target)
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "non-symlink"):
            analyzer._write_json(output, {"value": 1})
        self.assertEqual(target.read_text(encoding="utf-8"), "unchanged\n")

    def test_atomic_json_cleans_temp_and_preserves_destination_on_failure(self) -> None:
        output = self.root / "atomic.json"
        output.write_text("old\n", encoding="utf-8")
        with mock.patch.object(analyzer.os, "replace", side_effect=OSError("injected")):
            with self.assertRaisesRegex(analyzer.InvalidEvidence, "injected"):
                analyzer._write_json(output, {"value": 1})
        self.assertEqual(output.read_text(encoding="utf-8"), "old\n")
        self.assertEqual(list(self.root.glob(".atomic.json.tmp.*")), [])

    def test_json_rendering_is_deterministic(self) -> None:
        task = self.experiment["tasks"][0]
        self.add_run(task, 0)
        self.write_sacct([f"{self.job_id}|COMPLETED|0:0|0"])
        first = self.analyze()
        second = self.analyze()
        self.assertEqual(analyzer.render_json(first), analyzer.render_json(second))
        self.assertEqual(
            first["spec"]["sha256"],
            hashlib.sha256(self.spec_path.read_bytes()).hexdigest(),
        )

    def test_cli_exit_codes_are_complete_incomplete_invalid(self) -> None:
        task = self.experiment["tasks"][0]
        self.add_run(task, 0)
        self.write_sacct([f"{self.job_id}|COMPLETED|0:0|0"])
        output = self.root / "result.json"
        args = [
            "--spec",
            str(self.spec_path),
            "--experiment",
            self.job_id,
            "--sacct-file",
            str(self.sacct_path),
            "--json-out",
            str(output),
            "--skip-full-snapshot-check",
        ]
        with mock.patch.object(
            analyzer, "_read_tensorboard_scalars", self.scalar_loader
        ):
            with mock.patch("builtins.print"):
                self.assertEqual(analyzer.main(args), 4)
        complete_test = json.loads(output.read_text())
        self.assertEqual(complete_test["status"], "COMPLETE_TEST_ONLY")
        self.assertFalse(complete_test["decision"]["evidence_eligible"])

        self.write_sacct([f"{self.job_id}|PENDING|0:0|0"])
        with mock.patch("builtins.print"):
            self.assertEqual(analyzer.main(args), 4)
        self.assertEqual(
            json.loads(output.read_text())["status"], "INCOMPLETE_TEST_ONLY"
        )

        self.write_spec([task], manifest_sha256="0" * 64)
        with mock.patch("builtins.print"):
            self.assertEqual(analyzer.main(args), 2)
        invalid = json.loads(output.read_text())
        self.assertEqual(invalid["status"], "INVALID")
        self.assertIsNone(invalid["decision"])
        self.assertEqual(invalid["evidence"]["evidence_class"], "TEST_ONLY")

    def test_fail_on_decision_exit_is_opt_in_and_distinct(self) -> None:
        production_report = {
            "status": "COMPLETE",
            "evidence": {"evidence_class": "PRODUCTION"},
            "comparisons": [
                {"decision": {"status": "FAIL", "evidence_eligible": True}}
            ],
        }
        self.assertEqual(analyzer._report_exit_code(production_report, False), 0)
        self.assertEqual(analyzer._report_exit_code(production_report, True), 5)
        production_report["comparisons"][0]["decision"]["status"] = "PASS"
        self.assertEqual(analyzer._report_exit_code(production_report, True), 0)
        parsed = analyzer._parse_args(
            ["--experiment", self.job_id, "--fail-on-decision"]
        )
        self.assertTrue(parsed.fail_on_decision)


class RegistryTest(unittest.TestCase):
    def test_checked_in_registry_has_exact_experiment_shapes(self) -> None:
        path = Path(__file__).with_name("attention_result_specs.json")
        registry = json.loads(path.read_text(encoding="utf-8"))
        experiments = registry["experiments"]
        self.assertEqual(
            set(experiments),
            {"1671577", "1671578", "1671931", "1675026", "1675323", "1675705"},
        )
        self.assertEqual(len(experiments["1671578"]["tasks"]), 15)
        self.assertEqual(len(experiments["1671577"]["tasks"]), 4)
        self.assertEqual(len(experiments["1671931"]["tasks"]), 3)
        self.assertEqual(len(experiments["1675026"]["tasks"]), 1)
        self.assertEqual(len(experiments["1675323"]["tasks"]), 12)
        self.assertEqual(len(experiments["1675705"]["tasks"]), 12)
        for experiment in experiments.values():
            self.assertEqual(experiment["expected_steps"], {"start": 0, "end": 100})
            self.assertRegex(
                experiment["snapshot"]["manifest_sha256"], r"^[0-9a-f]{64}$"
            )
            self.assertTrue(experiment["run_root"].endswith("-l200"))
            variants = {task["variant"] for task in experiment["tasks"]}
            self.assertEqual(set(experiment["variant_metadata"]), variants)
            self.assertEqual(experiment["overall_policy"], "per_comparison_only")
            self.assertEqual(experiment["scheduler"]["qos"], "h200_mrs_shared")
            self.assertEqual(
                experiment["scheduler"]["snapshot_export"], "GR_CODE_SNAPSHOT"
            )
        for experiment_id, experiment in experiments.items():
            analyzer._validate_experiment(
                experiment_id, experiment, registry["policies"]
            )
        self.assertTrue(experiments["1675026"]["tasks"][0]["scalar"])
        self.assertFalse(
            experiments["1675323"]["variant_metadata"]["abscoef"][
                "linear_throughput_claim_eligible"
            ]
        )
        linear_eligible = {
            (experiment_id, variant)
            for experiment_id, experiment in experiments.items()
            for variant, metadata in experiment["variant_metadata"].items()
            if metadata["linear_throughput_claim_eligible"]
        }
        self.assertEqual(
            linear_eligible,
            {
                ("1675323", "identity"),
                ("1675323", "tanh"),
                ("1675323", "abs_tanh"),
            },
        )
        self.assertEqual(
            registry["policies"]["softmax_descriptive"]["minimum_pairs"], 3
        )
        self.assertEqual(
            experiments["1671578"]["screen_provenance"],
            "posthoc_fixed_screen",
        )
        self.assertEqual(
            experiments["1675705"]["screen_provenance"],
            "pre_submission_fixed_screen",
        )
        expected_legacy = {"GIT_COMMIT", "GIT_STATUS", "WORKTREE.patch"}
        self.assertEqual(
            set(experiments["1671577"]["snapshot"]["legacy_unmanifested_files"]),
            expected_legacy,
        )
        self.assertEqual(
            set(experiments["1671578"]["snapshot"]["legacy_unmanifested_files"]),
            expected_legacy,
        )
        for experiment_id in ("1671931", "1675026", "1675323", "1675705"):
            self.assertEqual(
                experiments[experiment_id]["snapshot"]["legacy_unmanifested_files"],
                {},
            )
        self.assertNotIn("preregister", path.read_text(encoding="utf-8").lower())

    def test_screen_provenance_must_match_fixed_policy_presence(self) -> None:
        path = Path(__file__).with_name("attention_result_specs.json")
        registry = json.loads(path.read_text(encoding="utf-8"))

        fixed_but_declared_none = json.loads(
            json.dumps(registry["experiments"]["1671578"])
        )
        fixed_but_declared_none["screen_provenance"] = "no_fixed_screen"
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "cannot declare"):
            analyzer._validate_experiment(
                "1671578", fixed_but_declared_none, registry["policies"]
            )

        no_fixed_but_declared_fixed = json.loads(
            json.dumps(registry["experiments"]["1671931"])
        )
        no_fixed_but_declared_fixed["screen_provenance"] = "posthoc_fixed_screen"
        with self.assertRaisesRegex(analyzer.InvalidEvidence, "requires a fixed"):
            analyzer._validate_experiment(
                "1671931", no_fixed_but_declared_fixed, registry["policies"]
            )


@unittest.skipUnless(
    os.environ.get("GR_RUN_ATTENTION_RESULTS_INTEGRATION") == "1",
    "set GR_RUN_ATTENTION_RESULTS_INTEGRATION=1 to read completed shared results",
)
class CompletedML1IntegrationTest(unittest.TestCase):
    def test_job_1671578_reproduces_paper_tail_values(self) -> None:
        report = analyzer.analyze_experiment(
            Path(__file__).with_name("attention_result_specs.json"), "1671578"
        )
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["evidence"]["evidence_class"], "PRODUCTION")
        self.assertTrue(report["evidence"]["scientific_evidence_eligible"])
        self.assertEqual(report["snapshot"]["full_snapshot_check"], "PASSED")
        aggregates = report["aggregates"]
        local = aggregates["local_w32"]["metrics"]["eval_epoch/ndcg@10"][
            "tail_mean_steps_96_100"
        ]["mean"]
        lift = aggregates["lift_w32"]["metrics"]["eval_epoch/ndcg@10"][
            "tail_mean_steps_96_100"
        ]["mean"]
        w32_comparison = next(
            item
            for item in report["comparisons"]
            if item["id"] == "lift_w32_vs_local_w32"
        )
        w64_comparison = next(
            item
            for item in report["comparisons"]
            if item["id"] == "lift_w64_vs_local_w64"
        )
        self.assertAlmostEqual(local, 0.19297, delta=0.00002)
        self.assertAlmostEqual(lift, 0.19511, delta=0.00002)
        self.assertAlmostEqual(
            w32_comparison["paired"]["mean_delta"], 0.002147, delta=2e-6
        )
        self.assertEqual(w32_comparison["decision"]["status"], "PASS")
        self.assertAlmostEqual(
            w64_comparison["paired"]["mean_delta"], 0.00027838, delta=2e-7
        )
        self.assertAlmostEqual(
            w64_comparison["paired"]["min_delta"], -0.00064756, delta=2e-7
        )
        self.assertEqual(w64_comparison["paired"]["count_positive"], 2)
        self.assertEqual(w64_comparison["paired"]["n_pairs"], 3)
        self.assertEqual(w64_comparison["decision"]["status"], "FAIL")
        self.assertEqual(report["decision"]["status"], "PER_COMPARISON_ONLY")
        self.assertEqual(analyzer._report_exit_code(report, False), 0)
        self.assertEqual(analyzer._report_exit_code(report, True), 5)


if __name__ == "__main__":
    unittest.main()
