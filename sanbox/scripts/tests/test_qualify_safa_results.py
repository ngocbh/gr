#!/usr/bin/env python3

import contextlib
import copy
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.qualify_safa_results import (
    EXPECTED_PARAMETER_INVENTORIES,
    EXPECTED_PARAMETER_COUNTS,
    ResultsError,
    load_results,
    main,
    operative_config_identities,
    parse_sacct_receipt,
    qualify_results,
    query_sacct_receipt,
)


DATASET = "ml-1m"
COMMIT = "a" * 40
TREE = "b" * 40
MANIFEST = "c" * 64
INVENTORY = EXPECTED_PARAMETER_INVENTORIES[DATASET]
ARRAY_JOB_ID = "123456"


def _operative_config(seed, arm, learning_rate=0.001, dropout_rate=0.2):
    return (
        "# Parameters for hstu_encoder:\n"
        "# ==============================================================================\n"
        f"hstu_encoder.attention_mode = '{arm}'\n"
        "\n"
        "# Parameters for train_fn:\n"
        "# ==============================================================================\n"
        f"train_fn.dropout_rate = {dropout_rate}\n"
        f"train_fn.learning_rate = {learning_rate}\n"
        f"train_fn.random_seed = {seed}\n"
    )


def _set_config_identity(run, resolved_gin_config):
    resolved_sha256, experiment_sha256 = operative_config_identities(
        resolved_gin_config,
        attention_mode=run["attention_mode"],
        random_seed=run["random_seed"],
    )
    run["resolved_gin_config"] = resolved_gin_config
    run["resolved_gin_config_sha256"] = resolved_sha256
    run["experiment_config_sha256"] = experiment_sha256


_, EXPECTED_EXPERIMENT_CONFIG_SHA256 = operative_config_identities(
    _operative_config(42, "hstu"),
    attention_mode="hstu",
    random_seed=42,
)


def _document(deltas=None):
    if deltas is None:
        deltas = {42: 0.003, 43: 0.003, 44: 0.0}
    runs = []
    for task_id, (seed, arm) in enumerate(
        (
            (42, "hstu"),
            (42, "safa"),
            (43, "hstu"),
            (43, "safa"),
            (44, "hstu"),
            (44, "safa"),
        )
    ):
        value = 0.5 if arm == "hstu" else 0.5 + deltas[seed]
        run = {
            "dataset": DATASET,
            "source_commit": COMMIT,
            "source_tree": TREE,
            "source_manifest": MANIFEST,
            "parameter_inventory_sha256": INVENTORY,
            "parameter_count": 313416,
            "metric": "ndcg@10",
            "seed": seed,
            "arm": arm,
            "attention_mode": arm,
            "random_seed": seed,
            "slurm_array_job_id": ARRAY_JOB_ID,
            "slurm_array_task_id": task_id,
            "slurm_job_id": str(123456 + task_id),
            "slurm_job_qos": "h200_dev",
            "slurm_restart_count": 0,
            "slurm_job_partition": "h200",
            "epochs": [{"epoch": epoch, "value": value} for epoch in range(96, 101)],
        }
        _set_config_identity(run, _operative_config(seed, arm))
        runs.append(run)
    return {"schema_version": 3, "runs": runs}


def _sacct_output(
    *,
    states=None,
    exit_codes=None,
    restarts=None,
    qos=None,
    partitions=None,
    raw_job_ids=None,
):
    states = states or {}
    exit_codes = exit_codes or {}
    restarts = restarts or {}
    qos = qos or {}
    partitions = partitions or {}
    raw_job_ids = raw_job_ids or {}
    return "\n".join(
        "|".join(
            (
                str(raw_job_ids.get(task_id, 123456 + task_id)),
                f"{ARRAY_JOB_ID}_{task_id}",
                states.get(task_id, "COMPLETED"),
                exit_codes.get(task_id, "0:0"),
                str(restarts.get(task_id, 0)),
                qos.get(task_id, "h200_dev"),
                partitions.get(task_id, "h200"),
            )
        )
        for task_id in range(6)
    )


def _scheduler_receipt(*, expected_qos="h200_dev", **kwargs):
    return parse_sacct_receipt(
        _sacct_output(**kwargs),
        expected_array_job_id=ARRAY_JOB_ID,
        expected_qos=expected_qos,
    )


def _qualify(document, scheduler_receipt=None):
    if scheduler_receipt is None:
        scheduler_receipt = _scheduler_receipt()
    return qualify_results(
        document,
        expected_dataset=DATASET,
        expected_source_commit=COMMIT,
        expected_source_tree=TREE,
        expected_source_manifest=MANIFEST,
        expected_experiment_config_sha256=EXPECTED_EXPERIMENT_CONFIG_SHA256,
        expected_array_job_id=ARRAY_JOB_ID,
        scheduler_receipt=scheduler_receipt,
    )


class SacctReceiptTest(unittest.TestCase):
    def test_valid_receipt_preserves_authoritative_fields(self) -> None:
        receipt = _scheduler_receipt(restarts={3: 2})
        self.assertEqual(set(receipt), set(range(6)))
        self.assertEqual(receipt[3]["slurm_job_id"], "123459")
        self.assertEqual(receipt[3]["slurm_restart_count"], 2)
        self.assertEqual(receipt[3]["slurm_job_qos"], "h200_dev")
        self.assertEqual(receipt[3]["slurm_job_partition"], "h200")

    def test_missing_extra_duplicate_aggregate_or_malformed_rows_are_rejected(
        self,
    ) -> None:
        valid_lines = _sacct_output().splitlines()
        cases = {
            "missing": "\n".join(valid_lines[:-1]),
            "extra": _sacct_output()
            + f"\n999999|{ARRAY_JOB_ID}_6|COMPLETED|0:0|0|h200_dev|h200",
            "duplicate": _sacct_output() + f"\n{valid_lines[0]}",
            "aggregate": (
                f"{ARRAY_JOB_ID}|{ARRAY_JOB_ID}_[0-5]|COMPLETED|0:0|0|" "h200_dev|h200"
            ),
            "malformed": "too|few|fields",
            "blank": "",
        }
        for label, output in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ResultsError):
                    parse_sacct_receipt(
                        output,
                        expected_array_job_id=ARRAY_JOB_ID,
                        expected_qos="h200_dev",
                    )

    def test_nonterminal_or_wrong_scheduler_values_are_rejected(self) -> None:
        cases = (
            {"states": {0: "RUNNING"}},
            {"states": {0: "FAILED"}},
            {"exit_codes": {0: "1:0"}},
            {"restarts": {0: "-1"}},
            {"qos": {0: "h200_mrs_shared"}},
            {"partitions": {0: "debug"}},
            {"raw_job_ids": {1: 123456}},
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ResultsError):
                    _scheduler_receipt(**mutation)

    def test_query_invokes_allocation_only_sacct_and_fails_closed(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_sacct_output(), stderr=""
        )
        with mock.patch(
            "scripts.qualify_safa_results.subprocess.run", return_value=completed
        ) as run:
            receipt = query_sacct_receipt(
                ARRAY_JOB_ID,
                expected_qos="h200_dev",
            )
        self.assertEqual(set(receipt), set(range(6)))
        command = run.call_args.args[0]
        self.assertEqual(command[0], "sacct")
        self.assertIn("--allocations", command)
        self.assertIn(
            "--format=JobIDRaw,JobID,State,ExitCode,Restarts,QOS,Partition", command
        )

        failures = (
            FileNotFoundError("sacct unavailable"),
            subprocess.TimeoutExpired("sacct", 30),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), mock.patch(
                "scripts.qualify_safa_results.subprocess.run", side_effect=failure
            ):
                with self.assertRaisesRegex(ResultsError, "could not query sacct"):
                    query_sacct_receipt(
                        ARRAY_JOB_ID,
                        expected_qos="h200_dev",
                    )

        failed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="accounting unavailable"
        )
        with mock.patch(
            "scripts.qualify_safa_results.subprocess.run", return_value=failed
        ):
            with self.assertRaisesRegex(ResultsError, "sacct query failed"):
                query_sacct_receipt(
                    ARRAY_JOB_ID,
                    expected_qos="h200_dev",
                )


class ThresholdTest(unittest.TestCase):
    def test_ml20_high_qos_passes_provenance_gate(self) -> None:
        document = _document()
        for run in document["runs"]:
            run["dataset"] = "ml-20m"
            run["parameter_count"] = EXPECTED_PARAMETER_COUNTS["ml-20m"]
            run["parameter_inventory_sha256"] = EXPECTED_PARAMETER_INVENTORIES["ml-20m"]
            run["slurm_job_qos"] = "h200_mrs_2_high"
        receipt = _scheduler_receipt(
            expected_qos="h200_mrs_2_high",
            qos={task_id: "h200_mrs_2_high" for task_id in range(6)},
        )
        summary = qualify_results(
            document,
            expected_dataset="ml-20m",
            expected_source_commit=COMMIT,
            expected_source_tree=TREE,
            expected_source_manifest=MANIFEST,
            expected_experiment_config_sha256=EXPECTED_EXPERIMENT_CONFIG_SHA256,
            expected_array_job_id=ARRAY_JOB_ID,
            scheduler_receipt=receipt,
        )
        self.assertTrue(summary["passed"])

    def test_exact_mean_and_two_positive_seed_boundary_passes(self) -> None:
        summary = _qualify(_document({42: 0.003, 43: 0.003, 44: 0.0}))
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["aggregate"]["mean_delta"], 0.002)
        self.assertEqual(summary["aggregate"]["positive_seed_count"], 2)

    def test_exact_minimum_delta_boundary_passes(self) -> None:
        summary = _qualify(_document({42: 0.004, 43: 0.004, 44: -0.001}))
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["aggregate"]["minimum_delta"], -0.001)

    def test_matching_nonzero_restart_count_passes(self) -> None:
        document = _document()
        document["runs"][3]["slurm_restart_count"] = 2
        summary = _qualify(
            document,
            scheduler_receipt=_scheduler_receipt(restarts={3: 2}),
        )
        self.assertTrue(summary["passed"])

    def test_mean_delta_below_threshold_fails(self) -> None:
        summary = _qualify(_document({42: 0.0019, 43: 0.0019, 44: 0.0019}))
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["gate_checks"]["mean_delta"])

    def test_only_one_positive_seed_fails(self) -> None:
        summary = _qualify(_document({42: 0.007, 43: 0.0, 44: 0.0}))
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["gate_checks"]["positive_seeds"])

    def test_minimum_delta_below_threshold_fails(self) -> None:
        summary = _qualify(_document({42: 0.004, 43: 0.004, 44: -0.0011}))
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["gate_checks"]["minimum_delta"])


class InvalidInputTest(unittest.TestCase):
    def test_all_hstu_runs_mislabeled_as_safa_are_rejected(self) -> None:
        document = _document()
        for run in document["runs"]:
            if run["arm"] == "safa":
                run["attention_mode"] = "hstu"
                _set_config_identity(run, _operative_config(run["random_seed"], "hstu"))
        with self.assertRaisesRegex(ResultsError, "attention_mode does not match arm"):
            _qualify(document)

    def test_wrong_recorded_seed_is_rejected(self) -> None:
        document = _document()
        run = document["runs"][0]
        run["random_seed"] = 43
        _set_config_identity(run, _operative_config(43, run["attention_mode"]))
        with self.assertRaisesRegex(ResultsError, "random_seed does not match seed"):
            _qualify(document)

    def test_learning_rate_or_dropout_config_drift_is_rejected(self) -> None:
        for field, learning_rate, dropout_rate in (
            ("learning rate", 0.002, 0.2),
            ("dropout rate", 0.001, 0.3),
        ):
            with self.subTest(field=field):
                document = _document()
                run = document["runs"][1]
                _set_config_identity(
                    run,
                    _operative_config(
                        run["random_seed"],
                        run["attention_mode"],
                        learning_rate=learning_rate,
                        dropout_rate=dropout_rate,
                    ),
                )
                with self.assertRaisesRegex(ResultsError, "experiment_config_sha256"):
                    _qualify(document)

    def test_uniform_learning_rate_or_dropout_drift_is_rejected(self) -> None:
        for field, learning_rate, dropout_rate in (
            ("learning rate", 0.002, 0.2),
            ("dropout rate", 0.001, 0.3),
        ):
            with self.subTest(field=field):
                document = _document()
                for run in document["runs"]:
                    _set_config_identity(
                        run,
                        _operative_config(
                            run["random_seed"],
                            run["attention_mode"],
                            learning_rate=learning_rate,
                            dropout_rate=dropout_rate,
                        ),
                    )
                with self.assertRaisesRegex(ResultsError, "externally pinned"):
                    _qualify(document)

    def test_missing_seed_or_arm_is_rejected(self) -> None:
        document = _document()
        document["runs"].pop()
        with self.assertRaisesRegex(ResultsError, "exactly six"):
            _qualify(document)

    def test_duplicate_seed_arm_is_rejected(self) -> None:
        document = _document()
        document["runs"][-1] = copy.deepcopy(document["runs"][0])
        with self.assertRaisesRegex(ResultsError, "duplicate (run|SLURM)"):
            _qualify(document)

    def test_missing_final_epoch_is_rejected(self) -> None:
        document = _document()
        document["runs"][0]["epochs"].pop()
        with self.assertRaisesRegex(ResultsError, "missing final epochs"):
            _qualify(document)

    def test_duplicate_epoch_is_rejected(self) -> None:
        document = _document()
        document["runs"][0]["epochs"].append(
            copy.deepcopy(document["runs"][0]["epochs"][0])
        )
        with self.assertRaisesRegex(ResultsError, "duplicate epoch"):
            _qualify(document)

    def test_nonfinite_value_is_rejected(self) -> None:
        document = _document()
        document["runs"][0]["epochs"][0]["value"] = float("inf")
        with self.assertRaisesRegex(ResultsError, "must be finite"):
            _qualify(document)

    def test_out_of_range_ndcg_is_rejected(self) -> None:
        document = _document()
        document["runs"][0]["epochs"][0]["value"] = 1.001
        with self.assertRaisesRegex(ResultsError, r"in \[0, 1\]"):
            _qualify(document)

    def test_mismatched_provenance_is_rejected(self) -> None:
        for field, value in (
            ("dataset", "ml-20m"),
            ("source_commit", "e" * 40),
            ("source_tree", "e" * 40),
            ("source_manifest", "e" * 64),
            ("parameter_inventory_sha256", "e" * 64),
            ("parameter_count", 313417),
            ("metric", "hr@10"),
        ):
            with self.subTest(field=field):
                document = _document()
                document["runs"][1][field] = value
                with self.assertRaises(ResultsError):
                    _qualify(document)

    def test_externally_expected_metadata_is_enforced(self) -> None:
        document = _document()
        with self.assertRaisesRegex(ResultsError, "wrong QoS"):
            qualify_results(
                document,
                expected_dataset="ml-20m",
                expected_source_commit=COMMIT,
                expected_source_tree=TREE,
                expected_source_manifest=MANIFEST,
                expected_experiment_config_sha256=(EXPECTED_EXPERIMENT_CONFIG_SHA256),
                expected_array_job_id=ARRAY_JOB_ID,
                scheduler_receipt=_scheduler_receipt(),
            )
        with self.assertRaisesRegex(ResultsError, "externally pinned"):
            qualify_results(
                document,
                expected_dataset=DATASET,
                expected_source_commit=COMMIT,
                expected_source_tree=TREE,
                expected_source_manifest="f" * 64,
                expected_experiment_config_sha256=(EXPECTED_EXPERIMENT_CONFIG_SHA256),
                expected_array_job_id=ARRAY_JOB_ID,
                scheduler_receipt=_scheduler_receipt(),
            )
        with self.assertRaisesRegex(ResultsError, "source commit"):
            qualify_results(
                document,
                expected_dataset=DATASET,
                expected_source_commit="f" * 40,
                expected_source_tree=TREE,
                expected_source_manifest=MANIFEST,
                expected_experiment_config_sha256=(EXPECTED_EXPERIMENT_CONFIG_SHA256),
                expected_array_job_id=ARRAY_JOB_ID,
                scheduler_receipt=_scheduler_receipt(),
            )
        with self.assertRaisesRegex(ResultsError, "source tree"):
            qualify_results(
                document,
                expected_dataset=DATASET,
                expected_source_commit=COMMIT,
                expected_source_tree="f" * 40,
                expected_source_manifest=MANIFEST,
                expected_experiment_config_sha256=(EXPECTED_EXPERIMENT_CONFIG_SHA256),
                expected_array_job_id=ARRAY_JOB_ID,
                scheduler_receipt=_scheduler_receipt(),
            )
        with self.assertRaisesRegex(ResultsError, "experiment config"):
            qualify_results(
                document,
                expected_dataset=DATASET,
                expected_source_commit=COMMIT,
                expected_source_tree=TREE,
                expected_source_manifest=MANIFEST,
                expected_experiment_config_sha256="f" * 64,
                expected_array_job_id=ARRAY_JOB_ID,
                scheduler_receipt=_scheduler_receipt(),
            )

        with self.assertRaisesRegex(ResultsError, "array job ID"):
            qualify_results(
                document,
                expected_dataset=DATASET,
                expected_source_commit=COMMIT,
                expected_source_tree=TREE,
                expected_source_manifest=MANIFEST,
                expected_experiment_config_sha256=EXPECTED_EXPERIMENT_CONFIG_SHA256,
                expected_array_job_id="654321",
                scheduler_receipt=_scheduler_receipt(),
            )

    def test_scheduler_provenance_is_fail_closed(self) -> None:
        mutations = (
            ("array ID", "slurm_array_job_id", "0"),
            ("job ID", "slurm_job_id", "job-123"),
            ("task ID", "slurm_array_task_id", 6),
            ("QoS", "slurm_job_qos", "h200_mrs_shared"),
            ("restart", "slurm_restart_count", -1),
            ("partition", "slurm_job_partition", "debug"),
        )
        for label, field, value in mutations:
            with self.subTest(label=label):
                document = _document()
                document["runs"][0][field] = value
                with self.assertRaises(ResultsError):
                    _qualify(document)

    def test_task_id_must_match_seed_arm_and_be_unique(self) -> None:
        document = _document()
        document["runs"][0]["slurm_array_task_id"] = 1
        with self.assertRaisesRegex(ResultsError, "seed/arm mapping"):
            _qualify(document)

        document = _document()
        document["runs"][1]["slurm_array_task_id"] = 0
        document["runs"][1]["seed"] = 42
        document["runs"][1]["arm"] = "hstu"
        document["runs"][1]["attention_mode"] = "hstu"
        _set_config_identity(document["runs"][1], _operative_config(42, "hstu"))
        with self.assertRaisesRegex(ResultsError, "duplicate run|duplicate SLURM"):
            _qualify(document)

    def test_run_scheduler_identity_must_match_receipt_and_use_distinct_jobs(
        self,
    ) -> None:
        document = _document()
        document["runs"][0]["slurm_restart_count"] = 1
        with self.assertRaisesRegex(ResultsError, "authoritative scheduler receipt"):
            _qualify(document)

        document = _document()
        document["runs"][1]["slurm_job_id"] = document["runs"][0]["slurm_job_id"]
        with self.assertRaisesRegex(ResultsError, "must be distinct"):
            _qualify(document)

    def test_legacy_schema_is_rejected(self) -> None:
        document = _document()
        document["schema_version"] = 2
        with self.assertRaisesRegex(ResultsError, "invalid results document"):
            _qualify(document)

    def test_uniformly_wrong_inventory_is_rejected(self) -> None:
        document = _document()
        for run in document["runs"]:
            run["parameter_count"] = 1
            run["parameter_inventory_sha256"] = "e" * 64
        with self.assertRaisesRegex(ResultsError, "parameter count"):
            _qualify(document)

    def test_noncanonical_git_id_lengths_are_rejected(self) -> None:
        for field, value in (("source_commit", "e" * 41), ("source_tree", "e" * 63)):
            with self.subTest(field=field):
                document = _document()
                for run in document["runs"]:
                    run[field] = value
                with self.assertRaisesRegex(ResultsError, "40-character"):
                    _qualify(document)

    def test_duplicate_json_key_and_nonfinite_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "results.json"
            path.write_text(
                '{"schema_version":3,"schema_version":3,"runs":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultsError, "duplicate JSON key"):
                load_results(path)
            path.write_text(
                '{"schema_version":3,"runs":[{"value":NaN}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultsError, "nonfinite"):
                load_results(path)


class CliTest(unittest.TestCase):
    def _run(self, document):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "results.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output), mock.patch(
                "scripts.qualify_safa_results.query_sacct_receipt",
                return_value=_scheduler_receipt(),
            ) as query:
                exit_code = main(
                    [
                        str(path),
                        "--expected-dataset",
                        DATASET,
                        "--expected-source-commit",
                        COMMIT,
                        "--expected-source-tree",
                        TREE,
                        "--expected-source-manifest",
                        MANIFEST,
                        "--expected-experiment-config-sha256",
                        EXPECTED_EXPERIMENT_CONFIG_SHA256,
                        "--expected-array-job-id",
                        ARRAY_JOB_ID,
                    ]
                )
            query.assert_called_once_with(
                ARRAY_JOB_ID,
                expected_qos="h200_dev",
            )
            return exit_code, json.loads(output.getvalue())

    def test_cli_exit_codes_and_machine_readable_summaries(self) -> None:
        pass_code, pass_summary = self._run(_document())
        self.assertEqual(pass_code, 0)
        self.assertEqual(pass_summary["status"], "pass")
        self.assertRegex(pass_summary["input_sha256"], r"^[0-9a-f]{64}$")

        fail_code, fail_summary = self._run(
            _document({42: 0.001, 43: 0.001, 44: 0.001})
        )
        self.assertEqual(fail_code, 1)
        self.assertEqual(fail_summary["status"], "fail")

        invalid = _document()
        invalid["runs"][0]["epochs"].pop()
        invalid_code, invalid_summary = self._run(invalid)
        self.assertEqual(invalid_code, 2)
        self.assertEqual(invalid_summary["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
