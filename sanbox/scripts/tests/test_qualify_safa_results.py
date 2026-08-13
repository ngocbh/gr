#!/usr/bin/env python3

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.qualify_safa_results import (
    EXPECTED_PARAMETER_INVENTORIES,
    ResultsError,
    load_results,
    main,
    operative_config_identities,
    qualify_results,
)


DATASET = "ml-1m"
COMMIT = "a" * 40
TREE = "b" * 40
MANIFEST = "c" * 64
INVENTORY = EXPECTED_PARAMETER_INVENTORIES[DATASET]


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
    for seed in (42, 43, 44):
        for arm in ("hstu", "safa"):
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
                "epochs": [
                    {"epoch": epoch, "value": value} for epoch in range(96, 101)
                ],
            }
            _set_config_identity(run, _operative_config(seed, arm))
            runs.append(run)
    return {"schema_version": 2, "runs": runs}


def _qualify(document):
    return qualify_results(
        document,
        expected_dataset=DATASET,
        expected_source_commit=COMMIT,
        expected_source_tree=TREE,
        expected_source_manifest=MANIFEST,
        expected_experiment_config_sha256=EXPECTED_EXPERIMENT_CONFIG_SHA256,
    )


class ThresholdTest(unittest.TestCase):
    def test_exact_mean_and_two_positive_seed_boundary_passes(self) -> None:
        summary = _qualify(_document({42: 0.003, 43: 0.003, 44: 0.0}))
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["aggregate"]["mean_delta"], 0.002)
        self.assertEqual(summary["aggregate"]["positive_seed_count"], 2)

    def test_exact_minimum_delta_boundary_passes(self) -> None:
        summary = _qualify(_document({42: 0.004, 43: 0.004, 44: -0.001}))
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["aggregate"]["minimum_delta"], -0.001)

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
        with self.assertRaisesRegex(ResultsError, "duplicate run"):
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
        with self.assertRaisesRegex(ResultsError, "dataset mismatch"):
            qualify_results(
                document,
                expected_dataset="ml-20m",
                expected_source_commit=COMMIT,
                expected_source_tree=TREE,
                expected_source_manifest=MANIFEST,
                expected_experiment_config_sha256=(EXPECTED_EXPERIMENT_CONFIG_SHA256),
            )
        with self.assertRaisesRegex(ResultsError, "externally pinned"):
            qualify_results(
                document,
                expected_dataset=DATASET,
                expected_source_commit=COMMIT,
                expected_source_tree=TREE,
                expected_source_manifest="f" * 64,
                expected_experiment_config_sha256=(EXPECTED_EXPERIMENT_CONFIG_SHA256),
            )
        with self.assertRaisesRegex(ResultsError, "source commit"):
            qualify_results(
                document,
                expected_dataset=DATASET,
                expected_source_commit="f" * 40,
                expected_source_tree=TREE,
                expected_source_manifest=MANIFEST,
                expected_experiment_config_sha256=(EXPECTED_EXPERIMENT_CONFIG_SHA256),
            )
        with self.assertRaisesRegex(ResultsError, "source tree"):
            qualify_results(
                document,
                expected_dataset=DATASET,
                expected_source_commit=COMMIT,
                expected_source_tree="f" * 40,
                expected_source_manifest=MANIFEST,
                expected_experiment_config_sha256=(EXPECTED_EXPERIMENT_CONFIG_SHA256),
            )
        with self.assertRaisesRegex(ResultsError, "experiment config"):
            qualify_results(
                document,
                expected_dataset=DATASET,
                expected_source_commit=COMMIT,
                expected_source_tree=TREE,
                expected_source_manifest=MANIFEST,
                expected_experiment_config_sha256="f" * 64,
            )

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
                '{"schema_version":2,"schema_version":2,"runs":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultsError, "duplicate JSON key"):
                load_results(path)
            path.write_text(
                '{"schema_version":2,"runs":[{"value":NaN}]}',
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
            with contextlib.redirect_stdout(output):
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
                    ]
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
