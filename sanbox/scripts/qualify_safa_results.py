#!/usr/bin/env python3
"""Apply the frozen paired-seed SAFA quality gate to exported run metrics.

Input is a JSON object with ``schema_version: 3`` and six entries in ``runs``.
Each run has the following shape::

    {
      "dataset": "ml-1m",
      "source_commit": "<git commit>",
      "source_tree": "<git tree>",
      "source_manifest": "<snapshot manifest sha256>",
      "parameter_inventory_sha256": "<named parameter inventory sha256>",
      "parameter_count": 313416,
      "metric": "ndcg@10",
      "seed": 42,
      "arm": "hstu",
      "attention_mode": "hstu",
      "random_seed": 42,
      "slurm_array_job_id": "123456",
      "slurm_array_task_id": 0,
      "slurm_job_id": "123456",
      "slurm_job_qos": "h200_dev",
      "slurm_restart_count": 0,
      "slurm_job_partition": "h200",
      "resolved_gin_config": "<operative Gin config>",
      "resolved_gin_config_sha256": "<exact config sha256>",
      "experiment_config_sha256": "<mode/seed-normalized config sha256>",
      "epochs": [{"epoch": 96, "value": 0.19}, ...]
    }

Extra epochs are allowed, but epochs 96 through 100 must be present exactly once.
All provenance and inventory fields must agree across the six runs.
The CLI independently verifies the six completed allocations with ``sacct``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SEEDS = (42, 43, 44)
ARMS = ("hstu", "safa")
EXPECTED_TASKS = (
    (42, "hstu"),
    (42, "safa"),
    (43, "hstu"),
    (43, "safa"),
    (44, "hstu"),
    (44, "safa"),
)
FINAL_EPOCHS = (96, 97, 98, 99, 100)
MEAN_DELTA_THRESHOLD = 0.002
POSITIVE_SEED_THRESHOLD = 2
MIN_DELTA_THRESHOLD = -0.001
HEX_GIT_ID = re.compile(r"[0-9a-f]{40}")
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
POSITIVE_DECIMAL_ID = re.compile(r"[1-9][0-9]*")
REQUIRED_QOS_BY_DATASET = {
    "ml-1m": "h200_dev",
    "ml-20m": "h200_mrs_2_high",
}
REQUIRED_PARTITION = "h200"
SACCT_FIELDS = "JobIDRaw,JobID,State,ExitCode,Restarts,QOS,Partition"
SCHEDULER_RECEIPT_KEYS = {
    "slurm_array_job_id",
    "slurm_array_task_id",
    "slurm_job_id",
    "slurm_restart_count",
    "slurm_job_qos",
    "slurm_job_partition",
    "state",
    "exit_code",
}
EXPECTED_PARAMETER_COUNTS = {
    "ml-1m": 313_416,
    "ml-20m": 38_917_344,
}
EXPECTED_PARAMETER_INVENTORIES = {
    "ml-1m": "2ca8f1559267c3a1741b2343092f2d2c55bcf2aff00265fa0dca8d628e6cf6c8",
    "ml-20m": "38636c03bbbbb842fd4a6fb81fa3f21e93ddf39d6509bb8d96bec42667c7f4d5",
}
RUN_KEYS = {
    "dataset",
    "source_commit",
    "source_tree",
    "source_manifest",
    "parameter_inventory_sha256",
    "parameter_count",
    "metric",
    "seed",
    "arm",
    "attention_mode",
    "random_seed",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "slurm_job_id",
    "slurm_job_qos",
    "slurm_restart_count",
    "slurm_job_partition",
    "resolved_gin_config",
    "resolved_gin_config_sha256",
    "experiment_config_sha256",
    "epochs",
}
CONSISTENT_METADATA_KEYS = (
    "dataset",
    "source_commit",
    "source_tree",
    "source_manifest",
    "parameter_inventory_sha256",
    "parameter_count",
    "metric",
    "experiment_config_sha256",
    "slurm_array_job_id",
    "slurm_job_qos",
    "slurm_job_partition",
)
EXPERIMENT_IDENTITY_BINDINGS = (
    "hstu_encoder.attention_mode",
    "train_fn.random_seed",
)
OPERATIVE_BINDING_PATTERN = re.compile(
    r"^(?P<prefix>[ \t]*(?P<binding>"
    + "|".join(re.escape(binding) for binding in EXPERIMENT_IDENTITY_BINDINGS)
    + r")[ \t]*=[ \t]*)(?P<value>[^\r\n]*?)(?P<ending>\r?\n)?$"
)


class ResultsError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ResultsError(f"could not checksum results JSON: {error}") from error
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ResultsError(f"nonfinite JSON number is forbidden: {value}")


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResultsError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_results(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            document = json.load(
                source,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except (OSError, json.JSONDecodeError) as error:
        raise ResultsError(f"could not parse results JSON: {error}") from error
    if not isinstance(document, dict):
        raise ResultsError("results document must be a JSON object")
    if set(document) != {"schema_version", "runs"}:
        raise ResultsError("document keys must be exactly schema_version and runs")
    if document["schema_version"] != 3:
        raise ResultsError("schema_version must be 3")
    if not isinstance(document["runs"], list):
        raise ResultsError("runs must be a JSON list")
    return document


def _require_string(run: Mapping[str, Any], key: str) -> str:
    value = run[key]
    if not isinstance(value, str) or not value:
        raise ResultsError(f"{key} must be a nonempty string")
    return value


def _require_job_id(run: Mapping[str, Any], key: str) -> str:
    value = _require_string(run, key)
    if POSITIVE_DECIMAL_ID.fullmatch(value) is None:
        raise ResultsError(f"{key} must be a positive decimal job ID string")
    return value


def _require_nonnegative_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResultsError(f"{key} must be a nonnegative integer")
    return value


def validate_scheduler_receipt(
    receipt: Mapping[int, Mapping[str, Any]],
    *,
    expected_array_job_id: str,
    expected_qos: str,
) -> None:
    if (
        not isinstance(expected_array_job_id, str)
        or POSITIVE_DECIMAL_ID.fullmatch(expected_array_job_id) is None
    ):
        raise ResultsError("expected array job ID must be a positive decimal string")
    if (
        not isinstance(receipt, Mapping)
        or any(type(task_id) is not int for task_id in receipt)
        or set(receipt) != set(range(6))
    ):
        raise ResultsError("scheduler receipt must contain exactly tasks 0 through 5")

    raw_job_ids = set()
    for task_id in range(6):
        record = receipt[task_id]
        if not isinstance(record, Mapping) or set(record) != SCHEDULER_RECEIPT_KEYS:
            raise ResultsError(f"scheduler receipt task {task_id} has invalid fields")
        if record["slurm_array_job_id"] != expected_array_job_id:
            raise ResultsError("scheduler receipt has the wrong array job ID")
        if (
            type(record["slurm_array_task_id"]) is not int
            or record["slurm_array_task_id"] != task_id
        ):
            raise ResultsError("scheduler receipt task ID does not match its key")
        raw_job_id = record["slurm_job_id"]
        if (
            not isinstance(raw_job_id, str)
            or POSITIVE_DECIMAL_ID.fullmatch(raw_job_id) is None
        ):
            raise ResultsError("scheduler JobIDRaw must be a positive decimal string")
        if raw_job_id in raw_job_ids:
            raise ResultsError("scheduler JobIDRaw values must be distinct")
        raw_job_ids.add(raw_job_id)
        _require_nonnegative_int(record["slurm_restart_count"], "scheduler Restarts")
        if record["state"] != "COMPLETED":
            raise ResultsError(f"scheduler task {task_id} did not complete")
        if record["exit_code"] != "0:0":
            raise ResultsError(f"scheduler task {task_id} has a nonzero exit code")
        if record["slurm_job_qos"] != expected_qos:
            raise ResultsError(f"scheduler task {task_id} used the wrong QoS")
        if record["slurm_job_partition"] != REQUIRED_PARTITION:
            raise ResultsError(f"scheduler task {task_id} used the wrong partition")


def parse_sacct_receipt(
    output: str,
    *,
    expected_array_job_id: str,
    expected_qos: str,
) -> Dict[int, Dict[str, Any]]:
    """Parse allocation-only, pipe-delimited sacct output for one array."""
    if (
        not isinstance(expected_array_job_id, str)
        or POSITIVE_DECIMAL_ID.fullmatch(expected_array_job_id) is None
    ):
        raise ResultsError("expected array job ID must be a positive decimal string")
    if not isinstance(output, str) or not output:
        raise ResultsError("sacct returned no scheduler records")
    records: Dict[int, Dict[str, Any]] = {}
    lines = output.splitlines()
    if not lines or any(not line for line in lines):
        raise ResultsError("sacct returned malformed scheduler records")
    for line in lines:
        fields = line.split("|")
        if len(fields) != 7:
            raise ResultsError(
                "sacct scheduler record must contain exactly seven fields"
            )
        (
            job_id_raw,
            job_id,
            state,
            exit_code,
            restarts,
            qos,
            partition,
        ) = fields
        match = re.fullmatch(rf"{re.escape(expected_array_job_id)}_([0-9]+)", job_id)
        if match is None:
            raise ResultsError("sacct JobID does not identify the expected array task")
        task_id = int(match.group(1))
        if task_id not in range(6):
            raise ResultsError(f"sacct returned unexpected array task {task_id}")
        if task_id in records:
            raise ResultsError(f"sacct returned duplicate array task {task_id}")
        if re.fullmatch(r"0|[1-9][0-9]*", restarts) is None:
            raise ResultsError("sacct Restarts must be a nonnegative integer")
        records[task_id] = {
            "slurm_array_job_id": expected_array_job_id,
            "slurm_array_task_id": task_id,
            "slurm_job_id": job_id_raw,
            "slurm_restart_count": int(restarts),
            "slurm_job_qos": qos,
            "slurm_job_partition": partition,
            "state": state,
            "exit_code": exit_code,
        }
    validate_scheduler_receipt(
        records,
        expected_array_job_id=expected_array_job_id,
        expected_qos=expected_qos,
    )
    return records


def query_sacct_receipt(
    expected_array_job_id: str, *, expected_qos: str
) -> Dict[int, Dict[str, Any]]:
    if (
        not isinstance(expected_array_job_id, str)
        or POSITIVE_DECIMAL_ID.fullmatch(expected_array_job_id) is None
    ):
        raise ResultsError("expected array job ID must be a positive decimal string")
    command = [
        "sacct",
        "--jobs",
        expected_array_job_id,
        "--allocations",
        "--noheader",
        "--parsable2",
        f"--format={SACCT_FIELDS}",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as error:
        raise ResultsError(f"could not query sacct: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise ResultsError(f"sacct query failed: {detail}")
    return parse_sacct_receipt(
        completed.stdout,
        expected_array_job_id=expected_array_job_id,
        expected_qos=expected_qos,
    )


def operative_config_identities(
    resolved_gin_config: str,
    *,
    attention_mode: str,
    random_seed: int,
) -> Tuple[str, str]:
    """Recompute exact and mode/seed-normalized operative-config identities."""
    expected_values = {
        "hstu_encoder.attention_mode": attention_mode,
        "train_fn.random_seed": random_seed,
    }
    found_values: Dict[str, Any] = {}
    normalized_lines = []
    for line in resolved_gin_config.splitlines(keepends=True):
        match = OPERATIVE_BINDING_PATTERN.fullmatch(line)
        if match is None:
            normalized_lines.append(line)
            continue
        binding = match.group("binding")
        if binding in found_values:
            raise ResultsError(f"duplicate operative Gin binding: {binding}")
        try:
            found_values[binding] = ast.literal_eval(match.group("value").strip())
        except (SyntaxError, ValueError) as error:
            raise ResultsError(
                f"operative Gin binding is not a literal: {binding}"
            ) from error
        normalized_lines.append(
            f"{match.group('prefix')}<redacted>{match.group('ending') or ''}"
        )

    missing = sorted(set(expected_values) - set(found_values))
    if missing:
        raise ResultsError(f"missing operative Gin identity bindings: {missing}")
    for binding, expected_value in expected_values.items():
        if found_values[binding] != expected_value:
            raise ResultsError(
                f"operative Gin binding {binding} does not match recorded metadata"
            )

    exact_sha256 = hashlib.sha256(resolved_gin_config.encode("utf-8")).hexdigest()
    experiment_sha256 = hashlib.sha256(
        "".join(normalized_lines).encode("utf-8")
    ).hexdigest()
    return exact_sha256, experiment_sha256


def _validate_run_metadata(run: Mapping[str, Any]) -> None:
    if set(run) != RUN_KEYS:
        missing = sorted(RUN_KEYS - set(run))
        extra = sorted(set(run) - RUN_KEYS)
        raise ResultsError(f"run keys mismatch: missing={missing}, extra={extra}")
    if _require_string(run, "metric") != "ndcg@10":
        raise ResultsError("metric must be ndcg@10")
    dataset = _require_string(run, "dataset")
    if dataset not in REQUIRED_QOS_BY_DATASET:
        raise ResultsError(f"unsupported dataset: {dataset}")
    if HEX_GIT_ID.fullmatch(_require_string(run, "source_commit")) is None:
        raise ResultsError("source_commit must be a 40-character lowercase Git ID")
    if HEX_GIT_ID.fullmatch(_require_string(run, "source_tree")) is None:
        raise ResultsError("source_tree must be a 40-character lowercase Git ID")
    if HEX_SHA256.fullmatch(_require_string(run, "source_manifest")) is None:
        raise ResultsError("source_manifest must be a lowercase SHA-256")
    if HEX_SHA256.fullmatch(_require_string(run, "parameter_inventory_sha256")) is None:
        raise ResultsError("parameter_inventory_sha256 must be a lowercase SHA-256")
    if HEX_SHA256.fullmatch(_require_string(run, "resolved_gin_config_sha256")) is None:
        raise ResultsError("resolved_gin_config_sha256 must be a lowercase SHA-256")
    if HEX_SHA256.fullmatch(_require_string(run, "experiment_config_sha256")) is None:
        raise ResultsError("experiment_config_sha256 must be a lowercase SHA-256")
    parameter_count = run["parameter_count"]
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
    ):
        raise ResultsError("parameter_count must be a positive integer")
    seed = run["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ResultsError(f"seed must be one of {SEEDS}")
    if run["arm"] not in ARMS:
        raise ResultsError(f"arm must be one of {ARMS}")
    attention_mode = _require_string(run, "attention_mode")
    if attention_mode not in ARMS:
        raise ResultsError(f"attention_mode must be one of {ARMS}")
    if attention_mode != run["arm"]:
        raise ResultsError("attention_mode does not match arm")
    random_seed = run["random_seed"]
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed not in SEEDS
    ):
        raise ResultsError(f"random_seed must be one of {SEEDS}")
    if random_seed != seed:
        raise ResultsError("random_seed does not match seed")
    _require_job_id(run, "slurm_array_job_id")
    _require_job_id(run, "slurm_job_id")
    task_id = run["slurm_array_task_id"]
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or not 0 <= task_id < 6
    ):
        raise ResultsError("slurm_array_task_id must be an integer in [0, 5]")
    expected_seed, expected_arm = EXPECTED_TASKS[task_id]
    if (seed, run["arm"]) != (expected_seed, expected_arm):
        raise ResultsError(
            "slurm_array_task_id does not match the recorded seed/arm mapping"
        )
    required_qos = REQUIRED_QOS_BY_DATASET[dataset]
    if _require_string(run, "slurm_job_qos") != required_qos:
        raise ResultsError(f"slurm_job_qos must be {required_qos} for {dataset}")
    _require_nonnegative_int(run["slurm_restart_count"], "slurm_restart_count")
    if _require_string(run, "slurm_job_partition") != REQUIRED_PARTITION:
        raise ResultsError(f"slurm_job_partition must be {REQUIRED_PARTITION}")

    resolved_gin_config = _require_string(run, "resolved_gin_config")
    resolved_sha256, experiment_sha256 = operative_config_identities(
        resolved_gin_config,
        attention_mode=attention_mode,
        random_seed=random_seed,
    )
    if resolved_sha256 != run["resolved_gin_config_sha256"]:
        raise ResultsError("resolved Gin config does not match its recorded SHA-256")
    if experiment_sha256 != run["experiment_config_sha256"]:
        raise ResultsError("normalized experiment config does not match its SHA-256")


def _epoch_values(run: Mapping[str, Any]) -> Dict[int, Decimal]:
    records = run["epochs"]
    if not isinstance(records, list):
        raise ResultsError("epochs must be a JSON list")
    values: Dict[int, Decimal] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"epoch", "value"}:
            raise ResultsError("each epoch record must contain exactly epoch and value")
        epoch = record["epoch"]
        value = record["value"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ResultsError("epoch must be a non-negative integer")
        if epoch in values:
            raise ResultsError(
                f"duplicate epoch {epoch} for seed={run['seed']} arm={run['arm']}"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ResultsError("NDCG@10 value must be numeric")
        numeric_value = Decimal(str(value))
        if not numeric_value.is_finite():
            raise ResultsError("NDCG@10 value must be finite")
        if not Decimal(0) <= numeric_value <= Decimal(1):
            raise ResultsError("NDCG@10 value must be in [0, 1]")
        values[epoch] = numeric_value
    missing_epochs = sorted(set(FINAL_EPOCHS) - set(values))
    if missing_epochs:
        raise ResultsError(
            f"missing final epochs for seed={run['seed']} arm={run['arm']}: "
            f"{missing_epochs}"
        )
    return values


def qualify_results(
    document: Mapping[str, Any],
    *,
    expected_dataset: str,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_source_manifest: str,
    expected_experiment_config_sha256: str,
    expected_array_job_id: str,
    scheduler_receipt: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    if document.get("schema_version") != 3 or not isinstance(
        document.get("runs"), list
    ):
        raise ResultsError("invalid results document")
    runs: List[Mapping[str, Any]] = document["runs"]
    if expected_dataset not in EXPECTED_PARAMETER_COUNTS:
        raise ResultsError(f"unsupported dataset: {expected_dataset}")
    if len(runs) != len(SEEDS) * len(ARMS):
        raise ResultsError("runs must contain exactly six paired seed/arm entries")
    validate_scheduler_receipt(
        scheduler_receipt,
        expected_array_job_id=expected_array_job_id,
        expected_qos=REQUIRED_QOS_BY_DATASET[expected_dataset],
    )

    baseline_metadata: Optional[Dict[str, Any]] = None
    final_means: Dict[Tuple[int, str], Decimal] = {}
    task_ids = set()
    run_job_ids = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ResultsError("every run must be a JSON object")
        _validate_run_metadata(run)
        metadata = {key: run[key] for key in CONSISTENT_METADATA_KEYS}
        if baseline_metadata is None:
            baseline_metadata = metadata
        elif metadata != baseline_metadata:
            differing = [
                key
                for key in CONSISTENT_METADATA_KEYS
                if metadata[key] != baseline_metadata[key]
            ]
            raise ResultsError(f"mismatched run metadata: {differing}")

        run_key = (int(run["seed"]), str(run["arm"]))
        task_id = int(run["slurm_array_task_id"])
        if task_id in task_ids:
            raise ResultsError(f"duplicate SLURM array task ID: {task_id}")
        task_ids.add(task_id)
        run_job_id = str(run["slurm_job_id"])
        if run_job_id in run_job_ids:
            raise ResultsError("run SLURM job IDs must be distinct")
        run_job_ids.add(run_job_id)
        scheduler_record = scheduler_receipt[task_id]
        for key in (
            "slurm_array_job_id",
            "slurm_array_task_id",
            "slurm_job_id",
            "slurm_restart_count",
            "slurm_job_qos",
            "slurm_job_partition",
        ):
            if run[key] != scheduler_record[key]:
                raise ResultsError(
                    f"run {key} does not match the authoritative scheduler receipt"
                )
        if run_key in final_means:
            raise ResultsError(f"duplicate run for seed={run_key[0]} arm={run_key[1]}")
        values = _epoch_values(run)
        final_means[run_key] = sum(
            (values[epoch] for epoch in FINAL_EPOCHS), start=Decimal(0)
        ) / Decimal(len(FINAL_EPOCHS))

    expected_keys = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(final_means) != expected_keys:
        missing = sorted(expected_keys - set(final_means))
        raise ResultsError(f"missing seed/arm runs: {missing}")
    if task_ids != set(range(6)):
        missing_tasks = sorted(set(range(6)) - task_ids)
        raise ResultsError(f"missing SLURM array task IDs: {missing_tasks}")
    assert baseline_metadata is not None
    if baseline_metadata["dataset"] != expected_dataset:
        raise ResultsError(
            f"dataset mismatch: expected {expected_dataset}, "
            f"found {baseline_metadata['dataset']}"
        )
    if baseline_metadata["source_manifest"] != expected_source_manifest:
        raise ResultsError("source manifest does not match the externally pinned value")
    if baseline_metadata["source_commit"] != expected_source_commit:
        raise ResultsError("source commit does not match the externally pinned value")
    if baseline_metadata["source_tree"] != expected_source_tree:
        raise ResultsError("source tree does not match the externally pinned value")
    if (
        not isinstance(expected_array_job_id, str)
        or POSITIVE_DECIMAL_ID.fullmatch(expected_array_job_id) is None
    ):
        raise ResultsError("expected array job ID must be a positive decimal string")
    if baseline_metadata["slurm_array_job_id"] != expected_array_job_id:
        raise ResultsError(
            "SLURM array job ID does not match the externally pinned value"
        )
    if HEX_SHA256.fullmatch(expected_experiment_config_sha256) is None:
        raise ResultsError(
            "expected experiment config identity must be a lowercase SHA-256"
        )
    if (
        baseline_metadata["experiment_config_sha256"]
        != expected_experiment_config_sha256
    ):
        raise ResultsError(
            "experiment config does not match the externally pinned identity"
        )
    if (
        baseline_metadata["parameter_count"]
        != EXPECTED_PARAMETER_COUNTS[expected_dataset]
    ):
        raise ResultsError(
            "parameter count does not match the frozen dataset inventory"
        )
    if baseline_metadata["parameter_inventory_sha256"] != (
        EXPECTED_PARAMETER_INVENTORIES[expected_dataset]
    ):
        raise ResultsError(
            "parameter inventory does not match the frozen dataset inventory"
        )

    deltas = {
        seed: final_means[(seed, "safa")] - final_means[(seed, "hstu")]
        for seed in SEEDS
    }
    mean_delta = sum(deltas.values(), start=Decimal(0)) / Decimal(len(SEEDS))
    positive_seed_count = sum(delta > Decimal(0) for delta in deltas.values())
    minimum_delta = min(deltas.values())
    gate_checks = {
        "mean_delta": mean_delta >= Decimal(str(MEAN_DELTA_THRESHOLD)),
        "positive_seeds": positive_seed_count >= POSITIVE_SEED_THRESHOLD,
        "minimum_delta": minimum_delta >= Decimal(str(MIN_DELTA_THRESHOLD)),
    }
    passed = all(gate_checks.values())
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "metadata": baseline_metadata,
        "scheduler_receipt": {
            str(task_id): dict(scheduler_receipt[task_id]) for task_id in range(6)
        },
        "final_epochs": list(FINAL_EPOCHS),
        "thresholds": {
            "mean_delta_min": MEAN_DELTA_THRESHOLD,
            "positive_seed_count_min": POSITIVE_SEED_THRESHOLD,
            "minimum_delta_min": MIN_DELTA_THRESHOLD,
        },
        "per_seed": {
            str(seed): {
                "hstu_mean_ndcg@10": float(final_means[(seed, "hstu")]),
                "safa_mean_ndcg@10": float(final_means[(seed, "safa")]),
                "delta": float(deltas[seed]),
            }
            for seed in SEEDS
        },
        "aggregate": {
            "mean_delta": float(mean_delta),
            "sample_std_delta": statistics.stdev(
                float(value) for value in deltas.values()
            ),
            "positive_seed_count": positive_seed_count,
            "minimum_delta": float(minimum_delta),
        },
        "gate_checks": gate_checks,
    }


def _write_summary(summary: Mapping[str, Any], output: Optional[Path]) -> None:
    serialized = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    sys.stdout.write(serialized)
    if output is not None:
        output = output.expanduser().absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, output)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--expected-dataset", choices=sorted(EXPECTED_PARAMETER_COUNTS), required=True
    )
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-source-manifest", required=True)
    parser.add_argument("--expected-experiment-config-sha256", required=True)
    parser.add_argument("--expected-array-job-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        input_sha256 = _sha256(args.results)
        document = load_results(args.results)
        scheduler_receipt = query_sacct_receipt(
            args.expected_array_job_id,
            expected_qos=REQUIRED_QOS_BY_DATASET[args.expected_dataset],
        )
        summary = qualify_results(
            document,
            expected_dataset=args.expected_dataset,
            expected_source_commit=args.expected_source_commit,
            expected_source_tree=args.expected_source_tree,
            expected_source_manifest=args.expected_source_manifest,
            expected_experiment_config_sha256=(args.expected_experiment_config_sha256),
            expected_array_job_id=args.expected_array_job_id,
            scheduler_receipt=scheduler_receipt,
        )
    except ResultsError as error:
        summary = {"status": "invalid", "passed": False, "error": str(error)}
        _write_summary(summary, args.output)
        return 2
    summary["input_sha256"] = input_sha256
    _write_summary(summary, args.output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
