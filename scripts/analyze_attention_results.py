#!/usr/bin/env python3
"""Produce restart-safe, provenance-bound summaries for attention experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import stat
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


REPORT_SCHEMA_VERSION = "attention-results/v1"
DEFAULT_SPEC = Path(__file__).with_name("attention_result_specs.json")
EVAL_TAGS = (
    "eval_epoch/ndcg@10",
    "eval_epoch/hr@10",
    "eval_epoch/mrr",
)
THROUGHPUT_TAG = "performance/train_examples_per_second"
REQUIRED_TAGS = EVAL_TAGS + (THROUGHPUT_TAG,)
TAIL_FIELD = "tail_mean_steps_96_100"


class InvalidEvidence(RuntimeError):
    """The inputs cannot support a trustworthy result."""


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float


ScalarLoader = Callable[[Path], Mapping[str, Sequence[ScalarPoint]]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidEvidence(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_registry(path: Path) -> Tuple[Dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise InvalidEvidence(f"cannot read spec {path}: {error}") from error
    spec_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        registry = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, InvalidEvidence) as error:
        raise InvalidEvidence(f"invalid spec {path}: {error}") from error
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise InvalidEvidence("spec schema_version must be 1")
    experiments = registry.get("experiments")
    policies = registry.get("policies")
    if not isinstance(experiments, dict) or not isinstance(policies, dict):
        raise InvalidEvidence(
            "spec must contain object-valued experiments and policies"
        )
    return registry, spec_sha256


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidEvidence(f"{field} must be a non-empty string")
    return value


def _require_positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidEvidence(f"{field} must be a positive integer")
    return value


def _require_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidEvidence(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidEvidence(f"{field} must be finite")
    return result


def _validate_policy(name: str, policy: Mapping[str, Any]) -> None:
    policy_type = _require_string(policy.get("type"), f"policies[{name!r}].type")
    _require_string(policy.get("metric"), f"policies[{name!r}].metric")
    if policy_type == "fixed_tail_screen":
        _require_positive_integer(
            policy.get("required_pairs"), f"policies[{name!r}].required_pairs"
        )
        _require_finite_number(
            policy.get("mean_delta_min"), f"policies[{name!r}].mean_delta_min"
        )
        _require_finite_number(
            policy.get("min_delta_min"), f"policies[{name!r}].min_delta_min"
        )
    elif policy_type == "fixed_positive_fraction_screen":
        _require_positive_integer(
            policy.get("required_pairs"), f"policies[{name!r}].required_pairs"
        )
    elif policy_type == "insufficient_seeds":
        _require_positive_integer(
            policy.get("minimum_pairs"), f"policies[{name!r}].minimum_pairs"
        )
    elif policy_type not in (
        "no_fixed_decision_rule",
        "cross_snapshot_ineligible",
    ):
        raise InvalidEvidence(f"unknown policy type {policy_type!r}")
    if policy_type in ("fixed_tail_screen", "fixed_positive_fraction_screen"):
        fraction = _require_finite_number(
            policy.get("positive_fraction_min"),
            f"policies[{name!r}].positive_fraction_min",
        )
        if not 0.0 <= fraction <= 1.0:
            raise InvalidEvidence(
                f"policies[{name!r}].positive_fraction_min must be in [0, 1]"
            )


def _validate_experiment(
    experiment_id: str,
    experiment: Mapping[str, Any],
    policies: Mapping[str, Any],
) -> None:
    job_id = _require_string(experiment.get("job_id"), "job_id")
    if job_id != experiment_id or re.fullmatch(r"[0-9]+", job_id) is None:
        raise InvalidEvidence("experiment key and numeric job_id must match")

    snapshot = experiment.get("snapshot")
    if not isinstance(snapshot, dict):
        raise InvalidEvidence("snapshot must be an object")
    root = Path(_require_string(snapshot.get("root"), "snapshot.root"))
    if not root.is_absolute():
        raise InvalidEvidence("snapshot.root must be absolute")
    manifest_sha256 = _require_string(
        snapshot.get("manifest_sha256"), "snapshot.manifest_sha256"
    )
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise InvalidEvidence("snapshot.manifest_sha256 must be lowercase SHA256")
    legacy_files = snapshot.get("legacy_unmanifested_files")
    if not isinstance(legacy_files, dict):
        raise InvalidEvidence("snapshot.legacy_unmanifested_files must be an object")
    allowed_legacy_files = {"GIT_COMMIT", "GIT_STATUS", "WORKTREE.patch"}
    if legacy_files and set(legacy_files) != allowed_legacy_files:
        raise InvalidEvidence(
            "nonempty legacy_unmanifested_files must declare exactly "
            "GIT_COMMIT, GIT_STATUS, and WORKTREE.patch"
        )
    for path_text, digest in legacy_files.items():
        if "/" in path_text or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise InvalidEvidence("invalid legacy_unmanifested_files entry")

    run_root = Path(_require_string(experiment.get("run_root"), "run_root"))
    if not run_root.is_absolute():
        raise InvalidEvidence("run_root must be absolute")
    expected_steps = experiment.get("expected_steps")
    if expected_steps != {"start": 0, "end": 100}:
        raise InvalidEvidence("expected_steps must be exactly start=0, end=100")

    scheduler = experiment.get("scheduler")
    if not isinstance(scheduler, dict):
        raise InvalidEvidence("scheduler must be an object")
    for field in ("job_name", "qos", "work_dir", "snapshot_export"):
        _require_string(scheduler.get(field), f"scheduler.{field}")
    if scheduler["qos"] != "h200_mrs_shared":
        raise InvalidEvidence("scheduler.qos must be h200_mrs_shared")
    if scheduler["snapshot_export"] != "GR_CODE_SNAPSHOT":
        raise InvalidEvidence("scheduler.snapshot_export must be GR_CODE_SNAPSHOT")
    if not Path(str(scheduler["work_dir"])).is_absolute():
        raise InvalidEvidence("scheduler.work_dir must be absolute")
    wrapper = scheduler.get("wrapper")
    if not isinstance(wrapper, dict):
        raise InvalidEvidence("scheduler.wrapper must be an object")
    wrapper_form = _require_string(wrapper.get("form"), "scheduler.wrapper.form")
    wrapper_token = _require_string(wrapper.get("token"), "scheduler.wrapper.token")
    raw_wrapper_parts = wrapper_token.split("/")
    normalized_parts = (
        raw_wrapper_parts[1:] if wrapper_token.startswith("/") else raw_wrapper_parts
    )
    if (
        any(part in ("", ".", "..") for part in normalized_parts)
        or "\\" in wrapper_token
        or PurePosixPath(wrapper_token).as_posix() != wrapper_token
    ):
        raise InvalidEvidence(
            "scheduler.wrapper.token must be normalized without traversal"
        )
    if wrapper_form == "relative_workdir":
        if Path(wrapper_token).is_absolute():
            raise InvalidEvidence("relative_workdir wrapper token must be relative")
    elif wrapper_form == "absolute_snapshot":
        if not Path(wrapper_token).is_absolute():
            raise InvalidEvidence("absolute_snapshot wrapper token must be absolute")
        try:
            resolved_root = root.resolve(strict=True)
            resolved_wrapper = Path(wrapper_token).resolve(strict=True)
            wrapper_relative = resolved_wrapper.relative_to(resolved_root).as_posix()
        except (OSError, ValueError) as error:
            raise InvalidEvidence(
                "absolute wrapper must resolve to a file inside snapshot root"
            ) from error
        try:
            wrapper_mode = resolved_wrapper.lstat().st_mode
        except OSError as error:
            raise InvalidEvidence("cannot stat absolute snapshot wrapper") from error
        if resolved_wrapper.is_symlink() or not stat.S_ISREG(wrapper_mode):
            raise InvalidEvidence("absolute snapshot wrapper must be a regular file")
        manifest_path = resolved_root / "SOURCE_SHA256SUMS"
        try:
            manifest_files = _manifest_paths(manifest_path.read_bytes())
        except OSError as error:
            raise InvalidEvidence(
                "cannot read snapshot manifest for wrapper binding"
            ) from error
        if wrapper_relative not in manifest_files:
            raise InvalidEvidence("absolute snapshot wrapper is not manifest-listed")
    else:
        raise InvalidEvidence(f"unknown scheduler.wrapper.form {wrapper_form!r}")
    if not isinstance(scheduler.get("manifest_export_required"), bool):
        raise InvalidEvidence("scheduler.manifest_export_required must be boolean")
    if experiment.get("overall_policy") != "per_comparison_only":
        raise InvalidEvidence("overall_policy must be per_comparison_only")
    screen_provenance = experiment.get("screen_provenance")
    if screen_provenance not in (
        "posthoc_fixed_screen",
        "pre_submission_fixed_screen",
        "no_fixed_screen",
    ):
        raise InvalidEvidence("invalid screen_provenance")

    tasks = experiment.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise InvalidEvidence("tasks must be a non-empty array")
    task_ids: set[int] = set()
    run_names: set[str] = set()
    variant_seeds: set[Tuple[str, int]] = set()
    variants: set[str] = set()
    scalar_count = 0
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise InvalidEvidence(f"tasks[{index}] must be an object")
        has_task_id = "task_id" in task
        is_scalar = task.get("scalar") is True
        if has_task_id == is_scalar:
            raise InvalidEvidence(
                f"tasks[{index}] must have exactly one of task_id or scalar=true"
            )
        if has_task_id:
            task_id = task["task_id"]
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise InvalidEvidence(f"tasks[{index}].task_id must be non-negative")
            if task_id in task_ids:
                raise InvalidEvidence(f"duplicate task_id {task_id}")
            task_ids.add(task_id)
        else:
            scalar_count += 1
        variant = _require_string(task.get("variant"), f"tasks[{index}].variant")
        run_name = _require_string(task.get("run_name"), f"tasks[{index}].run_name")
        seed = task.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise InvalidEvidence(f"tasks[{index}].seed must be an integer")
        if run_name in run_names:
            raise InvalidEvidence(f"duplicate run_name {run_name!r}")
        if (variant, seed) in variant_seeds:
            raise InvalidEvidence(f"duplicate variant/seed pair {variant!r}/{seed}")
        run_names.add(run_name)
        variant_seeds.add((variant, seed))
        variants.add(variant)
    if scalar_count not in (0, 1) or (scalar_count == 1 and len(tasks) != 1):
        raise InvalidEvidence(
            "a scalar experiment must contain exactly one scalar task"
        )
    seeds_by_variant: Dict[str, set[int]] = {}
    for variant, seed in variant_seeds:
        seeds_by_variant.setdefault(variant, set()).add(seed)

    variant_metadata = experiment.get("variant_metadata", {})
    if not isinstance(variant_metadata, dict):
        raise InvalidEvidence("variant_metadata must be an object")
    if not set(variant_metadata).issubset(variants):
        raise InvalidEvidence("variant_metadata names an unknown variant")
    for variant, metadata in variant_metadata.items():
        if not isinstance(metadata, dict):
            raise InvalidEvidence(f"variant_metadata[{variant!r}] must be an object")
        if "linear_throughput_claim_eligible" in metadata and not isinstance(
            metadata["linear_throughput_claim_eligible"], bool
        ):
            raise InvalidEvidence(
                f"variant_metadata[{variant!r}].linear_throughput_claim_eligible "
                "must be boolean"
            )

    comparisons = experiment.get("comparisons", [])
    if not isinstance(comparisons, list):
        raise InvalidEvidence("comparisons must be an array")
    comparison_ids: set[str] = set()
    fixed_policy_count = 0
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            raise InvalidEvidence(f"comparisons[{index}] must be an object")
        comparison_id = _require_string(
            comparison.get("id"), f"comparisons[{index}].id"
        )
        if comparison_id in comparison_ids:
            raise InvalidEvidence(f"duplicate comparison id {comparison_id!r}")
        comparison_ids.add(comparison_id)
        candidate = _require_string(
            comparison.get("candidate"), f"comparisons[{index}].candidate"
        )
        baseline = _require_string(
            comparison.get("baseline"), f"comparisons[{index}].baseline"
        )
        policy_name = _require_string(
            comparison.get("policy"), f"comparisons[{index}].policy"
        )
        if policy_name not in policies or not isinstance(policies[policy_name], dict):
            raise InvalidEvidence(f"unknown comparison policy {policy_name!r}")
        _validate_policy(policy_name, policies[policy_name])
        if candidate not in variants:
            raise InvalidEvidence(f"unknown candidate variant {candidate!r}")
        policy_type = policies[policy_name].get("type")
        if policy_type != "cross_snapshot_ineligible" and baseline not in variants:
            raise InvalidEvidence(f"unknown baseline variant {baseline!r}")
        if policy_type in (
            "fixed_tail_screen",
            "fixed_positive_fraction_screen",
        ):
            fixed_policy_count += 1
            expected_seeds = comparison.get("expected_seeds")
            if (
                not isinstance(expected_seeds, list)
                or any(
                    isinstance(seed, bool) or not isinstance(seed, int)
                    for seed in expected_seeds
                )
                or len(set(expected_seeds)) != len(expected_seeds)
            ):
                raise InvalidEvidence(
                    f"comparisons[{index}].expected_seeds must be unique integers"
                )
            required_pairs = int(policies[policy_name]["required_pairs"])
            if len(expected_seeds) != required_pairs:
                raise InvalidEvidence(
                    f"comparisons[{index}].expected_seeds must have "
                    f"{required_pairs} entries"
                )
            expected_set = set(expected_seeds)
            if seeds_by_variant.get(candidate) != expected_set:
                raise InvalidEvidence(
                    f"candidate {candidate!r} seed set differs from expected_seeds"
                )
            if seeds_by_variant.get(baseline) != expected_set:
                raise InvalidEvidence(
                    f"baseline {baseline!r} seed set differs from expected_seeds"
                )
    if screen_provenance == "no_fixed_screen" and fixed_policy_count != 0:
        raise InvalidEvidence(
            "no_fixed_screen provenance cannot declare a fixed comparison policy"
        )
    if (
        screen_provenance in ("posthoc_fixed_screen", "pre_submission_fixed_screen")
        and fixed_policy_count == 0
    ):
        raise InvalidEvidence(
            f"{screen_provenance} provenance requires a fixed comparison policy"
        )


def _manifest_paths(manifest_bytes: bytes) -> set[str]:
    try:
        lines = manifest_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise InvalidEvidence("snapshot manifest is not UTF-8") from error
    paths: set[str] = set()
    pattern = re.compile(r"^([0-9a-f]{64}) ([ *])(.+)$")
    for line_number, line in enumerate(lines, start=1):
        match = pattern.fullmatch(line)
        if match is None:
            raise InvalidEvidence(f"malformed snapshot manifest line {line_number}")
        path_text = match.group(3)
        path = PurePosixPath(path_text)
        if (
            path.is_absolute()
            or path_text in ("", ".")
            or any(part in ("", ".", "..") for part in path.parts)
            or str(path) != path_text
        ):
            raise InvalidEvidence(
                f"unsafe snapshot manifest path on line {line_number}: {path_text!r}"
            )
        if path_text == "SOURCE_SHA256SUMS":
            raise InvalidEvidence("snapshot manifest must not list itself")
        if path_text in paths:
            raise InvalidEvidence(f"duplicate snapshot manifest path {path_text!r}")
        paths.add(path_text)
    if not paths:
        raise InvalidEvidence("snapshot manifest is empty")
    return paths


def _snapshot_tree(root: Path) -> Tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory: Path, relative: PurePosixPath) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise InvalidEvidence(
                f"cannot scan snapshot directory {directory}: {error}"
            ) from error
        for entry in entries:
            relative_path = relative / entry.name
            relative_text = str(relative_path)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as error:
                raise InvalidEvidence(
                    f"cannot stat snapshot node {entry.path}: {error}"
                ) from error
            if stat.S_ISLNK(mode):
                raise InvalidEvidence(f"snapshot contains symlink: {relative_text}")
            if stat.S_ISREG(mode):
                files.add(relative_text)
            elif stat.S_ISDIR(mode):
                directories.add(relative_text)
                visit(Path(entry.path), relative_path)
            else:
                raise InvalidEvidence(
                    f"snapshot contains special node: {relative_text}"
                )

    visit(root, PurePosixPath())
    return files, directories


def _expected_snapshot_directories(files: Iterable[str]) -> set[str]:
    directories: set[str] = set()
    for file_path in files:
        parent = PurePosixPath(file_path).parent
        while str(parent) != ".":
            directories.add(str(parent))
            parent = parent.parent
    return directories


def _verify_snapshot(
    snapshot: Mapping[str, Any], skip_full_snapshot_check: bool
) -> Dict[str, Any]:
    root = Path(str(snapshot["root"]))
    if root.is_symlink() or not root.is_dir():
        raise InvalidEvidence(f"snapshot root is not a non-symlink directory: {root}")
    manifest = root / "SOURCE_SHA256SUMS"
    if manifest.is_symlink() or not manifest.is_file():
        raise InvalidEvidence(
            f"snapshot manifest is not a non-symlink regular file: {manifest}"
        )
    expected_sha256 = str(snapshot["manifest_sha256"])
    try:
        manifest_bytes = manifest.read_bytes()
    except OSError as error:
        raise InvalidEvidence(
            f"cannot read snapshot manifest {manifest}: {error}"
        ) from error
    observed_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise InvalidEvidence(
            "snapshot manifest SHA256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    legacy_files = snapshot["legacy_unmanifested_files"]
    for relative_path, expected_digest in legacy_files.items():
        legacy_path = root / relative_path
        try:
            mode = legacy_path.lstat().st_mode
        except OSError as error:
            raise InvalidEvidence(
                f"missing legacy unmanifested file {relative_path}: {error}"
            ) from error
        if legacy_path.is_symlink() or not stat.S_ISREG(mode):
            raise InvalidEvidence(
                f"legacy unmanifested file is not regular: {relative_path}"
            )
        observed_digest = _sha256(legacy_path)
        if observed_digest != expected_digest:
            raise InvalidEvidence(
                f"legacy unmanifested file hash mismatch: {relative_path}"
            )
    if not skip_full_snapshot_check:
        manifest_files = _manifest_paths(manifest_bytes)
        expected_files = manifest_files | {"SOURCE_SHA256SUMS"} | set(legacy_files)
        expected_directories = _expected_snapshot_directories(expected_files)
        actual_files, actual_directories = _snapshot_tree(root)
        if actual_files != expected_files:
            raise InvalidEvidence(
                "snapshot regular-file set mismatch: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"extra={sorted(actual_files - expected_files)}"
            )
        if actual_directories != expected_directories:
            raise InvalidEvidence(
                "snapshot directory set mismatch: "
                f"missing={sorted(expected_directories - actual_directories)}, "
                f"extra={sorted(actual_directories - expected_directories)}"
            )
        try:
            result = subprocess.run(
                [
                    "sha256sum",
                    "--check",
                    "--strict",
                    "--quiet",
                    "SOURCE_SHA256SUMS",
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise InvalidEvidence(f"could not execute sha256sum: {error}") from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise InvalidEvidence(
                f"full snapshot checksum validation failed: {detail or 'no detail'}"
            )
    return {
        "root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": observed_sha256,
        "legacy_unmanifested_files": dict(legacy_files),
        "legacy_exception_applied": bool(legacy_files),
        "full_snapshot_check": (
            "SKIPPED_FOR_TESTS" if skip_full_snapshot_check else "PASSED"
        ),
        "sealed_regular_file_count": (
            None if skip_full_snapshot_check else len(expected_files)
        ),
        "sealed_directory_count": (
            None if skip_full_snapshot_check else len(expected_directories)
        ),
    }


def _read_sacct(job_id: str, sacct_file: Optional[Path]) -> Tuple[str, Dict[str, Any]]:
    if sacct_file is not None:
        if sacct_file.is_symlink() or not sacct_file.is_file():
            raise InvalidEvidence(
                f"sacct file is not a non-symlink regular file: {sacct_file}"
            )
        try:
            raw = sacct_file.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise InvalidEvidence(
                f"cannot read sacct file {sacct_file}: {error}"
            ) from error
        return text, {
            "source": "file",
            "source_path": str(sacct_file),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
        }
    command = [
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        job_id,
        "--format=JobID,JobName,QOS,State,ExitCode,Restarts,SubmitLine,WorkDir",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except OSError as error:
        raise InvalidEvidence(f"could not execute sacct: {error}") from error
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise InvalidEvidence(f"sacct failed with exit {result.returncode}: {stderr}")
    try:
        stdout = result.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidEvidence("sacct output is not UTF-8") from error
    return stdout, {
        "source": "sacct",
        "command": command,
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
    }


def _validate_submit_line(
    submit_line: str, experiment: Mapping[str, Any]
) -> Dict[str, Any]:
    try:
        tokens = shlex.split(submit_line, posix=True)
    except ValueError as error:
        raise InvalidEvidence(f"cannot parse scheduler SubmitLine: {error}") from error
    scheduler = experiment["scheduler"]
    wrapper = scheduler["wrapper"]
    expected_wrapper = str(wrapper["token"])
    if len(tokens) < 4 or tokens[0] != "sbatch":
        raise InvalidEvidence("scheduler SubmitLine is not an sbatch invocation")
    if tokens.count("--parsable") != 1:
        raise InvalidEvidence(
            "scheduler SubmitLine must contain exactly one --parsable"
        )
    export_tokens = [token for token in tokens if token.startswith("--export=")]
    if len(export_tokens) != 1:
        raise InvalidEvidence("scheduler SubmitLine must contain one --export= token")
    if tokens[-1] != expected_wrapper or tokens.count(expected_wrapper) != 1:
        raise InvalidEvidence(
            f"scheduler SubmitLine wrapper mismatch: expected {expected_wrapper!r}"
        )
    if tokens != ["sbatch", "--parsable", export_tokens[0], expected_wrapper]:
        raise InvalidEvidence("scheduler SubmitLine contains unexpected arguments")
    export_parts = export_tokens[0][len("--export=") :].split(",")
    if not export_parts or export_parts[0] != "ALL":
        raise InvalidEvidence("scheduler SubmitLine export must begin with ALL")
    exports: Dict[str, str] = {}
    for part in export_parts[1:]:
        if "=" not in part:
            raise InvalidEvidence(f"malformed scheduler export {part!r}")
        key, value = part.split("=", 1)
        if not key or key in exports:
            raise InvalidEvidence(f"duplicate or empty scheduler export {key!r}")
        exports[key] = value
    snapshot_export = str(scheduler["snapshot_export"])
    expected_snapshot = str(experiment["snapshot"]["root"])
    if exports.get(snapshot_export) != expected_snapshot:
        raise InvalidEvidence(
            f"scheduler {snapshot_export} export does not bind the registered snapshot"
        )
    manifest_key = "GR_SNAPSHOT_MANIFEST_SHA256"
    manifest_required = bool(scheduler["manifest_export_required"])
    expected_manifest = str(experiment["snapshot"]["manifest_sha256"])
    if manifest_required:
        if exports.get(manifest_key) != expected_manifest:
            raise InvalidEvidence(
                "scheduler manifest export does not bind the registered manifest"
            )
    elif manifest_key in exports:
        raise InvalidEvidence("unexpected scheduler manifest export for legacy job")
    expected_keys = {snapshot_export} | ({manifest_key} if manifest_required else set())
    if set(exports) != expected_keys:
        raise InvalidEvidence(
            f"unexpected scheduler exports: {sorted(set(exports) - expected_keys)}"
        )
    return {
        "tokens": tokens,
        "wrapper_form": wrapper["form"],
        "wrapper_token": expected_wrapper,
        "exports": exports,
    }


def _validated_scheduler_row(
    fields: Sequence[str], experiment: Mapping[str, Any]
) -> Dict[str, Any]:
    (
        row_job_id,
        job_name,
        qos,
        state,
        exit_code,
        restarts_text,
        submit_line,
        work_dir,
    ) = fields
    scheduler = experiment["scheduler"]
    if job_name != scheduler["job_name"]:
        raise InvalidEvidence(
            f"scheduler JobName mismatch for {row_job_id}: {job_name!r}"
        )
    if qos != scheduler["qos"]:
        raise InvalidEvidence(f"scheduler QOS mismatch for {row_job_id}: {qos!r}")
    if work_dir != scheduler["work_dir"]:
        raise InvalidEvidence(
            f"scheduler WorkDir mismatch for {row_job_id}: {work_dir!r}"
        )
    if re.fullmatch(r"[0-9]+", restarts_text) is None:
        raise InvalidEvidence(
            f"invalid scheduler Restarts for {row_job_id}: {restarts_text!r}"
        )
    submit_binding = _validate_submit_line(submit_line, experiment)
    return {
        "job_id": row_job_id,
        "job_name": job_name,
        "qos": qos,
        "state": state,
        "exit_code": exit_code,
        "restarts": int(restarts_text),
        "submit_line": submit_line,
        "work_dir": work_dir,
        "submit_binding": submit_binding,
    }


def _scheduler_ref(job_id: str, task: Mapping[str, Any]) -> str:
    if task.get("scalar") is True:
        return job_id
    return f"{job_id}_{task['task_id']}"


def _task_sort_key(task: Mapping[str, Any]) -> Tuple[int, int]:
    if task.get("scalar") is True:
        return (-1, 0)
    return (0, int(task["task_id"]))


def _parse_scheduler(
    text: str, experiment: Mapping[str, Any], source: Mapping[str, Any]
) -> Tuple[Dict[str, Any], bool]:
    job_id = str(experiment["job_id"])
    tasks = list(experiment["tasks"])
    expected_refs = {_scheduler_ref(job_id, task) for task in tasks}
    records_by_ref: Dict[str, Dict[str, Any]] = {}
    compressed_records: List[Dict[str, Any]] = []
    seen_job_ids: set[str] = set()
    compressed_task_range_seen = False
    array_range = re.compile(rf"^{re.escape(job_id)}_\[[^]]+\](?:%[0-9]+)?$")

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split("|")]
        if len(fields) == 9 and fields[-1] == "":
            fields.pop()
        if len(fields) != 8:
            raise InvalidEvidence(
                f"malformed sacct line {line_number}: expected eight fields"
            )
        row_job_id = fields[0]
        if row_job_id in seen_job_ids:
            raise InvalidEvidence(f"duplicate sacct record for {row_job_id}")
        seen_job_ids.add(row_job_id)
        record = _validated_scheduler_row(fields, experiment)
        if row_job_id in expected_refs:
            records_by_ref[row_job_id] = record
        elif array_range.fullmatch(row_job_id):
            compressed_task_range_seen = True
            compressed_records.append(record)
        elif row_job_id == job_id and len(tasks) > 1:
            # Some Slurm versions include an array-parent row with -X. It is not
            # evidence for any task and is therefore retained only as context.
            compressed_records.append(record)
        else:
            raise InvalidEvidence(f"unexpected sacct JobID binding: {row_job_id!r}")

    ordered_records: List[Dict[str, Any]] = []
    incomplete = compressed_task_range_seen
    for task in sorted(tasks, key=_task_sort_key):
        scheduler_job_id = _scheduler_ref(job_id, task)
        record = records_by_ref.get(scheduler_job_id)
        if record is None:
            incomplete = True
            ordered_records.append(
                {
                    "job_id": scheduler_job_id,
                    "state": "MISSING_EXACT_TASK_RECORD",
                    "exit_code": None,
                    "restarts": None,
                }
            )
            continue
        ordered_records.append(record)
        if record["state"] != "COMPLETED" or record["exit_code"] != "0:0":
            incomplete = True

    scheduler = dict(source)
    scheduler.update(
        {
            "records": ordered_records,
            "compressed_or_parent_records": sorted(
                compressed_records, key=lambda record: record["job_id"]
            ),
        }
    )
    return scheduler, incomplete


def _attempt_suffix(
    job_id: str, task: Mapping[str, Any], scheduler_record: Mapping[str, Any]
) -> str:
    restart = scheduler_record["restarts"]
    if not isinstance(restart, int):
        raise InvalidEvidence(
            "cannot form an attempt suffix without scheduler Restarts"
        )
    task_component = "" if task.get("scalar") is True else f"-t{task['task_id']}"
    return f"{task['run_name']}-j{job_id}{task_component}-r{restart}"


def _find_run_directories(run_root: Path, suffixes: Iterable[str]) -> Dict[str, Path]:
    if run_root.is_symlink() or not run_root.is_dir():
        raise InvalidEvidence(f"run_root is not a non-symlink directory: {run_root}")
    suffix_list = sorted(set(suffixes))
    candidates: Dict[str, List[Path]] = {suffix: [] for suffix in suffix_list}
    try:
        children = list(run_root.iterdir())
    except OSError as error:
        raise InvalidEvidence(f"cannot list run_root {run_root}: {error}") from error
    for path in children:
        for suffix in suffix_list:
            if path.name.endswith(suffix):
                candidates[suffix].append(path)

    result: Dict[str, Path] = {}
    for suffix in suffix_list:
        paths = sorted(candidates[suffix], key=str)
        if len(paths) != 1:
            raise InvalidEvidence(
                f"expected exactly one run directory ending {suffix!r}, found {len(paths)}"
            )
        path = paths[0]
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise InvalidEvidence(
                f"cannot stat run directory {path}: {error}"
            ) from error
        if not stat.S_ISDIR(mode) or path.is_symlink():
            raise InvalidEvidence(
                f"run directory is not a non-symlink directory: {path}"
            )
        result[suffix] = path
    return result


def _event_file_provenance(run_dir: Path) -> List[Dict[str, Any]]:
    try:
        entries = sorted(run_dir.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise InvalidEvidence(
            f"cannot list run directory {run_dir}: {error}"
        ) from error
    event_files = [
        path for path in entries if path.name.startswith("events.out.tfevents")
    ]
    if not event_files:
        raise InvalidEvidence(
            f"run directory has no TensorBoard event files: {run_dir}"
        )
    provenance: List[Dict[str, Any]] = []
    for path in event_files:
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise InvalidEvidence(f"cannot stat event file {path}: {error}") from error
        if path.is_symlink() or not stat.S_ISREG(mode):
            raise InvalidEvidence(f"event file is not non-symlink regular file: {path}")
        provenance.append({"path": str(path), "sha256": _sha256(path)})
    return provenance


def _read_tensorboard_scalars(run_dir: Path) -> Mapping[str, Sequence[ScalarPoint]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            SCALARS,
            EventAccumulator,
        )
    except ImportError as error:
        raise InvalidEvidence(
            "tensorboard is required; run with the project environment"
        ) from error
    try:
        accumulator = EventAccumulator(
            str(run_dir),
            size_guidance={SCALARS: 0},
            purge_orphaned_data=False,
        )
        accumulator.Reload()
        available_tags = set(accumulator.Tags().get("scalars", []))
        return {
            tag: [
                ScalarPoint(point.step, float(point.value))
                for point in accumulator.Scalars(tag)
            ]
            for tag in REQUIRED_TAGS
            if tag in available_tags
        }
    except InvalidEvidence:
        raise
    except Exception as error:
        raise InvalidEvidence(
            f"could not load TensorBoard directory {run_dir}: {error}"
        ) from error


def _validated_series(
    raw_scalars: Mapping[str, Sequence[ScalarPoint]],
    tag: str,
    expected_steps: Sequence[int],
) -> Dict[int, float]:
    if tag not in raw_scalars:
        raise InvalidEvidence(f"missing required TensorBoard scalar {tag!r}")
    series: Dict[int, float] = {}
    duplicate_steps: List[int] = []
    for point in raw_scalars[tag]:
        step = point.step
        value = point.value
        if isinstance(step, bool) or not isinstance(step, int):
            raise InvalidEvidence(f"{tag!r} has non-integer step {step!r}")
        try:
            finite_value = float(value)
        except (TypeError, ValueError) as error:
            raise InvalidEvidence(
                f"{tag!r} has non-numeric value at step {step}"
            ) from error
        if not math.isfinite(finite_value):
            raise InvalidEvidence(f"{tag!r} has non-finite value at step {step}")
        if tag in EVAL_TAGS and not 0.0 <= finite_value <= 1.0:
            raise InvalidEvidence(f"{tag!r} has value outside [0, 1] at step {step}")
        if tag == THROUGHPUT_TAG and finite_value <= 0.0:
            raise InvalidEvidence(f"{tag!r} has non-positive value at step {step}")
        if step in series:
            duplicate_steps.append(step)
        else:
            series[step] = finite_value
    if duplicate_steps:
        raise InvalidEvidence(
            f"{tag!r} has duplicate scalar steps: {sorted(set(duplicate_steps))}"
        )
    observed_steps = set(series)
    required_steps = set(expected_steps)
    if observed_steps != required_steps:
        missing = sorted(required_steps - observed_steps)
        extra = sorted(observed_steps - required_steps)
        raise InvalidEvidence(
            f"{tag!r} scalar steps mismatch; missing={missing}, extra={extra}"
        )
    return series


def _summarize_scalars(
    raw_scalars: Mapping[str, Sequence[ScalarPoint]], expected_steps: Sequence[int]
) -> Dict[str, Any]:
    validated = {
        tag: _validated_series(raw_scalars, tag, expected_steps)
        for tag in REQUIRED_TAGS
    }
    metrics: Dict[str, Any] = {}
    for tag in EVAL_TAGS:
        series = validated[tag]
        best_value = max(series.values())
        best_step = min(step for step, value in series.items() if value == best_value)
        metrics[tag] = {
            TAIL_FIELD: statistics.fmean(series[step] for step in range(96, 101)),
            "final_step_100": series[100],
            "best_steps_0_100": best_value,
            "best_step": best_step,
        }
    throughput = validated[THROUGHPUT_TAG]
    epoch_rates = [throughput[step] for step in range(1, 101)]
    metrics[THROUGHPUT_TAG] = {
        "arithmetic_mean_of_epoch_rates_steps_1_100": statistics.fmean(epoch_rates),
        "harmonic_mean_equal_work_steps_1_100": statistics.harmonic_mean(epoch_rates),
    }
    return metrics


def _read_run(
    run_dir: Path,
    task: Mapping[str, Any],
    scheduler_record: Mapping[str, Any],
    expected_steps: Sequence[int],
    scalar_loader: ScalarLoader,
) -> Dict[str, Any]:
    before = _event_file_provenance(run_dir)
    raw_scalars = scalar_loader(run_dir)
    metrics = _summarize_scalars(raw_scalars, expected_steps)
    after = _event_file_provenance(run_dir)
    if before != after:
        raise InvalidEvidence(f"event files changed while being analyzed: {run_dir}")
    return {
        "task_id": task.get("task_id"),
        "scalar_job": task.get("scalar") is True,
        "variant": task["variant"],
        "seed": task["seed"],
        "run_name": task["run_name"],
        "scheduler": dict(scheduler_record),
        "run_directory": str(run_dir),
        "event_files": after,
        "metrics": metrics,
    }


def _aggregate(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def _aggregate_variants(
    runs: Sequence[Mapping[str, Any]], variant_metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    by_variant: Dict[str, List[Mapping[str, Any]]] = {}
    for run in runs:
        by_variant.setdefault(str(run["variant"]), []).append(run)
    aggregates: Dict[str, Any] = {}
    for variant in sorted(by_variant):
        variant_runs = sorted(by_variant[variant], key=lambda run: int(run["seed"]))
        metric_aggregates: Dict[str, Any] = {}
        for tag in EVAL_TAGS:
            metric_aggregates[tag] = {
                field: _aggregate(
                    [float(run["metrics"][tag][field]) for run in variant_runs]
                )
                for field in (
                    TAIL_FIELD,
                    "final_step_100",
                    "best_steps_0_100",
                    "best_step",
                )
            }
        metric_aggregates[THROUGHPUT_TAG] = {
            field: _aggregate(
                [float(run["metrics"][THROUGHPUT_TAG][field]) for run in variant_runs]
            )
            for field in (
                "arithmetic_mean_of_epoch_rates_steps_1_100",
                "harmonic_mean_equal_work_steps_1_100",
            )
        }
        metadata = variant_metadata.get(variant, {})
        if not isinstance(metadata, dict):
            raise InvalidEvidence(f"variant_metadata[{variant!r}] must be an object")
        aggregates[variant] = {
            "seeds": [int(run["seed"]) for run in variant_runs],
            "metrics": metric_aggregates,
            "complexity": metadata.get("complexity", "unspecified"),
            "linear_throughput_claim_eligible": metadata.get(
                "linear_throughput_claim_eligible", False
            )
            is True,
        }
    return aggregates


def _metric_value(run: Mapping[str, Any], metric: str) -> float:
    try:
        tag, field = metric.rsplit(".", 1)
        value = run["metrics"][tag][field]
    except (KeyError, ValueError, TypeError) as error:
        raise InvalidEvidence(f"invalid comparison metric path {metric!r}") from error
    return float(value)


def _paired_summary(
    comparison: Mapping[str, Any],
    policy: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    candidate = str(comparison["candidate"])
    baseline = str(comparison["baseline"])
    metric = _require_string(policy.get("metric"), "policy.metric")
    candidate_by_seed = {
        int(run["seed"]): run for run in runs if run["variant"] == candidate
    }
    baseline_by_seed = {
        int(run["seed"]): run for run in runs if run["variant"] == baseline
    }
    paired_seeds = sorted(set(candidate_by_seed) & set(baseline_by_seed))
    pairs: List[Dict[str, Any]] = []
    for seed in paired_seeds:
        candidate_value = _metric_value(candidate_by_seed[seed], metric)
        baseline_value = _metric_value(baseline_by_seed[seed], metric)
        pairs.append(
            {
                "seed": seed,
                "candidate": candidate_value,
                "baseline": baseline_value,
                "delta": candidate_value - baseline_value,
            }
        )
    deltas = [float(pair["delta"]) for pair in pairs]
    return {
        "metric": metric,
        "pairs": pairs,
        "candidate_seeds": sorted(candidate_by_seed),
        "baseline_seeds": sorted(baseline_by_seed),
        "paired_seeds": paired_seeds,
        "candidate_only_seeds": sorted(set(candidate_by_seed) - set(baseline_by_seed)),
        "baseline_only_seeds": sorted(set(baseline_by_seed) - set(candidate_by_seed)),
        "n_pairs": len(pairs),
        "mean_delta": statistics.fmean(deltas) if deltas else None,
        "sample_sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
        "min_delta": min(deltas) if deltas else None,
        "count_positive": sum(delta > 0.0 for delta in deltas),
        "positive_fraction": (
            sum(delta > 0.0 for delta in deltas) / len(deltas) if deltas else None
        ),
    }


def _apply_policy(
    policy: Mapping[str, Any],
    paired: Optional[Mapping[str, Any]],
    expected_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    policy_type = policy.get("type")
    if policy_type == "cross_snapshot_ineligible":
        return {
            "status": "INELIGIBLE_CROSS_SNAPSHOT",
            "checks": {},
        }
    if paired is None:
        raise InvalidEvidence(f"policy {policy_type!r} requires paired observations")
    n_pairs = int(paired["n_pairs"])
    if policy_type == "no_fixed_decision_rule":
        return {"status": "NO_FIXED_DECISION_RULE", "checks": {}}
    if policy_type == "insufficient_seeds":
        minimum_pairs = int(policy.get("minimum_pairs", 2))
        return {
            "status": (
                "INSUFFICIENT_SEEDS" if n_pairs < minimum_pairs else "DESCRIPTIVE_ONLY"
            ),
            "checks": {
                "minimum_pairs": {
                    "actual": n_pairs,
                    "threshold": minimum_pairs,
                    "passed": n_pairs >= minimum_pairs,
                }
            },
        }
    if policy_type not in (
        "fixed_tail_screen",
        "fixed_positive_fraction_screen",
    ):
        raise InvalidEvidence(f"unknown policy type {policy_type!r}")
    if expected_seeds is None:
        raise InvalidEvidence(f"fixed screen {policy_type!r} requires expected_seeds")

    required_pairs = int(policy.get("required_pairs", 0))
    expected_seed_list = sorted(int(seed) for seed in expected_seeds)
    checks: Dict[str, Any] = {
        "required_pairs": {
            "actual": n_pairs,
            "threshold": required_pairs,
            "passed": n_pairs == required_pairs,
        },
        "candidate_seed_set": {
            "actual": paired["candidate_seeds"],
            "expected": expected_seed_list,
            "passed": paired["candidate_seeds"] == expected_seed_list,
        },
        "baseline_seed_set": {
            "actual": paired["baseline_seeds"],
            "expected": expected_seed_list,
            "passed": paired["baseline_seeds"] == expected_seed_list,
        },
        "paired_seed_set": {
            "actual": paired["paired_seeds"],
            "expected": expected_seed_list,
            "passed": paired["paired_seeds"] == expected_seed_list,
        },
        "no_unpaired_seeds": {
            "candidate_only": paired["candidate_only_seeds"],
            "baseline_only": paired["baseline_only_seeds"],
            "passed": not paired["candidate_only_seeds"]
            and not paired["baseline_only_seeds"],
        },
    }
    positive_fraction = paired["positive_fraction"]
    positive_threshold = float(policy["positive_fraction_min"])
    checks["positive_fraction"] = {
        "actual": positive_fraction,
        "threshold": positive_threshold,
        "passed": positive_fraction is not None
        and float(positive_fraction) >= positive_threshold,
    }
    if policy_type == "fixed_tail_screen":
        mean_delta = paired["mean_delta"]
        min_delta = paired["min_delta"]
        mean_threshold = float(policy["mean_delta_min"])
        min_threshold = float(policy["min_delta_min"])
        checks["mean_delta"] = {
            "actual": mean_delta,
            "threshold": mean_threshold,
            "passed": mean_delta is not None and float(mean_delta) >= mean_threshold,
        }
        checks["min_delta"] = {
            "actual": min_delta,
            "threshold": min_threshold,
            "passed": min_delta is not None and float(min_delta) >= min_threshold,
        }
    passed = all(bool(check["passed"]) for check in checks.values())
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


def _build_comparisons(
    comparison_specs: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for comparison in comparison_specs:
        policy_name = str(comparison["policy"])
        policy = policies[policy_name]
        if policy.get("type") == "cross_snapshot_ineligible":
            paired = None
        else:
            paired = _paired_summary(comparison, policy, runs)
        results.append(
            {
                "id": comparison["id"],
                "candidate": comparison["candidate"],
                "baseline": comparison["baseline"],
                "policy": policy_name,
                "paired": paired,
                "decision": _apply_policy(
                    policy, paired, comparison.get("expected_seeds")
                ),
            }
        )
    return results


def _overall_decision(
    comparisons: Sequence[Mapping[str, Any]], evidence_eligible: bool
) -> Dict[str, Any]:
    return {
        "status": "PER_COMPARISON_ONLY",
        "evidence_eligible": evidence_eligible,
        "comparison_statuses": {
            str(comparison["id"]): str(comparison["decision"]["status"])
            for comparison in comparisons
        },
    }


def _evidence_descriptor(
    sacct_file: Optional[Path],
    skip_full_snapshot_check: bool,
    scalar_loader: Optional[ScalarLoader],
) -> Dict[str, Any]:
    test_only_reasons: List[str] = []
    if sacct_file is not None:
        test_only_reasons.append("injected_sacct_file")
    if skip_full_snapshot_check:
        test_only_reasons.append("skipped_full_snapshot_check")
    if scalar_loader is not None:
        test_only_reasons.append("injected_scalar_loader")
    return {
        "evidence_class": "TEST_ONLY" if test_only_reasons else "PRODUCTION",
        "scientific_evidence_eligible": not test_only_reasons,
        "test_only_reasons": test_only_reasons,
    }


def analyze_experiment(
    spec_path: Path,
    experiment_id: str,
    *,
    sacct_file: Optional[Path] = None,
    skip_full_snapshot_check: bool = False,
    scalar_loader: Optional[ScalarLoader] = None,
) -> Dict[str, Any]:
    registry, spec_sha256 = _load_registry(spec_path)
    experiments = registry["experiments"]
    if experiment_id not in experiments:
        raise InvalidEvidence(f"unknown experiment {experiment_id!r}")
    experiment = experiments[experiment_id]
    if not isinstance(experiment, dict):
        raise InvalidEvidence(f"experiment {experiment_id!r} must be an object")
    policies = registry["policies"]
    _validate_experiment(experiment_id, experiment, policies)
    evidence = _evidence_descriptor(sacct_file, skip_full_snapshot_check, scalar_loader)
    snapshot = _verify_snapshot(experiment["snapshot"], skip_full_snapshot_check)
    sacct_text, scheduler_source = _read_sacct(str(experiment["job_id"]), sacct_file)
    scheduler, incomplete = _parse_scheduler(sacct_text, experiment, scheduler_source)
    if incomplete:
        evidence["scientific_evidence_eligible"] = False
        evidence["ineligibility_reason"] = "incomplete_scheduler_evidence"
    status = "INCOMPLETE" if incomplete else "COMPLETE"
    if evidence["evidence_class"] == "TEST_ONLY":
        status += "_TEST_ONLY"
    report_base = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "experiment": experiment_id,
        "label": experiment.get("label"),
        "overall_policy": experiment["overall_policy"],
        "screen_provenance": experiment["screen_provenance"],
        "evidence": evidence,
        "spec": {
            "path": str(spec_path.absolute()),
            "sha256": spec_sha256,
        },
        "snapshot": snapshot,
        "scheduler": scheduler,
    }
    if incomplete:
        report_base.update(
            {
                "runs": [],
                "aggregates": {},
                "comparisons": [],
                "decision": None,
            }
        )
        return report_base

    tasks = sorted(experiment["tasks"], key=_task_sort_key)
    scheduler_by_ref = {record["job_id"]: record for record in scheduler["records"]}
    suffix_by_ref = {
        _scheduler_ref(str(experiment["job_id"]), task): _attempt_suffix(
            str(experiment["job_id"]),
            task,
            scheduler_by_ref[_scheduler_ref(str(experiment["job_id"]), task)],
        )
        for task in tasks
    }
    run_directories = _find_run_directories(
        Path(str(experiment["run_root"])), suffix_by_ref.values()
    )
    loader = scalar_loader or _read_tensorboard_scalars
    expected_steps = list(range(0, 101))
    runs: List[Dict[str, Any]] = []
    for task in tasks:
        scheduler_ref = _scheduler_ref(str(experiment["job_id"]), task)
        suffix = suffix_by_ref[scheduler_ref]
        runs.append(
            _read_run(
                run_directories[suffix],
                task,
                scheduler_by_ref[scheduler_ref],
                expected_steps,
                loader,
            )
        )
    aggregates = _aggregate_variants(runs, experiment.get("variant_metadata", {}))
    comparisons = _build_comparisons(experiment.get("comparisons", []), policies, runs)
    evidence_eligible = bool(evidence["scientific_evidence_eligible"])
    for comparison in comparisons:
        comparison["decision"]["evidence_eligible"] = evidence_eligible
    report_base.update(
        {
            "runs": runs,
            "aggregates": aggregates,
            "comparisons": comparisons,
            "decision": _overall_decision(comparisons, evidence_eligible),
        }
    )
    return report_base


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _format_mean_sd(aggregate: Mapping[str, Any]) -> str:
    mean = float(aggregate["mean"])
    sample_sd = aggregate["sample_sd"]
    return (
        f"{mean:.6f}" if sample_sd is None else f"{mean:.6f}+/-{float(sample_sd):.6f}"
    )


def _print_summary(report: Mapping[str, Any]) -> None:
    print(f"experiment {report['experiment']}: {report['status']}")
    print(
        f"evidence={report['evidence']['evidence_class']} "
        f"eligible={report['evidence']['scientific_evidence_eligible']}"
    )
    if str(report["status"]).startswith("INCOMPLETE"):
        for record in report["scheduler"]["records"]:
            if record["state"] != "COMPLETED" or record["exit_code"] != "0:0":
                print(
                    f"  {record['job_id']}: state={record['state']} "
                    f"exit={record['exit_code']} restarts={record['restarts']}"
                )
        print("decision: none (incomplete scheduler evidence)")
        return
    print(
        f"{'variant':24s} {'n':>2s} {'NDCG@10 tail':>22s} "
        f"{'HR@10 tail':>22s} {'MRR tail':>22s} "
        f"{'epoch-rate arith':>22s} {'equal-work harmonic':>22s} {'linear?':>8s}"
    )
    for variant, aggregate in report["aggregates"].items():
        metrics = aggregate["metrics"]
        ndcg = metrics["eval_epoch/ndcg@10"][TAIL_FIELD]
        hr = metrics["eval_epoch/hr@10"][TAIL_FIELD]
        mrr = metrics["eval_epoch/mrr"][TAIL_FIELD]
        arithmetic_throughput = metrics[THROUGHPUT_TAG][
            "arithmetic_mean_of_epoch_rates_steps_1_100"
        ]
        harmonic_throughput = metrics[THROUGHPUT_TAG][
            "harmonic_mean_equal_work_steps_1_100"
        ]
        print(
            f"{variant:24s} {ndcg['n']:2d} {_format_mean_sd(ndcg):>22s} "
            f"{_format_mean_sd(hr):>22s} {_format_mean_sd(mrr):>22s} "
            f"{_format_mean_sd(arithmetic_throughput):>22s} "
            f"{_format_mean_sd(harmonic_throughput):>22s} "
            f"{'yes' if aggregate['linear_throughput_claim_eligible'] else 'no':>8s}"
        )
    for comparison in report["comparisons"]:
        paired = comparison["paired"]
        if paired is None:
            detail = "no eligible within-snapshot pairing"
        elif paired["n_pairs"] == 0:
            detail = "no paired seeds"
        else:
            detail = (
                f"delta={paired['mean_delta']:+.6f} "
                f"sd={paired['sample_sd'] if paired['sample_sd'] is not None else 'null'} "
                f"min={paired['min_delta']:+.6f} "
                f"positive={paired['count_positive']}/{paired['n_pairs']}"
            )
        print(
            f"comparison {comparison['id']}: {comparison['decision']['status']} "
            f"({detail})"
        )
    print(f"decision: {report['decision']['status']}")


def _invalid_report(
    spec_path: Path,
    experiment_id: str,
    error: InvalidEvidence,
    evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    spec_sha256: Optional[str]
    try:
        spec_sha256 = _sha256(spec_path)
    except OSError:
        spec_sha256 = None
    invalid_evidence = dict(evidence) if evidence is not None else None
    if invalid_evidence is not None:
        invalid_evidence["scientific_evidence_eligible"] = False
        invalid_evidence["ineligibility_reason"] = "invalid_evidence"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "INVALID",
        "experiment": experiment_id,
        "spec": {"path": str(spec_path.absolute()), "sha256": spec_sha256},
        "error": str(error),
        "decision": None,
        "evidence": invalid_evidence,
    }


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    rendered = render_json(report).encode("utf-8")
    temporary: Optional[Path] = None
    file_descriptor: Optional[int] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination_mode = path.lstat().st_mode
        except FileNotFoundError:
            destination_mode = None
        if destination_mode is not None and (
            stat.S_ISLNK(destination_mode) or not stat.S_ISREG(destination_mode)
        ):
            raise InvalidEvidence(
                f"JSON destination is not a regular non-symlink file: {path}"
            )
        for _attempt in range(100):
            candidate = path.parent / (
                f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
            )
            try:
                file_descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                continue
            temporary = candidate
            break
        if temporary is None or file_descriptor is None:
            raise InvalidEvidence(f"could not allocate atomic JSON temp for {path}")
        with os.fdopen(file_descriptor, "wb") as output:
            file_descriptor = None
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
        try:
            destination_mode = path.lstat().st_mode
        except FileNotFoundError:
            destination_mode = None
        if destination_mode is not None and (
            stat.S_ISLNK(destination_mode) or not stat.S_ISREG(destination_mode)
        ):
            raise InvalidEvidence(
                f"JSON destination changed to a nonregular node: {path}"
            )
        os.replace(temporary, path)
        temporary = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise InvalidEvidence(f"cannot write JSON report {path}: {error}") from error
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--sacct-file",
        type=Path,
        help="tests only: inject sacct output; reports exit 4 and are noneligible",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--skip-full-snapshot-check",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fail-on-decision",
        action="store_true",
        help="exit 5 when any eligible fixed comparison has decision FAIL",
    )
    return parser.parse_args(argv)


def _report_exit_code(report: Mapping[str, Any], fail_on_decision: bool) -> int:
    if report.get("evidence", {}).get("evidence_class") == "TEST_ONLY":
        return 4
    if report["status"] == "INCOMPLETE":
        return 3
    if fail_on_decision and any(
        comparison["decision"]["status"] == "FAIL"
        and comparison["decision"].get("evidence_eligible") is True
        for comparison in report.get("comparisons", [])
    ):
        return 5
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        report = analyze_experiment(
            args.spec,
            args.experiment,
            sacct_file=args.sacct_file,
            skip_full_snapshot_check=args.skip_full_snapshot_check,
        )
        if args.json_out is not None:
            _write_json(args.json_out, report)
        _print_summary(report)
        return _report_exit_code(report, args.fail_on_decision)
    except InvalidEvidence as error:
        evidence = _evidence_descriptor(
            args.sacct_file, args.skip_full_snapshot_check, None
        )
        report = _invalid_report(args.spec, args.experiment, error, evidence)
        if args.json_out is not None:
            try:
                _write_json(args.json_out, report)
            except InvalidEvidence as write_error:
                print(f"INVALID: {write_error}", file=sys.stderr)
        print(f"INVALID: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
