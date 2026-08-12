#!/usr/bin/env python3
"""Probe Signed-LIFT feature geometry from exact ML-1M W32 checkpoints.

This diagnostic is CPU-only. It does not alter model behavior or register any
persistent operators: when FBGEMM's three layout helpers are unavailable, it
installs scoped one-dimensional PyTorch equivalents and restores the operator
namespace before returning.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import re
import secrets
import stat
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional


CANONICAL_TRAINING_SNAPSHOT = Path(
    "/checkpoints/ngocbh/longhstu/code_snapshots/"
    "attention_20260812T065504Z_3685881"
)
SNAPSHOT_MANIFEST_NAME = "SOURCE_SHA256SUMS"
SNAPSHOT_MANIFEST_SHA256 = (
    "8bc8b6af52cdae4873f8720d75bf46f3121b6f796a17975068a6efbc2b23fd68"
)
LEGACY_UNMANIFESTED_ROOT_FILES = {
    "GIT_COMMIT": "8e334b25223f66be7ddfe64ec469dd2d95c5c8da03ac2f5c4321fdecf7f248f4",
    "GIT_STATUS": "175cf34a60115590dfcc225612211c1f26189d39055eac3b472f6df7245c34fe",
    "WORKTREE.patch": "2d8a5579c75cd4ca4e9e1ae601e4f47447ecb14b46dd2a1c07029ca1167c87fc",
}


def _early_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonsymlink_path_components(path: Path) -> None:
    absolute = path.absolute()
    components = [absolute, *absolute.parents]
    for component in components:
        mode = os.lstat(component).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"path component is a symlink: {component}")


def _scan_regular_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory: Path, relative: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                child_relative = relative / entry.name
                relative_text = child_relative.as_posix()
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise ValueError(
                        f"training snapshot contains a symlink: {relative_text}"
                    )
                if stat.S_ISDIR(mode):
                    directories.add(relative_text)
                    visit(Path(entry.path), child_relative)
                elif stat.S_ISREG(mode):
                    files.add(relative_text)
                else:
                    raise ValueError(
                        "training snapshot contains a non-file/non-directory node: "
                        f"{relative_text}"
                    )

    visit(root, Path())
    return files, directories


def verify_training_snapshot(path: Path) -> dict[str, object]:
    requested = path.expanduser().absolute()
    if requested != CANONICAL_TRAINING_SNAPSHOT:
        raise ValueError(
            "training snapshot must be the exact canonical path "
            f"{CANONICAL_TRAINING_SNAPSHOT}, got {requested}"
        )
    _require_nonsymlink_path_components(requested)
    root_mode = os.lstat(requested).st_mode
    if not stat.S_ISDIR(root_mode):
        raise ValueError(f"training snapshot is not a directory: {requested}")

    manifest_path = requested / SNAPSHOT_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("training snapshot manifest is not a regular non-symlink file")
    manifest_sha = _early_sha256_file(manifest_path)
    if manifest_sha != SNAPSHOT_MANIFEST_SHA256:
        raise ValueError(
            f"training snapshot manifest SHA256 mismatch: {manifest_sha}"
        )

    manifest_entries: dict[str, str] = {}
    pattern = re.compile(r"^(?P<sha>[0-9a-f]{64})  (?P<path>.+)$")
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed snapshot manifest line {line_number}")
        relative = match.group("path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or relative in manifest_entries
        ):
            raise ValueError(f"unsafe or duplicate snapshot manifest path: {relative}")
        manifest_entries[relative] = match.group("sha")
    if len(manifest_entries) != 485:
        raise ValueError(
            f"snapshot manifest entry count mismatch: {len(manifest_entries)} != 485"
        )

    actual_files, actual_directories = _scan_regular_tree(requested)
    expected_files = set(manifest_entries) | {
        SNAPSHOT_MANIFEST_NAME,
        *LEGACY_UNMANIFESTED_ROOT_FILES,
    }
    if actual_files != expected_files:
        raise ValueError(
            "snapshot file tree mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    expected_directories: set[str] = set()
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_directories != expected_directories:
        raise ValueError(
            "snapshot directory tree mismatch: "
            f"missing={sorted(expected_directories - actual_directories)}, "
            f"extra={sorted(actual_directories - expected_directories)}"
        )

    verified_hashes = dict(manifest_entries)
    verified_hashes[SNAPSHOT_MANIFEST_NAME] = manifest_sha
    verified_hashes.update(LEGACY_UNMANIFESTED_ROOT_FILES)
    for relative, expected_sha in sorted(verified_hashes.items()):
        actual_sha = _early_sha256_file(requested / relative)
        if actual_sha != expected_sha:
            raise ValueError(
                f"snapshot checksum mismatch for {relative}: {actual_sha}"
            )

    tree_records = [f"d  {relative}" for relative in sorted(actual_directories)]
    tree_records.extend(
        f"f  {verified_hashes[relative]}  {relative}"
        for relative in sorted(actual_files)
    )
    tree_sha = hashlib.sha256(
        ("\n".join(tree_records) + "\n").encode("utf-8")
    ).hexdigest()
    research_python = {
        relative: manifest_entries[relative]
        for relative in sorted(manifest_entries)
        if relative.startswith("generative_recommenders/research/")
        and relative.endswith(".py")
    }
    research_records = [
        f"{sha}  {relative}" for relative, sha in research_python.items()
    ]
    research_sha = hashlib.sha256(
        ("\n".join(research_records) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "path": str(requested),
        "manifest": SNAPSHOT_MANIFEST_NAME,
        "manifest_sha256": manifest_sha,
        "manifest_entry_count": len(manifest_entries),
        "legacy_unmanifested_root_file_allowlist": dict(
            LEGACY_UNMANIFESTED_ROOT_FILES
        ),
        "file_count_including_manifest_and_legacy_metadata": len(actual_files),
        "directory_count": len(actual_directories),
        "tree_sha256": tree_sha,
        "research_python_file_count": len(research_python),
        "research_python_tree_sha256": research_sha,
        "research_python_inventory": research_python,
        "symlinks_allowed": False,
        "special_nodes_allowed": False,
    }


def _training_snapshot_argument(argv: Sequence[str]) -> Optional[Path]:
    values: list[str] = []
    for index, argument in enumerate(argv):
        if argument == "--training-snapshot":
            if index + 1 >= len(argv):
                raise ValueError("--training-snapshot requires a path")
            values.append(argv[index + 1])
        elif argument.startswith("--training-snapshot="):
            values.append(argument.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError("--training-snapshot may be specified only once")
    return Path(values[0]) if values else None


_BOOTSTRAP_SNAPSHOT_ARGUMENT = _training_snapshot_argument(sys.argv[1:])
_BOOTSTRAP_SNAPSHOT_PROVENANCE: Optional[dict[str, object]] = None
if _BOOTSTRAP_SNAPSHOT_ARGUMENT is not None:
    preloaded_gr = sorted(
        name
        for name in sys.modules
        if name == "generative_recommenders"
        or name.startswith("generative_recommenders.")
    )
    if preloaded_gr:
        raise RuntimeError(
            "GR modules were loaded before training-snapshot bootstrap: "
            f"{preloaded_gr}"
        )
    _BOOTSTRAP_SNAPSHOT_PROVENANCE = verify_training_snapshot(
        _BOOTSTRAP_SNAPSHOT_ARGUMENT
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(CANONICAL_TRAINING_SNAPSHOT))

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from generative_recommenders.research.data.dataset import DatasetV2
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.features import (
    SequentialFeatures,
    movielens_seq_features_from_row,
)
from generative_recommenders.research.modeling.sequential.hstu import HSTU
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
)
from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import (
    DotProductSimilarity,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
CPU_THREADS = 8
WINDOW_SIZE = 32
FEATURE_EPSILON = 1e-6
DEFAULT_GAMMAS = (0.5, 1.0, 2.0, 4.0)
EXPLORATORY_MAP_NAMES = (
    "per_vector_standardized_tanh",
    "signed_sqrt_unit_rms",
    "per_vector_standardized_tanh_rms_matched",
    "signed_sqrt_rms_matched",
)
CANONICAL_DATA_SHA256 = (
    "6e058859cd7c0e9bb2b7f17ad54056e7fbcc2eb12c6efed6c66bf4e6a4cf735a"
)
CANONICAL_DATA_ROWS = 6040
EXPECTED_PARAMETER_COUNT = 313_432
EXPECTED_STATE_TENSOR_COUNT = 67
EXPECTED_DATA_COLUMNS = (
    "index",
    "user_id",
    "sequence_item_ids",
    "sequence_ratings",
    "sequence_timestamps",
    "sex",
    "age_group",
    "occupation",
    "zip_code",
)
FBGEMM_LAYOUT_OPS = (
    "asynchronous_complete_cumsum",
    "dense_to_jagged",
    "jagged_to_padded_dense",
)

MODEL_SETTINGS: dict[str, Any] = {
    "dataset": "ml-1m",
    "max_sequence_length": 200,
    "max_output_length": 11,
    "padded_sequence_length": 211,
    "item_count_including_padding": 3953,
    "item_embedding_dim": 50,
    "num_blocks": 8,
    "num_heads": 2,
    "attention_dim": 25,
    "linear_dim": 25,
    "linear_activation": "silu",
    "linear_dropout_rate": 0.2,
    "attention_dropout_rate": 0.0,
    "relative_attention_bias": True,
    "forgetting_min_period": 8.0,
    "forgetting_max_period": 256.0,
    "hybrid_window_size": WINDOW_SIZE,
    "tail_feature_map_in_checkpoint": "identity",
}

GLOBAL_INTERPRETATION_LIMITS: dict[str, Any] = {
    "analysis_type": "post_hoc_descriptive_feature_geometry",
    "confirmatory_inference": False,
    "trained_candidate_arm_comparison": False,
    "accuracy_estimate": False,
    "causal_evidence": False,
    "predictive_quality_evidence": False,
    "candidate_learned_gain_evidence": False,
    "post_hoc_transform_disclosure": (
        "All tanh, abs-tanh, gamma-sweep, standardized-tanh, and signed-sqrt "
        "features are post-hoc transforms of Q/K learned by the source local "
        "or identity-tail LIFT checkpoint."
    ),
    "not_trained_arm_representations": True,
    "gamma_roles": {
        "1.0": "current_setting_diagnostic",
        "0.5": "post_hoc_exploratory",
        "2.0": "post_hoc_exploratory",
        "4.0": "post_hoc_exploratory",
    },
    "exploratory_feature_maps_are_post_hoc": list(EXPLORATORY_MAP_NAMES),
}


def checkpoint_interpretation_limits(kind: str) -> dict[str, Any]:
    if kind == "local":
        training_context = "local_arm_tail_disabled"
        learned_gain_scope = (
            "Tail gains were zero-connected and the tail was disabled during training."
        )
        tail_active_during_training = False
    elif kind == "lift":
        training_context = "lift_arm_identity_tail_active"
        learned_gain_scope = (
            "Tail gains were learned only with the identity feature tail."
        )
        tail_active_during_training = True
    else:
        raise ValueError(f"unknown checkpoint kind {kind!r}")
    return {
        "training_context": training_context,
        "tail_active_during_training": tail_active_during_training,
        "trained_tail_feature_map": "identity" if kind == "lift" else None,
        "learned_gain_scope": learned_gain_scope,
        "post_hoc_transforms_are_trained_representations": False,
        "post_hoc_transforms_have_learned_gains": False,
        "accuracy_estimate": False,
        "causal_or_predictive_quality_evidence": False,
        "confirmatory_inference": False,
    }


def global_interpretation_limits(gammas: Sequence[float]) -> dict[str, Any]:
    limits = json.loads(json.dumps(GLOBAL_INTERPRETATION_LIMITS))
    limits["gamma_roles"] = {
        str(float(gamma)): (
            "current_setting_diagnostic"
            if float(gamma) == 1.0
            else "post_hoc_exploratory"
        )
        for gamma in gammas
    }
    return limits

_CHECKPOINT_RE = re.compile(
    r"^HSTU-b8-h2-dqk25-dv25-lsilud0\.2-ad0\.0-"
    r"(?P<operator>localfohstu|hybridfohstu)-w32-t8-256_"
    r"DotProduct_local-l2-eps1e-06_ssl-t0\.05-n128-b128-"
    r"lr0\.001-wu0-wd0-2026-08-12-ml1m-fohstu-"
    r"(?P<run_kind>local|hybrid)-w32-seed(?P<seed>42|43|44)-"
    r"j(?P<job>1671578)-t(?P<task>1|2|6|7|11|12)-"
    r"r(?P<restart>0)_last\.pt$"
)


def _expected_checkpoint_filename(kind: str, seed: int, task: int) -> str:
    operator = "localfohstu" if kind == "local" else "hybridfohstu"
    run_kind = "local" if kind == "local" else "hybrid"
    return (
        "HSTU-b8-h2-dqk25-dv25-lsilud0.2-ad0.0-"
        f"{operator}-w32-t8-256_DotProduct_local-l2-eps1e-06_"
        "ssl-t0.05-n128-b128-lr0.001-wu0-wd0-2026-08-12-"
        f"ml1m-fohstu-{run_kind}-w32-seed{seed}-j1671578-t{task}-r0_last.pt"
    )


_EXPECTED_CHECKPOINT_ROWS = (
    ("local", 42, 1, "2879d336de3ca2a536d64492afbea0cc79a1664ed55db77bc8335621ed09ba17"),
    ("lift", 42, 2, "b86b64b3dad36da26e02284192caf4f8a644c3d5d89e713db501b2438b9fe332"),
    ("local", 43, 6, "8666ef4da24a0c261374a99f106868eb67437b1d42c17802d9c861a2838ed420"),
    ("lift", 43, 7, "535d2d9c5191edbdcf32d10b51b9b526bc12cd705fde6485e955f62a54b99966"),
    ("local", 44, 11, "7f09ff3a2f25cb37ab3079b2c54f553516d4c8853e4f962f47d71abbba2ee9f2"),
    ("lift", 44, 12, "84ac72d3cf8cf409018cfe91b4aad60943aec08c87548ba5557617101b6c5b12"),
)
EXPECTED_CHECKPOINTS = {
    _expected_checkpoint_filename(kind, seed, task): {
        "kind": kind,
        "seed": seed,
        "job_id": 1671578,
        "task_id": task,
        "restart_count": 0,
        "sha256": sha256,
    }
    for kind, seed, task, sha256 in _EXPECTED_CHECKPOINT_ROWS
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(result: os.stat_result) -> dict[str, int]:
    return {
        "device": int(result.st_dev),
        "inode": int(result.st_ino),
        "size": int(result.st_size),
        "mtime_ns": int(result.st_mtime_ns),
        "ctime_ns": int(result.st_ctime_ns),
    }


def read_verified_regular_file(
    path: Path,
    expected_sha256: Optional[str] = None,
) -> tuple[bytes, str, dict[str, int]]:
    resolved = path.expanduser().absolute()
    _require_nonsymlink_path_components(resolved)
    before = os.lstat(resolved)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"path is not a regular file: {resolved}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stat_identity(opened) != _stat_identity(
            before
        ):
            raise ValueError(f"file identity changed while opening: {resolved}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_read = os.fstat(descriptor)
        if _stat_identity(after_read) != _stat_identity(opened):
            raise ValueError(f"file identity changed while reading: {resolved}")
    finally:
        os.close(descriptor)
    after_close = os.lstat(resolved)
    identity = _stat_identity(opened)
    if _stat_identity(after_close) != identity:
        raise ValueError(f"file identity changed after reading: {resolved}")
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"file SHA256 mismatch for {resolved}: {digest} != {expected_sha256}"
        )
    return payload, digest, identity


def revalidate_regular_file(
    path: Path,
    expected_sha256: str,
    expected_identity: Mapping[str, int],
) -> None:
    _, digest, identity = read_verified_regular_file(path, expected_sha256)
    if digest != expected_sha256 or identity != dict(expected_identity):
        raise ValueError(f"file identity/content changed after consumption: {path}")


def load_checkpoint_from_verified_bytes(payload: bytes) -> object:
    return torch.load(
        io.BytesIO(payload),
        map_location=torch.device("cpu"),
        weights_only=True,
    )


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    ) + "\n"


def validate_gammas(gammas: Sequence[float]) -> tuple[float, ...]:
    if not gammas:
        raise ValueError("at least one gamma is required")
    normalized: list[float] = []
    for gamma in gammas:
        if isinstance(gamma, bool):
            raise ValueError("gamma values must be real numbers, not bool")
        value = float(gamma)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"gamma values must be finite and positive, got {gamma}")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"gamma values must be unique, got {normalized}")
    return tuple(sorted(normalized))


def validate_checkpoint_epoch(epoch: object) -> int:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != 100:
        raise ValueError(
            f"completed checkpoint epoch must equal exactly 100, got {epoch!r}"
        )
    return epoch


def select_user_indices(
    total_users: int,
    max_users: int,
    sampling: str,
    seed: int,
) -> list[int]:
    if total_users <= 0:
        raise ValueError(f"total_users must be positive, got {total_users}")
    if max_users <= 0 or max_users > total_users:
        raise ValueError(
            f"max_users must be in [1, {total_users}], got {max_users}"
        )
    if sampling == "first":
        return list(range(max_users))
    if sampling == "seeded":
        return random.Random(seed).sample(range(total_users), max_users)
    raise ValueError(f"sampling must be 'first' or 'seeded', got {sampling!r}")


def checkpoint_identity(path: Path) -> dict[str, Any]:
    match = _CHECKPOINT_RE.fullmatch(path.name)
    expected = EXPECTED_CHECKPOINTS.get(path.name)
    if match is None or expected is None:
        raise ValueError(
            "checkpoint filename is not one of the exact job-1671578 W32 "
            f"artifacts: {path.name}"
        )
    parsed_kind = "local" if match.group("operator") == "localfohstu" else "lift"
    parsed_run_kind = "local" if match.group("run_kind") == "local" else "lift"
    parsed = {
        "kind": parsed_kind,
        "seed": int(match.group("seed")),
        "job_id": int(match.group("job")),
        "task_id": int(match.group("task")),
        "restart_count": int(match.group("restart")),
    }
    for key, value in parsed.items():
        if expected[key] != value:
            raise ValueError(
                f"checkpoint filename/mapping mismatch for {key}: {value}"
            )
    if parsed_kind != parsed_run_kind:
        raise ValueError("checkpoint operator and run-kind tags disagree")
    return dict(expected)


def checkpoint_kind(path: Path) -> str:
    return str(checkpoint_identity(path)["kind"])


def validate_canonical_checkpoint_set(paths: Sequence[Path]) -> None:
    basenames = [path.name for path in paths]
    expected = set(EXPECTED_CHECKPOINTS)
    actual = set(basenames)
    if len(basenames) != len(actual):
        raise ValueError("canonical checkpoint basenames must be unique")
    if actual != expected:
        raise ValueError(
            "canonical checkpoint basename set mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def strip_ddp_prefix(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not state:
        raise ValueError("checkpoint state dictionary is empty")
    prefixed = [key.startswith("module.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("checkpoint state dictionary has a mixed DDP prefix")
    if all(prefixed):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return dict(state)


def validate_state_inventory(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> None:
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"checkpoint inventory mismatch: missing={missing}, extra={extra}")
    for name in sorted(expected):
        value = actual[name]
        reference = expected[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint entry {name} is not a tensor")
        if value.device.type != "cpu":
            raise ValueError(f"checkpoint entry {name} is not on CPU")
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"checkpoint shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(reference.shape)}"
            )
        if value.dtype != reference.dtype:
            raise ValueError(
                f"checkpoint dtype mismatch for {name}: {value.dtype} != {reference.dtype}"
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint entry {name} contains non-finite values")
        if name == "_attn_mask" and not torch.equal(value, reference):
            raise ValueError("checkpoint causal attention mask differs from exact W32 model")


def inventory_sha256(state: Mapping[str, torch.Tensor]) -> str:
    records = [
        f"{name}|{tuple(value.shape)}|{value.dtype}"
        for name, value in sorted(state.items())
    ]
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def validate_tail_gain_state(kind: str, rho_values: torch.Tensor) -> None:
    if kind not in ("local", "lift"):
        raise ValueError(f"unknown checkpoint kind {kind!r}")
    if rho_values.device.type != "cpu" or rho_values.numel() != 16:
        raise ValueError("checkpoint does not have exactly 8x2 CPU tail-gain values")
    nonzero = rho_values != 0
    if kind == "local" and bool(nonzero.any()):
        raise ValueError("local checkpoint has non-zero tail-gain parameters")
    if kind == "lift" and not bool(nonzero.all()):
        raise ValueError("completed LIFT checkpoint has an untrained zero tail gain")


def validate_data_csv(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().absolute()
    _, digest, identity = read_verified_regular_file(
        resolved, CANONICAL_DATA_SHA256
    )
    columns = tuple(pd.read_csv(resolved, nrows=0).columns.tolist())
    if columns != EXPECTED_DATA_COLUMNS:
        raise ValueError(f"data CSV columns mismatch: {columns}")
    user_frame = pd.read_csv(resolved, usecols=["user_id"])
    user_ids = [int(value) for value in user_frame["user_id"].tolist()]
    if len(user_ids) != CANONICAL_DATA_ROWS:
        raise ValueError(
            f"data CSV row count mismatch: {len(user_ids)} != {CANONICAL_DATA_ROWS}"
        )
    if len(set(user_ids)) != CANONICAL_DATA_ROWS or set(user_ids) != set(
        range(1, CANONICAL_DATA_ROWS + 1)
    ):
        raise ValueError("data CSV user IDs are not exactly 1..6040")
    return {
        "path": str(resolved),
        "sha256": digest,
        "file_identity": identity,
        "row_count": len(user_ids),
        "columns": list(columns),
    }


def _validate_cpu_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU, got {tensor.device}")


def _validated_offsets(offsets: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(offsets) != 1:
        raise ValueError("the CPU layout shim supports exactly one jagged dimension")
    offset = offsets[0]
    _validate_cpu_tensor(offset, "offsets[0]")
    if offset.dim() != 1 or offset.dtype not in (torch.int32, torch.int64):
        raise ValueError("offsets[0] must be a one-dimensional int32/int64 tensor")
    if offset.numel() < 1 or int(offset[0]) != 0:
        raise ValueError("offsets[0] must start at zero")
    if bool((offset[1:] < offset[:-1]).any()):
        raise ValueError("offsets[0] must be non-decreasing")
    return offset


def _shim_complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    _validate_cpu_tensor(lengths, "lengths")
    if lengths.dim() != 1 or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be a one-dimensional int32/int64 tensor")
    if bool((lengths < 0).any()):
        raise ValueError("lengths must be non-negative")
    return torch.cat((lengths.new_zeros(1), torch.cumsum(lengths, dim=0)))


def _shim_dense_to_jagged(
    dense: torch.Tensor,
    offsets: Sequence[torch.Tensor],
    total_L: Optional[int] = None,
) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
    _validate_cpu_tensor(dense, "dense")
    if dense.dim() < 2:
        raise ValueError("dense must have batch and sequence dimensions")
    offset = _validated_offsets(offsets)
    if offset.numel() != dense.size(0) + 1:
        raise ValueError("offset count does not match the dense batch size")
    lengths = offset[1:] - offset[:-1]
    if bool((lengths > dense.size(1)).any()):
        raise ValueError("an offset length exceeds the dense sequence dimension")
    final_length = int(offset[-1])
    if total_L is not None and int(total_L) != final_length:
        raise ValueError(f"total_L {total_L} does not match offsets {final_length}")
    pieces = [dense[index, : int(length)] for index, length in enumerate(lengths)]
    if pieces:
        values = torch.cat(pieces, dim=0)
    else:
        values = dense.new_empty((0, *dense.shape[2:]))
    return values, offsets


def _shim_jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: Sequence[torch.Tensor],
    max_lengths: Sequence[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    _validate_cpu_tensor(values, "values")
    if values.dim() < 1:
        raise ValueError("values must have a row dimension")
    offset = _validated_offsets(offsets)
    if len(max_lengths) != 1 or int(max_lengths[0]) < 0:
        raise ValueError("max_lengths must contain one non-negative length")
    if int(offset[-1]) != values.size(0):
        raise ValueError("the final offset does not match the jagged row count")
    batch_size = offset.numel() - 1
    max_length = int(max_lengths[0])
    output = values.new_full(
        (batch_size, max_length, *values.shape[1:]),
        padding_value,
    )
    for batch_index in range(batch_size):
        start = int(offset[batch_index])
        length = min(int(offset[batch_index + 1]) - start, max_length)
        if length:
            output[batch_index, :length] = values[start : start + length]
    return output


@contextlib.contextmanager
def scoped_fbgemm_layout_shims() -> Iterator[tuple[str, ...]]:
    namespace = torch.ops.fbgemm
    original = {
        name: (name in namespace.__dict__, namespace.__dict__.get(name))
        for name in FBGEMM_LAYOUT_OPS
    }
    implementations: dict[str, Callable[..., Any]] = {
        "asynchronous_complete_cumsum": _shim_complete_cumsum,
        "dense_to_jagged": _shim_dense_to_jagged,
        "jagged_to_padded_dense": _shim_jagged_to_padded_dense,
    }
    installed: list[str] = []
    try:
        for name in FBGEMM_LAYOUT_OPS:
            try:
                getattr(namespace, name)
            except AttributeError:
                setattr(namespace, name, implementations[name])
                installed.append(name)
        yield tuple(installed)
    finally:
        for name, (was_present, value) in original.items():
            if was_present:
                namespace.__dict__[name] = value
            else:
                namespace.__dict__.pop(name, None)


@contextlib.contextmanager
def scoped_cpu_thread_count(thread_count: int) -> Iterator[None]:
    if thread_count <= 0:
        raise ValueError(f"CPU thread count must be positive, got {thread_count}")
    original = torch.get_num_threads()
    torch.set_num_threads(thread_count)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def old_pair_mask(
    lengths: torch.Tensor,
    sequence_length: int,
    window_size: int = WINDOW_SIZE,
) -> torch.Tensor:
    _validate_cpu_tensor(lengths, "lengths")
    if lengths.dim() != 1 or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be a one-dimensional integer tensor")
    if sequence_length <= 0 or window_size <= 0:
        raise ValueError("sequence_length and window_size must be positive")
    if bool((lengths <= 0).any()) or bool((lengths > sequence_length).any()):
        raise ValueError("lengths must be in [1, sequence_length]")
    positions = torch.arange(sequence_length, device=lengths.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    distances = positions.unsqueeze(1) - positions.unsqueeze(0)
    return (
        (distances >= window_size).unsqueeze(0)
        & valid.unsqueeze(2)
        & valid.unsqueeze(1)
    )


def survival_from_log_forget(log_forget: torch.Tensor) -> torch.Tensor:
    _validate_cpu_tensor(log_forget, "log_forget")
    if log_forget.dim() != 3 or not log_forget.is_floating_point():
        raise ValueError("log_forget must be a floating [B, N, H] tensor")
    if not bool(torch.isfinite(log_forget).all()):
        raise ValueError("log_forget contains non-finite values")
    if bool((log_forget > 1e-7).any()):
        raise ValueError("log_forget must be non-positive")
    accumulation_dtype = (
        torch.float64 if log_forget.dtype == torch.float64 else torch.float32
    )
    prefix = torch.cumsum(log_forget.to(accumulation_dtype), dim=1).transpose(1, 2)
    log_survival = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
    return torch.exp(torch.clamp_max(log_survival, 0.0)).to(log_forget.dtype)


def coefficient_summary(values: torch.Tensor) -> dict[str, Any]:
    _validate_cpu_tensor(values, "coefficient values")
    flattened = values.reshape(-1)
    if flattened.numel() == 0:
        raise ValueError("coefficient metric received no old pairs")
    if not flattened.is_floating_point() or not bool(torch.isfinite(flattened).all()):
        raise ValueError("coefficient values must be finite floating point")
    absolute_mass = flattened.abs().sum()
    if float(absolute_mass) == 0.0:
        raise ValueError("coefficient metric has zero absolute mass")
    negative = flattened < 0
    return {
        "pair_count": int(flattened.numel()),
        "negative_pair_fraction": float(negative.to(torch.float64).mean()),
        "negative_l1_mass_fraction": float(
            (-flattened[negative]).sum() / absolute_mass
        ),
        "rms": float(flattened.to(torch.float64).square().mean().sqrt()),
    }


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    _validate_cpu_tensor(left, "left correlation values")
    _validate_cpu_tensor(right, "right correlation values")
    left_flat = left.reshape(-1).to(torch.float64)
    right_flat = right.reshape(-1).to(torch.float64)
    if left_flat.numel() == 0 or left_flat.shape != right_flat.shape:
        raise ValueError("correlation inputs must be non-empty and shape matched")
    if not bool(torch.isfinite(left_flat).all()) or not bool(
        torch.isfinite(right_flat).all()
    ):
        raise ValueError("correlation inputs contain non-finite values")
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = (
        left_centered.square().mean().sqrt()
        * right_centered.square().mean().sqrt()
    )
    if float(denominator) == 0.0:
        raise ValueError("correlation is undefined for zero variance")
    value = (left_centered * right_centered).mean() / denominator
    return float(value.clamp(min=-1.0, max=1.0))


def standardized_tanh_feature_map(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=-1, keepdim=True)
    variance = features.var(dim=-1, keepdim=True, unbiased=False)
    return torch.tanh((features - mean) / torch.sqrt(variance + FEATURE_EPSILON))


def signed_sqrt_rms_feature_map(features: torch.Tensor) -> torch.Tensor:
    mapped = torch.sign(features) * torch.sqrt(
        torch.abs(features) + FEATURE_EPSILON
    )
    rms = torch.sqrt(mapped.square().mean(dim=-1, keepdim=True)).clamp_min(
        FEATURE_EPSILON
    )
    return mapped / rms


def rms_match_feature_map(
    original: torch.Tensor,
    mapped: torch.Tensor,
) -> torch.Tensor:
    if original.shape != mapped.shape:
        raise ValueError("original and mapped features must have identical shapes")
    original_rms = torch.sqrt(original.square().mean(dim=-1, keepdim=True))
    mapped_rms = torch.sqrt(mapped.square().mean(dim=-1, keepdim=True)).clamp_min(
        FEATURE_EPSILON
    )
    return mapped * (original_rms / mapped_rms)


def standardized_tanh_rms_matched_feature_map(
    features: torch.Tensor,
) -> torch.Tensor:
    return rms_match_feature_map(features, standardized_tanh_feature_map(features))


def signed_sqrt_rms_matched_feature_map(features: torch.Tensor) -> torch.Tensor:
    return rms_match_feature_map(features, signed_sqrt_rms_feature_map(features))


def _coordinate_summary(
    features: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> dict[str, Any]:
    selected = features[valid_tokens]
    if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
        raise ValueError("coordinate metric has no finite valid features")
    return {
        "coordinate_count": int(selected.numel()),
        "positive_fraction": float((selected > 0).to(torch.float64).mean()),
        "negative_fraction": float((selected < 0).to(torch.float64).mean()),
        "zero_fraction": float((selected == 0).to(torch.float64).mean()),
        "mean": float(selected.to(torch.float64).mean()),
        "rms": float(selected.to(torch.float64).square().mean().sqrt()),
    }


def _saturation_summary(
    mapped: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> dict[str, float]:
    selected = mapped[valid_tokens].abs()
    return {
        "abs_ge_0_95_fraction": float((selected >= 0.95).to(torch.float64).mean()),
        "abs_ge_0_99_fraction": float((selected >= 0.99).to(torch.float64).mean()),
    }


def _pair_coefficients(
    q_features: torch.Tensor,
    k_features: torch.Tensor,
    survival: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("bnd,bmd->bnm", q_features, k_features) * survival


def _rms_ratio(numerator: Mapping[str, Any], denominator: Mapping[str, Any]) -> float:
    denominator_rms = float(denominator["rms"])
    if denominator_rms == 0.0:
        raise ValueError("RMS ratio denominator is zero")
    return float(numerator["rms"]) / denominator_rms


def head_feature_metrics(
    q: torch.Tensor,
    k: torch.Tensor,
    survival: torch.Tensor,
    old_mask: torch.Tensor,
    gammas: Sequence[float],
    valid_token_mask: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    _validate_cpu_tensor(q, "q")
    _validate_cpu_tensor(k, "k")
    _validate_cpu_tensor(survival, "survival")
    _validate_cpu_tensor(old_mask, "old_mask")
    if q.dim() != 3 or q.shape != k.shape:
        raise ValueError("q and k must have the same [B, N, D] shape")
    expected_pair_shape = (q.size(0), q.size(1), q.size(1))
    if tuple(survival.shape) != expected_pair_shape:
        raise ValueError("survival shape does not match q/k")
    if tuple(old_mask.shape) != expected_pair_shape or old_mask.dtype != torch.bool:
        raise ValueError("old_mask must be boolean and match q/k pair dimensions")
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(k).all()):
        raise ValueError("q/k contain non-finite values")
    if not bool(torch.isfinite(survival).all()) or bool((survival < 0).any()):
        raise ValueError("survival must be finite and non-negative")
    validated_gammas = validate_gammas(gammas)
    if valid_token_mask is None:
        valid_token_mask = torch.ones(
            q.shape[:2], dtype=torch.bool, device=q.device
        )
    if (
        tuple(valid_token_mask.shape) != tuple(q.shape[:2])
        or valid_token_mask.dtype != torch.bool
    ):
        raise ValueError("valid_token_mask must be boolean [B, N]")

    identity = _pair_coefficients(q, k, survival)
    identity_values = identity[old_mask]
    identity_summary = coefficient_summary(identity_values)

    gamma_entries: list[dict[str, Any]] = []
    tanh_gamma_one_values: Optional[torch.Tensor] = None
    tanh_gamma_one_summary: Optional[dict[str, Any]] = None
    for gamma in validated_gammas:
        q_tanh = torch.tanh(q * gamma)
        k_tanh = torch.tanh(k * gamma)
        tanh_coefficients = _pair_coefficients(q_tanh, k_tanh, survival)
        abs_tanh_coefficients = _pair_coefficients(
            q_tanh.abs(), k_tanh.abs(), survival
        )
        tanh_values = tanh_coefficients[old_mask]
        abs_tanh_values = abs_tanh_coefficients[old_mask]
        tanh_summary = coefficient_summary(tanh_values)
        abs_tanh_summary = coefficient_summary(abs_tanh_values)
        gamma_entries.append(
            {
                "gamma": gamma,
                "q_tanh_saturation": _saturation_summary(
                    q_tanh, valid_token_mask
                ),
                "k_tanh_saturation": _saturation_summary(
                    k_tanh, valid_token_mask
                ),
                "tanh_coefficients": tanh_summary,
                "abs_tanh_coefficients": abs_tanh_summary,
                "identity_tanh_correlation": pearson_correlation(
                    identity_values, tanh_values
                ),
                "tanh_identity_rms_ratio": _rms_ratio(
                    tanh_summary, identity_summary
                ),
                "abs_tanh_tanh_rms_ratio": _rms_ratio(
                    abs_tanh_summary, tanh_summary
                ),
            }
        )
        if gamma == 1.0:
            tanh_gamma_one_values = tanh_values
            tanh_gamma_one_summary = tanh_summary

    if tanh_gamma_one_values is None or tanh_gamma_one_summary is None:
        q_tanh_one = torch.tanh(q)
        k_tanh_one = torch.tanh(k)
        tanh_gamma_one_values = _pair_coefficients(
            q_tanh_one, k_tanh_one, survival
        )[old_mask]
        tanh_gamma_one_summary = coefficient_summary(tanh_gamma_one_values)

    exploratory: dict[str, Any] = {}
    for name, feature_map, formula in (
        (
            "per_vector_standardized_tanh",
            standardized_tanh_feature_map,
            "tanh((x-mean_d(x))/sqrt(var_d(x)+1e-6))",
        ),
        (
            "signed_sqrt_unit_rms",
            signed_sqrt_rms_feature_map,
            "rms_normalize(sign(x)*sqrt(abs(x)+1e-6))",
        ),
        (
            "per_vector_standardized_tanh_rms_matched",
            standardized_tanh_rms_matched_feature_map,
            (
                "rms_match_x(tanh((x-mean_d(x))/sqrt(var_d(x)+1e-6)))"
            ),
        ),
        (
            "signed_sqrt_rms_matched",
            signed_sqrt_rms_matched_feature_map,
            "rms_match_x(sign(x)*sqrt(abs(x)+1e-6))",
        ),
    ):
        mapped_values = _pair_coefficients(
            feature_map(q), feature_map(k), survival
        )[old_mask]
        mapped_summary = coefficient_summary(mapped_values)
        exploratory[name] = {
            "exploratory_only": True,
            "formula": formula,
            "coefficients": mapped_summary,
            "identity_correlation": pearson_correlation(
                identity_values, mapped_values
            ),
            "tanh_gamma_reference": 1.0,
            "rms_ratio_to_tanh_gamma_1": _rms_ratio(
                mapped_summary, tanh_gamma_one_summary
            ),
            "tanh_gamma_1_correlation": pearson_correlation(
                tanh_gamma_one_values, mapped_values
            ),
        }

    return {
        "q_coordinates": _coordinate_summary(q, valid_token_mask),
        "k_coordinates": _coordinate_summary(k, valid_token_mask),
        "identity_coefficients": identity_summary,
        "gamma_sweep": gamma_entries,
        "exploratory_feature_maps": exploratory,
    }


def scalar_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot aggregate an empty scalar sequence")
    normalized = [float(value) for value in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("cannot aggregate non-finite scalar values")
    return {
        "count": len(normalized),
        "min": min(normalized),
        "mean": math.fsum(normalized) / len(normalized),
        "max": max(normalized),
    }


def _checkpoint_head_entries(checkpoint: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    layers = checkpoint.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("checkpoint report has no layers to aggregate")
    heads: list[Mapping[str, Any]] = []
    for expected_layer, layer in enumerate(layers):
        if not isinstance(layer, Mapping) or layer.get("layer") != expected_layer:
            raise ValueError("checkpoint report layers are not contiguous and ordered")
        layer_heads = layer.get("heads")
        if not isinstance(layer_heads, list) or not layer_heads:
            raise ValueError(f"checkpoint report layer {expected_layer} has no heads")
        for expected_head, head in enumerate(layer_heads):
            if not isinstance(head, Mapping) or head.get("head") != expected_head:
                raise ValueError("checkpoint report heads are not contiguous and ordered")
            heads.append(head)
    return heads


def _summarize_heads(
    heads: Sequence[Mapping[str, Any]],
    gammas: tuple[float, ...],
    exploratory_names: tuple[str, ...],
) -> dict[str, Any]:
    if not heads:
        raise ValueError("cannot summarize an empty head collection")
    gamma_summaries: list[dict[str, Any]] = []
    for gamma_index, gamma in enumerate(gammas):
        entries: list[Mapping[str, Any]] = []
        for head in heads:
            sweep = head.get("gamma_sweep")
            if not isinstance(sweep, list) or len(sweep) != len(gammas):
                raise ValueError("head gamma sweep does not match configured gammas")
            if [float(entry["gamma"]) for entry in sweep] != list(gammas):
                raise ValueError("head gamma sweep is not ordered by configured gammas")
            entries.append(sweep[gamma_index])

        gamma_summaries.append(
            {
                "gamma": gamma,
                "identity_tanh_correlation": scalar_statistics(
                    [entry["identity_tanh_correlation"] for entry in entries]
                ),
                "tanh_identity_rms_ratio": scalar_statistics(
                    [entry["tanh_identity_rms_ratio"] for entry in entries]
                ),
                "abs_tanh_tanh_rms_ratio": scalar_statistics(
                    [entry["abs_tanh_tanh_rms_ratio"] for entry in entries]
                ),
                "tanh_negative_pair_fraction": scalar_statistics(
                    [
                        entry["tanh_coefficients"]["negative_pair_fraction"]
                        for entry in entries
                    ]
                ),
                "tanh_negative_l1_mass_fraction": scalar_statistics(
                    [
                        entry["tanh_coefficients"]["negative_l1_mass_fraction"]
                        for entry in entries
                    ]
                ),
                "q_tanh_saturation_abs_ge_0_95_fraction": scalar_statistics(
                    [
                        entry["q_tanh_saturation"]["abs_ge_0_95_fraction"]
                        for entry in entries
                    ]
                ),
                "q_tanh_saturation_abs_ge_0_99_fraction": scalar_statistics(
                    [
                        entry["q_tanh_saturation"]["abs_ge_0_99_fraction"]
                        for entry in entries
                    ]
                ),
                "k_tanh_saturation_abs_ge_0_95_fraction": scalar_statistics(
                    [
                        entry["k_tanh_saturation"]["abs_ge_0_95_fraction"]
                        for entry in entries
                    ]
                ),
                "k_tanh_saturation_abs_ge_0_99_fraction": scalar_statistics(
                    [
                        entry["k_tanh_saturation"]["abs_ge_0_99_fraction"]
                        for entry in entries
                    ]
                ),
            }
        )

    exploratory_summaries: dict[str, Any] = {}
    for name in exploratory_names:
        entries = []
        for head in heads:
            feature_maps = head.get("exploratory_feature_maps")
            if not isinstance(feature_maps, Mapping) or set(feature_maps) != set(
                exploratory_names
            ):
                raise ValueError(
                    "head exploratory feature maps do not match configured maps"
                )
            entries.append(feature_maps[name])
        exploratory_summaries[name] = {
            "identity_correlation": scalar_statistics(
                [entry["identity_correlation"] for entry in entries]
            ),
            "rms_ratio_to_tanh_gamma_1": scalar_statistics(
                [entry["rms_ratio_to_tanh_gamma_1"] for entry in entries]
            ),
            "negative_pair_fraction": scalar_statistics(
                [
                    entry["coefficients"]["negative_pair_fraction"]
                    for entry in entries
                ]
            ),
            "negative_l1_mass_fraction": scalar_statistics(
                [
                    entry["coefficients"]["negative_l1_mass_fraction"]
                    for entry in entries
                ]
            ),
        }
    return {
        "entry_count": len(heads),
        "gamma_sweep": gamma_summaries,
        "exploratory_feature_maps": exploratory_summaries,
    }


def build_aggregate_summary(
    checkpoints: Sequence[Mapping[str, Any]],
    gammas: Sequence[float],
    exploratory_names: Sequence[str] = EXPLORATORY_MAP_NAMES,
) -> dict[str, Any]:
    if not checkpoints:
        raise ValueError("cannot aggregate an empty checkpoint collection")
    validated_gammas = validate_gammas(gammas)
    names = tuple(exploratory_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("exploratory map names must be non-empty and unique")

    all_heads: list[Mapping[str, Any]] = []
    per_checkpoint: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        heads = _checkpoint_head_entries(checkpoint)
        all_heads.extend(heads)
        per_checkpoint.append(
            {
                "path": checkpoint["path"],
                "sha256": checkpoint["sha256"],
                "filename": checkpoint["filename"],
                "kind": checkpoint["kind"],
                "seed": checkpoint["seed"],
                "job_id": checkpoint["job_id"],
                "task_id": checkpoint["task_id"],
                "restart_count": checkpoint["restart_count"],
                **_summarize_heads(heads, validated_gammas, names),
            }
        )
    return {
        "all_checkpoints": _summarize_heads(
            all_heads, validated_gammas, names
        ),
        "per_checkpoint": per_checkpoint,
    }


def _build_exact_model(kind: str) -> HSTU:
    if kind not in ("local", "lift"):
        raise ValueError(f"unknown checkpoint kind {kind!r}")
    normalization = (
        "local_forgetting_rel_bias"
        if kind == "local"
        else "hybrid_forgetting_rel_bias"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model = HSTU(
            max_sequence_len=200,
            max_output_len=11,
            embedding_dim=50,
            num_blocks=8,
            num_heads=2,
            linear_dim=25,
            attention_dim=25,
            normalization=normalization,
            linear_config="uvqk",
            linear_activation="silu",
            linear_dropout_rate=0.2,
            attn_dropout_rate=0.0,
            embedding_module=LocalEmbeddingModule(
                num_items=3952,
                item_embedding_dim=50,
            ),
            similarity_module=DotProductSimilarity(),
            input_features_preproc_module=(
                LearnablePositionalEmbeddingInputFeaturesPreprocessor(
                    max_sequence_len=211,
                    embedding_dim=50,
                    dropout_rate=0.2,
                )
            ),
            output_postproc_module=L2NormEmbeddingPostprocessor(
                embedding_dim=50,
                eps=1e-6,
            ),
            enable_relative_attention_bias=True,
            concat_ua=False,
            forgetting_min_period=8.0,
            forgetting_max_period=256.0,
            hybrid_window_size=32,
            verbose=False,
        )
    model.to(torch.device("cpu"))
    model.eval()
    return model


def _load_exact_checkpoint(path: Path) -> tuple[HSTU, dict[str, Any]]:
    resolved = path.expanduser().absolute()
    expected_identity = checkpoint_identity(resolved)
    kind = str(expected_identity["kind"])
    checkpoint_bytes, checkpoint_sha, file_identity = read_verified_regular_file(
        resolved, str(expected_identity["sha256"])
    )
    checkpoint = load_checkpoint_from_verified_bytes(checkpoint_bytes)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint top level is not a mapping")
    required_top_keys = {"epoch", "model_state_dict", "optimizer_state_dict"}
    if set(checkpoint) != required_top_keys:
        raise ValueError(
            f"checkpoint top-level keys differ from {sorted(required_top_keys)}"
        )
    epoch = validate_checkpoint_epoch(checkpoint["epoch"])
    if not isinstance(checkpoint["optimizer_state_dict"], Mapping):
        raise ValueError("optimizer_state_dict is not a mapping")
    raw_state = checkpoint["model_state_dict"]
    if not isinstance(raw_state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in raw_state.items()
    ):
        raise ValueError("model_state_dict is not a string-to-tensor mapping")
    state = strip_ddp_prefix(raw_state)
    model = _build_exact_model(kind)
    expected_state = model.state_dict()
    if len(expected_state) != EXPECTED_STATE_TENSOR_COUNT:
        raise RuntimeError(
            "live exact model state inventory drifted: "
            f"{len(expected_state)} != {EXPECTED_STATE_TENSOR_COUNT}"
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"live exact model parameter count drifted: {parameter_count} "
            f"!= {EXPECTED_PARAMETER_COUNT}"
        )
    validate_state_inventory(state, expected_state)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"strict checkpoint load failed: {incompatible}")

    rho_values = torch.cat(
        [
            value.reshape(-1)
            for name, value in state.items()
            if name.endswith("._hybrid_tail_rho")
        ]
    )
    validate_tail_gain_state(kind, rho_values)

    model.eval()
    metadata = {
        "path": str(resolved),
        "sha256": checkpoint_sha,
        "file_identity": file_identity,
        "filename": resolved.name,
        "kind": kind,
        "seed": expected_identity["seed"],
        "job_id": expected_identity["job_id"],
        "task_id": expected_identity["task_id"],
        "restart_count": expected_identity["restart_count"],
        "normalization": (
            "local_forgetting_rel_bias"
            if kind == "local"
            else "hybrid_forgetting_rel_bias"
        ),
        "epoch": epoch,
        "state_tensor_count": len(state),
        "state_inventory_sha256": inventory_sha256(state),
        "trainable_parameter_count": parameter_count,
        "interpretation_limits": checkpoint_interpretation_limits(kind),
    }
    return model, metadata


def _load_selected_features(
    data_csv: Path,
    max_users: int,
    sampling: str,
    seed: int,
) -> tuple[SequentialFeatures, dict[str, Any]]:
    data_metadata = validate_data_csv(data_csv)
    dataset = DatasetV2(
        ratings_file=data_metadata["path"],
        padding_length=201,
        ignore_last_n=0,
        chronological=True,
        sample_ratio=1.0,
    )
    if len(dataset) != CANONICAL_DATA_ROWS:
        raise ValueError("DatasetV2 row count differs from validated CSV")
    indices = select_user_indices(len(dataset), max_users, sampling, seed)
    row = default_collate([dataset[index] for index in indices])
    features, _, _ = movielens_seq_features_from_row(
        row,
        device="cpu",  # pyre-ignore [6]
        max_output_length=11,
    )
    if tuple(features.past_ids.shape) != (max_users, 211):
        raise ValueError(
            f"selected feature shape is not {(max_users, 211)}: "
            f"{tuple(features.past_ids.shape)}"
        )
    if features.past_ids.device.type != "cpu":
        raise ValueError("selected features unexpectedly left CPU")
    lengths = features.past_lengths.to(torch.int64)
    valid_tokens = torch.arange(211).unsqueeze(0) < lengths.unsqueeze(1)
    if bool((features.past_ids[valid_tokens] <= 0).any()) or bool(
        (features.past_ids[valid_tokens] > 3952).any()
    ):
        raise ValueError("selected valid item IDs are outside ML-1M range")
    if bool((features.past_ids[~valid_tokens] != 0).any()):
        raise ValueError("selected sequences are not right padded with zero IDs")
    mask = old_pair_mask(lengths, 211)
    pair_count = int(mask.sum())
    if pair_count <= 0:
        raise ValueError("selected users produce no W32 old-history pairs")
    selected_user_ids = [int(value) for value in row["user_id"].tolist()]
    if len(set(selected_user_ids)) != len(selected_user_ids):
        raise ValueError("selected user IDs are not unique")
    revalidate_regular_file(
        Path(data_metadata["path"]),
        str(data_metadata["sha256"]),
        data_metadata["file_identity"],
    )
    data_metadata.update(
        {
            "sampling": sampling,
            "sampling_seed": seed if sampling == "seeded" else None,
            "requested_max_users": max_users,
            "selected_indices": indices,
            "selected_user_ids": selected_user_ids,
            "selected_user_count": len(selected_user_ids),
            "valid_token_count": int(lengths.sum()),
            "old_pair_count_per_head": pair_count,
            "length_min": int(lengths.min()),
            "length_max": int(lengths.max()),
            "length_mean": float(lengths.to(torch.float64).mean()),
            "selection_disclosure": {
                "selection_status": "post_hoc_deterministic_convenience_sample",
                "source": "processed_full_user_sequences",
                "held_out_sample": False,
                "predeclared_sample": False,
                "representative_sample_claim": False,
                "confirmatory_inference": False,
                "exact_selected_user_ids_recorded": True,
            },
        }
    )
    return features, data_metadata


@torch.inference_mode()
def _checkpoint_geometry(
    model: HSTU,
    metadata: dict[str, Any],
    features: SequentialFeatures,
    gammas: tuple[float, ...],
) -> dict[str, Any]:
    lengths = features.past_lengths.to(torch.int64)
    batch_size, sequence_length = features.past_ids.shape
    valid_tokens = torch.arange(sequence_length).unsqueeze(0) < lengths.unsqueeze(1)
    pair_mask = old_pair_mask(lengths, sequence_length)
    item_embeddings = model.get_item_embeddings(features.past_ids)
    output, caches = model.generate_user_embeddings(
        past_lengths=features.past_lengths,
        past_ids=features.past_ids,
        past_embeddings=item_embeddings,
        past_payloads=features.past_payloads,
        return_cache_states=True,
    )
    if output.device.type != "cpu" or not bool(torch.isfinite(output).all()):
        raise ValueError("model output is not finite CPU data")
    if len(caches) != 8:
        raise ValueError(f"model returned {len(caches)} layer caches instead of 8")

    layer_entries: list[dict[str, Any]] = []
    for layer_index, (cache, layer) in enumerate(
        zip(caches, model._hstu._attention_layers)
    ):
        padded_q = cache[1]
        padded_k = cache[2]
        expected_shape = (batch_size, sequence_length, 50)
        if tuple(padded_q.shape) != expected_shape or tuple(padded_k.shape) != expected_shape:
            raise ValueError(
                f"layer {layer_index} q/k cache shape differs from {expected_shape}"
            )
        if not bool(torch.isfinite(padded_q).all()) or not bool(
            torch.isfinite(padded_k).all()
        ):
            raise ValueError(f"layer {layer_index} q/k contain non-finite values")
        q = padded_q.view(batch_size, sequence_length, 2, 25).float()
        k = padded_k.view(batch_size, sequence_length, 2, 25).float()
        log_forget = F.logsigmoid(
            torch.einsum("bnhd,hd->bnh", k, layer._forget_weight.float())
            + layer._forget_bias.float().view(1, 1, 2)
        )
        survival = survival_from_log_forget(log_forget)
        alpha = 2.0 * torch.tanh(layer._hybrid_tail_rho.float() / 2.0)
        head_entries: list[dict[str, Any]] = []
        for head_index in range(2):
            metrics = head_feature_metrics(
                q=q[:, :, head_index],
                k=k[:, :, head_index],
                survival=survival[:, head_index],
                old_mask=pair_mask,
                gammas=gammas,
                valid_token_mask=valid_tokens,
            )
            metrics.update(
                {
                    "head": head_index,
                    "learned_tail_alpha": float(alpha[head_index]),
                }
            )
            head_entries.append(metrics)
        layer_entries.append({"layer": layer_index, "heads": head_entries})
    return {**metadata, "layers": layer_entries}


def _runtime_gr_module_inventory(snapshot: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for name, module in sorted(sys.modules.items()):
        if name != "generative_recommenders" and not name.startswith(
            "generative_recommenders."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            namespace_paths = getattr(module, "__path__", None)
            if namespace_paths is None:
                raise ValueError(f"loaded GR module has no source or namespace path: {name}")
            verified_paths: list[str] = []
            for namespace_path in namespace_paths:
                resolved_namespace = Path(namespace_path).resolve(strict=True)
                try:
                    resolved_namespace.relative_to(snapshot)
                except ValueError as error:
                    raise ValueError(
                        "loaded GR namespace is outside the frozen snapshot: "
                        f"{name}={resolved_namespace}"
                    ) from error
                verified_paths.append(str(resolved_namespace))
            inventory[name] = {
                "kind": "namespace_package",
                "paths": sorted(verified_paths),
            }
            continue
        source = Path(module_file).resolve(strict=True)
        if source.suffix == ".pyc":
            source = Path(importlib.util.source_from_cache(str(source))).resolve(
                strict=True
            )
        try:
            relative = source.relative_to(snapshot).as_posix()
        except ValueError as error:
            raise ValueError(
                f"loaded GR module is outside the frozen snapshot: {name}={source}"
            ) from error
        inventory[name] = {
            "kind": "source_module",
            "path": str(source),
            "snapshot_relative_path": relative,
            "sha256": sha256_file(source),
        }
    required_modules = {
        "generative_recommenders.research.data.dataset",
        "generative_recommenders.research.modeling.sequential.features",
        "generative_recommenders.research.modeling.sequential.hstu",
    }
    missing = sorted(required_modules - set(inventory))
    if missing:
        raise ValueError(f"required frozen GR modules were not imported: {missing}")
    return inventory


def _runtime_source_provenance(training_snapshot: Path) -> dict[str, Any]:
    if _BOOTSTRAP_SNAPSHOT_PROVENANCE is None:
        raise RuntimeError(
            "training snapshot was not bootstrapped before GR imports; invoke the "
            "CLI with --training-snapshot"
        )
    requested = training_snapshot.expanduser().absolute()
    if requested != CANONICAL_TRAINING_SNAPSHOT:
        raise ValueError("runtime training snapshot differs from bootstrap snapshot")
    final_snapshot_provenance = verify_training_snapshot(requested)
    if final_snapshot_provenance != _BOOTSTRAP_SNAPSHOT_PROVENANCE:
        raise ValueError("training snapshot changed after bootstrap verification")
    imported_modules = _runtime_gr_module_inventory(requested)
    return {
        "training_snapshot": final_snapshot_provenance,
        "bootstrap_before_gr_imports": True,
        "sys_path_precedence": str(Path(sys.path[0]).resolve(strict=True)),
        "loaded_gr_module_count": len(imported_modules),
        "loaded_gr_modules": imported_modules,
        "probe_script": {
            "path": str(Path(__file__).resolve(strict=True)),
            "sha256": sha256_file(Path(__file__).resolve(strict=True)),
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pandas": pd.__version__,
        },
    }


def _build_report_impl(
    checkpoint_paths: Sequence[Path],
    data_csv: Path,
    training_snapshot: Path,
    max_users: int = 128,
    gammas: Sequence[float] = DEFAULT_GAMMAS,
    sampling: str = "first",
    sampling_seed: int = 42,
) -> dict[str, Any]:
    if torch.get_num_threads() != CPU_THREADS:
        raise RuntimeError("build_report CPU thread scope was not installed")
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint path is required")
    if _BOOTSTRAP_SNAPSHOT_PROVENANCE is None:
        raise RuntimeError(
            "build_report requires pre-import --training-snapshot bootstrap"
        )
    if training_snapshot.expanduser().absolute() != CANONICAL_TRAINING_SNAPSHOT:
        raise ValueError("build_report training snapshot is not canonical")
    validated_gammas = validate_gammas(gammas)
    resolved_checkpoints = sorted(
        path.expanduser().absolute() for path in checkpoint_paths
    )
    if len(set(resolved_checkpoints)) != len(resolved_checkpoints):
        raise ValueError("checkpoint paths must be unique")
    validate_canonical_checkpoint_set(resolved_checkpoints)
    features, data_metadata = _load_selected_features(
        data_csv=data_csv,
        max_users=max_users,
        sampling=sampling,
        seed=sampling_seed,
    )

    checkpoint_reports: list[dict[str, Any]] = []
    with scoped_fbgemm_layout_shims() as installed_shims:
        for checkpoint_path in resolved_checkpoints:
            model, checkpoint_metadata = _load_exact_checkpoint(checkpoint_path)
            checkpoint_reports.append(
                _checkpoint_geometry(
                    model=model,
                    metadata=checkpoint_metadata,
                    features=features,
                    gammas=validated_gammas,
                )
            )

    aggregate_summary = build_aggregate_summary(
        checkpoint_reports,
        validated_gammas,
        EXPLORATORY_MAP_NAMES,
    )
    runtime_source_provenance = _runtime_source_provenance(training_snapshot)
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "signed_lift_feature_geometry",
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "runtime_source_provenance": runtime_source_provenance,
        "model_settings": dict(MODEL_SETTINGS),
        "interpretation_limits": global_interpretation_limits(validated_gammas),
        "metric_settings": {
            "gammas": list(validated_gammas),
            "window_size": WINDOW_SIZE,
            "coefficient_definition": (
                "F_ij * phi_q(q_i)^T phi_k(k_j), before common 0.5/N "
                "and learned tail alpha"
            ),
            "negative_mass_definition": (
                "sum(-c for c<0) / sum(abs(c)) over valid old pairs"
            ),
            "tanh_saturation_thresholds": [0.95, 0.99],
            "exploratory_only": list(EXPLORATORY_MAP_NAMES),
        },
        "layout_shims_installed": list(installed_shims),
        "data": data_metadata,
        "checkpoints": checkpoint_reports,
        "aggregate_summary": aggregate_summary,
    }


def build_report(
    checkpoint_paths: Sequence[Path],
    data_csv: Path,
    training_snapshot: Path,
    max_users: int = 128,
    gammas: Sequence[float] = DEFAULT_GAMMAS,
    sampling: str = "first",
    sampling_seed: int = 42,
) -> dict[str, Any]:
    with scoped_cpu_thread_count(CPU_THREADS):
        return _build_report_impl(
            checkpoint_paths=checkpoint_paths,
            data_csv=data_csv,
            training_snapshot=training_snapshot,
            max_users=max_users,
            gammas=gammas,
            sampling=sampling,
            sampling_seed=sampling_seed,
        )


def _write_atomic(path: Path, content: str) -> None:
    destination = path.expanduser().absolute()
    parent = destination.parent
    _require_nonsymlink_path_components(parent)
    parent_mode = os.lstat(parent).st_mode
    if not stat.S_ISDIR(parent_mode):
        raise ValueError(f"output parent is not a directory: {parent}")
    if os.path.lexists(destination):
        destination_mode = os.lstat(destination).st_mode
        if stat.S_ISLNK(destination_mode) or not stat.S_ISREG(destination_mode):
            raise ValueError(
                f"output destination is a symlink or nonregular file: {destination}"
            )
    temporary = destination.with_name(
        f".{destination.name}.tmp-{secrets.token_hex(16)}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    temporary_owned = False
    temporary_inode: Optional[tuple[int, int]] = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        created = os.fstat(descriptor)
        temporary_inode = (int(created.st_dev), int(created.st_ino))
        temporary_owned = True
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(content.encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary_owned = False
        destination_mode = os.lstat(destination).st_mode
        if stat.S_ISLNK(destination_mode) or not stat.S_ISREG(destination_mode):
            raise ValueError("atomic output did not produce a regular file")
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_owned and os.path.lexists(temporary):
            current = os.lstat(temporary)
            current_inode = (int(current.st_dev), int(current.st_ino))
            if current_inode == temporary_inode:
                os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--data-csv", required=True, type=Path)
    parser.add_argument("--training-snapshot", required=True, type=Path)
    parser.add_argument("--max-users", type=int, default=128)
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=list(DEFAULT_GAMMAS),
    )
    parser.add_argument(
        "--sampling",
        choices=("first", "seeded"),
        default="first",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(
        checkpoint_paths=args.checkpoints,
        data_csv=args.data_csv,
        training_snapshot=args.training_snapshot,
        max_users=args.max_users,
        gammas=args.gammas,
        sampling=args.sampling,
        sampling_seed=args.sampling_seed,
    )
    encoded = canonical_json(report)
    if args.json_output is None:
        sys.stdout.write(encoded)
    else:
        _write_atomic(args.json_output, encoded)
        print(str(args.json_output.expanduser().resolve()), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
