#!/usr/bin/env python3
"""Post-hoc, fail-closed diagnostics for matched HSTU/SAFA checkpoints."""

from __future__ import annotations

import argparse
import bisect
import contextlib
import hashlib
import importlib
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from scripts.snapshot import verify_snapshot
except ModuleNotFoundError:  # Direct execution from scripts/.
    from snapshot import verify_snapshot


SCHEMA_VERSION = 1
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
PROVENANCE_KEYS = ("source_commit", "source_tree", "source_manifest")
SLURM_KEYS = (
    "slurm_array_job_id",
    "slurm_array_task_id",
    "slurm_job_id",
    "slurm_job_qos",
    "slurm_job_partition",
    "slurm_restart_count",
)
IDENTITY_KEYS = (
    "attention_mode",
    "random_seed",
    "parameter_count",
    "parameter_inventory_sha256",
    "resolved_gin_config_sha256",
    "experiment_config_sha256",
    *PROVENANCE_KEYS,
)
METRIC_KEYS = ("ndcg@10", "ndcg@50", "hr@10", "hr@50", "mrr")
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
HISTORY_BINS = (
    (10, "1-10"),
    (20, "11-20"),
    (50, "21-50"),
    (100, "51-100"),
    (None, "101+"),
)
DAY = 24 * 60 * 60
GAP_BINS = (
    (DAY, "<=1d"),
    (7 * DAY, "(1d,7d]"),
    (30 * DAY, "(7d,30d]"),
    (90 * DAY, "(30d,90d]"),
    (365 * DAY, "(90d,365d]"),
    (None, ">365d"),
)


class DiagnosticError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_unchanged_file(path: Path, expected_sha256: str, description: str) -> None:
    try:
        actual_sha256 = _sha256_file(path)
    except OSError as error:
        raise DiagnosticError(f"could not re-read {description}: {error}") from error
    if actual_sha256 != expected_sha256:
        raise DiagnosticError(f"{description} changed during diagnostics")


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise DiagnosticError(f"{name} must be a lowercase SHA-256")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DiagnosticError(f"{name} must be an integer >= {minimum}")
    return value


def validate_final_epoch(checkpoint_epoch: Any, num_epochs: Any) -> None:
    checkpoint_epoch = _require_int(checkpoint_epoch, "checkpoint.epoch")
    num_epochs = _require_int(num_epochs, "train_fn.num_epochs", minimum=1)
    expected_final_epoch = num_epochs - 1
    if checkpoint_epoch != expected_final_epoch:
        raise DiagnosticError(
            "checkpoint is not final: "
            f"epoch={checkpoint_epoch} expected={expected_final_epoch}"
        )


def validate_checkpoint_bundle(
    checkpoint: Mapping[str, Any],
    resolved_config: str,
    run_metadata: Mapping[str, Any],
    verified_source: Mapping[str, str],
    *,
    require_slurm_provenance: bool = True,
) -> None:
    """Validates the independently stored checkpoint, run, and source identities."""
    required_checkpoint = {
        "epoch",
        "model_state_dict",
        "resolved_gin_config",
        "source_root",
        *IDENTITY_KEYS,
    }
    missing = sorted(required_checkpoint - set(checkpoint))
    if missing:
        raise DiagnosticError(f"checkpoint metadata is missing: {missing}")
    missing = sorted(set(IDENTITY_KEYS) - set(run_metadata))
    if missing:
        raise DiagnosticError(f"run metadata is missing: {missing}")

    if checkpoint["attention_mode"] not in ("hstu", "safa"):
        raise DiagnosticError("checkpoint attention_mode must be 'hstu' or 'safa'")
    if run_metadata["attention_mode"] not in ("hstu", "safa"):
        raise DiagnosticError("run metadata attention_mode must be 'hstu' or 'safa'")
    _require_int(checkpoint["epoch"], "checkpoint.epoch")
    _require_int(checkpoint["random_seed"], "checkpoint.random_seed")
    _require_int(checkpoint["parameter_count"], "checkpoint.parameter_count", minimum=1)
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise DiagnosticError("checkpoint.model_state_dict must be a mapping")
    if not isinstance(checkpoint["source_root"], str) or not checkpoint["source_root"]:
        raise DiagnosticError("checkpoint.source_root must be a non-empty string")

    for key in (
        "resolved_gin_config_sha256",
        "experiment_config_sha256",
        "parameter_inventory_sha256",
        "source_manifest",
    ):
        _require_sha256(checkpoint[key], f"checkpoint.{key}")
        _require_sha256(run_metadata[key], f"run_metadata.{key}")

    if not isinstance(checkpoint["resolved_gin_config"], str):
        raise DiagnosticError("checkpoint.resolved_gin_config must be text")
    if checkpoint["resolved_gin_config"] != resolved_config:
        raise DiagnosticError("run operative config differs from checkpoint config")
    actual_config_sha = _sha256_bytes(resolved_config.encode("utf-8"))
    if actual_config_sha != checkpoint["resolved_gin_config_sha256"]:
        raise DiagnosticError("checkpoint resolved Gin config checksum mismatch")

    for key in IDENTITY_KEYS:
        if checkpoint[key] != run_metadata[key]:
            raise DiagnosticError(f"checkpoint/run metadata mismatch: {key}")
    if run_metadata.get("source_root") != checkpoint["source_root"]:
        raise DiagnosticError("checkpoint/run metadata mismatch: source_root")
    for key in PROVENANCE_KEYS:
        if verified_source.get(key) != checkpoint[key]:
            raise DiagnosticError(f"checkpoint/source snapshot mismatch: {key}")

    scheduler_keys = set(SLURM_KEYS)
    scheduler_present = scheduler_keys & (set(checkpoint) | set(run_metadata))
    if require_slurm_provenance or scheduler_present:
        missing_checkpoint = sorted(scheduler_keys - set(checkpoint))
        missing_metadata = sorted(scheduler_keys - set(run_metadata))
        if missing_checkpoint or missing_metadata:
            raise DiagnosticError(
                "incomplete SLURM provenance: "
                f"checkpoint_missing={missing_checkpoint}, "
                f"run_metadata_missing={missing_metadata}"
            )
        for key in SLURM_KEYS:
            if checkpoint[key] != run_metadata[key]:
                raise DiagnosticError(f"checkpoint/run metadata mismatch: {key}")
        for key in ("slurm_array_job_id", "slurm_job_id"):
            value = checkpoint[key]
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[1-9][0-9]*", value) is None
            ):
                raise DiagnosticError(f"checkpoint.{key} must be a positive job ID")
        task_id = checkpoint["slurm_array_task_id"]
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or not 0 <= task_id < 6
        ):
            raise DiagnosticError("checkpoint.slurm_array_task_id must be in [0, 5]")
        if checkpoint["slurm_job_qos"] != "h200_mrs_shared":
            raise DiagnosticError("checkpoint SLURM QoS must be h200_mrs_shared")
        if checkpoint["slurm_job_partition"] != "h200":
            raise DiagnosticError("checkpoint SLURM partition must be h200")
        restart_count = checkpoint["slurm_restart_count"]
        if (
            isinstance(restart_count, bool)
            or not isinstance(restart_count, int)
            or restart_count < 0
        ):
            raise DiagnosticError(
                "checkpoint.slurm_restart_count must be a non-negative integer"
            )
        expected_runs = (
            (42, "hstu"),
            (42, "safa"),
            (43, "hstu"),
            (43, "safa"),
            (44, "hstu"),
            (44, "safa"),
        )
        if (
            checkpoint["random_seed"],
            checkpoint["attention_mode"],
        ) != expected_runs[task_id]:
            raise DiagnosticError(
                "checkpoint SLURM task does not match its seed/attention mode"
            )


@dataclass(frozen=True)
class RunBundle:
    checkpoint_path: Path
    checkpoint_sha256: str
    run_dir: Path
    source_root: Path
    checkpoint: Mapping[str, Any]
    resolved_config: str
    run_metadata: Mapping[str, Any]
    verified_source: Mapping[str, str]


def load_run_bundle(
    checkpoint_path: Path,
    run_dir: Path,
    source_root_override: Optional[Path] = None,
    *,
    require_slurm_provenance: bool = True,
) -> RunBundle:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    config_path = run_dir / "operative_config.gin"
    metadata_path = run_dir / "run_metadata.json"
    for path, description in (
        (checkpoint_path, "checkpoint"),
        (config_path, "operative config"),
        (metadata_path, "run metadata"),
    ):
        if not path.is_file():
            raise DiagnosticError(f"{description} is not a regular file: {path}")

    try:
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        require_unchanged_file(checkpoint_path, checkpoint_sha256, "checkpoint")
    except Exception as error:
        if isinstance(error, DiagnosticError):
            raise
        raise DiagnosticError(f"could not safely load checkpoint: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise DiagnosticError("checkpoint root must be a mapping")
    try:
        resolved_config = config_path.read_bytes().decode("utf-8")
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiagnosticError(f"could not read run artifacts: {error}") from error
    if not isinstance(run_metadata, Mapping):
        raise DiagnosticError("run_metadata.json must contain an object")

    recorded_source_root = checkpoint.get("source_root")
    if source_root_override is None:
        if not isinstance(recorded_source_root, str) or not recorded_source_root:
            raise DiagnosticError("checkpoint has no usable source_root")
        source_root = Path(recorded_source_root)
    else:
        source_root = source_root_override
    source_root = source_root.expanduser().resolve()
    expected_manifest = checkpoint.get("source_manifest")
    if not isinstance(expected_manifest, str):
        raise DiagnosticError("checkpoint has no usable source_manifest")
    try:
        verified_source = verify_snapshot(
            source_root,
            expected_manifest=expected_manifest,
        )
    except Exception as error:
        raise DiagnosticError(
            f"source snapshot verification failed: {error}"
        ) from error

    validate_checkpoint_bundle(
        checkpoint=checkpoint,
        resolved_config=resolved_config,
        run_metadata=run_metadata,
        verified_source=verified_source,
        require_slurm_provenance=require_slurm_provenance,
    )
    return RunBundle(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        run_dir=run_dir,
        source_root=source_root,
        checkpoint=checkpoint,
        resolved_config=resolved_config,
        run_metadata=run_metadata,
        verified_source=verified_source,
    )


class QuantileHistogram:
    """A fixed-width [0, 1] histogram with deterministic quantiles."""

    def __init__(self, num_bins: int) -> None:
        if num_bins < 100:
            raise DiagnosticError("histogram_bins must be >= 100")
        self.num_bins = num_bins
        self.counts = torch.zeros(num_bins, dtype=torch.float64)

    def update(self, values: torch.Tensor, weight: float = 1.0) -> None:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        if not bool(torch.isfinite(values).all()):
            raise DiagnosticError("non-finite attention diagnostic value")
        if bool(((values < 0) | (values > 1)).any()):
            raise DiagnosticError("attention probability lies outside [0, 1]")
        indices = torch.clamp(
            torch.floor(values * self.num_bins).to(torch.int64),
            max=self.num_bins - 1,
        )
        counts = torch.bincount(indices, minlength=self.num_bins).cpu().double()
        self.counts += counts * weight

    @property
    def total(self) -> float:
        return float(self.counts.sum())

    def quantiles(self, probabilities: Iterable[float] = QUANTILES) -> Dict[str, float]:
        if self.total <= 0:
            return {}
        cumulative = torch.cumsum(self.counts, dim=0)
        result: Dict[str, float] = {}
        for probability in probabilities:
            target = max(
                float(probability) * self.total, torch.finfo(torch.float64).eps
            )
            index = int(torch.searchsorted(cumulative, torch.tensor(target)).item())
            index = min(index, self.num_bins - 1)
            result[f"p{round(probability * 100):02d}"] = (index + 0.5) / self.num_bins
        return result


def _update_head_histograms(
    histograms: Sequence[QuantileHistogram],
    values: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    weight: float = 1.0,
) -> None:
    if values.ndim < 2 or values.shape[-1] != len(histograms):
        raise DiagnosticError("head histogram tensor has an incompatible shape")
    if valid_mask is not None:
        if tuple(valid_mask.shape) != tuple(values.shape[:-1]):
            raise DiagnosticError("head histogram mask has an incompatible shape")
        values = values[
            valid_mask.unsqueeze(-1).expand(*valid_mask.shape, values.shape[-1])
        ].reshape(-1, values.shape[-1])
    else:
        values = values.reshape(-1, values.shape[-1])
    if values.numel() == 0:
        return
    values = values.detach().float()
    invalid = (~torch.isfinite(values)) | (values < 0) | (values > 1)
    if bool(invalid.any()):
        raise DiagnosticError("invalid attention probability in head histogram")
    num_bins = histograms[0].num_bins
    if any(histogram.num_bins != num_bins for histogram in histograms):
        raise DiagnosticError("head histograms use inconsistent bin counts")
    indices = torch.clamp(
        torch.floor(values * num_bins).to(torch.int64),
        max=num_bins - 1,
    )
    offsets = torch.arange(
        len(histograms), device=values.device, dtype=torch.int64
    ).view(1, -1)
    counts = torch.bincount(
        (indices + offsets * num_bins).reshape(-1),
        minlength=len(histograms) * num_bins,
    ).reshape(len(histograms), num_bins)
    counts = counts.cpu().double() * weight
    for head_index, histogram in enumerate(histograms):
        histogram.counts += counts[head_index]


def _bucket(value: int, bins: Sequence[Tuple[Optional[int], str]]) -> str:
    for upper_bound, label in bins:
        if upper_bound is None or value <= upper_bound:
            return label
    raise AssertionError("terminal bucket is missing")


class MetricStrata:
    def __init__(self, labels: Sequence[str]) -> None:
        self._labels = tuple(labels)
        self._counts: Dict[str, int] = defaultdict(int)
        self._sums: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def update(
        self,
        labels: Sequence[Optional[str]],
        metrics: Mapping[str, torch.Tensor],
    ) -> None:
        batch_size = len(labels)
        metric_values: Dict[str, torch.Tensor] = {}
        for key in METRIC_KEYS:
            if key not in metrics:
                raise DiagnosticError(f"evaluation did not return {key}")
            values = metrics[key].detach().float().cpu().reshape(-1)
            if values.numel() != batch_size:
                raise DiagnosticError(f"metric {key} is not per-example")
            if not bool(torch.isfinite(values).all()):
                raise DiagnosticError(f"metric {key} contains non-finite values")
            metric_values[key] = values

        for index, label in enumerate(labels):
            if label is None:
                continue
            if label not in self._labels:
                raise DiagnosticError(f"unexpected metric stratum: {label}")
            self._counts[label] += 1
            for key, values in metric_values.items():
                self._sums[label][key] += float(values[index])

    def result(self) -> List[Dict[str, Any]]:
        output = []
        for label in self._labels:
            count = self._counts[label]
            output.append(
                {
                    "label": label,
                    "count": count,
                    "metrics": {
                        key: (self._sums[label][key] / count if count else None)
                        for key in METRIC_KEYS
                    },
                }
            )
        return output


def _uniform_hash_sample(
    population_size: int,
    sample_size: int,
    *,
    seed: int,
    namespace: str,
) -> List[int]:
    """Deterministic uniform ranks using rejection-sampled SHA-256 words."""
    if population_size < 0 or sample_size < 0 or sample_size > population_size:
        raise DiagnosticError("invalid deterministic sample dimensions")
    if sample_size == population_size:
        return list(range(population_size))
    if sample_size == 0:
        return []
    modulus = 1 << 64
    cutoff = modulus - (modulus % population_size)
    selected = set()
    counter = 0
    prefix = f"{seed}:{namespace}:".encode("utf-8")
    while len(selected) < sample_size:
        digest = hashlib.sha256(prefix + str(counter).encode("ascii")).digest()
        counter += 1
        for offset in range(0, len(digest), 8):
            candidate = int.from_bytes(digest[offset : offset + 8], "big")
            if candidate < cutoff:
                selected.add(candidate % population_size)
                if len(selected) == sample_size:
                    break
    return sorted(selected)


def deterministic_dataset_indices(size: int, limit: int, seed: int) -> List[int]:
    if size <= 0:
        raise DiagnosticError("evaluation dataset is empty")
    if limit <= 0:
        raise DiagnosticError("max_examples must be positive")
    return _uniform_hash_sample(
        size,
        min(size, limit),
        seed=seed,
        namespace="dataset-v1",
    )


@dataclass
class HeadAttentionStats:
    histogram_bins: int
    gate_histogram: QuantileHistogram = field(init=False)
    survival_histogram: QuantileHistogram = field(init=False)
    transition_log_gate_sum: float = 0.0
    transition_gate_count: int = 0
    survival_population_count: int = 0
    survival_sample_count: int = 0
    negative_abs_mass: float = 0.0
    positive_abs_mass: float = 0.0
    negative_count: int = 0
    coefficient_count: int = 0

    def __post_init__(self) -> None:
        self.gate_histogram = QuantileHistogram(self.histogram_bins)
        self.survival_histogram = QuantileHistogram(self.histogram_bins)


class AttentionDiagnostics:
    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        *,
        histogram_bins: int,
        pair_samples_per_batch: int,
        sample_seed: int,
        attention_mode: str = "safa",
    ) -> None:
        if pair_samples_per_batch <= 0:
            raise DiagnosticError("pair_samples_per_batch must be positive")
        if attention_mode not in ("hstu", "safa"):
            raise DiagnosticError("diagnostic attention_mode must be hstu or safa")
        self._layers = list(layers)
        self._attention_mode = attention_mode
        self._stats = [
            [HeadAttentionStats(histogram_bins) for _ in range(layer._num_heads)]
            for layer in self._layers
        ]
        self._pair_samples_per_batch = pair_samples_per_batch
        self._sample_seed = sample_seed
        self._batch_index = 0
        self._layer_call_index = 0
        self._lengths = torch.empty(0, dtype=torch.int64)
        self._pair_coordinates = (
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
        )
        self._pair_population = 0
        self._pair_sample_weight = 0.0

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def start_batch(self, lengths: torch.Tensor) -> None:
        if self._layer_call_index != 0:
            raise DiagnosticError("previous attention batch was not finalized")
        lengths = lengths.detach().to(torch.int64).cpu().reshape(-1)
        if lengths.numel() == 0 or bool((lengths <= 0).any()):
            raise DiagnosticError("all diagnostic histories must be non-empty")
        self._lengths = lengths
        pair_counts = [int(length) * (int(length) - 1) // 2 for length in lengths]
        cumulative = []
        running = 0
        for count in pair_counts:
            running += count
            cumulative.append(running)
        self._pair_population = running
        sample_size = min(self._pair_samples_per_batch, running)
        ranks = _uniform_hash_sample(
            running,
            sample_size,
            seed=self._sample_seed,
            namespace=f"attention-pairs-v1:{self._batch_index}",
        )
        batch_indices: List[int] = []
        query_indices: List[int] = []
        key_indices: List[int] = []
        for rank in ranks:
            batch_index = bisect.bisect_right(cumulative, rank)
            previous = cumulative[batch_index - 1] if batch_index else 0
            local_rank = rank - previous
            query_index = (1 + math.isqrt(1 + 8 * local_rank)) // 2
            key_index = local_rank - query_index * (query_index - 1) // 2
            if not (0 <= key_index < query_index < int(lengths[batch_index])):
                raise AssertionError("triangular pair-index inversion failed")
            batch_indices.append(batch_index)
            query_indices.append(query_index)
            key_indices.append(key_index)
        self._pair_coordinates = tuple(
            torch.tensor(values, dtype=torch.int64)
            for values in (batch_indices, query_indices, key_indices)
        )
        self._pair_sample_weight = running / sample_size if sample_size else 0.0

    def observe(
        self,
        *,
        padded_k: torch.Tensor,
        attention_mode: str,
        forget_weight: torch.Tensor,
        forget_bias: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> None:
        if self._layer_call_index >= len(self._layers):
            raise DiagnosticError("model made more attention calls than expected")
        if attention_mode != self._attention_mode:
            raise DiagnosticError(
                "runtime attention helper mode disagrees with the checkpoint"
            )
        layer_index = self._layer_call_index
        layer = self._layers[layer_index]
        self._layer_call_index += 1
        if (
            forget_weight.data_ptr() != layer._forget_weight.data_ptr()
            or forget_bias.data_ptr() != layer._forget_bias.data_ptr()
        ):
            raise DiagnosticError("attention layer order or gate parameters mismatch")

        batch_size, sequence_length, num_heads, _ = padded_k.shape
        if batch_size != self._lengths.numel() or num_heads != layer._num_heads:
            raise DiagnosticError(
                "attention tensor shape disagrees with reconstructed model"
            )
        if int(self._lengths.max()) > sequence_length:
            raise DiagnosticError("history length exceeds attention sequence length")
        if tuple(attention_weights.shape) != (
            batch_size,
            num_heads,
            sequence_length,
            sequence_length,
        ):
            raise DiagnosticError("unexpected attention coefficient shape")

        if self._attention_mode == "safa":
            forget_logits = torch.einsum(
                "bnhd,hd->bnh", padded_k, forget_weight
            ) + forget_bias.view(1, 1, num_heads)
            log_gate = F.logsigmoid(forget_logits).float()
            if not bool(torch.isfinite(log_gate).all()):
                raise DiagnosticError("forget gates contain non-finite log values")
            gate = torch.exp(log_gate)
            positions = torch.arange(sequence_length, device=padded_k.device).view(
                1, -1
            )
            valid_positions = positions < self._lengths.to(padded_k.device).view(-1, 1)
            transition_positions = valid_positions & (positions > 0)

            prefix = torch.cumsum(log_gate, dim=1)
            pair_batch, pair_query, pair_key = (
                coordinates.to(padded_k.device)
                for coordinates in self._pair_coordinates
            )
            _update_head_histograms(
                [stats.gate_histogram for stats in self._stats[layer_index]],
                gate,
                valid_mask=valid_positions,
            )
            transition_log_sums = torch.where(
                transition_positions.unsqueeze(-1),
                log_gate,
                torch.zeros((), dtype=log_gate.dtype, device=log_gate.device),
            ).sum(dim=(0, 1))
            transition_log_sums = transition_log_sums.cpu()
            transition_count = int(torch.clamp_min(self._lengths - 1, 0).sum())
            sampled_survival = None
            if pair_batch.numel():
                sampled_survival = torch.exp(
                    torch.clamp_max(
                        prefix[pair_batch, pair_query, :]
                        - prefix[pair_batch, pair_key, :],
                        0.0,
                    )
                )
                _update_head_histograms(
                    [stats.survival_histogram for stats in self._stats[layer_index]],
                    sampled_survival,
                    weight=self._pair_sample_weight,
                )
            for head_index, stats in enumerate(self._stats[layer_index]):
                stats.transition_log_gate_sum += float(transition_log_sums[head_index])
                stats.transition_gate_count += transition_count
                if sampled_survival is not None:
                    stats.survival_sample_count += sampled_survival.shape[0]
                stats.survival_population_count += self._pair_population

        lengths = self._lengths.to(attention_weights.device)
        positions = torch.arange(sequence_length, device=attention_weights.device)
        causal_mask = positions.view(-1, 1) >= positions.view(1, -1)
        upper_mask = ~causal_mask
        negative_abs = torch.zeros(
            num_heads, dtype=torch.float64, device=attention_weights.device
        )
        positive_abs = torch.zeros_like(negative_abs)
        negative_count = torch.zeros(
            num_heads, dtype=torch.int64, device=attention_weights.device
        )
        nonfinite_count = torch.zeros(
            (), dtype=torch.int64, device=attention_weights.device
        )
        noncausal_count = torch.zeros_like(nonfinite_count)
        reduction_batch_size = 8
        for batch_start in range(0, batch_size, reduction_batch_size):
            batch_end = min(batch_start + reduction_batch_size, batch_size)
            coefficients = attention_weights[batch_start:batch_end].float()
            nonfinite_count += (~torch.isfinite(coefficients)).sum()
            noncausal_count += torch.count_nonzero(coefficients[..., upper_mask])
            chunk_lengths = lengths[batch_start:batch_end]
            valid_positions = positions.view(1, -1) < chunk_lengths.view(-1, 1)
            valid_pairs = (
                valid_positions.unsqueeze(-1)
                & valid_positions.unsqueeze(-2)
                & causal_mask.unsqueeze(0)
            ).unsqueeze(1)
            negative_abs += (
                -coefficients.clamp(max=0)
                .masked_fill(~valid_pairs, 0)
                .sum(dim=(0, 2, 3))
            ).double()
            positive_abs += (
                coefficients.clamp(min=0)
                .masked_fill(~valid_pairs, 0)
                .sum(dim=(0, 2, 3))
                .double()
            )
            negative_count += ((coefficients < 0) & valid_pairs).sum(dim=(0, 2, 3))

        summary = torch.cat(
            (
                negative_abs,
                positive_abs,
                negative_count.double(),
                nonfinite_count.reshape(1).double(),
                noncausal_count.reshape(1).double(),
            )
        ).cpu()
        if int(summary[3 * num_heads]) != 0:
            raise DiagnosticError("attention coefficients contain non-finite values")
        if int(summary[3 * num_heads + 1]) != 0:
            raise DiagnosticError("attention coefficients violate the causal mask")
        coefficient_count = int(
            sum(int(length) * (int(length) + 1) // 2 for length in self._lengths)
        )
        for head_index, stats in enumerate(self._stats[layer_index]):
            stats.negative_abs_mass += float(summary[head_index])
            stats.positive_abs_mass += float(summary[num_heads + head_index])
            stats.negative_count += int(summary[2 * num_heads + head_index])
            stats.coefficient_count += coefficient_count

    def finish_batch(self) -> None:
        if self._layer_call_index != len(self._layers):
            raise DiagnosticError(
                "model made fewer attention calls than reconstructed SAFA layers"
            )
        self._layer_call_index = 0
        self._batch_index += 1

    def result(self) -> List[Dict[str, Any]]:
        output = []
        for layer_index, (layer, head_stats) in enumerate(
            zip(self._layers, self._stats)
        ):
            heads = []
            for head_index, stats in enumerate(head_stats):
                mean_log_gate = (
                    stats.transition_log_gate_sum / stats.transition_gate_count
                    if stats.transition_gate_count
                    else None
                )
                half_life = (
                    math.log(0.5) / mean_log_gate
                    if mean_log_gate is not None and mean_log_gate < 0
                    else None
                )
                total_abs_mass = stats.negative_abs_mass + stats.positive_abs_mass
                if self._attention_mode == "safa":
                    forgetting = {
                        "available": True,
                        "forget_weight_l2": float(
                            torch.linalg.vector_norm(
                                layer._forget_weight[head_index].detach().float()
                            )
                        ),
                        "forget_bias": float(
                            layer._forget_bias[head_index].detach().float()
                        ),
                        "gate_valid_position_count": int(stats.gate_histogram.total),
                        "gate_quantiles": stats.gate_histogram.quantiles(),
                        "transition_gate_count": stats.transition_gate_count,
                        "mean_transition_log_gate": mean_log_gate,
                        "effective_half_life_events": half_life,
                        "survival_positive_lag_population_count": (
                            stats.survival_population_count
                        ),
                        "survival_positive_lag_sample_count": (
                            stats.survival_sample_count
                        ),
                        "survival_quantiles": stats.survival_histogram.quantiles(),
                    }
                else:
                    forgetting = {
                        "available": False,
                        "reason": (
                            "matched HSTU does not apply forget gates; its dormant "
                            "parameter inventory is intentionally not interpreted"
                        ),
                    }
                heads.append(
                    {
                        "head": head_index,
                        "forgetting": forgetting,
                        "signed_coefficients": {
                            "coefficient_count": stats.coefficient_count,
                            "negative_count_fraction": (
                                stats.negative_count / stats.coefficient_count
                                if stats.coefficient_count
                                else None
                            ),
                            "negative_absolute_mass_fraction": (
                                stats.negative_abs_mass / total_abs_mass
                                if total_abs_mass > 0
                                else None
                            ),
                            "negative_absolute_mass": stats.negative_abs_mass,
                            "positive_absolute_mass": stats.positive_abs_mass,
                        },
                    }
                )
            output.append({"layer": layer_index, "heads": heads})
        return output


@contextlib.contextmanager
def instrument_attention(hstu_module: Any, collector: AttentionDiagnostics):
    original = hstu_module._signed_additive_attention_weights

    def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
        output = original(*args, **kwargs)
        if args:
            raise DiagnosticError("unexpected positional SAFA helper invocation")
        collector.observe(
            padded_k=kwargs["padded_k"],
            attention_mode=kwargs["attention_mode"],
            forget_weight=kwargs["forget_weight"],
            forget_bias=kwargs["forget_bias"],
            attention_weights=output,
        )
        return output

    hstu_module._signed_additive_attention_weights = wrapped
    try:
        yield
    finally:
        hstu_module._signed_additive_attention_weights = original


def _parameter_counts(model: torch.nn.Module) -> Tuple[int, int, str]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    inventory = "\n".join(
        f"{name}\t{tuple(parameter.shape)}\t{parameter.dtype}\t"
        f"{parameter.requires_grad}"
        for name, parameter in sorted(model.named_parameters())
    )
    return total, trainable, _sha256_bytes(inventory.encode("utf-8"))


def validate_state_dict_schema(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, Any],
) -> None:
    expected_keys = set(expected)
    actual_keys = set(actual)
    if expected_keys != actual_keys:
        raise DiagnosticError(
            "checkpoint state keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        if not isinstance(actual_tensor, torch.Tensor):
            raise DiagnosticError(f"checkpoint state value is not a tensor: {name}")
        if actual_tensor.shape != expected_tensor.shape:
            raise DiagnosticError(f"checkpoint state shape mismatch: {name}")
        if actual_tensor.dtype != expected_tensor.dtype:
            raise DiagnosticError(f"checkpoint state dtype mismatch: {name}")


def _module_is_within(module: Any, source_root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).resolve().relative_to(source_root)
    except ValueError:
        return False
    return True


def _load_source_modules(source_root: Path) -> Dict[str, Any]:
    for name, module in tuple(sys.modules.items()):
        if name == "generative_recommenders" or name.startswith(
            "generative_recommenders."
        ):
            if not _module_is_within(module, source_root):
                raise DiagnosticError(
                    "a generative_recommenders module was imported before the exact "
                    "training source snapshot"
                )
    sys.path.insert(0, str(source_root))
    try:
        importlib.import_module("fbgemm_gpu")
        modules = {
            "gin": importlib.import_module("gin"),
            "train": importlib.import_module(
                "generative_recommenders.research.trainer.train"
            ),
            "reco": importlib.import_module(
                "generative_recommenders.research.data.reco_dataset"
            ),
            "loader": importlib.import_module(
                "generative_recommenders.research.trainer.data_loader"
            ),
            "features": importlib.import_module(
                "generative_recommenders.research.modeling.sequential.features"
            ),
            "embedding": importlib.import_module(
                "generative_recommenders.research.modeling.sequential.embedding_modules"
            ),
            "input": importlib.import_module(
                "generative_recommenders.research.modeling.sequential.input_features_preprocessors"
            ),
            "output": importlib.import_module(
                "generative_recommenders.research.modeling.sequential.output_postprocessors"
            ),
            "encoder": importlib.import_module(
                "generative_recommenders.research.modeling.sequential.encoder_utils"
            ),
            "similarity": importlib.import_module(
                "generative_recommenders.research.modeling.similarity_utils"
            ),
            "losses": importlib.import_module(
                "generative_recommenders.research.modeling.sequential.autoregressive_losses"
            ),
            "eval": importlib.import_module(
                "generative_recommenders.research.data.eval"
            ),
            "topk": importlib.import_module(
                "generative_recommenders.research.indexing.utils"
            ),
            "hstu": importlib.import_module(
                "generative_recommenders.research.modeling.sequential.hstu"
            ),
        }
    except Exception as error:
        raise DiagnosticError(
            f"could not import the exact training source environment: {error}"
        ) from error
    for name, module in modules.items():
        if name != "gin" and not _module_is_within(module, source_root):
            raise DiagnosticError(
                f"module {name} did not load from the training snapshot"
            )
    return modules


def _gin_query(gin: Any, name: str) -> Any:
    try:
        return gin.query_parameter(name)
    except ValueError as error:
        raise DiagnosticError(f"operative Gin config is missing {name}") from error


def _expect_type(value: Any, expected: type, name: str) -> Any:
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise DiagnosticError(f"{name} must be an integer")
    elif not isinstance(value, expected):
        raise DiagnosticError(f"{name} must be {expected.__name__}")
    return value


def _build_model_and_dataset(
    bundle: RunBundle,
    modules: Mapping[str, Any],
    device: torch.device,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    gin = modules["gin"]
    gin.clear_config()
    try:
        gin.parse_config(bundle.resolved_config)
    except Exception as error:
        raise DiagnosticError(
            f"could not parse checkpoint operative Gin config: {error}"
        )

    mode = _gin_query(gin, "hstu_encoder.attention_mode")
    seed = _gin_query(gin, "train_fn.random_seed")
    if mode != bundle.checkpoint["attention_mode"]:
        raise DiagnosticError("Gin/checkpoint attention_mode mismatch")
    if seed != bundle.checkpoint["random_seed"]:
        raise DiagnosticError("Gin/checkpoint random_seed mismatch")
    exact_sha, experiment_sha = modules["train"]._config_identities(
        bundle.resolved_config,
        attention_mode=mode,
        random_seed=seed,
    )
    if exact_sha != bundle.checkpoint["resolved_gin_config_sha256"]:
        raise DiagnosticError("recomputed exact Gin identity mismatch")
    if experiment_sha != bundle.checkpoint["experiment_config_sha256"]:
        raise DiagnosticError("recomputed experiment Gin identity mismatch")

    config = {
        name: _gin_query(gin, f"train_fn.{name}")
        for name in (
            "dataset_name",
            "max_sequence_length",
            "positional_sampling_ratio",
            "eval_batch_size",
            "eval_user_max_batch_size",
            "main_module",
            "main_module_bf16",
            "dropout_rate",
            "user_embedding_norm",
            "sampling_strategy",
            "item_l2_norm",
            "top_k_method",
            "embedding_module_type",
            "item_embedding_dim",
            "interaction_module_type",
            "gr_output_length",
            "l2_norm_eps",
            "enable_tf32",
            "random_seed",
            "num_epochs",
        )
    }
    for name in (
        "dataset_name",
        "main_module",
        "user_embedding_norm",
        "sampling_strategy",
        "top_k_method",
        "embedding_module_type",
        "interaction_module_type",
    ):
        _expect_type(config[name], str, f"train_fn.{name}")
    for name in (
        "max_sequence_length",
        "eval_batch_size",
        "item_embedding_dim",
        "gr_output_length",
        "random_seed",
        "num_epochs",
    ):
        _expect_type(config[name], int, f"train_fn.{name}")
    if config["main_module"] != "HSTU" or mode not in ("hstu", "safa"):
        raise DiagnosticError("diagnostics require matched HSTU or SAFA")
    validate_final_epoch(bundle.checkpoint["epoch"], config["num_epochs"])
    if config["embedding_module_type"] != "local":
        raise DiagnosticError("only the trained local embedding module is supported")
    if config["sampling_strategy"] != "local":
        raise DiagnosticError("only the trained local negative sampler is supported")

    modules["train"]._seed_everything(config["random_seed"])
    torch.backends.cuda.matmul.allow_tf32 = bool(config["enable_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["enable_tf32"])
    dataset = modules["reco"].get_reco_dataset(
        dataset_name=config["dataset_name"],
        max_sequence_length=config["max_sequence_length"],
        chronological=True,
        positional_sampling_ratio=config["positional_sampling_ratio"],
    )
    embedding = modules["embedding"].LocalEmbeddingModule(
        num_items=dataset.max_item_id,
        item_embedding_dim=config["item_embedding_dim"],
    )
    interaction, _ = modules["similarity"].get_similarity_function(
        module_type=config["interaction_module_type"],
        query_embedding_dim=config["item_embedding_dim"],
        item_embedding_dim=config["item_embedding_dim"],
    )
    if config["user_embedding_norm"] == "l2_norm":
        output_postprocessor = modules["output"].L2NormEmbeddingPostprocessor(
            embedding_dim=config["item_embedding_dim"],
            eps=1e-6,
        )
    elif config["user_embedding_norm"] == "layer_norm":
        output_postprocessor = modules["output"].LayerNormEmbeddingPostprocessor(
            embedding_dim=config["item_embedding_dim"],
            eps=1e-6,
        )
    else:
        raise DiagnosticError("unsupported trained user_embedding_norm")
    input_preprocessor = modules[
        "input"
    ].LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=(dataset.max_sequence_length + config["gr_output_length"] + 1),
        embedding_dim=config["item_embedding_dim"],
        dropout_rate=config["dropout_rate"],
    )
    model = modules["encoder"].get_sequential_encoder(
        module_type=config["main_module"],
        max_sequence_length=dataset.max_sequence_length,
        max_output_length=config["gr_output_length"] + 1,
        embedding_module=embedding,
        interaction_module=interaction,
        input_preproc_module=input_preprocessor,
        output_postproc_module=output_postprocessor,
        verbose=False,
    )
    total, trainable, inventory_sha = _parameter_counts(model)
    if total != bundle.checkpoint["parameter_count"]:
        raise DiagnosticError("reconstructed model parameter count mismatch")
    if inventory_sha != bundle.checkpoint["parameter_inventory_sha256"]:
        raise DiagnosticError("reconstructed model parameter inventory mismatch")
    if getattr(model, "_attention_mode", None) != mode:
        raise DiagnosticError("reconstructed model attention_mode mismatch")
    if config["main_module_bf16"]:
        model = model.to(torch.bfloat16)
    validate_state_dict_schema(
        model.state_dict(),
        bundle.checkpoint["model_state_dict"],
    )
    try:
        model.load_state_dict(bundle.checkpoint["model_state_dict"], strict=True)
    except Exception as error:
        raise DiagnosticError(
            f"strict checkpoint state load failed: {error}"
        ) from error
    model = model.to(device).eval()
    negatives_sampler = (
        modules["losses"]
        .LocalNegativesSampler(
            num_items=dataset.max_item_id,
            item_emb=model._embedding_module._item_emb,
            all_item_ids=dataset.all_item_ids,
            l2_norm=config["item_l2_norm"],
            l2_norm_eps=config["l2_norm_eps"],
        )
        .to(device)
    )
    return model, negatives_sampler, dataset, config


def _index_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(int(index).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def data_fingerprint(data_root: Path, dataset_name: str) -> Dict[str, Any]:
    if dataset_name not in ("ml-1m", "ml-20m"):
        raise DiagnosticError(
            "data fingerprinting is implemented only for the paired ML-1M/ML-20M runs"
        )
    relative_paths = (
        Path(dataset_name) / "sasrec_format.csv",
        Path("processed") / dataset_name / "movies.csv",
    )
    files = []
    combined = hashlib.sha256()
    for relative_path in relative_paths:
        path = data_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise DiagnosticError(f"required data file is not a regular file: {path}")
        try:
            digest = _sha256_file(path)
            size = path.stat().st_size
        except OSError as error:
            raise DiagnosticError(f"could not fingerprint data file {path}: {error}")
        relative = relative_path.as_posix()
        combined.update(f"{relative}\t{size}\t{digest}\n".encode("utf-8"))
        files.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return {
        "algorithm": "sha256-v1",
        "combined_sha256": combined.hexdigest(),
        "files": files,
        "authenticated_by_training_artifact": False,
    }


def _gap_labels(
    row: Mapping[str, torch.Tensor], lengths: torch.Tensor
) -> Tuple[List[Optional[str]], int, Optional[str]]:
    required = ("historical_timestamps", "target_timestamps")
    missing = [key for key in required if key not in row]
    if missing:
        return [None] * lengths.numel(), lengths.numel(), f"missing fields: {missing}"
    historical = row["historical_timestamps"]
    targets = row["target_timestamps"].reshape(-1)
    if historical.ndim != 2 or targets.numel() != lengths.numel():
        return (
            [None] * lengths.numel(),
            lengths.numel(),
            "timestamp tensors have incompatible shapes",
        )
    labels: List[Optional[str]] = []
    invalid = 0
    for index, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        valid_history = historical[index, :length].to(torch.int64)
        last_timestamp = int(valid_history[-1])
        target_timestamp = int(targets[index])
        gap = target_timestamp - last_timestamp
        history_is_positive = bool((valid_history > 0).all())
        history_is_chronological = bool((valid_history[1:] >= valid_history[:-1]).all())
        if (
            not history_is_positive
            or not history_is_chronological
            or target_timestamp <= 0
            or gap < 0
        ):
            labels.append(None)
            invalid += 1
        else:
            labels.append(_bucket(gap, GAP_BINS))
    reason = (
        "non-positive or non-chronological timestamps were excluded"
        if invalid
        else None
    )
    return labels, invalid, reason


@torch.inference_mode()
def _evaluate(
    *,
    model: Any,
    negatives_sampler: Any,
    dataset: Any,
    config: Mapping[str, Any],
    modules: Mapping[str, Any],
    device: torch.device,
    indices: Sequence[int],
    batch_size: int,
    collector: AttentionDiagnostics,
) -> Dict[str, Any]:
    subset = torch.utils.data.Subset(dataset.eval_dataset, list(indices))
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    eval_state = modules["eval"].get_eval_state(
        model=model,
        all_item_ids=dataset.all_item_ids,
        negatives_sampler=negatives_sampler,
        top_k_module_fn=lambda embeddings, ids: modules["topk"].get_top_k_module(
            top_k_method=config["top_k_method"],
            model=model,
            item_embeddings=embeddings,
            item_ids=ids,
        ),
        device=device,
        float_dtype=torch.bfloat16 if config["main_module_bf16"] else None,
    )
    history_labels = tuple(label for _, label in HISTORY_BINS)
    gap_labels = tuple(label for _, label in GAP_BINS)
    overall = MetricStrata(("all",))
    by_history = MetricStrata(history_labels)
    by_gap = MetricStrata(gap_labels)
    gap_invalid_count = 0
    gap_reasons = set()
    processed = 0

    with instrument_attention(modules["hstu"], collector):
        for row in loader:
            lengths = row["history_lengths"].to(torch.int64).reshape(-1)
            if bool((lengths <= 0).any()):
                raise DiagnosticError("evaluation row has an empty history")
            collector.start_batch(lengths)
            features, target_ids, target_ratings = modules[
                "features"
            ].movielens_seq_features_from_row(
                row,
                device=device,
                max_output_length=config["gr_output_length"] + 1,
            )
            metrics = modules["eval"].eval_metrics_v2_from_tensors(
                eval_state,
                model,
                features,
                target_ids=target_ids,
                target_ratings=target_ratings,
                user_max_batch_size=config["eval_user_max_batch_size"],
                dtype=torch.bfloat16 if config["main_module_bf16"] else None,
            )
            collector.finish_batch()
            batch_length = lengths.numel()
            overall.update(["all"] * batch_length, metrics)
            by_history.update(
                [_bucket(int(length), HISTORY_BINS) for length in lengths],
                metrics,
            )
            labels, invalid_count, reason = _gap_labels(row, lengths)
            gap_invalid_count += invalid_count
            if reason:
                gap_reasons.add(reason)
            by_gap.update(labels, metrics)
            processed += batch_length
    if processed != len(indices):
        raise DiagnosticError("evaluation processed an unexpected number of examples")

    valid_gap_count = processed - gap_invalid_count
    return {
        "overall": overall.result()[0],
        "by_history_length": {
            "unit": "events",
            "bins": by_history.result(),
        },
        "by_elapsed_time_gap": {
            "available": valid_gap_count > 0,
            "unit": "seconds",
            "definition": "target_timestamp - last_valid_history_timestamp",
            "valid_count": valid_gap_count,
            "invalid_count": gap_invalid_count,
            "limitations": sorted(gap_reasons),
            "bins": by_gap.result() if valid_gap_count else [],
        },
    }


def run_diagnostics(args: argparse.Namespace) -> Dict[str, Any]:
    bundle = load_run_bundle(
        checkpoint_path=args.checkpoint,
        run_dir=args.run_dir,
        source_root_override=args.source_root,
        require_slurm_provenance=not args.allow_missing_slurm_provenance,
    )
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise DiagnosticError(f"data root is not a directory: {data_root}")
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as error:
        raise DiagnosticError(f"invalid device {args.device!r}: {error}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DiagnosticError("CUDA diagnostics requested but CUDA is unavailable")
    modules = _load_source_modules(bundle.source_root)

    with tempfile.TemporaryDirectory(prefix="safa-diagnostics-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        (temporary_root / "tmp").symlink_to(data_root, target_is_directory=True)
        previous_cwd = Path.cwd()
        try:
            os.chdir(temporary_root)
            model, negatives_sampler, dataset, config = _build_model_and_dataset(
                bundle=bundle,
                modules=modules,
                device=device,
            )
            dataset_identity = data_fingerprint(data_root, config["dataset_name"])
            indices = deterministic_dataset_indices(
                len(dataset.eval_dataset),
                args.max_examples,
                args.sample_seed,
            )
            batch_size = args.batch_size or int(config["eval_batch_size"])
            if batch_size <= 0:
                raise DiagnosticError("batch_size must be positive")
            layers = list(model._hstu._attention_layers)
            collector = AttentionDiagnostics(
                layers,
                histogram_bins=args.histogram_bins,
                pair_samples_per_batch=args.pair_samples_per_batch,
                sample_seed=args.sample_seed,
                attention_mode=bundle.checkpoint["attention_mode"],
            )
            recommendation_metrics = _evaluate(
                model=model,
                negatives_sampler=negatives_sampler,
                dataset=dataset,
                config=config,
                modules=modules,
                device=device,
                indices=indices,
                batch_size=batch_size,
                collector=collector,
            )
        finally:
            os.chdir(previous_cwd)

    require_unchanged_file(
        bundle.checkpoint_path,
        bundle.checkpoint_sha256,
        "checkpoint",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {
            "path": str(bundle.checkpoint_path),
            "epoch": bundle.checkpoint["epoch"],
            "sha256": bundle.checkpoint_sha256,
        },
        "run_dir": str(bundle.run_dir),
        "source": {key: bundle.verified_source[key] for key in PROVENANCE_KEYS},
        "scheduler": {key: bundle.checkpoint.get(key) for key in SLURM_KEYS},
        "config": {
            "dataset_name": config["dataset_name"],
            "attention_mode": bundle.checkpoint["attention_mode"],
            "random_seed": bundle.checkpoint["random_seed"],
            "resolved_gin_config_sha256": bundle.checkpoint[
                "resolved_gin_config_sha256"
            ],
            "experiment_config_sha256": bundle.checkpoint["experiment_config_sha256"],
            "parameter_count": bundle.checkpoint["parameter_count"],
            "parameter_inventory_sha256": bundle.checkpoint[
                "parameter_inventory_sha256"
            ],
        },
        "data": dataset_identity,
        "evaluation_sample": {
            "dataset_size": len(dataset.eval_dataset),
            "requested_max_examples": args.max_examples,
            "evaluated_examples": len(indices),
            "selection": "uniform-without-replacement-sha256-v1",
            "sample_seed": args.sample_seed,
            "selected_indices_sha256": _index_sha256(indices),
            "batch_size": batch_size,
        },
        "attention_diagnostics": {
            "gate_quantiles": (
                "fixed-width histogram over every valid sampled position; value "
                "resolution is 1 / histogram_bins"
            ),
            "survival_quantiles": (
                "batch-population-weighted deterministic sample of valid positive-lag "
                "causal pairs"
            ),
            "histogram_bins": args.histogram_bins,
            "pair_samples_per_batch": args.pair_samples_per_batch,
            "half_life_definition": (
                "log(0.5) / mean(log(forget_gate)) over valid transitions; unit events"
            ),
            "signed_mass_definition": (
                "sum(abs(negative final attention coefficients)) / "
                "sum(abs(all final attention coefficients)) over valid causal pairs"
            ),
            "layers": collector.result(),
        },
        "recommendation_metrics": recommendation_metrics,
        "limitations": [
            "Results describe the deterministic evaluation sample, not unsampled users.",
            "Gate histogram counts, half-life sufficient statistics, and signed "
            "coefficient mass are exact within the evaluated example sample. Gate "
            "quantiles are histogram-resolved; survival quantiles are both sampled "
            "and histogram-resolved.",
            "Effective half-life is in event steps under the observed mean log gate; "
            "it is not a wall-clock half-life or a constant-gate claim.",
            "Elapsed-time strata exclude rows without positive, chronological history "
            "and target timestamps and report the excluded count.",
            "Matched HSTU has no active forgetting mechanism; gate and survival "
            "statistics are unavailable by design, while signed-mass and recommendation "
            "strata remain directly comparable to SAFA.",
            "Training artifacts did not record a data checksum. Current input hashes are "
            "reported for paired-report equality but cannot retrospectively authenticate "
            "the bytes used during training.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing operative_config.gin and run_metadata.json.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Immutable training snapshot. Defaults to the checkpoint source_root and "
            "must match its commit, tree, and manifest."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            Path(os.environ["GR_DATA_ROOT"]) if "GR_DATA_ROOT" in os.environ else None
        ),
        required="GR_DATA_ROOT" not in os.environ,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=20260813)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pair-samples-per-batch", type=int, default=4096)
    parser.add_argument("--histogram-bins", type=int, default=10000)
    parser.add_argument(
        "--allow-missing-slurm-provenance",
        action="store_true",
        help=(
            "Permit a legacy/local checkpoint with no scheduler fields. Any partially "
            "present scheduler identity is still rejected."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_examples <= 0:
        raise SystemExit("--max-examples must be positive")
    if args.sample_seed < 0:
        raise SystemExit("--sample-seed must be non-negative")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.pair_samples_per_batch <= 0:
        raise SystemExit("--pair-samples-per-batch must be positive")
    if args.histogram_bins < 100:
        raise SystemExit("--histogram-bins must be at least 100")
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        if args.output.exists() and not args.force:
            raise SystemExit(f"output exists (use --force): {args.output}")
    try:
        result = run_diagnostics(args)
    except DiagnosticError as error:
        raise SystemExit(f"attention diagnostics refused to run: {error}") from error
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
