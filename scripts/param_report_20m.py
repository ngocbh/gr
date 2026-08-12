#!/usr/bin/env python3
"""Deterministic parameter audit for the HSTU attention experiments.

The default report is pure Python: it derives named tensor shapes directly from
the constructors in ``research/modeling/sequential/hstu.py`` and verifies that
the dimensions still agree with each experiment's gin file.  This keeps the
report usable on login nodes where importing FLA probes for a CUDA driver.

``--verify-constructors`` performs an independent CPU cross-check against real
``nn.Module.named_parameters()`` inventories.  It installs process-local stubs
for kernel callables only; no forward pass is possible or attempted.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import io
import json
import math
import re
import sys
import warnings
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Iterator, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MAX_SEQUENCE_LENGTH = 200
CANONICAL_GR_OUTPUT_LENGTH = 10
ML1_MAX_ITEM_ID = 3952
ML20_MAX_ITEM_ID = 131262

Shape = Tuple[int, ...]
Inventory = Dict[str, Shape]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    dataset: str
    config: str
    embedding_dim: int
    max_item_id: int
    blocks: int
    heads: int
    dqk: int
    dv: int
    normalization: str = "rel_bias"
    relative_bias: bool = True
    gate_rank: int = 0
    output_rank: int = 0
    time_gate: str = "continuous"
    window_size: int = 64
    reference: str | None = None
    assessment: str = "reference"


@dataclass(frozen=True)
class ResolvedSettings:
    """Validated settings that reproduce the model construction in train.py."""

    spec: ModelSpec
    dataset_name: str
    max_sequence_length: int
    gr_output_length: int
    main_module: str
    main_module_bf16: bool
    embedding_module_type: str
    item_embedding_dim: int
    dropout_rate: float
    user_embedding_norm: str
    interaction_module_type: str
    l2_norm_eps: float
    activation_checkpoint: bool
    blocks: int
    heads: int
    dqk: int
    dv: int
    linear_dropout_rate: float
    attn_dropout_rate: float
    normalization: str
    linear_config: str
    linear_activation: str
    concat_ua: bool
    relative_bias: bool
    time_gate: str
    gate_rank: int
    output_rank: int
    kla_omega_coupling: bool
    forgetting_min_period: float
    forgetting_max_period: float
    window_size: int
    softmax_temperature: float
    max_item_id: int

    @property
    def max_output_length(self) -> int:
        return self.gr_output_length + 1

    @property
    def position_rows(self) -> int:
        return self.max_sequence_length + self.max_output_length


SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        "ml20_hstu",
        "HSTU",
        "ml-20m",
        "configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin",
        256,
        ML20_MAX_ITEM_ID,
        16,
        8,
        32,
        32,
    ),
    ModelSpec(
        "ml20_softmax",
        "Softmax",
        "ml-20m",
        "configs/ml-20m/hstu-softmax-20m.gin",
        256,
        ML20_MAX_ITEM_ID,
        16,
        8,
        32,
        32,
        normalization="softmax_rel_bias",
        reference="ml20_hstu",
        assessment="exact named inventory",
    ),
    ModelSpec(
        "ml20_fohstu",
        "Full FoHSTU",
        "ml-20m",
        "configs/ml-20m/hstu-forgetting-20m.gin",
        256,
        ML20_MAX_ITEM_ID,
        16,
        8,
        32,
        32,
        normalization="forgetting_rel_bias",
        reference="ml20_hstu",
        assessment="close count; not exact",
    ),
    ModelSpec(
        "ml20_local_w32",
        "Local W32",
        "ml-20m",
        "configs/ml-20m/hstu-local-forgetting-w32-20m.gin",
        256,
        ML20_MAX_ITEM_ID,
        16,
        8,
        32,
        32,
        normalization="local_forgetting_rel_bias",
        window_size=32,
        reference="ml20_hstu",
        assessment="close count; not exact vs HSTU",
    ),
    ModelSpec(
        "ml20_lift_w32",
        "LIFT W32",
        "ml-20m",
        "configs/ml-20m/hstu-hybrid-forgetting-w32-20m.gin",
        256,
        ML20_MAX_ITEM_ID,
        16,
        8,
        32,
        32,
        normalization="hybrid_forgetting_rel_bias",
        window_size=32,
        reference="ml20_local_w32",
        assessment="exact named inventory vs Local W32",
    ),
    ModelSpec(
        "ml20_kda",
        "KDA",
        "ml-20m",
        "configs/ml-20m/hstu-kda-core-20m.gin",
        256,
        ML20_MAX_ITEM_ID,
        16,
        8,
        32,
        32,
        normalization="kda",
        relative_bias=False,
        gate_rank=64,
        output_rank=64,
        time_gate="none",
        reference="ml20_hstu",
        assessment="close count; structurally confounded",
    ),
    ModelSpec(
        "ml20_iso_kla",
        "IsoKLA",
        "ml-20m",
        "configs/ml-20m/hstu-iso-kla-core-20m.gin",
        256,
        ML20_MAX_ITEM_ID,
        16,
        8,
        32,
        32,
        normalization="iso_kla",
        relative_bias=False,
        gate_rank=64,
        output_rank=64,
        time_gate="none",
        reference="ml20_hstu",
        assessment="close count; structurally confounded",
    ),
    ModelSpec(
        "ml20_diag_kla",
        "DiagKLA-14",
        "ml-20m",
        "configs/ml-20m/hstu-diag-kla-core-20m.gin",
        256,
        ML20_MAX_ITEM_ID,
        14,
        8,
        32,
        32,
        normalization="diag_kla",
        relative_bias=False,
        gate_rank=32,
        output_rank=32,
        time_gate="none",
        reference="ml20_hstu",
        assessment="materially confounded (depth/projections)",
    ),
    ModelSpec(
        "ml20_diag_kla_r64",
        "DiagKLA-16-r64",
        "ml-20m",
        "configs/ml-20m/hstu-diag-kla-core-20m-r64.gin",
        256,
        ML20_MAX_ITEM_ID,
        16,
        8,
        32,
        32,
        normalization="diag_kla",
        relative_bias=False,
        gate_rank=64,
        output_rank=64,
        time_gate="none",
        reference="ml20_hstu",
        assessment="materially confounded (parameter count)",
    ),
    ModelSpec(
        "ml1_large_hstu",
        "HSTU-large",
        "ml-1m",
        "configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin",
        50,
        ML1_MAX_ITEM_ID,
        8,
        2,
        25,
        25,
    ),
    ModelSpec(
        "ml1_large_kda",
        "KDA-large",
        "ml-1m",
        "configs/ml-1m/hstu-kda-core-large-final.gin",
        50,
        ML1_MAX_ITEM_ID,
        8,
        2,
        25,
        25,
        normalization="kda",
        relative_bias=False,
        reference="ml1_large_hstu",
        assessment="materially confounded (parameter count)",
    ),
    ModelSpec(
        "ml1_large_kda_balanced",
        "KDA-large-balanced",
        "ml-1m",
        "configs/ml-1m/hstu-kda-core-large-balanced-final.gin",
        50,
        ML1_MAX_ITEM_ID,
        8,
        2,
        25,
        25,
        normalization="kda",
        relative_bias=False,
        gate_rank=15,
        output_rank=15,
        reference="ml1_large_hstu",
        assessment="close count; structurally confounded",
    ),
    ModelSpec(
        "ml1_large_iso_kla",
        "IsoKLA-large",
        "ml-1m",
        "configs/ml-1m/hstu-iso-kla-core-large-final.gin",
        50,
        ML1_MAX_ITEM_ID,
        8,
        2,
        25,
        25,
        normalization="iso_kla",
        relative_bias=False,
        time_gate="none",
        reference="ml1_large_hstu",
        assessment="materially confounded (parameter count)",
    ),
    ModelSpec(
        "ml1_large_diag_kla",
        "DiagKLA-large",
        "ml-1m",
        "configs/ml-1m/hstu-diag-kla-core-large-final.gin",
        50,
        ML1_MAX_ITEM_ID,
        8,
        2,
        32,
        32,
        normalization="diag_kla",
        relative_bias=False,
        time_gate="none",
        reference="ml1_large_hstu",
        assessment="materially confounded (width/count)",
    ),
    ModelSpec(
        "ml1_hstu",
        "HSTU-2block",
        "ml-1m",
        "configs/ml-1m/hstu-sampled-softmax-n128-final.gin",
        50,
        ML1_MAX_ITEM_ID,
        2,
        1,
        50,
        50,
    ),
    ModelSpec(
        "ml1_kda_matched",
        "KDA-matched",
        "ml-1m",
        "configs/ml-1m/hstu-kda-core-matched-final.gin",
        50,
        ML1_MAX_ITEM_ID,
        2,
        1,
        43,
        43,
        normalization="kda",
        relative_bias=False,
        time_gate="none",
        reference="ml1_hstu",
        assessment="near count only; width/bias confounded",
    ),
    ModelSpec(
        "ml1_kda_time_matched",
        "KDA-time-matched",
        "ml-1m",
        "configs/ml-1m/hstu-kda-core-time-matched-final.gin",
        50,
        ML1_MAX_ITEM_ID,
        2,
        1,
        43,
        43,
        normalization="kda",
        relative_bias=False,
        time_gate="continuous",
        reference="ml1_hstu",
        assessment="near count only; width/bias confounded",
    ),
)

SPEC_BY_KEY = {spec.key: spec for spec in SPECS}
GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "MovieLens-20M live attention configs",
        tuple(spec.key for spec in SPECS if spec.key.startswith("ml20_")),
    ),
    (
        "MovieLens-1M large shell",
        (
            "ml1_large_hstu",
            "ml1_large_kda",
            "ml1_large_kda_balanced",
            "ml1_large_iso_kla",
            "ml1_large_diag_kla",
        ),
    ),
    (
        "MovieLens-1M configs named *-matched",
        ("ml1_hstu", "ml1_kda_matched", "ml1_kda_time_matched"),
    ),
)

TRAIN_SOURCE = ROOT / "generative_recommenders/research/trainer/train.py"
ENCODER_SOURCE = (
    ROOT / "generative_recommenders/research/modeling/sequential/encoder_utils.py"
)
PREPROCESSOR_SOURCE = ROOT / "generative_recommenders/research/data/preprocessor.py"


def _parse_python_source(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid escape sequence", category=DeprecationWarning
        )
        return ast.parse(path.read_text(), filename=str(path))


def _function_literal_defaults(path: Path, function_name: str) -> Dict[str, object]:
    tree = _parse_python_source(path)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"{path}: missing function {function_name}")
    positional = function.args.posonlyargs + function.args.args
    defaults: Dict[str, object] = {}
    for argument, default in zip(
        positional[-len(function.args.defaults) :], function.args.defaults
    ):
        defaults[argument.arg] = ast.literal_eval(default)
    for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
        if default is not None:
            defaults[argument.arg] = ast.literal_eval(default)
    return defaults


@lru_cache(maxsize=1)
def source_defaults() -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    for prefix, path, function_name in (
        ("train_fn", TRAIN_SOURCE, "train_fn"),
        ("hstu_encoder", ENCODER_SOURCE, "hstu_encoder"),
        (
            "get_sequential_encoder",
            ENCODER_SOURCE,
            "get_sequential_encoder",
        ),
    ):
        defaults.update(
            {
                f"{prefix}.{name}": value
                for name, value in _function_literal_defaults(
                    path, function_name
                ).items()
            }
        )
    return defaults


@lru_cache(maxsize=1)
def dataset_max_item_ids_from_source() -> Dict[str, int]:
    tree = _parse_python_source(PREPROCESSOR_SOURCE)
    variable_to_dataset = {"ml_1m_dp": "ml-1m", "ml_20m_dp": "ml-20m"}
    values: Dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in variable_to_dataset:
                continue
            keyword = next(
                (
                    entry
                    for entry in value.keywords
                    if entry.arg == "expected_max_item_id"
                ),
                None,
            )
            if keyword is not None:
                values[variable_to_dataset[target.id]] = int(
                    ast.literal_eval(keyword.value)
                )
    return values


def canonical_bindings(spec: ModelSpec) -> Dict[str, object]:
    """Every training/encoder setting assumed by the inventory formula."""
    return {
        "train_fn.dataset_name": spec.dataset,
        "train_fn.max_sequence_length": CANONICAL_MAX_SEQUENCE_LENGTH,
        "train_fn.main_module": "HSTU",
        "train_fn.main_module_bf16": False,
        "train_fn.dropout_rate": 0.2,
        "train_fn.user_embedding_norm": "l2_norm",
        "train_fn.embedding_module_type": "local",
        "train_fn.item_embedding_dim": spec.embedding_dim,
        "train_fn.interaction_module_type": "DotProduct",
        "train_fn.gr_output_length": CANONICAL_GR_OUTPUT_LENGTH,
        "train_fn.l2_norm_eps": 1e-6,
        "get_sequential_encoder.activation_checkpoint": False,
        "hstu_encoder.num_blocks": spec.blocks,
        "hstu_encoder.num_heads": spec.heads,
        "hstu_encoder.dqk": spec.dqk,
        "hstu_encoder.dv": spec.dv,
        "hstu_encoder.linear_dropout_rate": 0.2,
        "hstu_encoder.attn_dropout_rate": 0.0,
        "hstu_encoder.normalization": spec.normalization,
        "hstu_encoder.linear_config": "uvqk",
        "hstu_encoder.linear_activation": "silu",
        "hstu_encoder.concat_ua": False,
        "hstu_encoder.enable_relative_attention_bias": spec.relative_bias,
        "hstu_encoder.kda_time_gate": spec.time_gate,
        "hstu_encoder.kda_gate_rank": spec.gate_rank,
        "hstu_encoder.kda_o_rank": spec.output_rank,
        "hstu_encoder.kla_omega_coupling": False,
        "hstu_encoder.forgetting_min_period": 8.0,
        "hstu_encoder.forgetting_max_period": 256.0,
        "hstu_encoder.hybrid_window_size": spec.window_size,
        "hstu_encoder.softmax_temperature": 0.0,
        "hstu_encoder.signed_feature_gamma": 1.0,
        "hstu_encoder.hybrid_tail_feature_map": "identity",
    }


def source_assumption_errors() -> Tuple[str, ...]:
    errors = []
    defaults = source_defaults()
    tracked_hstu = {
        key for key in canonical_bindings(SPECS[0]) if key.startswith("hstu_encoder.")
    }
    actual_hstu = {key for key in defaults if key.startswith("hstu_encoder.")}
    if actual_hstu != tracked_hstu:
        errors.append(
            "hstu_encoder optional arguments changed: "
            f"untracked={sorted(actual_hstu - tracked_hstu)!r} "
            f"missing={sorted(tracked_hstu - actual_hstu)!r}"
        )
    source_ids = dataset_max_item_ids_from_source()
    expected_ids = {spec.dataset: spec.max_item_id for spec in SPECS}
    if source_ids != expected_ids:
        errors.append(
            f"dataset max item IDs changed: source={source_ids!r} expected={expected_ids!r}"
        )
    for key in canonical_bindings(SPECS[0]):
        if key not in defaults:
            errors.append(f"source default missing for tracked binding {key}")
    return tuple(errors)


def _numel(shape: Shape) -> int:
    return math.prod(shape)


def inventory_total(inventory: Mapping[str, Shape]) -> int:
    return sum(_numel(shape) for shape in inventory.values())


def mixer_total(inventory: Mapping[str, Shape]) -> int:
    return sum(
        _numel(shape) for name, shape in inventory.items() if name.startswith("_hstu.")
    )


def expected_inventory(settings: ResolvedSettings) -> Inventory:
    """Return the exact trainable name/shape inventory implied by ``hstu.py``."""
    inventory: Inventory = {
        "_embedding_module._item_emb.weight": (
            settings.max_item_id + 1,
            settings.item_embedding_dim,
        ),
        "_input_features_preproc._pos_emb.weight": (
            settings.position_rows,
            settings.item_embedding_dim,
        ),
    }
    h = settings.heads
    dk = settings.dqk
    dv = settings.dv
    dim = settings.item_embedding_dim
    linear_attention = settings.normalization in {"kda", "iso_kla", "diag_kla"}

    for layer in range(settings.blocks):
        prefix = f"_hstu._attention_layers.{layer}."
        inventory[prefix + "_uvqk"] = (dim, 2 * h * (dv + dk))

        if settings.relative_bias:
            inventory[prefix + "_rel_attn_bias._ts_w"] = (129,)
            inventory[prefix + "_rel_attn_bias._pos_w"] = (
                2 * settings.position_rows - 1,
            )

        output_dim = h * dv * (3 if settings.concat_ua else 1)
        if settings.output_rank > 0:
            inventory[prefix + "_o.0.weight"] = (settings.output_rank, output_dim)
            inventory[prefix + "_o.1.weight"] = (dim, settings.output_rank)
            inventory[prefix + "_o.1.bias"] = (dim,)
        else:
            inventory[prefix + "_o.weight"] = (dim, output_dim)
            inventory[prefix + "_o.bias"] = (dim,)

        if settings.normalization in {
            "forgetting_rel_bias",
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        }:
            inventory[prefix + "_forget_weight"] = (h, dk)
            inventory[prefix + "_forget_bias"] = (h,)
        if settings.normalization in {
            "local_forgetting_rel_bias",
            "hybrid_forgetting_rel_bias",
        }:
            inventory[prefix + "_hybrid_tail_rho"] = (h,)

        if not linear_attention:
            continue
        if settings.gate_rank > 0:
            inventory[prefix + "_kda_f_proj.0.weight"] = (
                settings.gate_rank,
                dim,
            )
            inventory[prefix + "_kda_f_proj.1.weight"] = (
                h * dk,
                settings.gate_rank,
            )
        else:
            inventory[prefix + "_kda_f_proj.weight"] = (h * dk, dim)
        inventory[prefix + "_kda_A_log"] = (h,)
        inventory[prefix + "_kda_dt_bias"] = (h * dk,)
        if settings.time_gate == "continuous":
            inventory[prefix + "_kda_time_w"] = (h, dk)

        if settings.normalization == "kda":
            inventory[prefix + "_kda_b_proj.weight"] = (h, dim)
        elif settings.normalization == "iso_kla":
            inventory[prefix + "_kla_r_proj.weight"] = (h, dim)
            inventory[prefix + "_kla_r_proj.bias"] = (h,)
            inventory[prefix + "_kla_qn_proj.weight"] = (h, dim)
            inventory[prefix + "_kla_qn_proj.bias"] = (h,)
            inventory[prefix + "_kla_mu_param"] = (h,)
        else:
            inventory[prefix + "_kla_r_proj.weight"] = (h, dim)
            inventory[prefix + "_kla_r_proj.bias"] = (h,)
            inventory[prefix + "_kla_qn_proj.weight"] = (h * dk, dim)
            inventory[prefix + "_kla_qn_proj.bias"] = (h * dk,)
            inventory[prefix + "_kla_mu_param"] = (h,)
    return inventory


def expected_buffers(settings: ResolvedSettings) -> Inventory:
    # Fixed-forgetting variants would add two buffers per layer; none are in this audit.
    return {"_attn_mask": (settings.position_rows, settings.position_rows)}


def parse_gin_text(text: str, source: str = "<gin>") -> Dict[str, object]:
    assignments: Dict[str, object] = {}
    assignment = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*$")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = assignment.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        try:
            assignments[key] = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"{source}:{line_number}: cannot parse {raw_value!r}"
            ) from error
    return assignments


def parse_gin_assignments(path: Path) -> Dict[str, object]:
    return parse_gin_text(path.read_text(), source=str(path))


def _config_assignments(spec: ModelSpec) -> Dict[str, object]:
    path = ROOT / spec.config
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {spec.config}")
    return parse_gin_assignments(path)


def resolved_binding_values(
    spec: ModelSpec, assignments: Mapping[str, object] | None = None
) -> Dict[str, object]:
    values = dict(source_defaults())
    values.update(_config_assignments(spec) if assignments is None else assignments)
    return values


def config_errors(
    spec: ModelSpec, assignments: Mapping[str, object] | None = None
) -> Tuple[str, ...]:
    try:
        explicit = (
            _config_assignments(spec) if assignments is None else dict(assignments)
        )
    except FileNotFoundError as error:
        return (str(error),)
    values = resolved_binding_values(spec, explicit)
    expected = canonical_bindings(spec)
    errors = []
    for key, expected_value in expected.items():
        actual = values.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            errors.append(
                f"{spec.config}: {key}={actual!r}, report expects {expected_value!r}"
            )
    tracked_hstu = {key for key in expected if key.startswith("hstu_encoder.")}
    for key in sorted(explicit):
        if key.startswith("hstu_encoder.") and key not in tracked_hstu:
            errors.append(f"{spec.config}: untracked HSTU binding {key}")
    return tuple(errors)


def resolve_settings(
    spec: ModelSpec, assignments: Mapping[str, object] | None = None
) -> ResolvedSettings:
    errors = config_errors(spec, assignments)
    if errors:
        raise ValueError("; ".join(errors))
    values = resolved_binding_values(spec, assignments)
    max_item_ids = dataset_max_item_ids_from_source()
    dataset_name = str(values["train_fn.dataset_name"])
    return ResolvedSettings(
        spec=spec,
        dataset_name=dataset_name,
        max_sequence_length=int(values["train_fn.max_sequence_length"]),
        gr_output_length=int(values["train_fn.gr_output_length"]),
        main_module=str(values["train_fn.main_module"]),
        main_module_bf16=bool(values["train_fn.main_module_bf16"]),
        embedding_module_type=str(values["train_fn.embedding_module_type"]),
        item_embedding_dim=int(values["train_fn.item_embedding_dim"]),
        dropout_rate=float(values["train_fn.dropout_rate"]),
        user_embedding_norm=str(values["train_fn.user_embedding_norm"]),
        interaction_module_type=str(values["train_fn.interaction_module_type"]),
        l2_norm_eps=float(values["train_fn.l2_norm_eps"]),
        activation_checkpoint=bool(
            values["get_sequential_encoder.activation_checkpoint"]
        ),
        blocks=int(values["hstu_encoder.num_blocks"]),
        heads=int(values["hstu_encoder.num_heads"]),
        dqk=int(values["hstu_encoder.dqk"]),
        dv=int(values["hstu_encoder.dv"]),
        linear_dropout_rate=float(values["hstu_encoder.linear_dropout_rate"]),
        attn_dropout_rate=float(values["hstu_encoder.attn_dropout_rate"]),
        normalization=str(values["hstu_encoder.normalization"]),
        linear_config=str(values["hstu_encoder.linear_config"]),
        linear_activation=str(values["hstu_encoder.linear_activation"]),
        concat_ua=bool(values["hstu_encoder.concat_ua"]),
        relative_bias=bool(values["hstu_encoder.enable_relative_attention_bias"]),
        time_gate=str(values["hstu_encoder.kda_time_gate"]),
        gate_rank=int(values["hstu_encoder.kda_gate_rank"]),
        output_rank=int(values["hstu_encoder.kda_o_rank"]),
        kla_omega_coupling=bool(values["hstu_encoder.kla_omega_coupling"]),
        forgetting_min_period=float(values["hstu_encoder.forgetting_min_period"]),
        forgetting_max_period=float(values["hstu_encoder.forgetting_max_period"]),
        window_size=int(values["hstu_encoder.hybrid_window_size"]),
        softmax_temperature=float(values["hstu_encoder.softmax_temperature"]),
        max_item_id=max_item_ids[dataset_name],
    )


def compare_inventories(
    reference: Mapping[str, Shape], candidate: Mapping[str, Shape]
) -> Dict[str, object]:
    reference_names, candidate_names = set(reference), set(candidate)
    return {
        "added": [
            {
                "name": name,
                "shape": list(candidate[name]),
                "numel": _numel(candidate[name]),
            }
            for name in sorted(candidate_names - reference_names)
        ],
        "removed": [
            {
                "name": name,
                "shape": list(reference[name]),
                "numel": _numel(reference[name]),
            }
            for name in sorted(reference_names - candidate_names)
        ],
        "reshaped": [
            {
                "name": name,
                "from_shape": list(reference[name]),
                "to_shape": list(candidate[name]),
                "from_numel": _numel(reference[name]),
                "to_numel": _numel(candidate[name]),
            }
            for name in sorted(reference_names & candidate_names)
            if reference[name] != candidate[name]
        ],
    }


def inventory_sha256(inventory: Mapping[str, Shape]) -> str:
    payload = [
        {"name": name, "shape": list(shape), "numel": _numel(shape)}
        for name, shape in sorted(inventory.items())
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def comparison_sha256(
    reference: Mapping[str, Shape], candidate: Mapping[str, Shape]
) -> str:
    payload = compare_inventories(reference, candidate)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _grouped_inventory(inventory: Mapping[str, Shape]) -> Counter[Tuple[str, Shape]]:
    grouped: Counter[Tuple[str, Shape]] = Counter()
    for name, shape in inventory.items():
        pattern = re.sub(
            r"_attention_layers\.\d+\.", "_attention_layers.{layer}.", name
        )
        grouped[(pattern, shape)] += 1
    return grouped


def _format_delta(value: int, reference: int) -> str:
    percent = 100.0 * value / reference
    return f"{value:+,} ({percent:+.3f}%)"


def print_human_report(
    settings_by_key: Mapping[str, ResolvedSettings],
    inventories: Mapping[str, Inventory],
    details: bool,
) -> None:
    print(
        "Static source: hstu.py constructor formulas; gin bindings checked. "
        "Buffers are excluded from all counts."
    )
    print("Dataset IDs: ml-1m max=3952/rows=3953; " "ml-20m max=131262/rows=131263.")
    for title, keys in GROUPS:
        print(f"\n=== {title} ===")
        header = (
            f"{'model':20s} {'blk':>3s} {'H':>2s} {'d':>3s} {'r':>3s} | "
            f"{'mixer':>10s} {'total':>11s} | {'delta vs reference':>24s} | assessment"
        )
        print(header)
        print("-" * len(header))
        for key in keys:
            spec = SPEC_BY_KEY[key]
            settings = settings_by_key[key]
            inventory = inventories[key]
            rank = (
                settings.gate_rank if settings.gate_rank == settings.output_rank else -1
            )
            if spec.reference is None:
                delta = "reference"
            else:
                reference_total = inventory_total(inventories[spec.reference])
                delta = _format_delta(
                    inventory_total(inventory) - reference_total, reference_total
                )
            print(
                f"{spec.label:20s} {settings.blocks:3d} {settings.heads:2d} "
                f"{settings.dqk:3d} "
                f"{rank:3d} | {mixer_total(inventory):10,d} {inventory_total(inventory):11,d} | "
                f"{delta:>24s} | {spec.assessment}"
            )

    print("\n=== KDA/KLA component deltas (exact layer-pattern aggregation) ===")
    for spec in SPECS:
        if spec.normalization not in {"kda", "iso_kla", "diag_kla"}:
            continue
        assert spec.reference is not None
        reference = inventories[spec.reference]
        candidate = inventories[spec.key]
        delta = _grouped_inventory(candidate)
        delta.subtract(_grouped_inventory(reference))
        print(f"\n[{spec.label} vs {SPEC_BY_KEY[spec.reference].label}]")
        for (name, shape), occurrences in sorted(delta.items()):
            if occurrences:
                count = f"{occurrences:+d} tensor" + (
                    "s" if abs(occurrences) != 1 else ""
                )
                print(f"  {count:>11s}  {name} shape={shape} each={_numel(shape):,}")

    print("\n=== buffers (not trainable; excluded above) ===")
    common = expected_buffers(settings_by_key[SPECS[0].key])
    for name, shape in common.items():
        print(f"  every model: {name} shape={shape} numel={_numel(shape):,}")

    if details:
        print("\n=== exact unaggregated name/shape deltas ===")
        for spec in SPECS:
            if spec.normalization not in {"kda", "iso_kla", "diag_kla"}:
                continue
            assert spec.reference is not None
            diff = compare_inventories(
                inventories[spec.reference], inventories[spec.key]
            )
            print(f"\n[{spec.label}]")
            for kind in ("removed", "added", "reshaped"):
                for entry in diff[kind]:  # type: ignore[index]
                    print(f"  {kind[:-1]:8s} {json.dumps(entry, sort_keys=True)}")


def json_report(
    settings_by_key: Mapping[str, ResolvedSettings],
    inventories: Mapping[str, Inventory],
) -> Dict[str, object]:
    models = []
    for spec in SPECS:
        settings = settings_by_key[spec.key]
        inventory = inventories[spec.key]
        binding_values = resolved_binding_values(spec)
        reference_inventory = (
            inventories[spec.reference] if spec.reference is not None else None
        )
        models.append(
            {
                "key": spec.key,
                "label": spec.label,
                "dataset": spec.dataset,
                "config": spec.config,
                "resolved_bindings": {
                    key: binding_values[key] for key in sorted(canonical_bindings(spec))
                },
                "dimensions": {
                    "embedding_dim": settings.item_embedding_dim,
                    "max_item_id": settings.max_item_id,
                    "embedding_rows": settings.max_item_id + 1,
                    "max_sequence_length": settings.max_sequence_length,
                    "gr_output_length": settings.gr_output_length,
                    "position_rows": settings.position_rows,
                    "blocks": settings.blocks,
                    "heads": settings.heads,
                    "dqk": settings.dqk,
                    "dv": settings.dv,
                    "gate_rank": settings.gate_rank,
                    "output_rank": settings.output_rank,
                },
                "parameter_total": inventory_total(inventory),
                "mixer_total": mixer_total(inventory),
                "parameter_inventory_sha256": inventory_sha256(inventory),
                "parameters": [
                    {"name": name, "shape": list(shape), "numel": _numel(shape)}
                    for name, shape in sorted(inventory.items())
                ],
                "buffers": [
                    {"name": name, "shape": list(shape), "numel": _numel(shape)}
                    for name, shape in sorted(expected_buffers(settings).items())
                ],
                "reference": spec.reference,
                "assessment": spec.assessment,
                "delta": (
                    compare_inventories(reference_inventory, inventory)
                    if reference_inventory is not None
                    else None
                ),
                "delta_inventory_sha256": (
                    comparison_sha256(reference_inventory, inventory)
                    if reference_inventory is not None
                    else None
                ),
            }
        )
    return {
        "schema_version": 2,
        "counting_method": "AST-resolved train.py/hstu.py constructor formula",
        "buffers_excluded_from_parameter_counts": True,
        "models": models,
    }


def _function_node(
    tree: ast.AST, function_name: str, class_name: str | None = None
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    scope: Iterable[ast.AST]
    if class_name is None:
        scope = getattr(tree, "body", ())
    else:
        class_node = next(
            (
                node
                for node in getattr(tree, "body", ())
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ),
            None,
        )
        if class_node is None:
            raise ValueError(f"missing class {class_name}")
        scope = class_node.body
    function = next(
        (
            node
            for node in scope
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"missing function {function_name}")
    return function


def _expression(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _same_expression(actual: ast.AST, expected: str) -> bool:
    return ast.dump(actual, include_attributes=False) == ast.dump(
        _expression(expected), include_attributes=False
    )


def _assignment_targets(node: ast.AST) -> Tuple[ast.AST, ...]:
    if isinstance(node, ast.Assign):
        return tuple(node.targets)
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        return (node.target,)
    return ()


def _writes_name(node: ast.AST, target_name: str) -> bool:
    return any(
        any(
            isinstance(part, ast.Name) and part.id == target_name
            for part in ast.walk(target)
        )
        for target in _assignment_targets(node)
    )


def _write_nodes(scope: ast.AST, target_name: str) -> Tuple[ast.AST, ...]:
    return tuple(node for node in ast.walk(scope) if _writes_name(node, target_name))


def _direct_assignment_values(
    statements: Sequence[ast.stmt], target_name: str
) -> Tuple[ast.expr, ...]:
    values = []
    for node in statements:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in targets
        ):
            values.append(node.value)
    return tuple(values)


def _is_self_method_call(call: ast.Call, method_name: str) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == method_name
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
    )


def _all_self_method_calls(scope: ast.AST, method_name: str) -> Tuple[ast.Call, ...]:
    return tuple(
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and _is_self_method_call(node, method_name)
    )


def _unconditional_statements(statements: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield statements guaranteed to execute if their containing scope executes.

    Context-manager bodies are transparent. Conditional/loop/try/function bodies
    are deliberately opaque so a matching expression in dead or optional code
    cannot satisfy the proof.
    """
    for statement in statements:
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            yield from _unconditional_statements(statement.body)
        else:
            yield statement


def _unconditional_self_method_calls(
    statements: Sequence[ast.stmt], method_name: str
) -> Tuple[ast.Call, ...]:
    calls = []
    for statement in _unconditional_statements(statements):
        if isinstance(
            statement,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.FunctionDef),
        ):
            continue
        calls.extend(
            node
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and _is_self_method_call(node, method_name)
        )
    return tuple(calls)


def _string_constants(expression: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(expression)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _direct_if_with_constants(
    statements: Sequence[ast.stmt], constants: set[str]
) -> ast.If | None:
    matches = [
        statement
        for statement in statements
        if isinstance(statement, ast.If)
        and _string_constants(statement.test) == constants
    ]
    return matches[0] if len(matches) == 1 else None


def _call_semantic_errors(
    label: str, call: ast.Call, expected_keywords: Mapping[str, str]
) -> Tuple[str, ...]:
    errors = []
    if call.args:
        errors.append(f"{label} unexpectedly uses positional arguments")
    if any(keyword.arg is None for keyword in call.keywords):
        errors.append(f"{label} unexpectedly expands **kwargs")
    keywords = {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }
    if set(keywords) != set(expected_keywords):
        errors.append(
            f"{label} keyword set changed: actual={sorted(keywords)!r} "
            f"expected={sorted(expected_keywords)!r}"
        )
    for name, expected in expected_keywords.items():
        if name not in keywords or not _same_expression(keywords[name], expected):
            errors.append(f"{label} {name} is not {expected}")
    return tuple(errors)


def fla_kda_source_path() -> Path:
    for package in ("fla-core", "flash-linear-attention"):
        try:
            candidate = Path(distribution(package).locate_file("fla/ops/kda/chunk.py"))
        except PackageNotFoundError:
            continue
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("installed FLA KDA source not found")


def head_layout_source_errors(
    fla_source: Path | None = None, hstu_source: Path | None = None
) -> Tuple[str, ...]:
    """Prove head preservation and scaling from the executed source expressions."""
    errors = []
    hstu_path = hstu_source or (
        ROOT / "generative_recommenders/research/modeling/sequential/hstu.py"
    )
    hstu_tree = _parse_python_source(hstu_path)
    forward = _function_node(
        hstu_tree, "forward", class_name="SequentialTransductionUnitJagged"
    )
    linear_matches = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.If)
        and _string_constants(node.test) == {"kda", "iso_kla", "diag_kla", "exact_kla"}
    ]
    linear_branch = linear_matches[0] if len(linear_matches) == 1 else None
    if linear_branch is None:
        errors.append("hstu.py must contain exactly one KDA/KLA normalization branch")
        return tuple(errors)

    def is_head_dimension_assignment(node: ast.AST) -> bool:
        return isinstance(node, ast.Assign) and any(
            ast.unparse(target) == "(H, dk, dv)" for target in node.targets
        )

    all_dimension_assignments = [
        node for node in ast.walk(linear_branch) if is_head_dimension_assignment(node)
    ]
    direct_dimension_assignments = [
        node for node in linear_branch.body if is_head_dimension_assignment(node)
    ]
    unique_dimension_writes = all(
        len(_write_nodes(linear_branch, name)) == 1 for name in ("H", "dk", "dv")
    )
    if (
        len(all_dimension_assignments) != 1
        or len(direct_dimension_assignments) != 1
        or not unique_dimension_writes
        or not _same_expression(
            direct_dimension_assignments[0].value,
            "(self._num_heads, self._attention_dim, self._linear_dim)",
        )
    ):
        errors.append(
            "hstu.py must bind H/dk/dv once and unconditionally from per-head dimensions"
        )

    for target, expected in {
        "padded_q": "_pad(q).view(B, n, H, dk)",
        "padded_k": "_pad(k).view(B, n, H, dk)",
        "padded_v": "_pad(v).view(B, n, H, dv)",
    }.items():
        all_writes = _write_nodes(linear_branch, target)
        direct_values = _direct_assignment_values(linear_branch.body, target)
        if (
            len(all_writes) != 1
            or len(direct_values) != 1
            or not _same_expression(direct_values[0], expected)
        ):
            errors.append(
                f"hstu.py {target} must have one unconditional definition {expected}"
            )

    kda_iso_branch = _direct_if_with_constants(linear_branch.body, {"kda", "iso_kla"})
    if kda_iso_branch is None:
        errors.append("hstu.py must contain one direct KDA/IsoKLA branch")
    else:
        all_calls = _all_self_method_calls(kda_iso_branch, "_chunk_kda")
        effective_calls = _unconditional_self_method_calls(
            kda_iso_branch.body, "_chunk_kda"
        )
        if len(all_calls) != 1 or len(effective_calls) != 1:
            errors.append(
                "hstu.py KDA/IsoKLA must have one unconditional _chunk_kda call"
            )
        else:
            errors.extend(
                _call_semantic_errors(
                    "hstu.py _chunk_kda",
                    effective_calls[0],
                    {
                        "q": "padded_q.bfloat16()",
                        "k": "padded_k.bfloat16()",
                        "v": "padded_v.bfloat16()",
                        "g": "g.bfloat16()",
                        "beta": "beta.bfloat16()",
                        "A_log": "self._kda_A_log",
                        "dt_bias": "self._kda_dt_bias",
                        "use_qk_l2norm_in_kernel": "True",
                        "use_gate_in_kernel": "True",
                        "use_beta_sigmoid_in_kernel": "use_beta_sigmoid",
                    },
                )
            )

        diag_statements = kda_iso_branch.orelse
        for target, expected in {
            "qn": "F.normalize(padded_q.float(), p=2, dim=-1)",
            "kn": "F.normalize(padded_k.float(), p=2, dim=-1)",
        }.items():
            all_writes = tuple(
                node
                for statement in diag_statements
                for node in _write_nodes(statement, target)
            )
            direct_values = _direct_assignment_values(diag_statements, target)
            if (
                len(all_writes) != 1
                or len(direct_values) != 1
                or not _same_expression(direct_values[0], expected)
            ):
                errors.append(
                    f"hstu.py {target} must have one unconditional definition {expected}"
                )

        diag_scope = ast.Module(body=diag_statements, type_ignores=[])
        all_kalman_calls = _all_self_method_calls(diag_scope, "_chunk_kalman")
        effective_kalman_calls = _unconditional_self_method_calls(
            diag_statements, "_chunk_kalman"
        )
        if len(all_kalman_calls) != 1 or len(effective_kalman_calls) != 1:
            errors.append(
                "hstu.py DiagKLA must have one unconditional _chunk_kalman call"
            )
        else:
            errors.extend(
                _call_semantic_errors(
                    "hstu.py _chunk_kalman",
                    effective_kalman_calls[0],
                    {
                        "q": "qn",
                        "k": "kn",
                        "kappa": "kappa.float()",
                        "v": "padded_v.float()",
                        "g": "alpha.clamp_min(1e-6).log()",
                        "scale": "dk ** -0.5",
                        "use_qk_l2norm_in_kernel": "False",
                        "output_final_state": "False",
                    },
                )
            )

    try:
        fla_path = fla_source or fla_kda_source_path()
        fla_tree = _parse_python_source(fla_path)
        chunk_kda = _function_node(fla_tree, "chunk_kda")
    except (FileNotFoundError, ValueError, SyntaxError) as error:
        errors.append(f"cannot inspect FLA chunk_kda: {error}")
        return tuple(errors)

    def is_shape_assignment(node: ast.AST) -> bool:
        return isinstance(node, ast.Assign) and any(
            ast.unparse(target) == "(B, T, H, K, HV)" for target in node.targets
        )

    all_shape_assignments = [
        node for node in ast.walk(chunk_kda) if is_shape_assignment(node)
    ]
    direct_shape_assignments = [
        node for node in chunk_kda.body if is_shape_assignment(node)
    ]
    if (
        len(all_shape_assignments) != 1
        or len(direct_shape_assignments) != 1
        or len(_write_nodes(chunk_kda, "K")) != 1
        or any(_write_nodes(chunk_kda, name) for name in ("q", "k", "v"))
        or not _same_expression(
            direct_shape_assignments[0].value, "(*q.shape, v.shape[2])"
        )
    ):
        errors.append(
            "FLA chunk_kda must bind K once and unconditionally from q.shape's last dimension"
        )

    all_scale_guards = [
        node
        for node in ast.walk(chunk_kda)
        if isinstance(node, ast.If) and _same_expression(node.test, "scale is None")
    ]
    direct_scale_guards = [
        node
        for node in chunk_kda.body
        if isinstance(node, ast.If) and _same_expression(node.test, "scale is None")
    ]
    all_scale_writes = _write_nodes(chunk_kda, "scale")
    scale_guard_valid = False
    if len(all_scale_guards) == 1 and len(direct_scale_guards) == 1:
        guard = direct_scale_guards[0]
        guard_assignments = _direct_assignment_values(guard.body, "scale")
        scale_guard_valid = (
            len(guard.body) == 1
            and len(guard_assignments) == 1
            and len(all_scale_writes) == 1
            and _same_expression(guard_assignments[0], "K ** -0.5")
        )
    if not scale_guard_valid:
        errors.append(
            "FLA chunk_kda must assign scale exactly once as K ** -0.5 in its direct None guard"
        )

    direct_returns = [
        statement for statement in chunk_kda.body if isinstance(statement, ast.Return)
    ]
    if len(direct_returns) != 1:
        errors.append("FLA chunk_kda must have one direct return")
    else:
        return_value = direct_returns[0].value
        valid_return = (
            isinstance(return_value, ast.Call)
            and isinstance(return_value.func, ast.Attribute)
            and isinstance(return_value.func.value, ast.Name)
            and return_value.func.value.id == "ChunkKDAFunction"
            and return_value.func.attr == "apply"
            and len(return_value.args) > 7
            and _same_expression(return_value.args[7], "scale")
        )
        if not valid_return:
            errors.append("FLA chunk_kda does not pass the guarded scale to its kernel")
    return tuple(errors)


def _unavailable_kernel(*args: object, **kwargs: object) -> None:
    raise RuntimeError("parameter-audit kernel stub was called")


@contextmanager
def _counting_kernel_stubs() -> Iterator[None]:
    """Stub only dynamically imported kernel modules during constructor checks."""
    missing = object()
    saved: Dict[str, object] = {}

    def install(name: str, **attributes: object) -> ModuleType:
        saved[name] = sys.modules.get(name, missing)
        module = ModuleType(name)
        module.__dict__.update(attributes)
        if name in {"fla", "fla.ops"}:
            module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
        return module

    fla = install("fla")
    fla_ops = install("fla.ops")
    fla_kda = install("fla.ops.kda", chunk_kda=_unavailable_kernel)
    fla.ops = fla_ops  # type: ignore[attr-defined]
    fla_ops.kda = fla_kda  # type: ignore[attr-defined]

    prefix = "generative_recommenders.research.modeling.sequential.kla.kla_ops."
    install(prefix + "iso_chunk", iso_beta_chunk=_unavailable_kernel)
    install(prefix + "kalman_chunk", chunk_kalman=_unavailable_kernel)
    install(prefix + "diag_chunk", kla_kappa_chunk=_unavailable_kernel)
    install(prefix + "gain_recurrent", gain_recurrent=_unavailable_kernel)
    try:
        yield
    finally:
        for name, previous in reversed(tuple(saved.items())):
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous  # type: ignore[assignment]


def constructor_inventory(settings: ResolvedSettings) -> Tuple[Inventory, Inventory]:
    """Instantiate a real HSTU shell on CPU without importing GPU kernels."""
    spec = settings.spec
    if settings.main_module != "HSTU":
        raise ValueError(f"unsupported main module {settings.main_module}")
    if settings.embedding_module_type != "local":
        raise ValueError(f"unsupported embedding type {settings.embedding_module_type}")
    with _counting_kernel_stubs():
        from generative_recommenders.research.modeling.sequential.embedding_modules import (
            LocalEmbeddingModule,
        )
        from generative_recommenders.research.modeling.sequential.encoder_utils import (
            hstu_encoder,
        )
        from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
            LearnablePositionalEmbeddingInputFeaturesPreprocessor,
        )
        from generative_recommenders.research.modeling.sequential.output_postprocessors import (
            L2NormEmbeddingPostprocessor,
            LayerNormEmbeddingPostprocessor,
        )
        from generative_recommenders.research.modeling.similarity_utils import (
            get_similarity_function,
        )

        with redirect_stdout(io.StringIO()):
            embedding = LocalEmbeddingModule(
                num_items=settings.max_item_id,
                item_embedding_dim=settings.item_embedding_dim,
            )
            similarity, _ = get_similarity_function(
                module_type=settings.interaction_module_type,
                query_embedding_dim=settings.item_embedding_dim,
                item_embedding_dim=settings.item_embedding_dim,
            )
            preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
                max_sequence_len=settings.position_rows,
                embedding_dim=settings.item_embedding_dim,
                dropout_rate=settings.dropout_rate,
            )
            postprocessor_type = (
                L2NormEmbeddingPostprocessor
                if settings.user_embedding_norm == "l2_norm"
                else LayerNormEmbeddingPostprocessor
            )
            postprocessor = postprocessor_type(
                embedding_dim=settings.item_embedding_dim,
                eps=settings.l2_norm_eps,
            )
            model = hstu_encoder(
                max_sequence_length=settings.max_sequence_length,
                max_output_length=settings.max_output_length,
                embedding_module=embedding,
                similarity_module=similarity,
                input_preproc_module=preprocessor,
                output_postproc_module=postprocessor,
                activation_checkpoint=settings.activation_checkpoint,
                verbose=False,
                num_blocks=settings.blocks,
                num_heads=settings.heads,
                dqk=settings.dqk,
                dv=settings.dv,
                linear_dropout_rate=settings.linear_dropout_rate,
                attn_dropout_rate=settings.attn_dropout_rate,
                normalization=settings.normalization,
                linear_config=settings.linear_config,
                linear_activation=settings.linear_activation,
                concat_ua=settings.concat_ua,
                enable_relative_attention_bias=settings.relative_bias,
                kda_gate_rank=settings.gate_rank,
                kda_o_rank=settings.output_rank,
                kda_time_gate=settings.time_gate,
                kla_omega_coupling=settings.kla_omega_coupling,
                forgetting_min_period=settings.forgetting_min_period,
                forgetting_max_period=settings.forgetting_max_period,
                hybrid_window_size=settings.window_size,
                softmax_temperature=settings.softmax_temperature,
            )
            if settings.main_module_bf16:
                model = model.bfloat16()
        parameters = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                raise AssertionError(f"unexpected frozen parameter: {spec.key}:{name}")
            parameters[name] = tuple(parameter.shape)
        buffers = {name: tuple(buffer.shape) for name, buffer in model.named_buffers()}
        del model, embedding, similarity, preprocessor, postprocessor
    gc.collect()
    return parameters, buffers


def verify_constructors(
    settings_by_key: Mapping[str, ResolvedSettings],
    inventories: Mapping[str, Inventory],
) -> Tuple[str, ...]:
    errors = []
    for spec in SPECS:
        error_count = len(errors)
        settings = settings_by_key[spec.key]
        try:
            actual_parameters, actual_buffers = constructor_inventory(settings)
        except Exception as error:
            errors.append(
                f"{spec.key}: constructor unavailable: {type(error).__name__}: {error}"
            )
            continue
        if actual_parameters != inventories[spec.key]:
            errors.append(
                f"{spec.key}: parameter schema mismatch: "
                + json.dumps(
                    compare_inventories(inventories[spec.key], actual_parameters),
                    sort_keys=True,
                )
            )
        if actual_buffers != expected_buffers(settings):
            errors.append(
                f"{spec.key}: buffer mismatch: expected={expected_buffers(settings)!r} "
                f"actual={actual_buffers!r}"
            )
        if len(errors) == error_count:
            print(
                f"[constructor verified] {spec.key}: "
                f"{inventory_total(actual_parameters):,} trainable; "
                f"{inventory_total(actual_buffers):,} buffer elements",
                file=sys.stderr,
            )
    return tuple(errors)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-constructors",
        action="store_true",
        help="cross-check formulas against CPU nn.Module inventories using kernel stubs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit exact deterministic name/shape inventories as JSON",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="print every unaggregated KDA/KLA name/shape delta",
    )
    args = parser.parse_args(argv)

    errors = list(source_assumption_errors())
    errors.extend(error for spec in SPECS for error in config_errors(spec))
    if errors:
        for error in errors:
            print(f"[config mismatch] {error}", file=sys.stderr)
        return 1

    settings_by_key = {spec.key: resolve_settings(spec) for spec in SPECS}
    inventories = {
        key: expected_inventory(settings) for key, settings in settings_by_key.items()
    }
    if args.verify_constructors:
        errors = list(head_layout_source_errors())
        errors.extend(verify_constructors(settings_by_key, inventories))
        if errors:
            for error in errors:
                print(f"[constructor mismatch] {error}", file=sys.stderr)
            return 1

    if args.json:
        print(
            json.dumps(
                json_report(settings_by_key, inventories), indent=2, sort_keys=True
            )
        )
    else:
        print_human_report(settings_by_key, inventories, details=args.details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
