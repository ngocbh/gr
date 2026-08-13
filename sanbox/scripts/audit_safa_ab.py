#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

"""Audit configuration and parameter parity for the HSTU/SAFA experiment."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gin  # noqa: E402
from generative_recommenders.research.modeling.sequential.embedding_modules import (  # noqa: E402
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (  # noqa: E402
    hstu_encoder,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (  # noqa: E402
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (  # noqa: E402
    L2NormEmbeddingPostprocessor,
)
from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import (  # noqa: E402
    DotProductSimilarity,
)

# These imports register every configurable referenced by the paired files.
from generative_recommenders.research.trainer import (
    data_loader as _data_loader,
)  # noqa: E402,F401
from generative_recommenders.research.trainer import train as _train  # noqa: E402,F401


BindingKey = Tuple[str, str, str]
ParameterSignature = Dict[str, Tuple[Tuple[int, ...], bool, int]]


@dataclass(frozen=True)
class PairSpec:
    dataset: str
    max_item_id: int
    upstream_config: Path
    hstu_config: Path
    safa_config: Path
    expected_backbone_parameters: int
    expected_forget_parameters: int
    expected_total_parameters: int


PAIR_SPECS: Mapping[str, PairSpec] = {
    "amzn-books": PairSpec(
        dataset="amzn-books",
        max_item_id=695762,
        upstream_config=REPO_ROOT
        / "configs/amzn-books/hstu-sampled-softmax-n512-large-final.gin",
        hstu_config=REPO_ROOT
        / "configs/amzn-books/hstu-matched-sampled-softmax-n512-large-final.gin",
        safa_config=REPO_ROOT
        / "configs/amzn-books/safa-sampled-softmax-n512-large-final.gin",
        expected_backbone_parameters=44_865_440,
        expected_forget_parameters=1_152,
        expected_total_parameters=44_866_592,
    ),
    "ml-1m": PairSpec(
        dataset="ml-1m",
        max_item_id=3952,
        upstream_config=REPO_ROOT
        / "configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin",
        hstu_config=REPO_ROOT
        / "configs/ml-1m/hstu-matched-sampled-softmax-n128-large-final.gin",
        safa_config=REPO_ROOT
        / "configs/ml-1m/safa-sampled-softmax-n128-large-final.gin",
        expected_backbone_parameters=313_000,
        expected_forget_parameters=416,
        expected_total_parameters=313_416,
    ),
    "ml-20m": PairSpec(
        dataset="ml-20m",
        max_item_id=131262,
        upstream_config=REPO_ROOT
        / "configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin",
        hstu_config=REPO_ROOT
        / "configs/ml-20m/hstu-matched-sampled-softmax-n128-large-final.gin",
        safa_config=REPO_ROOT
        / "configs/ml-20m/safa-sampled-softmax-n128-large-final.gin",
        expected_backbone_parameters=38_913_120,
        expected_forget_parameters=4_224,
        expected_total_parameters=38_917_344,
    ),
}

TRAIN_GR_OUTPUT_LENGTH_DEFAULT = (
    inspect.signature(_train.train_fn).parameters["gr_output_length"].default
)


def _explicit_bindings() -> Dict[BindingKey, Any]:
    bindings: Dict[BindingKey, Any] = {}
    for (scope, selector), parameters in gin.config._CONFIG.items():
        for parameter, value in parameters.items():
            bindings[(scope, selector, parameter)] = value
    return bindings


def load_config_bindings(config_path: Path) -> Dict[BindingKey, Any]:
    gin.clear_config()
    gin.parse_config_file(str(config_path))
    return _explicit_bindings()


def _parameter_keys(
    bindings: Mapping[BindingKey, Any], parameter: str
) -> set[BindingKey]:
    return {key for key in bindings if key[2] == parameter}


def _pop_unique_parameter(
    bindings: Dict[BindingKey, Any], parameter: str, context: str
) -> Any:
    keys = _parameter_keys(bindings, parameter)
    if len(keys) != 1:
        raise AssertionError(
            f"{context}: expected one {parameter} binding, found {keys}"
        )
    return bindings.pop(keys.pop())


def _remove_equivalent_output_length_default(
    bindings: Dict[BindingKey, Any], context: str, required: bool
) -> None:
    keys = _parameter_keys(bindings, "gr_output_length")
    allowed_counts = {1} if required else {0, 1}
    if len(keys) not in allowed_counts:
        expectation = "one" if required else "at most one"
        raise AssertionError(
            f"{context}: expected {expectation} gr_output_length binding, "
            f"found {keys}"
        )
    if not keys:
        return
    value = bindings.pop(keys.pop())
    if value != TRAIN_GR_OUTPUT_LENGTH_DEFAULT:
        raise AssertionError(
            f"{context}: gr_output_length={value!r} changes the trainer default "
            f"{TRAIN_GR_OUTPUT_LENGTH_DEFAULT!r}"
        )


def assert_config_pair(spec: PairSpec) -> None:
    hstu_bindings = load_config_bindings(spec.hstu_config)
    safa_bindings = load_config_bindings(spec.safa_config)
    hstu_mode = _pop_unique_parameter(
        hstu_bindings, "attention_mode", f"{spec.dataset} HSTU"
    )
    safa_mode = _pop_unique_parameter(
        safa_bindings, "attention_mode", f"{spec.dataset} SAFA"
    )
    if (hstu_mode, safa_mode) != ("hstu", "safa"):
        raise AssertionError(
            f"{spec.dataset}: expected paired modes ('hstu', 'safa'), "
            f"found {(hstu_mode, safa_mode)}"
        )
    if hstu_bindings != safa_bindings:
        differing_keys = sorted(
            key
            for key in hstu_bindings.keys() | safa_bindings.keys()
            if hstu_bindings.get(key) != safa_bindings.get(key)
        )
        raise AssertionError(
            f"{spec.dataset}: non-mode gin bindings differ: {differing_keys}"
        )


def assert_upstream_fidelity(spec: PairSpec) -> None:
    upstream_bindings = load_config_bindings(spec.upstream_config)
    if _parameter_keys(upstream_bindings, "attention_mode"):
        raise AssertionError(
            f"{spec.dataset}: canonical upstream config unexpectedly binds attention_mode"
        )
    _remove_equivalent_output_length_default(
        upstream_bindings, f"{spec.dataset} upstream", required=False
    )

    for arm, config_path, expected_mode in (
        ("HSTU", spec.hstu_config, "hstu"),
        ("SAFA", spec.safa_config, "safa"),
    ):
        candidate_bindings = load_config_bindings(config_path)
        mode = _pop_unique_parameter(
            candidate_bindings, "attention_mode", f"{spec.dataset} {arm}"
        )
        if mode != expected_mode:
            raise AssertionError(
                f"{spec.dataset} {arm}: expected attention_mode={expected_mode!r}, "
                f"found {mode!r}"
            )
        _remove_equivalent_output_length_default(
            candidate_bindings, f"{spec.dataset} {arm}", required=True
        )
        if candidate_bindings != upstream_bindings:
            differing_keys = sorted(
                key
                for key in candidate_bindings.keys() | upstream_bindings.keys()
                if candidate_bindings.get(key) != upstream_bindings.get(key)
            )
            raise AssertionError(
                f"{spec.dataset} {arm}: drift from canonical upstream LARGE "
                f"config: {differing_keys}"
            )


def _query_int(binding: str) -> int:
    return int(gin.query_parameter(binding))


def build_model(config_path: Path, max_item_id: int) -> torch.nn.Module:
    gin.clear_config()
    gin.parse_config_file(str(config_path))

    max_sequence_length = _query_int("train_fn.max_sequence_length")
    gr_output_length = _query_int("train_fn.gr_output_length")
    item_embedding_dim = _query_int("train_fn.item_embedding_dim")
    max_output_length = gr_output_length + 1

    embedding_module = LocalEmbeddingModule(
        num_items=max_item_id,
        item_embedding_dim=item_embedding_dim,
    )
    return hstu_encoder(
        max_sequence_length=max_sequence_length,
        max_output_length=max_output_length,
        embedding_module=embedding_module,
        similarity_module=DotProductSimilarity(),
        input_preproc_module=LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=max_sequence_length + max_output_length,
            embedding_dim=item_embedding_dim,
            dropout_rate=float(gin.query_parameter("train_fn.dropout_rate")),
        ),
        output_postproc_module=L2NormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=float(gin.query_parameter("train_fn.l2_norm_eps")),
        ),
        activation_checkpoint=False,
        verbose=False,
    )


def parameter_signature(model: torch.nn.Module) -> ParameterSignature:
    return {
        name: (tuple(parameter.shape), parameter.requires_grad, parameter.numel())
        for name, parameter in model.named_parameters()
    }


def _inventory(config_path: Path, max_item_id: int) -> ParameterSignature:
    model = build_model(config_path=config_path, max_item_id=max_item_id)
    signature = parameter_signature(model)
    del model
    gc.collect()
    return signature


def audit_pair(spec: PairSpec) -> Dict[str, int | str]:
    assert_config_pair(spec)
    assert_upstream_fidelity(spec)
    hstu_signature = _inventory(spec.hstu_config, spec.max_item_id)
    safa_signature = _inventory(spec.safa_config, spec.max_item_id)

    if hstu_signature != safa_signature:
        differing_names = sorted(
            name
            for name in hstu_signature.keys() | safa_signature.keys()
            if hstu_signature.get(name) != safa_signature.get(name)
        )
        raise AssertionError(
            f"{spec.dataset}: parameter names/shapes differ: {differing_names}"
        )

    total_parameters = sum(entry[2] for entry in hstu_signature.values())
    trainable_parameters = sum(
        entry[2] for entry in hstu_signature.values() if entry[1]
    )
    forget_parameters = sum(
        entry[2] for name, entry in hstu_signature.items() if "forget" in name
    )
    backbone_parameters = total_parameters - forget_parameters
    actual_counts = (
        backbone_parameters,
        forget_parameters,
        total_parameters,
        trainable_parameters,
    )
    expected_counts = (
        spec.expected_backbone_parameters,
        spec.expected_forget_parameters,
        spec.expected_total_parameters,
        spec.expected_total_parameters,
    )
    if actual_counts != expected_counts:
        raise AssertionError(
            f"{spec.dataset}: count mismatch; expected {expected_counts}, "
            f"found {actual_counts}"
        )

    return {
        "dataset": spec.dataset,
        "parameter_tensors": len(hstu_signature),
        "backbone_parameters": backbone_parameters,
        "forget_parameters": forget_parameters,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *PAIR_SPECS.keys()],
        default="all",
    )
    args = parser.parse_args()

    datasets = PAIR_SPECS.keys() if args.dataset == "all" else [args.dataset]
    results = [audit_pair(PAIR_SPECS[dataset]) for dataset in datasets]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
