# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import ast
import hashlib
import json
import logging
import os
import random
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gin
import numpy as np
import torch
import torch.distributed as dist

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore
from generative_recommenders.research.data.eval import (
    _avg,
    add_to_summary_writer,
    eval_metrics_v2_from_tensors,
    get_eval_state,
)
from generative_recommenders.research.data.reco_dataset import get_reco_dataset
from generative_recommenders.research.indexing.utils import get_top_k_module
from generative_recommenders.research.modeling.sequential.autoregressive_losses import (
    BCELoss,
    InBatchNegativesSampler,
    LocalNegativesSampler,
)
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    get_sequential_encoder,
)
from generative_recommenders.research.modeling.sequential.features import (
    movielens_seq_features_from_row,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.losses.sampled_softmax import (
    SampledSoftmaxLoss,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.similarity_utils import (
    get_similarity_function,
)
from generative_recommenders.research.trainer.data_loader import create_data_loader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


def setup(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _seed_everything(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def _source_provenance() -> Dict[str, str]:
    source_root = Path(os.environ.get("GR_SOURCE_ROOT", Path.cwd())).resolve()
    identifiers = {
        "source_commit": ("GR_SOURCE_COMMIT", "SOURCE_COMMIT"),
        "source_tree": ("GR_SOURCE_TREE", "SOURCE_TREE"),
        "source_manifest": (
            "GR_SOURCE_MANIFEST",
            "SOURCE_MANIFEST_SHA256",
        ),
    }
    provenance = {"source_root": str(source_root)}
    for key, (environment_name, filename) in identifiers.items():
        value = os.environ.get(environment_name)
        metadata_path = source_root / filename
        if value is None and metadata_path.is_file():
            value = metadata_path.read_text(encoding="utf-8").strip()
        provenance[key] = value or "unavailable"
    return provenance


def _slurm_provenance(
    *, attention_mode: str, random_seed: int, dataset_name: str
) -> Dict[str, Any]:
    """Validate scheduler identity for a full SAFA A/B array run."""
    required = os.environ.get("GR_REQUIRE_SLURM_PROVENANCE", "0")
    if required not in ("0", "1"):
        raise ValueError("GR_REQUIRE_SLURM_PROVENANCE must be 0 or 1")
    if required == "0":
        return {}

    values = {
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_qos": os.environ.get("SLURM_JOB_QOS"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    missing = sorted(key for key, value in values.items() if not value)
    if missing:
        raise ValueError(f"missing required SLURM provenance: {missing}")

    for key in ("slurm_array_job_id", "slurm_job_id"):
        if re.fullmatch(r"[1-9][0-9]*", str(values[key])) is None:
            raise ValueError(f"{key} must be a positive decimal job ID")
    task_id_string = str(values["slurm_array_task_id"])
    if re.fullmatch(r"[0-5]", task_id_string) is None:
        raise ValueError("slurm_array_task_id must be an integer in [0, 5]")
    required_qos = {
        "amzn-books": "h200_mrs_2_high",
        "ml-1m": "h200_dev",
        "ml-20m": "h200_mrs_2_high",
    }.get(dataset_name)
    if required_qos is None:
        raise ValueError(f"unsupported SAFA A/B dataset: {dataset_name}")
    if values["slurm_job_qos"] != required_qos:
        raise ValueError(f"{dataset_name} SAFA A/B runs require QoS {required_qos}")
    restart_count_string = str(values["slurm_restart_count"])
    if re.fullmatch(r"0|[1-9][0-9]*", restart_count_string) is None:
        raise ValueError("slurm_restart_count must be a nonnegative integer")
    if values["slurm_job_partition"] != "h200":
        raise ValueError("full SAFA A/B runs require partition h200")

    expected_runs = (
        (42, "hstu"),
        (42, "safa"),
        (43, "hstu"),
        (43, "safa"),
        (44, "hstu"),
        (44, "safa"),
    )
    task_id = int(task_id_string)
    expected_seed, expected_mode = expected_runs[task_id]
    if (random_seed, attention_mode) != (expected_seed, expected_mode):
        raise ValueError(
            "SLURM array task does not match the configured seed/attention mode"
        )

    return {
        "slurm_array_job_id": str(values["slurm_array_job_id"]),
        "slurm_array_task_id": task_id,
        "slurm_job_id": str(values["slurm_job_id"]),
        "slurm_job_qos": str(values["slurm_job_qos"]),
        "slurm_restart_count": int(restart_count_string),
        "slurm_job_partition": str(values["slurm_job_partition"]),
    }


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
    inventory_sha256 = hashlib.sha256(inventory.encode("utf-8")).hexdigest()
    return total, trainable, inventory_sha256


def _attention_mode(main_module: str) -> str:
    try:
        mode = gin.query_parameter("hstu_encoder.attention_mode")
    except ValueError:
        mode = os.environ.get("GR_ATTENTION_MODE", main_module.lower())
    return str(mode)


_EXPERIMENT_IDENTITY_BINDINGS = (
    "hstu_encoder.attention_mode",
    "train_fn.random_seed",
)
_OPERATIVE_BINDING_PATTERN = re.compile(
    r"^(?P<prefix>[ \t]*(?P<binding>"
    + "|".join(re.escape(binding) for binding in _EXPERIMENT_IDENTITY_BINDINGS)
    + r")[ \t]*=[ \t]*)(?P<value>[^\r\n]*?)(?P<ending>\r?\n)?$"
)


def _config_identities(
    resolved_gin_config: str,
    *,
    attention_mode: str,
    random_seed: int,
) -> Tuple[str, str]:
    """Hash the exact Gin config and one redacting only the A/B dimensions."""
    expected_values = {
        "hstu_encoder.attention_mode": attention_mode,
        "train_fn.random_seed": random_seed,
    }
    found_values: Dict[str, Any] = {}
    normalized_lines = []
    for line in resolved_gin_config.splitlines(keepends=True):
        match = _OPERATIVE_BINDING_PATTERN.fullmatch(line)
        if match is None:
            normalized_lines.append(line)
            continue
        binding = match.group("binding")
        if binding in found_values:
            raise ValueError(f"duplicate operative Gin binding: {binding}")
        try:
            found_values[binding] = ast.literal_eval(match.group("value").strip())
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"operative Gin binding is not a literal: {binding}"
            ) from error
        normalized_lines.append(
            f"{match.group('prefix')}<redacted>{match.group('ending') or ''}"
        )

    missing = sorted(set(expected_values) - set(found_values))
    if missing:
        raise ValueError(f"missing operative Gin identity bindings: {missing}")
    for binding, expected_value in expected_values.items():
        if found_values[binding] != expected_value:
            raise ValueError(
                f"operative Gin binding {binding} does not match runtime metadata"
            )

    exact_sha256 = hashlib.sha256(resolved_gin_config.encode("utf-8")).hexdigest()
    normalized_config = "".join(normalized_lines)
    experiment_sha256 = hashlib.sha256(normalized_config.encode("utf-8")).hexdigest()
    return exact_sha256, experiment_sha256


def _synchronize(device: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _wandb_requirement(*, wandb_enabled: bool, wandb_mode: Optional[str]) -> bool:
    value = os.environ.get("GR_REQUIRE_WANDB", "0")
    if value not in ("0", "1"):
        raise ValueError("GR_REQUIRE_WANDB must be 0 or 1")
    required = value == "1"
    if required and not wandb_enabled:
        raise ValueError("GR_REQUIRE_WANDB=1 requires wandb_enabled=True")
    effective_mode = wandb_mode or os.environ.get("WANDB_MODE", "online")
    if required and effective_mode != "online":
        raise ValueError("required W&B runs must use WANDB_MODE=online")
    return required


def _wandb_log(
    run: Any,
    payload: Dict[str, Any],
    step: int,
    *,
    required: bool = False,
) -> None:
    try:
        run.log(payload, step=step)
    except Exception as error:
        if required:
            raise RuntimeError(f"required W&B logging failed at step {step}") from error
        logging.warning("Weights & Biases logging failed at step %d: %s", step, error)


def _wandb_initialize(*, required: bool, init_kwargs: Dict[str, Any]) -> Any:
    if wandb is None:
        message = "wandb_enabled=True but wandb is not installed"
        if required:
            raise RuntimeError(message)
        logging.warning("%s; continuing without it", message)
        return None
    try:
        run = wandb.init(**init_kwargs)
    except Exception as error:
        if required:
            raise RuntimeError("required W&B initialization failed") from error
        logging.warning(
            "Weights & Biases initialization failed; continuing without it: %s",
            error,
        )
        return None
    if required and run is None:
        raise RuntimeError("required W&B initialization returned no run")
    return run


def _wandb_finish(run: Any, *, required: bool) -> None:
    try:
        run.finish()
    except Exception as error:
        if required:
            raise RuntimeError("required W&B finalization failed") from error
        logging.warning("Weights & Biases finalization failed: %s", error)


@gin.configurable
def get_weighted_loss(
    main_loss: torch.Tensor,
    aux_losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    weighted_loss = main_loss
    for key, weight in weights.items():
        cur_weighted_loss = aux_losses[key] * weight
        weighted_loss = weighted_loss + cur_weighted_loss
    return weighted_loss


@gin.configurable
def train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    dataset_name: str = "ml-20m",
    max_sequence_length: int = 200,
    positional_sampling_ratio: float = 1.0,
    local_batch_size: int = 128,
    eval_batch_size: int = 128,
    eval_user_max_batch_size: Optional[int] = None,
    main_module: str = "SASRec",
    main_module_bf16: bool = False,
    dropout_rate: float = 0.2,
    user_embedding_norm: str = "l2_norm",
    sampling_strategy: str = "in-batch",
    loss_module: str = "SampledSoftmaxLoss",
    loss_weights: Optional[Dict[str, float]] = {},
    num_negatives: int = 1,
    loss_activation_checkpoint: bool = False,
    item_l2_norm: bool = False,
    temperature: float = 0.05,
    num_epochs: int = 101,
    learning_rate: float = 1e-3,
    num_warmup_steps: int = 0,
    weight_decay: float = 1e-3,
    top_k_method: str = "MIPSBruteForceTopK",
    eval_interval: int = 100,
    full_eval_every_n: int = 1,
    save_ckpt_every_n: int = 1000,
    partial_eval_num_iters: int = 32,
    embedding_module_type: str = "local",
    item_embedding_dim: int = 240,
    interaction_module_type: str = "",
    gr_output_length: int = 10,
    l2_norm_eps: float = 1e-6,
    enable_tf32: bool = False,
    random_seed: int = 42,
    experiment_name: Optional[str] = None,
    max_train_batches_per_epoch: Optional[int] = None,
    max_eval_batches_per_epoch: Optional[int] = None,
    save_final_checkpoint: bool = True,
    wandb_enabled: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    wandb_mode: Optional[str] = None,
) -> None:
    wandb_required = _wandb_requirement(
        wandb_enabled=wandb_enabled,
        wandb_mode=wandb_mode,
    )
    # Seed before dataset, sampler, and model construction so paired runs share
    # initialization and data order. Kernel determinism remains backend-specific.
    _seed_everything(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    logging.info(f"cuda.matmul.allow_tf32: {enable_tf32}")
    logging.info(f"cudnn.allow_tf32: {enable_tf32}")
    logging.info(f"Training model on rank {rank}.")
    if world_size > 1:
        setup(rank, world_size, master_port)

    dataset = get_reco_dataset(
        dataset_name=dataset_name,
        max_sequence_length=max_sequence_length,
        chronological=True,
        positional_sampling_ratio=positional_sampling_ratio,
    )

    train_data_sampler, train_data_loader = create_data_loader(
        dataset.train_dataset,
        batch_size=local_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=world_size > 1,
        seed=random_seed,
    )
    eval_data_sampler, eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,  # needed for partial eval
        drop_last=world_size > 1,
        seed=random_seed,
    )

    model_debug_str = main_module
    if embedding_module_type == "local":
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            num_items=dataset.max_item_id,
            item_embedding_dim=item_embedding_dim,
        )
    else:
        raise ValueError(f"Unknown embedding_module_type {embedding_module_type}")
    model_debug_str += f"-{embedding_module.debug_str()}"

    interaction_module, interaction_module_debug_str = get_similarity_function(
        module_type=interaction_module_type,
        query_embedding_dim=item_embedding_dim,
        item_embedding_dim=item_embedding_dim,
    )

    assert (
        user_embedding_norm == "l2_norm" or user_embedding_norm == "layer_norm"
    ), f"Not implemented for {user_embedding_norm}"
    output_postproc_module = (
        L2NormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
        if user_embedding_norm == "l2_norm"
        else LayerNormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
    )
    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=dataset.max_sequence_length + gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        dropout_rate=dropout_rate,
    )

    model = get_sequential_encoder(
        module_type=main_module,
        max_sequence_length=dataset.max_sequence_length,
        max_output_length=gr_output_length + 1,
        embedding_module=embedding_module,
        interaction_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        verbose=True,
    )
    model_debug_str = model.debug_str()
    parameter_count, trainable_parameter_count, parameter_inventory_sha256 = (
        _parameter_counts(model)
    )
    attention_mode = _attention_mode(main_module)
    source_provenance = _source_provenance()
    slurm_provenance = _slurm_provenance(
        attention_mode=attention_mode,
        random_seed=random_seed,
        dataset_name=dataset_name,
    )
    resolved_gin_config = gin.operative_config_str()
    resolved_gin_config_sha256, experiment_config_sha256 = _config_identities(
        resolved_gin_config,
        attention_mode=attention_mode,
        random_seed=random_seed,
    )
    expected_experiment_config_sha256 = os.environ.get(
        "GR_EXPECTED_EXPERIMENT_CONFIG_SHA256"
    )
    if expected_experiment_config_sha256 is not None:
        if re.fullmatch(r"[0-9a-f]{64}", expected_experiment_config_sha256) is None:
            raise ValueError(
                "GR_EXPECTED_EXPERIMENT_CONFIG_SHA256 must be a lowercase SHA-256"
            )
        if experiment_config_sha256 != expected_experiment_config_sha256:
            raise ValueError(
                "operative Gin config does not match the externally pinned "
                "experiment identity"
            )

    config_identity_only = os.environ.get("GR_CONFIG_IDENTITY_ONLY", "0")
    if config_identity_only not in ("0", "1"):
        raise ValueError("GR_CONFIG_IDENTITY_ONLY must be 0 or 1")
    if config_identity_only == "1":
        output = os.environ.get("GR_CONFIG_IDENTITY_OUTPUT")
        if not output:
            raise ValueError(
                "GR_CONFIG_IDENTITY_OUTPUT is required in config-identity-only mode"
            )
        if rank == 0:
            output_path = Path(output).expanduser().absolute()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "attention_mode": attention_mode,
                        "random_seed": random_seed,
                        "resolved_gin_config_sha256": resolved_gin_config_sha256,
                        "experiment_config_sha256": experiment_config_sha256,
                        **source_provenance,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if world_size > 1:
            cleanup()
        return
    logging.info(
        "Model inventory: mode=%s parameters=%d trainable=%d sha256=%s",
        attention_mode,
        parameter_count,
        trainable_parameter_count,
        parameter_inventory_sha256,
    )

    # loss
    loss_debug_str = loss_module
    if loss_module == "BCELoss":
        loss_debug_str = loss_debug_str[:-4]
        assert temperature == 1.0
        ar_loss = BCELoss(temperature=temperature, model=model)
    elif loss_module == "SampledSoftmaxLoss":
        loss_debug_str = "ssl"
        if temperature != 1.0:
            loss_debug_str += f"-t{temperature}"
        ar_loss = SampledSoftmaxLoss(
            num_to_sample=num_negatives,
            softmax_temperature=temperature,
            model=model,
            activation_checkpoint=loss_activation_checkpoint,
        )
        loss_debug_str += (
            f"-n{num_negatives}{'-ac' if loss_activation_checkpoint else ''}"
        )
    else:
        raise ValueError(f"Unrecognized loss module {loss_module}.")

    # sampling
    if sampling_strategy == "in-batch":
        negatives_sampler = InBatchNegativesSampler(
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
            dedup_embeddings=True,
        )
        sampling_debug_str = (
            f"in-batch{f'-l2-eps{l2_norm_eps}' if item_l2_norm else ''}-dedup"
        )
    elif sampling_strategy == "local":
        negatives_sampler = LocalNegativesSampler(
            num_items=dataset.max_item_id,
            item_emb=model._embedding_module._item_emb,
            all_item_ids=dataset.all_item_ids,
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
        )
    else:
        raise ValueError(f"Unrecognized sampling strategy {sampling_strategy}.")
    sampling_debug_str = negatives_sampler.debug_str()

    # Creates model and moves it to GPU with id rank
    device = rank
    if main_module_bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    ar_loss = ar_loss.to(device)
    negatives_sampler = negatives_sampler.to(device)
    model_module = model
    if world_size > 1:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    # TODO: wrap in create_optimizer.
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=weight_decay,
    )

    date_str = date.today().strftime("%Y-%m-%d")
    model_subfolder = f"{dataset_name}-l{max_sequence_length}"
    model_desc = (
        f"{model_subfolder}"
        + f"/{model_debug_str}_{interaction_module_debug_str}_{sampling_debug_str}_{loss_debug_str}"
        + f"{f'-ddp{world_size}' if world_size > 1 else ''}-b{local_batch_size}-lr{learning_rate}-wu{num_warmup_steps}-wd{weight_decay}{'' if enable_tf32 else '-notf32'}-{date_str}"
    )
    if full_eval_every_n > 1:
        model_desc += f"-fe{full_eval_every_n}"
    if positional_sampling_ratio is not None and positional_sampling_ratio < 1:
        model_desc += f"-d{positional_sampling_ratio}"
    experiment_name = os.environ.get("GR_EXPERIMENT_NAME") or experiment_name
    wandb_run_name = os.environ.get("WANDB_NAME") or wandb_run_name
    wandb_group = os.environ.get("WANDB_RUN_GROUP") or wandb_group
    if wandb_tags is None and os.environ.get("WANDB_TAGS"):
        wandb_tags = [
            tag.strip() for tag in os.environ["WANDB_TAGS"].split(",") if tag.strip()
        ]
    if experiment_name:
        if Path(experiment_name).name != experiment_name:
            raise ValueError("experiment_name must not contain path separators")
        model_desc += f"-{experiment_name}"

    exps_root = Path(os.environ.get("GR_EXPS_ROOT", "./exps")).expanduser()
    ckpts_root = Path(os.environ.get("GR_CKPTS_ROOT", "./ckpts")).expanduser()
    log_dir = exps_root / model_desc
    ckpt_prefix = ckpts_root / model_desc
    (exps_root / model_subfolder).mkdir(parents=True, exist_ok=True)
    (ckpts_root / model_subfolder).mkdir(parents=True, exist_ok=True)
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "operative_config.gin").write_text(
            resolved_gin_config, encoding="utf-8"
        )
        (log_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "dataset_name": dataset_name,
                    "attention_mode": attention_mode,
                    "random_seed": random_seed,
                    "resolved_gin_config_sha256": resolved_gin_config_sha256,
                    "experiment_config_sha256": experiment_config_sha256,
                    "parameter_count": parameter_count,
                    "trainable_parameter_count": trainable_parameter_count,
                    "parameter_inventory_sha256": parameter_inventory_sha256,
                    **source_provenance,
                    **slurm_provenance,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        writer = SummaryWriter(log_dir=str(log_dir))
        logging.info(f"Rank {rank}: writing logs to {log_dir}")
        _wandb_run = None
        if wandb_enabled:
            wandb_config = {
                "dataset_name": dataset_name,
                "attention_mode": attention_mode,
                "random_seed": random_seed,
                "world_size": world_size,
                "max_sequence_length": max_sequence_length,
                "local_batch_size": local_batch_size,
                "eval_batch_size": eval_batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "parameter_inventory_sha256": parameter_inventory_sha256,
                "resolved_gin_config": resolved_gin_config,
                "resolved_gin_config_sha256": resolved_gin_config_sha256,
                "experiment_config_sha256": experiment_config_sha256,
                "data_root": os.environ.get("GR_DATA_ROOT", "./tmp"),
                "exps_root": str(exps_root),
                "ckpts_root": str(ckpts_root),
                **source_provenance,
                **slurm_provenance,
            }
            _wandb_run = _wandb_initialize(
                required=wandb_required,
                init_kwargs={
                    "project": wandb_project or os.environ.get("WANDB_PROJECT", "gr"),
                    "entity": wandb_entity or os.environ.get("WANDB_ENTITY"),
                    "name": wandb_run_name or experiment_name or model_desc,
                    "group": wandb_group,
                    "tags": wandb_tags,
                    "mode": wandb_mode or os.environ.get("WANDB_MODE", "online"),
                    "dir": str(log_dir),
                    "config": wandb_config,
                },
            )
    else:
        writer = None
        _wandb_run = None
        logging.info(f"Rank {rank}: disabling summary writer")

    last_training_time = time.time()
    torch.autograd.set_detect_anomaly(True)

    batch_id = 0
    epoch = 0
    for epoch in range(num_epochs):
        if train_data_sampler is not None:
            train_data_sampler.set_epoch(epoch)
        if eval_data_sampler is not None:
            eval_data_sampler.set_epoch(epoch)
        _synchronize(device)
        epoch_wall_start = time.perf_counter()
        epoch_periodic_eval_seconds = 0.0
        epoch_train_examples = 0
        model.train()
        for train_iter, row in enumerate(iter(train_data_loader)):
            if (
                max_train_batches_per_epoch is not None
                and train_iter >= max_train_batches_per_epoch
            ):
                break
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                row,
                device=device,
                max_output_length=gr_output_length + 1,
            )
            epoch_train_examples += int(seq_features.past_lengths.numel()) * world_size

            if (batch_id % eval_interval) == 0:
                _synchronize(device)
                periodic_eval_start = time.perf_counter()
                model.eval()

                eval_state = get_eval_state(
                    model=model_module,
                    all_item_ids=dataset.all_item_ids,
                    negatives_sampler=negatives_sampler,
                    top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                        top_k_method=top_k_method,
                        model=model_module,
                        item_embeddings=item_embeddings,
                        item_ids=item_ids,
                    ),
                    device=device,
                    float_dtype=torch.bfloat16 if main_module_bf16 else None,
                )
                # pyrefly: ignore [bad-specialization]
                eval_dict = eval_metrics_v2_from_tensors(
                    eval_state,
                    # pyrefly: ignore [bad-argument-count]
                    model_module,
                    seq_features,
                    # pyrefly: ignore [unexpected-keyword]
                    target_ids=target_ids,
                    # pyrefly: ignore [unexpected-keyword]
                    target_ratings=target_ratings,
                    # pyrefly: ignore [unexpected-keyword]
                    user_max_batch_size=eval_user_max_batch_size,
                    # pyrefly: ignore [unexpected-keyword]
                    dtype=torch.bfloat16 if main_module_bf16 else None,
                )
                add_to_summary_writer(
                    # pyrefly: ignore [bad-argument-type]
                    writer,
                    batch_id,
                    # pyrefly: ignore [bad-argument-type]
                    eval_dict,
                    prefix="eval",
                    world_size=world_size,
                )
                # _avg performs a collective, so every rank computes these in
                # the same order before rank 0 emits external logs.
                eval_ndcg_10 = _avg(eval_dict["ndcg@10"], world_size)
                eval_hr_10 = _avg(eval_dict["hr@10"], world_size)
                eval_hr_50 = _avg(eval_dict["hr@50"], world_size)
                eval_mrr = _avg(eval_dict["mrr"], world_size)
                logging.info(
                    f"rank {rank}:  batch-stat (eval): iter {batch_id} (epoch {epoch}): "
                    + f"NDCG@10 {eval_ndcg_10:.4f}, "
                    f"HR@10 {eval_hr_10:.4f}, "
                    f"HR@50 {eval_hr_50:.4f}, " + f"MRR {eval_mrr:.4f} "
                )
                if rank == 0 and _wandb_run is not None:
                    _wandb_log(
                        _wandb_run,
                        {
                            "eval/ndcg@10": float(eval_ndcg_10),
                            "eval/hr@10": float(eval_hr_10),
                            "eval/hr@50": float(eval_hr_50),
                            "eval/mrr": float(eval_mrr),
                            "epoch": epoch,
                        },
                        batch_id,
                        required=wandb_required,
                    )
                model.train()
                _synchronize(device)
                epoch_periodic_eval_seconds += time.perf_counter() - periodic_eval_start

            # TODO: consider separating this out?
            B, N = seq_features.past_ids.shape
            seq_features.past_ids.scatter_(
                dim=1,
                index=seq_features.past_lengths.view(-1, 1),
                src=target_ids.view(-1, 1),
            )

            opt.zero_grad()
            input_embeddings = model_module.get_item_embeddings(seq_features.past_ids)
            seq_embeddings = model(
                past_lengths=seq_features.past_lengths,
                past_ids=seq_features.past_ids,
                past_embeddings=input_embeddings,
                past_payloads=seq_features.past_payloads,
            )  # [B, X]

            supervision_ids = seq_features.past_ids

            if sampling_strategy == "in-batch":
                # get_item_embeddings currently assume 1-d tensor.
                in_batch_ids = supervision_ids.view(-1)
                negatives_sampler.process_batch(
                    ids=in_batch_ids,
                    presences=(in_batch_ids != 0),
                    embeddings=model_module.get_item_embeddings(in_batch_ids),
                )
            else:
                # pyre-fixme[16]: `InBatchNegativesSampler` has no attribute
                #  `_item_emb`.
                negatives_sampler._item_emb = model_module._embedding_module._item_emb

            ar_mask = supervision_ids[:, 1:] != 0
            loss, aux_losses = ar_loss(
                lengths=seq_features.past_lengths,  # [B],
                output_embeddings=seq_embeddings[:, :-1, :],  # [B, N-1, D]
                supervision_ids=supervision_ids[:, 1:],  # [B, N-1]
                supervision_embeddings=input_embeddings[:, 1:, :],  # [B, N - 1, D]
                supervision_weights=ar_mask.float(),
                negatives_sampler=negatives_sampler,
                **seq_features.past_payloads,
            )  # [B, N]

            main_loss = loss.detach().clone()
            loss = get_weighted_loss(loss, aux_losses, weights=loss_weights or {})

            if rank == 0:
                assert writer is not None
                writer.add_scalar("losses/ar_loss", loss, batch_id)
                writer.add_scalar("losses/main_loss", main_loss, batch_id)

            loss.backward()

            # Optional linear warmup.
            if batch_id < num_warmup_steps:
                lr_scalar = min(1.0, float(batch_id + 1) / num_warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = lr_scalar * learning_rate
                lr = lr_scalar * learning_rate
            else:
                lr = learning_rate

            if (batch_id % eval_interval) == 0:
                logging.info(
                    f" rank: {rank}, batch-stat (train): step {batch_id} "
                    f"(epoch {epoch} in {time.time() - last_training_time:.2f}s): {loss:.6f}"
                )
                last_training_time = time.time()
                if rank == 0:
                    assert writer is not None
                    writer.add_scalar("loss/train", loss, batch_id)
                    writer.add_scalar("lr", lr, batch_id)
                    if _wandb_run is not None:
                        _wandb_log(
                            _wandb_run,
                            {
                                "loss/train": float(loss.detach()),
                                "loss/main": float(main_loss.detach()),
                                "learning_rate": lr,
                                "epoch": epoch,
                            },
                            batch_id,
                            required=wandb_required,
                        )

            opt.step()

            batch_id += 1

        _synchronize(device)
        epoch_train_seconds = max(
            time.perf_counter() - epoch_wall_start - epoch_periodic_eval_seconds,
            0.0,
        )
        epoch_examples_per_second = epoch_train_examples / max(
            epoch_train_seconds, 1e-9
        )
        if rank == 0:
            assert writer is not None
            writer.add_scalar(
                "perf/train_examples_per_second",
                epoch_examples_per_second,
                epoch,
            )

        def is_full_eval(epoch: int) -> bool:
            return (epoch % full_eval_every_n) == 0

        # eval per epoch
        eval_dict_all = None
        _synchronize(device)
        eval_start_time = time.perf_counter()
        model.eval()
        eval_state = get_eval_state(
            model=model_module,
            all_item_ids=dataset.all_item_ids,
            negatives_sampler=negatives_sampler,
            top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                top_k_method=top_k_method,
                model=model_module,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
            ),
            device=device,
            float_dtype=torch.bfloat16 if main_module_bf16 else None,
        )
        for eval_iter, row in enumerate(iter(eval_data_loader)):
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                row, device=device, max_output_length=gr_output_length + 1
            )
            # pyrefly: ignore [bad-specialization]
            eval_dict = eval_metrics_v2_from_tensors(
                eval_state,
                # pyrefly: ignore [bad-argument-count]
                model_module,
                seq_features,
                # pyrefly: ignore [unexpected-keyword]
                target_ids=target_ids,
                # pyrefly: ignore [unexpected-keyword]
                target_ratings=target_ratings,
                # pyrefly: ignore [unexpected-keyword]
                user_max_batch_size=eval_user_max_batch_size,
                # pyrefly: ignore [unexpected-keyword]
                dtype=torch.bfloat16 if main_module_bf16 else None,
            )

            if eval_dict_all is None:
                eval_dict_all = {}
                # pyrefly: ignore [missing-attribute]
                for k, v in eval_dict.items():
                    eval_dict_all[k] = []

            # pyrefly: ignore [missing-attribute]
            for k, v in eval_dict.items():
                eval_dict_all[k] = eval_dict_all[k] + [v]
            del eval_dict

            if (
                max_eval_batches_per_epoch is not None
                and eval_iter + 1 >= max_eval_batches_per_epoch
            ):
                logging.info(
                    "Truncating epoch %d eval to %d iters for a smoke run.",
                    epoch,
                    eval_iter + 1,
                )
                break
            if (eval_iter + 1 >= partial_eval_num_iters) and not is_full_eval(epoch):
                logging.info(
                    f"Truncating epoch {epoch} eval to {eval_iter + 1} iters to save cost.."
                )
                break

        assert eval_dict_all is not None
        for k, v in eval_dict_all.items():
            # pyrefly: ignore [unsupported-operation]
            eval_dict_all[k] = torch.cat(v, dim=-1)

        # pyrefly: ignore [bad-argument-type]
        ndcg_10 = _avg(eval_dict_all["ndcg@10"], world_size=world_size)
        # pyrefly: ignore [bad-argument-type]
        ndcg_50 = _avg(eval_dict_all["ndcg@50"], world_size=world_size)
        # pyrefly: ignore [bad-argument-type]
        hr_10 = _avg(eval_dict_all["hr@10"], world_size=world_size)
        # pyrefly: ignore [bad-argument-type]
        hr_50 = _avg(eval_dict_all["hr@50"], world_size=world_size)
        # pyrefly: ignore [bad-argument-type]
        mrr = _avg(eval_dict_all["mrr"], world_size=world_size)

        add_to_summary_writer(
            writer,
            batch_id=epoch,
            # pyrefly: ignore [bad-argument-type]
            metrics=eval_dict_all,
            prefix="eval_epoch",
            world_size=world_size,
        )
        if full_eval_every_n > 1 and is_full_eval(epoch):
            add_to_summary_writer(
                writer,
                batch_id=epoch,
                # pyrefly: ignore [bad-argument-type]
                metrics=eval_dict_all,
                prefix="eval_epoch_full",
                world_size=world_size,
            )
        if rank == 0 and epoch > 0 and (epoch % save_ckpt_every_n) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "dataset_name": dataset_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "attention_mode": attention_mode,
                    "random_seed": random_seed,
                    "parameter_count": parameter_count,
                    "parameter_inventory_sha256": parameter_inventory_sha256,
                    "resolved_gin_config": resolved_gin_config,
                    "resolved_gin_config_sha256": resolved_gin_config_sha256,
                    "experiment_config_sha256": experiment_config_sha256,
                    **source_provenance,
                    **slurm_provenance,
                },
                f"{ckpt_prefix}_ep{epoch}",
            )

        _synchronize(device)
        eval_seconds = time.perf_counter() - eval_start_time
        logging.info(
            f"rank {rank}: eval @ epoch {epoch} in {eval_seconds:.2f}s: "
            f"NDCG@10 {ndcg_10:.4f}, NDCG@50 {ndcg_50:.4f}, HR@10 {hr_10:.4f}, HR@50 {hr_50:.4f}, MRR {mrr:.4f}"
        )
        if rank == 0 and _wandb_run is not None:
            _wandb_log(
                _wandb_run,
                {
                    "eval_epoch/ndcg@10": float(ndcg_10),
                    "eval_epoch/ndcg@50": float(ndcg_50),
                    "eval_epoch/hr@10": float(hr_10),
                    "eval_epoch/hr@50": float(hr_50),
                    "eval_epoch/mrr": float(mrr),
                    "perf/train_examples_per_second": epoch_examples_per_second,
                    "perf/train_seconds": epoch_train_seconds,
                    "perf/eval_seconds": eval_seconds,
                    "epoch": epoch,
                },
                batch_id,
                required=wandb_required,
            )
        last_training_time = time.time()

    if rank == 0:
        if writer is not None:
            writer.flush()
            writer.close()

        if save_final_checkpoint:
            torch.save(
                {
                    "epoch": epoch,
                    "dataset_name": dataset_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "attention_mode": attention_mode,
                    "random_seed": random_seed,
                    "parameter_count": parameter_count,
                    "parameter_inventory_sha256": parameter_inventory_sha256,
                    "resolved_gin_config": resolved_gin_config,
                    "resolved_gin_config_sha256": resolved_gin_config_sha256,
                    "experiment_config_sha256": experiment_config_sha256,
                    **source_provenance,
                    **slurm_provenance,
                },
                f"{ckpt_prefix}_ep{epoch}",
            )
        if _wandb_run is not None:
            _wandb_finish(_wandb_run, required=wandb_required)

    if world_size > 1:
        cleanup()
