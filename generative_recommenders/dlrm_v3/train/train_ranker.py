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

# pyre-strict
import argparse
import logging

logging.basicConfig(level=logging.INFO)
import os
import random
import sys
import traceback
from datetime import datetime
from typing import Optional

import gin
import numpy as np
import torch
from generative_recommenders.dlrm_v3.checkpoint import load_dmp_checkpoint
from generative_recommenders.dlrm_v3.train.utils import (
    cleanup,
    eval_loop,
    make_model,
    make_optimizer_and_shard,
    make_train_test_dataloaders,
    setup,
    streaming_train_eval_loop,
    train_eval_loop,
    train_loop,
)
from generative_recommenders.dlrm_v3.utils import MetricsLogger
from torch import multiprocessing as mp
from torchrec.test_utils import get_free_port

logger: logging.Logger = logging.getLogger(__name__)


SUPPORTED_CONFIGS = {
    "debug": "debug.gin",
    "kuairand-1k": "kuairand_1k.gin",
    "kuairand-27k": "kuairand_27k.gin",
    "movielens-1m": "movielens_1m.gin",
    "movielens-20m": "movielens_20m.gin",
    "movielens-13b": "movielens_13b.gin",
    "movielens-18b": "movielens_18b.gin",
    "streaming-400m": "streaming_400m.gin",
    "streaming-200b": "streaming_200b.gin",
    "streaming-100b": "streaming_100b.gin",
}


def _set_global_seed(seed: int) -> None:
    """Seed all RNGs for reproducible model init / dropout / data order.

    The same seed is set on every rank: DDP broadcasts rank-0 params at wrap
    time, and ChunkDistributedSampler adds its own per-rank offset, so identical
    global seeding across ranks is correct.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _bind_env_paths(run_name: str) -> None:
    """Bind data/checkpoint/exp roots from .env onto gin-configurable targets.

    Gin configs hold only relative parts (per-dataset subpaths live in
    get_dataset; run_name is the per-run leaf). The absolute roots come from
    the environment (GR_DATA_ROOT / GR_CKPTS_ROOT / GR_EXPS_ROOT) so nothing
    machine-specific is committed to gin.
    """
    data_root = os.environ.get("GR_DATA_ROOT")
    if data_root:
        # gin may hold a relative dataset subfolder (e.g. "KuaiRand-1K"); join it
        # onto the absolute GR_DATA_ROOT. If the config leaves new_path_prefix
        # unset, fall back to the root alone (ml-1m/ml-20m bake their per-dataset
        # subpath into get_dataset instead).
        try:
            rel_prefix = gin.query_parameter(
                "make_train_test_dataloaders.new_path_prefix"
            )
        except ValueError:
            rel_prefix = None
        full_prefix = (
            os.path.join(data_root, rel_prefix) if rel_prefix else data_root
        )
        gin.bind_parameter(
            "make_train_test_dataloaders.new_path_prefix", full_prefix
        )

    exps_root = os.environ.get("GR_EXPS_ROOT")
    if exps_root:
        gin.bind_parameter(
            "MetricsLogger.tensorboard_log_path",
            os.path.join(exps_root, run_name),
        )

    ckpts_root = os.environ.get("GR_CKPTS_ROOT")
    if ckpts_root:
        gin.bind_parameter(
            "save_dmp_checkpoint.path", os.path.join(ckpts_root, run_name)
        )


def _main_func(
    rank: int,
    world_size: int,
    master_port: int,
    gin_file: str,
    mode: str,
    seed: int,
    num_epochs: int,
    run_name: str,
    max_seq_len: int,
    max_attn_len: int,
    stu_module: Optional[str],
    neutreno_lambda: Optional[float],
    neutreno_after_norm: Optional[bool],
    attnres_block_size: Optional[int],
    mhc_num_streams: Optional[int],
    mhc_num_iters: Optional[int],
    mhc_tau: Optional[float],
) -> None:
    device = torch.device(f"cuda:{rank}")
    logger.info(f"rank: {rank}, world_size: {world_size}, device: {device}")
    setup(
        rank=rank,
        world_size=world_size,
        master_port=master_port,
        device=device,
    )
    # parse all arguments
    gin.parse_config_file(gin_file)

    # Env-driven roots + per-run overrides take precedence over gin defaults.
    _set_global_seed(seed)
    _bind_env_paths(run_name)
    gin.bind_parameter("make_train_test_dataloaders.seed", seed)
    if os.environ.get("GR_WANDB_ENABLED", "0") == "1":
        gin.bind_parameter("MetricsLogger.wandb_enabled", True)
        gin.bind_parameter("MetricsLogger.wandb_run_name", run_name)
    if num_epochs > 0:
        gin.bind_parameter("train_loop.num_epochs", num_epochs)
        gin.bind_parameter("train_eval_loop.num_epochs", num_epochs)
    if max_seq_len > 0:
        gin.bind_parameter("make_model.max_seq_len", max_seq_len)
    if max_attn_len > 0:
        gin.bind_parameter("make_model.max_attn_len", max_attn_len)
    if stu_module is not None:
        gin.bind_parameter("make_model.stu_module_type", stu_module)
    if neutreno_lambda is not None:
        gin.bind_parameter("make_model.neutreno_lambda", neutreno_lambda)
    if neutreno_after_norm is not None:
        gin.bind_parameter("make_model.neutreno_after_norm", neutreno_after_norm)
    if attnres_block_size is not None:
        gin.bind_parameter("make_model.attnres_block_size", attnres_block_size)
    if mhc_num_streams is not None:
        gin.bind_parameter("make_model.mhc_num_streams", mhc_num_streams)
    if mhc_num_iters is not None:
        gin.bind_parameter("make_model.mhc_num_iters", mhc_num_iters)
    if mhc_tau is not None:
        gin.bind_parameter("make_model.mhc_tau", mhc_tau)

    model, model_configs, embedding_table_configs = make_model()
    model, optimizer = make_optimizer_and_shard(
        model=model, device=device, world_size=world_size
    )
    train_dataloader, test_dataloader = make_train_test_dataloaders(
        hstu_config=model_configs,
        embedding_table_configs=embedding_table_configs,
    )
    metrics = MetricsLogger(
        multitask_configs=model_configs.multitask_configs,
        batch_size=train_dataloader.batch_size,
        window_size=2500,
        device=device,
        rank=rank,
    )
    load_dmp_checkpoint(
        model=model, optimizer=optimizer, metric_logger=metrics, device=device
    )

    # train loop
    try:
        if mode == "train":
            train_loop(
                rank=rank,
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "eval":
            # reinit metrics logger for eval
            metrics = MetricsLogger(
                multitask_configs=model_configs.multitask_configs,
                batch_size=train_dataloader.batch_size,
                window_size=1000,
                device=device,
                rank=rank,
            )
            eval_loop(
                rank=rank,
                model=model,
                dataloader=test_dataloader,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "train-eval":
            train_eval_loop(
                rank=rank,
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=test_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "streaming-train-eval":
            streaming_train_eval_loop(
                rank=rank,
                model=model,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
                hstu_config=model_configs,
                embedding_table_configs=embedding_table_configs,
            )
    except Exception as e:
        logger.info(traceback.format_exc())
        cleanup()
        raise Exception(e)


def get_args():  # pyre-ignore [3]
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="debug", choices=SUPPORTED_CONFIGS.keys(), help="dataset"
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "train-eval", "streaming-train-eval"],
        help="mode",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="global + sampler seed for reproducibility"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=0,
        help="override num_epochs for train/train-eval loops; 0 = use gin value",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="leaf name for exp/checkpoint dirs and wandb run; auto-generated if unset",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="override model max_seq_len (HSTU max sequence length); 0 = use config default",
    )
    parser.add_argument(
        "--max-attn-len",
        type=int,
        default=0,
        help="sliding-window attention span (each token attends to prev N tokens); 0 = full causal attention",
    )
    parser.add_argument(
        "--stu-module",
        default=None,
        choices=["STU", "STU_PYTORCH", "STU_DELTANET", "STU_PDELTANET", "NeuTRENO", "AttnRes", "mHC"],
        help="STU variant: STU (vanilla fused), STU_PYTORCH (vanilla eager-PyTorch), "
        "STU_DELTANET (windowed attn + gated-delta long memory, overlapping), "
        "STU_PDELTANET (non-overlapping: window=recent W, delta=history older than W, "
        "candidate-conditioned gate), "
        "NeuTRENO, AttnRes, or mHC; None = use config default",
    )
    parser.add_argument(
        "--neutreno-lambda",
        type=float,
        default=None,
        help="NeuTRENO anti-oversmoothing strength (only used when --stu-module=NeuTRENO)",
    )
    parser.add_argument(
        "--neutreno-after-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="inject the NeuTRENO term AFTER the output norm (default: before norm); "
        "only used when --stu-module=NeuTRENO",
    )
    parser.add_argument(
        "--attnres-block-size",
        type=int,
        default=None,
        help="AttnRes layers per block; 1 = Full AttnRes (only used when --stu-module=AttnRes)",
    )
    parser.add_argument(
        "--mhc-num-streams",
        type=int,
        default=None,
        help="mHC residual stream count; 1 ~ vanilla residual (only used when --stu-module=mHC)",
    )
    parser.add_argument(
        "--mhc-num-iters",
        type=int,
        default=None,
        help="mHC Sinkhorn iterations for H_res projection (only used when --stu-module=mHC)",
    )
    parser.add_argument(
        "--mhc-tau",
        type=float,
        default=None,
        help="mHC Sinkhorn temperature; smaller = sharper H_res (only used when --stu-module=mHC)",
    )
    args, unknown_args = parser.parse_known_args()
    logger.warning(f"unknown_args: {unknown_args}")
    return args


def main() -> None:
    args = get_args()
    logger.info(args)
    assert args.dataset in SUPPORTED_CONFIGS, f"Unsupported dataset: {args.dataset}"
    assert args.mode in [
        "train",
        "eval",
        "train-eval",
        "streaming-train-eval",
    ], f"Unsupported mode: {args.mode}"
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    MASTER_PORT = str(get_free_port())
    gin_path = f"{os.path.dirname(__file__)}/gin/{SUPPORTED_CONFIGS[args.dataset]}"

    # Generate run_name once in the parent so all ranks share identical paths.
    run_name = args.run_name or (
        f"{args.dataset}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    logger.info(f"run_name: {run_name}")

    mp.start_processes(
        _main_func,
        args=(
            WORLD_SIZE,
            MASTER_PORT,
            gin_path,
            args.mode,
            args.seed,
            args.num_epochs,
            run_name,
            args.max_seq_len,
            args.max_attn_len,
            args.stu_module,
            args.neutreno_lambda,
            args.neutreno_after_norm,
            args.attnres_block_size,
            args.mhc_num_streams,
            args.mhc_num_iters,
            args.mhc_tau,
        ),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )


if __name__ == "__main__":
    main()
