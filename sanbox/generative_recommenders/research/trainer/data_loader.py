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

import os
import random
from typing import Optional, Tuple

import gin
import numpy as np
import torch


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


@gin.configurable
def create_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    world_size: int,
    rank: int,
    shuffle: bool,
    prefetch_factor: int = 128,
    num_workers: Optional[int] = os.cpu_count(),
    drop_last: bool = False,
    seed: int = 0,
) -> Tuple[
    Optional[torch.utils.data.distributed.DistributedSampler[torch.utils.data.Dataset]],
    torch.utils.data.DataLoader,
]:
    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=drop_last,
        )
    else:
        sampler = None
    workers = num_workers or 0
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    data_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "sampler": sampler,
        "generator": generator,
        "worker_init_fn": _seed_worker,
    }
    if workers > 0:
        data_loader_kwargs["prefetch_factor"] = prefetch_factor
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_kwargs)
    return sampler, data_loader
