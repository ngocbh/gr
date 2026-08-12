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

import random
import unittest

import numpy as np
import torch
from generative_recommenders.research.trainer.data_loader import create_data_loader
from generative_recommenders.research.trainer.train import _seed_everything


class ReproducibilityTest(unittest.TestCase):
    def test_seed_everything_resets_python_numpy_and_torch(self) -> None:
        _seed_everything(17)
        first = (random.random(), float(np.random.rand()), float(torch.rand(())))

        _seed_everything(17)
        second = (random.random(), float(np.random.rand()), float(torch.rand(())))

        self.assertEqual(first, second)

    def test_data_sampler_uses_requested_seed(self) -> None:
        dataset = torch.utils.data.TensorDataset(torch.arange(32))

        sampler_a, _ = create_data_loader(
            dataset=dataset,
            batch_size=4,
            world_size=1,
            rank=0,
            shuffle=True,
            seed=23,
            num_workers=0,
        )
        sampler_b, _ = create_data_loader(
            dataset=dataset,
            batch_size=4,
            world_size=1,
            rank=0,
            shuffle=True,
            seed=23,
            num_workers=0,
        )
        sampler_c, _ = create_data_loader(
            dataset=dataset,
            batch_size=4,
            world_size=1,
            rank=0,
            shuffle=True,
            seed=29,
            num_workers=0,
        )

        assert sampler_a is not None
        assert sampler_b is not None
        assert sampler_c is not None
        self.assertEqual(list(sampler_a), list(sampler_b))
        self.assertNotEqual(list(sampler_a), list(sampler_c))


if __name__ == "__main__":
    unittest.main()
