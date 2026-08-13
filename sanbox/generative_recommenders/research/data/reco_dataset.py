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

from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from generative_recommenders.research.data.dataset import DatasetV2, MultiFileDatasetV2
from generative_recommenders.research.data.item_features import ItemFeatures
from generative_recommenders.research.data.preprocessor import get_common_preprocessors


@dataclass
class RecoDataset:
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset


def get_reco_dataset(
    dataset_name: str,
    max_sequence_length: int,
    chronological: bool,
    positional_sampling_ratio: float = 1.0,
) -> RecoDataset:
    if dataset_name == "ml-1m":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
            sample_ratio=positional_sampling_ratio,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
            sample_ratio=1.0,  # do not sample
        )
    elif dataset_name == "ml-20m":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
        )
    elif dataset_name == "ml-3b":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = MultiFileDatasetV2(
            file_prefix="tmp/ml-3b/16x32",
            num_files=16,
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
        )
        eval_dataset = MultiFileDatasetV2(
            file_prefix="tmp/ml-3b/16x32",
            num_files=16,
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
        )
    elif dataset_name == "amzn-books":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
    elif dataset_name == "kuairand-1k":
        dp = get_common_preprocessors()[dataset_name]
        metadata = dp.load_metadata()
        if max_sequence_length != metadata["max_sequence_length"]:
            raise ValueError(
                "kuairand-1k requires max_sequence_length="
                f"{metadata['max_sequence_length']}, got {max_sequence_length}"
            )
        num_unique_items = int(metadata["statistics"]["num_items"])
        train_dataset = DatasetV2(
            ratings_file=dp.train_format_csv(),
            padding_length=max_sequence_length + 1,
            ignore_last_n=0,
            shift_id_by=1,  # dense [0..n-1] -> model ids [1..n]
            chronological=chronological,
            sample_ratio=positional_sampling_ratio,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.eval_format_csv(),
            padding_length=max_sequence_length + 1,
            ignore_last_n=0,
            shift_id_by=1,
            chronological=chronological,
            sample_ratio=1.0,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if dataset_name == "ml-1m" or dataset_name == "ml-20m":
        items = pd.read_csv(dp.processed_item_csv(), delimiter=",")
        max_jagged_dimension = 16
        expected_max_item_id = dp.expected_max_item_id()
        assert expected_max_item_id is not None
        item_features: ItemFeatures = ItemFeatures(
            max_ind_range=[63, 16383, 511],
            num_items=expected_max_item_id + 1,
            max_jagged_dimension=max_jagged_dimension,
            lengths=[
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
            ],
            values=[
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
            ],
        )
        all_item_ids = []
        for df_index, row in items.iterrows():
            # print(f"index {df_index}: {row}")
            movie_id = int(row["movie_id"])
            genres = row["genres"].split("|")
            titles = row["cleaned_title"].split(" ")
            # print(f"{index}: genres{genres}, title{titles}")
            genres_vector = [hash(x) % item_features.max_ind_range[0] for x in genres]
            titles_vector = [hash(x) % item_features.max_ind_range[1] for x in titles]
            years_vector = [hash(row["year"]) % item_features.max_ind_range[2]]
            item_features.lengths[0][movie_id] = min(
                len(genres_vector), max_jagged_dimension
            )
            item_features.lengths[1][movie_id] = min(
                len(titles_vector), max_jagged_dimension
            )
            item_features.lengths[2][movie_id] = min(
                len(years_vector), max_jagged_dimension
            )
            for f, f_values in enumerate([genres_vector, titles_vector, years_vector]):
                for j in range(min(len(f_values), max_jagged_dimension)):
                    item_features.values[f][movie_id][j] = f_values[j]
            all_item_ids.append(movie_id)
        max_item_id = dp.expected_max_item_id()
        num_unique_items = dp.expected_num_unique_items()
        assert num_unique_items is not None
        for x in all_item_ids:
            assert x > 0, "x in all_item_ids should be positive"
    elif dataset_name == "kuairand-1k":
        # KuaiRand artifacts use dense zero-based IDs and DatasetV2 shifts them
        # by one, reserving model ID zero for padding.
        item_features = None
        max_item_id = num_unique_items
        all_item_ids = list(range(1, num_unique_items + 1))
    else:
        # expected_max_item_id and item_features are not set for Amazon/synthetic data.
        # pyrefly: ignore [bad-assignment]
        item_features = None
        max_item_id = dp.expected_num_unique_items()
        num_unique_items = dp.expected_num_unique_items()
        assert num_unique_items is not None
        all_item_ids = [x + 1 for x in range(max_item_id)]  # pyre-ignore [6]

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=num_unique_items,
        max_item_id=max_item_id,  # pyre-ignore [6]
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
