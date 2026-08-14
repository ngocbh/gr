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

"""
Usage: mkdir -p tmp/ && python3 preprocess_public_data.py [--dataset DATASET]
"""

import argparse
from typing import List, Optional

from generative_recommenders.research.data.preprocessor import get_common_preprocessors


LEGACY_DATASETS = ("ml-1m", "ml-20m", "amzn-books")
SELECTABLE_DATASETS = (*LEGACY_DATASETS, "kuairand-1k")


def _selected_datasets(selections: Optional[List[str]]) -> List[str]:
    if not selections:
        return list(LEGACY_DATASETS)
    if "all" in selections:
        if len(selections) != 1:
            raise ValueError("--dataset all cannot be combined with another dataset")
        return list(SELECTABLE_DATASETS)
    return list(dict.fromkeys(selections))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[*SELECTABLE_DATASETS, "all"],
        help="dataset to preprocess; repeat to select multiple datasets",
    )
    args = parser.parse_args(argv)
    preprocessors = get_common_preprocessors()
    for dataset in _selected_datasets(args.dataset):
        preprocessors[dataset].preprocess_rating()


if __name__ == "__main__":
    main()
