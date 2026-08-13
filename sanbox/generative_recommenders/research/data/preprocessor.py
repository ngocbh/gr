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

import abc
import hashlib
import json
import logging
import os
import platform
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


KUAI_RAND_MAX_SEQUENCE_LENGTH = 2_048
KUAI_RAND_MIN_CORE_DEGREE = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_training_window_bounds(
    sequence_length: int,
    max_sequence_length: int = KUAI_RAND_MAX_SEQUENCE_LENGTH,
) -> Iterator[Tuple[int, int]]:
    """Yield rows whose one-event overlap covers every prefix transition once."""
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")
    if sequence_length < 2:
        return

    max_events_per_row = max_sequence_length + 1
    start = 0
    while start < sequence_length - 1:
        end = min(start + max_events_per_row, sequence_length)
        yield start, end
        if end == sequence_length:
            break
        start = end - 1


def _iterative_k_core(
    interactions: pd.DataFrame,
    min_degree: int = KUAI_RAND_MIN_CORE_DEGREE,
) -> Tuple[pd.DataFrame, int]:
    """Apply simultaneous user/item event-count filtering until convergence."""
    if min_degree < 1:
        raise ValueError("min_degree must be positive")
    filtered = interactions
    iterations = 0
    while not filtered.empty:
        user_counts = filtered["user_id"].value_counts(sort=False)
        item_counts = filtered["video_id"].value_counts(sort=False)
        keep = filtered["user_id"].map(user_counts).ge(min_degree) & filtered[
            "video_id"
        ].map(item_counts).ge(min_degree)
        if bool(keep.all()):
            break
        filtered = filtered.loc[keep].copy()
        iterations += 1
    return filtered, iterations


class DataProcessor:
    """
    This preprocessor does not remap item_ids. This is intended so that we can easily join other
    side-information based on item_ids later.
    """

    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int],
        expected_max_item_id: Optional[int],
    ) -> None:
        self._prefix: str = prefix
        self._expected_num_unique_items = expected_num_unique_items
        self._expected_max_item_id = expected_max_item_id

    @abc.abstractmethod
    def expected_num_unique_items(self) -> Optional[int]:
        return self._expected_num_unique_items

    @abc.abstractmethod
    def expected_max_item_id(self) -> Optional[int]:
        return self._expected_max_item_id

    @abc.abstractmethod
    def processed_item_csv(self) -> str:
        pass

    def output_format_csv(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format.csv"

    def to_seq_data(
        self,
        ratings_data: pd.DataFrame,
        user_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if user_data is not None:
            ratings_data_transformed = ratings_data.join(
                user_data.set_index("user_id"), on="user_id"
            )
        else:
            ratings_data_transformed = ratings_data
        ratings_data_transformed.item_ids = ratings_data_transformed.item_ids.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.timestamps = ratings_data_transformed.timestamps.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.rename(
            columns={
                "item_ids": "sequence_item_ids",
                "ratings": "sequence_ratings",
                "timestamps": "sequence_timestamps",
            },
            inplace=True,
        )
        return ratings_data_transformed

    def file_exists(self, name: str) -> bool:
        return os.path.isfile("%s/%s" % (os.getcwd(), name))


class MovielensSyntheticDataProcessor(DataProcessor):
    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int] = None,
        expected_max_item_id: Optional[int] = None,
    ) -> None:
        super().__init__(prefix, expected_num_unique_items, expected_max_item_id)

    def preprocess_rating(self) -> None:
        return


class MovielensDataProcessor(DataProcessor):
    def __init__(
        self,
        download_path: str,
        saved_name: str,
        prefix: str,
        convert_timestamp: bool,
        expected_num_unique_items: Optional[int] = None,
        expected_max_item_id: Optional[int] = None,
    ) -> None:
        super().__init__(prefix, expected_num_unique_items, expected_max_item_id)
        self._download_path = download_path
        self._saved_name = saved_name
        self._convert_timestamp: bool = convert_timestamp

    def download(self) -> None:
        if not self.file_exists(self._saved_name):
            urlretrieve(self._download_path, self._saved_name)
        if self._saved_name[-4:] == ".zip":
            ZipFile(self._saved_name, "r").extractall(path="tmp/")
        else:
            with tarfile.open(self._saved_name, "r:*") as tar_ref:
                tar_ref.extractall("tmp/")

    def processed_item_csv(self) -> str:
        return f"tmp/processed/{self._prefix}/movies.csv"

    def sasrec_format_csv_by_user_train(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format_by_user_train.csv"

    def sasrec_format_csv_by_user_test(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format_by_user_test.csv"

    def preprocess_rating(self) -> int:
        self.download()

        if self._prefix == "ml-1m":
            users = pd.read_csv(
                f"tmp/{self._prefix}/users.dat",
                sep="::",
                names=["user_id", "sex", "age_group", "occupation", "zip_code"],
            )
            ratings = pd.read_csv(
                f"tmp/{self._prefix}/ratings.dat",
                sep="::",
                names=["user_id", "movie_id", "rating", "unix_timestamp"],
            )
            movies = pd.read_csv(
                f"tmp/{self._prefix}/movies.dat",
                sep="::",
                names=["movie_id", "title", "genres"],
                encoding="iso-8859-1",
            )
        elif self._prefix == "ml-20m":
            # ml-20m
            # ml-20m doesn't have user data.
            users = None
            # ratings: userId,movieId,rating,timestamp
            ratings = pd.read_csv(
                f"tmp/{self._prefix}/ratings.csv",
                sep=",",
            )
            ratings.rename(
                columns={
                    "userId": "user_id",
                    "movieId": "movie_id",
                    "timestamp": "unix_timestamp",
                },
                inplace=True,
            )
            # movieId,title,genres
            # 1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
            # 2,Jumanji (1995),Adventure|Children|Fantasy
            movies = pd.read_csv(
                f"tmp/{self._prefix}/movies.csv",
                sep=",",
                encoding="iso-8859-1",
            )
            movies.rename(columns={"movieId": "movie_id"}, inplace=True)
        else:
            assert self._prefix == "ml-20mx16x32"
            # ml-1b
            user_ids = []
            movie_ids = []
            for i in range(16):
                train_file = f"tmp/{self._prefix}/trainx16x32_{i}.npz"
                with np.load(train_file) as data:
                    user_ids.extend([x[0] for x in data["arr_0"]])
                    movie_ids.extend([x[1] for x in data["arr_0"]])
            ratings = pd.DataFrame(
                data={
                    "user_id": user_ids,
                    "movie_id": movie_ids,
                    "rating": user_ids,  # placeholder
                    "unix_timestamp": movie_ids,  # placeholder
                }
            )
            users = None
            movies = None

        if movies is not None:
            # ML-1M and ML-20M only
            movies["year"] = movies["title"].apply(lambda x: x[-5:-1])
            movies["cleaned_title"] = movies["title"].apply(lambda x: x[:-7])
            # movies.year = pd.Categorical(movies.year)
            # movies["year"] = movies.year.cat.codes

        if users is not None:
            ## Users (ml-1m only)
            users.sex = pd.Categorical(users.sex)
            users["sex"] = users.sex.cat.codes

            users.age_group = pd.Categorical(users.age_group)
            users["age_group"] = users.age_group.cat.codes

            users.occupation = pd.Categorical(users.occupation)
            users["occupation"] = users.occupation.cat.codes

            users.zip_code = pd.Categorical(users.zip_code)
            users["zip_code"] = users.zip_code.cat.codes

        # Normalize movie ids to speed up training
        print(
            f"{self._prefix} #item before normalize: {len(set(ratings['movie_id'].values))}"
        )
        print(
            f"{self._prefix} max item id before normalize: {max(set(ratings['movie_id'].values))}"
        )
        # print(f"ratings.movie_id.cat.categories={ratings.movie_id.cat.categories}; {type(ratings.movie_id.cat.categories)}")
        # print(f"ratings.movie_id.cat.codes={ratings.movie_id.cat.codes}; {type(ratings.movie_id.cat.codes)}")
        # print(movie_id_to_cat)
        # ratings["movie_id"] = ratings.movie_id.cat.codes
        # print(f"{self._prefix} #item after normalize: {len(set(ratings['movie_id'].values))}")
        # print(f"{self._prefix} max item id after normalize: {max(set(ratings['movie_id'].values))}")
        # movies["remapped_id"] = movies["movie_id"].apply(lambda x: movie_id_to_cat[x])

        if self._convert_timestamp:
            ratings["unix_timestamp"] = pd.to_datetime(
                ratings["unix_timestamp"], unit="s"
            )

        # Save primary csv's
        if not os.path.exists(f"tmp/processed/{self._prefix}"):
            os.makedirs(f"tmp/processed/{self._prefix}")
        if users is not None:
            users.to_csv(f"tmp/processed/{self._prefix}/users.csv", index=False)
        if movies is not None:
            movies.to_csv(f"tmp/processed/{self._prefix}/movies.csv", index=False)
        ratings.to_csv(f"tmp/processed/{self._prefix}/ratings.csv", index=False)

        num_unique_users = len(set(ratings["user_id"].values))
        num_unique_items = len(set(ratings["movie_id"].values))

        # SASRec version
        ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")
        seq_ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.movie_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.unix_timestamp.apply(list)),
            }
        )

        result = pd.DataFrame([[]])
        for col in ["item_ids"]:
            result[col + "_mean"] = seq_ratings_data[col].apply(len).mean()
            result[col + "_min"] = seq_ratings_data[col].apply(len).min()
            result[col + "_max"] = seq_ratings_data[col].apply(len).max()
        print(self._prefix)
        print(result)

        seq_ratings_data = self.to_seq_data(seq_ratings_data, users)
        seq_ratings_data.sample(frac=1).reset_index().to_csv(
            self.output_format_csv(), index=False, sep=","
        )

        # Split by user ids (not tested yet)
        user_id_split = int(num_unique_users * 0.9)
        seq_ratings_data_train = seq_ratings_data[
            seq_ratings_data["user_id"] <= user_id_split
        ]
        seq_ratings_data_train.sample(frac=1).reset_index().to_csv(
            self.sasrec_format_csv_by_user_train(),
            index=False,
            sep=",",
        )
        seq_ratings_data_test = seq_ratings_data[
            seq_ratings_data["user_id"] > user_id_split
        ]
        seq_ratings_data_test.sample(frac=1).reset_index().to_csv(
            self.sasrec_format_csv_by_user_test(), index=False, sep=","
        )
        print(
            f"{self._prefix}: train num user: {len(set(seq_ratings_data_train['user_id'].values))}"
        )
        print(
            f"{self._prefix}: test num user: {len(set(seq_ratings_data_test['user_id'].values))}"
        )

        # print(seq_ratings_data)
        if self.expected_num_unique_items() is not None:
            assert (
                self.expected_num_unique_items() == num_unique_items
            ), f"Expected items: {self.expected_num_unique_items()}, got: {num_unique_items}"

        return num_unique_items


class AmazonDataProcessor(DataProcessor):
    def __init__(
        self,
        download_path: str,
        saved_name: str,
        prefix: str,
        expected_num_unique_items: Optional[int],
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=None,
        )
        self._download_path = download_path
        self._saved_name = saved_name
        self._prefix = prefix

    def download(self) -> None:
        if not self.file_exists(self._saved_name):
            urlretrieve(self._download_path, self._saved_name)

    def preprocess_rating(self) -> int:
        self.download()

        ratings = pd.read_csv(
            self._saved_name,
            sep=",",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        print(f"{self._prefix} #data points before filter: {ratings.shape[0]}")
        print(
            f"{self._prefix} #user before filter: {len(set(ratings['user_id'].values))}"
        )
        print(
            f"{self._prefix} #item before filter: {len(set(ratings['item_id'].values))}"
        )

        # filter users and items with presence < 5
        item_id_count = (
            ratings["item_id"]
            .value_counts()
            .rename_axis("unique_values")
            .reset_index(name="item_count")
        )
        user_id_count = (
            ratings["user_id"]
            .value_counts()
            .rename_axis("unique_values")
            .reset_index(name="user_count")
        )
        ratings = ratings.join(item_id_count.set_index("unique_values"), on="item_id")
        ratings = ratings.join(user_id_count.set_index("unique_values"), on="user_id")
        ratings = ratings[ratings["item_count"] >= 5]
        ratings = ratings[ratings["user_count"] >= 5]
        print(f"{self._prefix} #data points after filter: {ratings.shape[0]}")

        # categorize user id and item id
        ratings["item_id"] = pd.Categorical(ratings["item_id"])
        # pyrefly: ignore [missing-attribute]
        ratings["item_id"] = ratings["item_id"].cat.codes
        ratings["user_id"] = pd.Categorical(ratings["user_id"])
        # pyrefly: ignore [missing-attribute]
        ratings["user_id"] = ratings["user_id"].cat.codes
        print(
            f"{self._prefix} #user after filter: {len(set(ratings['user_id'].values))}"
        )
        print(
            f"{self._prefix} #item ater filter: {len(set(ratings['item_id'].values))}"
        )

        num_unique_items = len(set(ratings["item_id"].values))

        # SASRec version
        ratings_group = ratings.sort_values(by=["timestamp"]).groupby("user_id")

        seq_ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.item_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.timestamp.apply(list)),
            }
        )

        seq_ratings_data = seq_ratings_data[
            seq_ratings_data["item_ids"].apply(len) >= 5
        ]

        result = pd.DataFrame([[]])
        for col in ["item_ids"]:
            result[col + "_mean"] = seq_ratings_data[col].apply(len).mean()
            result[col + "_min"] = seq_ratings_data[col].apply(len).min()
            result[col + "_max"] = seq_ratings_data[col].apply(len).max()
        print(self._prefix)
        print(result)

        if not os.path.exists(f"tmp/{self._prefix}"):
            os.makedirs(f"tmp/{self._prefix}")

        seq_ratings_data = self.to_seq_data(seq_ratings_data)
        seq_ratings_data.sample(frac=1).reset_index().to_csv(
            self.output_format_csv(), index=False, sep=","
        )

        if self.expected_num_unique_items() is not None:
            assert (
                self.expected_num_unique_items() == num_unique_items
            ), f"expected: {self.expected_num_unique_items()}, actual: {num_unique_items}"
            logging.info(f"{self.expected_num_unique_items()} unique items.")

        return num_unique_items


class KuaiRandDataProcessor(DataProcessor):
    """Build the frozen KuaiRand-1K sequential-retrieval artifacts."""

    _DOWNLOAD_URL = "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz"
    _EXPECTED_ARCHIVE_BYTES = 1_135_436_720
    _EXPECTED_ARCHIVE_MD5 = "6b0b9c8222d67fcd4c676218edca3f1f"
    _EXPECTED_ARCHIVE_SHA256 = (
        "dfaafbb5fd16e9e6d2f9a6adaa4ea25df20a14bc26a90961c136e26c00a7bb2c"
    )
    _PROTOCOL_VERSION = "kuairand-1k-sequential-v2"
    _LOG_FILENAMES: Tuple[str, ...] = (
        "log_standard_4_08_to_4_21_1k.csv",
        "log_standard_4_22_to_5_08_1k.csv",
        "log_random_4_22_to_5_08_1k.csv",
    )
    _EXPECTED_LOG_SHA256 = {
        "log_standard_4_08_to_4_21_1k.csv": (
            "355355897a84baa4df26b78b0271bb9a27b127a39cd7aa4a898cf86db0bc1810"
        ),
        "log_standard_4_22_to_5_08_1k.csv": (
            "548daf771e54e2b73086cc8e7c6f56787d44421f5026c2100421810e47ae9dd4"
        ),
        "log_random_4_22_to_5_08_1k.csv": (
            "e98841eabb3b078c812c4646bf0c5e3f0c947a4183dd950cb091e32377b9a849"
        ),
    }
    _EXPECTED_STATISTICS = {
        "raw_interactions": 11_756_073,
        "positive_interactions": 4_434_766,
        "core_interactions": 2_128_834,
        "num_users": 999,
        "num_items": 192_120,
        "num_train_windows": 1_584,
        "num_eval_rows": 999,
        "min_user_sequence_length": 14,
        "max_user_sequence_length": 12_034,
        "core_filter_iterations": 3,
    }
    _EXPECTED_ARTIFACT_ROWS = {
        "train.csv": 1_584,
        "eval.csv": 999,
        "item_id_map.csv": 192_120,
    }
    _PROTOCOL = {
        "interaction_filter": "is_click == 1 and is_hate == 0",
        "core_filter": "iterative user/item event-count 5-core",
        "item_id_mapping": "ascending original ID to dense zero-based ID",
        "ordering": [
            "user_id",
            "time_ms",
            "source_file_rank",
            "source_row",
        ],
        "training_prefix": "all retained events except final user event",
        "train_window_events": KUAI_RAND_MAX_SEQUENCE_LENGTH + 1,
        "train_window_stride": KUAI_RAND_MAX_SEQUENCE_LENGTH,
        "eval_split": "one leave-last-out row per retained user",
        "rating_encoding": "constant 5 for retained positive events",
    }
    _INPUT_DTYPES = {
        "user_id": "int64",
        "video_id": "int64",
        "time_ms": "int64",
        "is_click": "int8",
        "is_hate": "int8",
    }

    def __init__(
        self,
        archive_path: Optional[str] = None,
        prefix: str = "kuairand-1k",
        output_root: Optional[str] = None,
        validate_official: bool = True,
    ) -> None:
        super().__init__(
            prefix=prefix,
            expected_num_unique_items=192_120,
            expected_max_item_id=192_120,
        )
        resolved_output_root = output_root or os.environ.get("GR_DATA_ROOT", "tmp")
        self._output_dir = Path(resolved_output_root) / prefix
        configured_archive = archive_path or os.environ.get("GR_KUAIRAND_ARCHIVE")
        self._archive_path = Path(
            configured_archive
            if configured_archive is not None
            else self._output_dir.parent / "KuaiRand-1K.tar.gz"
        )
        self._extract_dir = self._output_dir / "raw"
        self._validate_official = validate_official

    def processed_item_csv(self) -> str:
        return self.item_id_map_csv()

    def output_format_csv(self) -> str:
        return self.eval_format_csv()

    def train_format_csv(self) -> str:
        return str(self._output_dir / "train.csv")

    def eval_format_csv(self) -> str:
        return str(self._output_dir / "eval.csv")

    def item_id_map_csv(self) -> str:
        return str(self._output_dir / "item_id_map.csv")

    def metadata_json(self) -> str:
        return str(self._output_dir / "metadata.json")

    def checksums_file(self) -> str:
        return str(self._output_dir / "checksums.sha256")

    def _find_extracted_logs(self) -> Optional[List[Path]]:
        paths: List[Path] = []
        for filename in self._LOG_FILENAMES:
            matches = sorted(self._extract_dir.rglob(filename))
            if not matches:
                return None
            if len(matches) != 1:
                raise ValueError(
                    f"expected one extracted {filename}, found {len(matches)}"
                )
            if matches[0].is_symlink() or not matches[0].is_file():
                raise ValueError(
                    f"extracted KuaiRand log must be a regular file: {matches[0]}"
                )
            paths.append(matches[0])
        return paths

    def _extract_archive(self) -> List[Path]:
        existing = self._find_extracted_logs()
        if existing is not None:
            return existing

        self._extract_dir.mkdir(parents=True, exist_ok=True)
        extraction_root = self._extract_dir.resolve()
        with tarfile.open(self._archive_path, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if Path(member.name).name in self._LOG_FILENAMES
            ]
            found_names = [Path(member.name).name for member in members]
            if sorted(found_names) != sorted(self._LOG_FILENAMES):
                raise FileNotFoundError(
                    "KuaiRand archive must contain each required log exactly once"
                )
            for member in members:
                if not member.isfile():
                    raise ValueError(
                        "KuaiRand archive logs must be regular files: " f"{member.name}"
                    )
                destination = (self._extract_dir / member.name).resolve()
                if not destination.is_relative_to(extraction_root):
                    raise ValueError(
                        f"KuaiRand archive path escapes extraction root: {member.name}"
                    )
            archive.extractall(self._extract_dir, members=members)

        extracted = self._find_extracted_logs()
        if extracted is None:
            raise FileNotFoundError("KuaiRand archive does not contain all three logs")
        return extracted

    def download(self) -> List[Path]:
        if not self._archive_path.is_file():
            self._archive_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info("Downloading KuaiRand-1K to %s", self._archive_path)
            partial_archive = self._archive_path.with_name(
                f"{self._archive_path.name}.part"
            )
            try:
                urlretrieve(self._DOWNLOAD_URL, partial_archive)
                self._validate_archive(partial_archive)
                os.replace(partial_archive, self._archive_path)
            finally:
                partial_archive.unlink(missing_ok=True)
        self._validate_archive(self._archive_path)
        return self._extract_archive()

    def _validate_archive(self, archive_path: Path) -> None:
        if not self._validate_official:
            return
        archive_size = archive_path.stat().st_size
        if archive_size != self._EXPECTED_ARCHIVE_BYTES:
            raise ValueError(
                "unexpected KuaiRand archive size: "
                f"expected {self._EXPECTED_ARCHIVE_BYTES}, found {archive_size}"
            )
        archive_md5 = _md5(archive_path)
        if archive_md5 != self._EXPECTED_ARCHIVE_MD5:
            raise ValueError(
                "unexpected KuaiRand archive MD5: "
                f"expected {self._EXPECTED_ARCHIVE_MD5}, found {archive_md5}"
            )

    def _read_positive_interactions(
        self, log_paths: Sequence[Path]
    ) -> Tuple[pd.DataFrame, int, Dict[str, str]]:
        frames: List[pd.DataFrame] = []
        raw_interactions = 0
        log_sha256: Dict[str, str] = {}
        columns = list(self._INPUT_DTYPES)
        for source_file_rank, log_path in enumerate(log_paths):
            digest = _sha256(log_path)
            log_sha256[log_path.name] = digest
            if self._validate_official:
                expected = self._EXPECTED_LOG_SHA256[log_path.name]
                if digest != expected:
                    raise ValueError(
                        f"unexpected SHA-256 for {log_path.name}: {digest}"
                    )

            source_row = 0
            for chunk in pd.read_csv(
                log_path,
                usecols=columns,
                dtype=self._INPUT_DTYPES,
                chunksize=1_000_000,
            ):
                chunk_size = len(chunk)
                raw_interactions += chunk_size
                keep = chunk["is_click"].eq(1) & chunk["is_hate"].eq(0)
                selected = chunk.loc[keep, ["user_id", "video_id", "time_ms"]].copy()
                selected["source_file_rank"] = source_file_rank
                selected["source_row"] = np.arange(
                    source_row, source_row + chunk_size, dtype=np.int64
                )[keep.to_numpy()]
                frames.append(selected)
                source_row += chunk_size

        if not frames:
            raise ValueError("KuaiRand logs contained no readable chunks")
        interactions = pd.concat(frames, ignore_index=True)
        if interactions.empty:
            raise ValueError("KuaiRand click/non-hate filter removed every interaction")
        return interactions, raw_interactions, log_sha256

    @staticmethod
    def _serialize(values: Sequence[int]) -> str:
        return ",".join(str(int(value)) for value in values)

    @classmethod
    def _sequence_row(
        cls,
        user_id: int,
        item_ids: Sequence[int],
        timestamps: Sequence[int],
    ) -> Dict[str, Any]:
        if len(item_ids) != len(timestamps) or len(item_ids) < 2:
            raise ValueError("sequence rows require aligned item/timestamp pairs")
        return {
            "user_id": int(user_id),
            "sequence_item_ids": cls._serialize(item_ids),
            "sequence_ratings": ",".join(["5"] * len(item_ids)),
            "sequence_timestamps": cls._serialize(timestamps),
        }

    @staticmethod
    def _write_csv(frame: pd.DataFrame, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)

    def load_metadata(self, verify_artifacts: bool = True) -> Dict[str, Any]:
        metadata_path = Path(self.metadata_json())
        if metadata_path.is_symlink():
            raise ValueError(
                f"KuaiRand metadata must not be a symlink: {metadata_path}"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "KuaiRand-1K is not preprocessed; run "
                "preprocess_public_data.py --dataset kuairand-1k"
            ) from error
        expected_metadata_keys = {
            "artifacts",
            "dataset",
            "implementation",
            "max_sequence_length",
            "protocol",
            "schema_version",
            "source",
            "statistics",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected_metadata_keys:
            raise ValueError(
                "KuaiRand metadata schema mismatch: expected keys "
                f"{sorted(expected_metadata_keys)}, found {sorted(metadata)}"
            )
        if metadata.get("schema_version") != 1:
            raise ValueError("KuaiRand metadata has an unexpected schema version")
        if metadata.get("dataset") != "kuairand-1k":
            raise ValueError("KuaiRand metadata has an unexpected dataset key")
        if metadata.get("max_sequence_length") != KUAI_RAND_MAX_SEQUENCE_LENGTH:
            raise ValueError("KuaiRand metadata has an unexpected sequence length")
        if metadata.get("protocol") != self._PROTOCOL:
            raise ValueError("KuaiRand metadata has an unexpected protocol definition")

        implementation = metadata.get("implementation")
        expected_implementation_keys = {
            "numpy_version",
            "pandas_version",
            "preprocessor_sha256",
            "protocol_version",
            "python_version",
        }
        if not isinstance(implementation, dict) or set(implementation) != (
            expected_implementation_keys
        ):
            raise ValueError("KuaiRand metadata has an invalid implementation record")
        if implementation.get("protocol_version") != self._PROTOCOL_VERSION:
            raise ValueError("KuaiRand metadata has an unexpected protocol version")
        current_preprocessor_sha256 = _sha256(Path(__file__).resolve())
        if implementation.get("preprocessor_sha256") != current_preprocessor_sha256:
            raise ValueError(
                "KuaiRand metadata was generated by a different preprocessor source"
            )
        for version_key in ("python_version", "numpy_version", "pandas_version"):
            if not isinstance(implementation.get(version_key), str):
                raise ValueError(
                    f"KuaiRand metadata has an invalid {version_key} record"
                )

        statistics = metadata.get("statistics")
        expected_statistic_keys = {
            *self._EXPECTED_STATISTICS,
            "mean_user_sequence_length",
        }
        if (
            not isinstance(statistics, dict)
            or set(statistics) != expected_statistic_keys
        ):
            raise ValueError("KuaiRand metadata has an invalid statistics record")
        if self._validate_official:
            for key, expected in self._EXPECTED_STATISTICS.items():
                if statistics.get(key) != expected:
                    raise ValueError(
                        f"unexpected KuaiRand statistic {key}: "
                        f"expected {expected}, found {statistics.get(key)}"
                    )

        source = metadata.get("source")
        if not isinstance(source, dict) or set(source) != {
            "archive_name",
            "archive_sha256",
            "logs",
        }:
            raise ValueError("KuaiRand metadata has an invalid source record")
        logs = source.get("logs")
        if not isinstance(logs, list) or any(
            not isinstance(record, dict) or set(record) != {"filename", "sha256"}
            for record in logs
        ):
            raise ValueError("KuaiRand metadata has an invalid source log record")
        if self._validate_official:
            if source.get("archive_sha256") != self._EXPECTED_ARCHIVE_SHA256:
                raise ValueError("KuaiRand metadata has an unexpected archive SHA-256")
            expected_logs = [
                {
                    "filename": filename,
                    "sha256": self._EXPECTED_LOG_SHA256[filename],
                }
                for filename in self._LOG_FILENAMES
            ]
            if logs != expected_logs:
                raise ValueError("KuaiRand metadata has unexpected source log hashes")

        artifacts = metadata.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(
            self._EXPECTED_ARTIFACT_ROWS
        ):
            raise ValueError("KuaiRand metadata has an invalid artifact inventory")
        for filename, record in artifacts.items():
            if not isinstance(record, dict) or set(record) != {
                "bytes",
                "rows",
                "sha256",
            }:
                raise ValueError(f"KuaiRand metadata has an invalid {filename} record")
            if not isinstance(record.get("bytes"), int) or record["bytes"] < 1:
                raise ValueError(f"KuaiRand metadata has invalid bytes for {filename}")
            if not isinstance(record.get("rows"), int) or record["rows"] < 1:
                raise ValueError(f"KuaiRand metadata has invalid rows for {filename}")
            if self._validate_official and (
                record["rows"] != self._EXPECTED_ARTIFACT_ROWS[filename]
            ):
                raise ValueError(
                    f"KuaiRand metadata has unexpected rows for {filename}"
                )
        if verify_artifacts:
            expected_checksums = {"metadata.json": _sha256(metadata_path)}
            for filename, record in artifacts.items():
                artifact = self._output_dir / filename
                if not artifact.is_file() or artifact.is_symlink():
                    raise FileNotFoundError(f"missing KuaiRand artifact: {artifact}")
                actual_sha256 = _sha256(artifact)
                if actual_sha256 != record.get("sha256"):
                    raise ValueError(f"KuaiRand artifact checksum mismatch: {artifact}")
                if artifact.stat().st_size != record["bytes"]:
                    raise ValueError(
                        f"KuaiRand artifact byte count mismatch: {artifact}"
                    )
                expected_checksums[filename] = actual_sha256

            checksums_path = Path(self.checksums_file())
            if not checksums_path.is_file() or checksums_path.is_symlink():
                raise FileNotFoundError(
                    f"missing KuaiRand checksum manifest: {checksums_path}"
                )
            checksum_records: Dict[str, str] = {}
            for line in checksums_path.read_text(encoding="utf-8").splitlines():
                try:
                    digest, filename = line.split("  ", 1)
                except ValueError as error:
                    raise ValueError(
                        "KuaiRand checksum manifest has an invalid record"
                    ) from error
                if filename in checksum_records:
                    raise ValueError(f"KuaiRand checksum manifest repeats {filename}")
                checksum_records[filename] = digest
            if checksum_records != expected_checksums:
                raise ValueError("KuaiRand checksum manifest does not match metadata")
        return metadata

    def preprocess_rating(self) -> int:
        log_paths = self.download()
        interactions, raw_interactions, log_sha256 = self._read_positive_interactions(
            log_paths
        )
        positive_interactions = len(interactions)
        interactions, core_iterations = _iterative_k_core(interactions)
        if interactions.empty:
            raise ValueError("KuaiRand iterative 5-core is empty")

        interactions.sort_values(
            by=["user_id", "time_ms", "source_file_rank", "source_row"],
            kind="mergesort",
            inplace=True,
            ignore_index=True,
        )
        original_item_ids = np.sort(interactions["video_id"].unique())
        dense_item_ids = np.searchsorted(
            original_item_ids, interactions["video_id"].to_numpy()
        )
        if not np.array_equal(
            original_item_ids[dense_item_ids], interactions["video_id"]
        ):
            raise AssertionError("KuaiRand dense item remap is inconsistent")
        interactions["dense_item_id"] = dense_item_ids

        train_rows: List[Dict[str, Any]] = []
        eval_rows: List[Dict[str, Any]] = []
        sequence_lengths: List[int] = []
        for user_id, sequence in interactions.groupby("user_id", sort=True):
            item_ids = sequence["dense_item_id"].to_numpy(dtype=np.int64)
            timestamps = sequence["time_ms"].to_numpy(dtype=np.int64)
            sequence_lengths.append(len(item_ids))
            train_item_ids = item_ids[:-1]
            train_timestamps = timestamps[:-1]
            for start, end in _iter_training_window_bounds(len(train_item_ids)):
                train_rows.append(
                    self._sequence_row(
                        user_id=int(user_id),
                        item_ids=train_item_ids[start:end],
                        timestamps=train_timestamps[start:end],
                    )
                )

            eval_start = max(0, len(item_ids) - KUAI_RAND_MAX_SEQUENCE_LENGTH - 1)
            eval_rows.append(
                self._sequence_row(
                    user_id=int(user_id),
                    item_ids=item_ids[eval_start:],
                    timestamps=timestamps[eval_start:],
                )
            )

        item_map = pd.DataFrame(
            {
                "original_item_id": original_item_ids,
                "dense_item_id": np.arange(len(original_item_ids), dtype=np.int64),
                "model_item_id": np.arange(
                    1, len(original_item_ids) + 1, dtype=np.int64
                ),
            }
        )
        output_columns = [
            "user_id",
            "sequence_item_ids",
            "sequence_ratings",
            "sequence_timestamps",
        ]
        train_frame = pd.DataFrame(train_rows, columns=output_columns)
        eval_frame = pd.DataFrame(eval_rows, columns=output_columns)

        statistics = {
            "raw_interactions": raw_interactions,
            "positive_interactions": positive_interactions,
            "core_interactions": len(interactions),
            "num_users": len(sequence_lengths),
            "num_items": len(original_item_ids),
            "num_train_windows": len(train_frame),
            "num_eval_rows": len(eval_frame),
            "min_user_sequence_length": min(sequence_lengths),
            "max_user_sequence_length": max(sequence_lengths),
            "mean_user_sequence_length": float(np.mean(sequence_lengths)),
            "core_filter_iterations": core_iterations,
        }
        if self._validate_official:
            for key, expected in self._EXPECTED_STATISTICS.items():
                if statistics[key] != expected:
                    raise ValueError(
                        f"unexpected KuaiRand statistic {key}: "
                        f"expected {expected}, found {statistics[key]}"
                    )

        user_counts = interactions["user_id"].value_counts()
        item_counts = interactions["video_id"].value_counts()
        if int(user_counts.min()) < KUAI_RAND_MIN_CORE_DEGREE:
            raise AssertionError("KuaiRand user 5-core invariant failed")
        if int(item_counts.min()) < KUAI_RAND_MIN_CORE_DEGREE:
            raise AssertionError("KuaiRand item 5-core invariant failed")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        train_path = Path(self.train_format_csv())
        eval_path = Path(self.eval_format_csv())
        item_map_path = Path(self.item_id_map_csv())
        self._write_csv(train_frame, train_path)
        self._write_csv(eval_frame, eval_path)
        self._write_csv(item_map, item_map_path)

        artifacts: Dict[str, Dict[str, Union[int, str]]] = {}
        for path, rows in (
            (train_path, len(train_frame)),
            (eval_path, len(eval_frame)),
            (item_map_path, len(item_map)),
        ):
            artifacts[path.name] = {
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        metadata = {
            "schema_version": 1,
            "dataset": "kuairand-1k",
            "max_sequence_length": KUAI_RAND_MAX_SEQUENCE_LENGTH,
            "protocol": self._PROTOCOL,
            "implementation": {
                "protocol_version": self._PROTOCOL_VERSION,
                "preprocessor_sha256": _sha256(Path(__file__).resolve()),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
            "source": {
                "archive_name": self._archive_path.name,
                "archive_sha256": _sha256(self._archive_path),
                "logs": [
                    {"filename": filename, "sha256": log_sha256[filename]}
                    for filename in self._LOG_FILENAMES
                ],
            },
            "statistics": statistics,
            "artifacts": artifacts,
        }
        metadata_path = Path(self.metadata_json())
        temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata, metadata_path)

        checksummed_paths = [train_path, eval_path, item_map_path, metadata_path]
        checksums_path = Path(self.checksums_file())
        temporary_checksums = checksums_path.with_name(f".{checksums_path.name}.tmp")
        temporary_checksums.write_text(
            "".join(f"{_sha256(path)}  {path.name}\n" for path in checksummed_paths),
            encoding="utf-8",
        )
        os.replace(temporary_checksums, checksums_path)
        logging.info(
            "Prepared KuaiRand-1K: %d users, %d items, %d interactions, %d windows",
            statistics["num_users"],
            statistics["num_items"],
            statistics["core_interactions"],
            statistics["num_train_windows"],
        )
        return len(original_item_ids)


def get_common_preprocessors() -> Dict[
    str,
    Union[
        AmazonDataProcessor,
        KuaiRandDataProcessor,
        MovielensDataProcessor,
        MovielensSyntheticDataProcessor,
    ],
]:
    ml_1m_dp = MovielensDataProcessor(  # pyre-ignore [45]
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "tmp/movielens1m.zip",
        prefix="ml-1m",
        convert_timestamp=False,
        expected_num_unique_items=3706,
        expected_max_item_id=3952,
    )
    ml_20m_dp = MovielensDataProcessor(  # pyre-ignore [45]
        "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "tmp/movielens20m.zip",
        prefix="ml-20m",
        convert_timestamp=False,
        expected_num_unique_items=26744,
        expected_max_item_id=131262,
    )
    ml_1b_dp = MovielensDataProcessor(  # pyre-ignore [45]
        "https://files.grouplens.org/datasets/movielens/ml-20mx16x32.tar",
        "tmp/movielens1b.tar",
        prefix="ml-20mx16x32",
        convert_timestamp=False,
    )
    ml_3b_dp = MovielensSyntheticDataProcessor(  # pyre-ignore [45]
        prefix="ml-3b",
        expected_num_unique_items=26743 * 32,
        expected_max_item_id=26743 * 32,
    )
    amzn_books_dp = AmazonDataProcessor(  # pyre-ignore [45]
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv",
        "tmp/ratings_Books.csv",
        prefix="amzn_books",
        expected_num_unique_items=695762,
    )
    kuairand_1k_dp = KuaiRandDataProcessor()
    return {
        "ml-1m": ml_1m_dp,
        "ml-20m": ml_20m_dp,
        "ml-1b": ml_1b_dp,
        "ml-3b": ml_3b_dp,
        "amzn-books": amzn_books_dp,
        "kuairand-1k": kuairand_1k_dp,
    }
