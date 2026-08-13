# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

from generative_recommenders.research.data.dataset import DatasetV2
from generative_recommenders.research.data.eval import _seen_ids_excluding_target
from generative_recommenders.research.data.preprocessor import (
    KUAI_RAND_MAX_SEQUENCE_LENGTH,
    KuaiRandDataProcessor,
    _iter_training_window_bounds,
)
from generative_recommenders.research.data.reco_dataset import get_reco_dataset
from generative_recommenders.research.indexing.candidate_index import CandidateIndex
from generative_recommenders.research.rails.indexing.mips_top_k import (
    MIPSBruteForceTopK,
)
from preprocess_public_data import (
    LEGACY_DATASETS,
    SELECTABLE_DATASETS,
    _selected_datasets,
)


LOG_COLUMNS = ["user_id", "video_id", "time_ms", "is_click", "is_hate"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class KuaiRandDataTest(unittest.TestCase):
    def _make_archive(self, root: Path) -> Path:
        source_root = root / "source/KuaiRand-1K/data"
        source_root.mkdir(parents=True)
        filenames = KuaiRandDataProcessor._LOG_FILENAMES
        rows = {filename: [] for filename in filenames}

        # Five users x five items is the stable 5-core. Equal timestamps for
        # items 100/101 test the source-row tie break; 103 precedes 102 in the
        # file to ensure chronological sorting is not inherited from input order.
        for user_id in range(5):
            base_time = 10_000 + 100 * user_id
            rows[filenames[0]].extend(
                [
                    [user_id, 100, base_time, 1, 0],
                    [user_id, 101, base_time, 1, 0],
                ]
            )
            rows[filenames[1]].extend(
                [
                    [user_id, 103, base_time + 3, 1, 0],
                    [user_id, 102, base_time + 2, 1, 0],
                ]
            )
            rows[filenames[2]].append([user_id, 104, base_time + 4, 1, 0])

        # This component passes the first simultaneous degree check, then
        # cascades away over three iterations.
        for user_id in range(4):
            rows[filenames[0]].append([user_id, 200, 20_000 + user_id, 1, 0])
        rows[filenames[1]].extend(
            [[99, item_id, 30_000 + item_id, 1, 0] for item_id in range(200, 205)]
        )

        # Both filters are necessary: neither row may affect the core.
        rows[filenames[2]].extend([[0, 999, 40_000, 0, 0], [0, 998, 40_001, 1, 1]])
        for filename in filenames:
            pd.DataFrame(rows[filename], columns=LOG_COLUMNS).to_csv(
                source_root / filename, index=False, lineterminator="\n"
            )
        (source_root / "video_features_should_not_extract.csv").write_text(
            "unused\n", encoding="utf-8"
        )

        archive_path = root / "KuaiRand-1K.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(root / "source/KuaiRand-1K", arcname="KuaiRand-1K")
        return archive_path

    def test_boundary_overlap_covers_each_transition_once(self) -> None:
        sequence_length = 2 * KUAI_RAND_MAX_SEQUENCE_LENGTH + 6
        bounds = list(_iter_training_window_bounds(sequence_length))
        self.assertEqual(
            bounds,
            [
                (0, KUAI_RAND_MAX_SEQUENCE_LENGTH + 1),
                (
                    KUAI_RAND_MAX_SEQUENCE_LENGTH,
                    2 * KUAI_RAND_MAX_SEQUENCE_LENGTH + 1,
                ),
                (2 * KUAI_RAND_MAX_SEQUENCE_LENGTH, sequence_length),
            ],
        )
        covered = [
            transition for start, end in bounds for transition in range(start, end - 1)
        ]
        self.assertEqual(covered, list(range(sequence_length - 1)))

    def test_archive_rejects_non_regular_log_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "KuaiRand-1K.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for index, filename in enumerate(KuaiRandDataProcessor._LOG_FILENAMES):
                    member = tarfile.TarInfo(f"KuaiRand-1K/data/{filename}")
                    if index == 0:
                        member.type = tarfile.FIFOTYPE
                    else:
                        payload = b"user_id,video_id,time_ms,is_click,is_hate\n"
                        member.size = len(payload)
                        archive.addfile(member, fileobj=io.BytesIO(payload))
                        continue
                    archive.addfile(member)
            processor = KuaiRandDataProcessor(
                archive_path=str(archive_path),
                output_root=str(root / "data"),
                validate_official=False,
            )
            with self.assertRaisesRegex(ValueError, "must be regular files"):
                processor.download()

    def test_long_histories_preserve_boundaries_and_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "source/KuaiRand-1K/data"
            source_root.mkdir(parents=True)
            filenames = KuaiRandDataProcessor._LOG_FILENAMES
            num_events = 2 * KUAI_RAND_MAX_SEQUENCE_LENGTH + 7
            long_rows = [
                [user_id, 1_000 + offset, 10_000 + offset, 1, 0]
                for user_id in range(5)
                for offset in range(num_events)
            ]
            pd.DataFrame(long_rows, columns=LOG_COLUMNS).to_csv(
                source_root / filenames[0], index=False, lineterminator="\n"
            )
            for filename in filenames[1:]:
                pd.DataFrame(columns=LOG_COLUMNS).to_csv(
                    source_root / filename, index=False, lineterminator="\n"
                )
            archive_path = root / "KuaiRand-1K.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(root / "source/KuaiRand-1K", arcname="KuaiRand-1K")

            processor = KuaiRandDataProcessor(
                archive_path=str(archive_path),
                output_root=str(root / "data"),
                validate_official=False,
            )
            self.assertEqual(processor.preprocess_rating(), num_events)
            train = pd.read_csv(processor.train_format_csv())
            evaluation = pd.read_csv(processor.eval_format_csv())
            self.assertEqual(len(train), 15)
            self.assertEqual(len(evaluation), 5)

            user_train = train[train.user_id == 0]
            train_sequences = [
                [int(value) for value in sequence.split(",")]
                for sequence in user_train.sequence_item_ids
            ]
            self.assertEqual(
                [len(sequence) for sequence in train_sequences],
                [
                    KUAI_RAND_MAX_SEQUENCE_LENGTH + 1,
                    KUAI_RAND_MAX_SEQUENCE_LENGTH + 1,
                    6,
                ],
            )
            self.assertEqual(train_sequences[0][-1], train_sequences[1][0])
            self.assertEqual(train_sequences[1][-1], train_sequences[2][0])
            held_out_item_id = num_events - 1
            self.assertNotIn(
                held_out_item_id, [item for row in train_sequences for item in row]
            )

            eval_sequence = [
                int(value) for value in evaluation.iloc[0].sequence_item_ids.split(",")
            ]
            self.assertEqual(len(eval_sequence), KUAI_RAND_MAX_SEQUENCE_LENGTH + 1)
            self.assertEqual(train_sequences[-1][-1], eval_sequence[-2])
            self.assertEqual(eval_sequence[-1], held_out_item_id)

            with mock.patch(
                "generative_recommenders.research.data.reco_dataset."
                "get_common_preprocessors",
                return_value={"kuairand-1k": processor},
            ):
                dataset = get_reco_dataset(
                    dataset_name="kuairand-1k",
                    max_sequence_length=KUAI_RAND_MAX_SEQUENCE_LENGTH,
                    chronological=True,
                )
            self.assertEqual(
                int(dataset.train_dataset[2]["target_ids"]), num_events - 1
            )
            self.assertEqual(int(dataset.eval_dataset[0]["target_ids"]), num_events)

    def test_preprocess_and_load_synthetic_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = self._make_archive(root)
            data_root = root / "data"
            processor = KuaiRandDataProcessor(
                archive_path=str(archive_path),
                output_root=str(data_root),
                validate_official=False,
            )

            self.assertEqual(processor.preprocess_rating(), 5)
            self.assertFalse(
                (
                    data_root / "kuairand-1k/raw/KuaiRand-1K/data/"
                    "video_features_should_not_extract.csv"
                ).exists()
            )
            metadata = processor.load_metadata()
            statistics = metadata["statistics"]
            self.assertEqual(statistics["core_interactions"], 25)
            self.assertEqual(statistics["num_users"], 5)
            self.assertEqual(statistics["num_items"], 5)
            self.assertEqual(statistics["num_train_windows"], 5)
            self.assertEqual(statistics["num_eval_rows"], 5)
            self.assertEqual(statistics["core_filter_iterations"], 3)
            self.assertEqual(
                metadata["implementation"]["protocol_version"],
                "kuairand-1k-sequential-v2",
            )
            self.assertEqual(len(metadata["implementation"]["preprocessor_sha256"]), 64)

            item_map = pd.read_csv(processor.item_id_map_csv())
            self.assertEqual(
                item_map["original_item_id"].tolist(), list(range(100, 105))
            )
            self.assertEqual(item_map["dense_item_id"].tolist(), list(range(5)))
            self.assertEqual(item_map["model_item_id"].tolist(), list(range(1, 6)))

            train = pd.read_csv(processor.train_format_csv())
            evaluation = pd.read_csv(processor.eval_format_csv())
            self.assertEqual(train.iloc[0].sequence_item_ids, "0,1,2,3")
            self.assertEqual(evaluation.iloc[0].sequence_item_ids, "0,1,2,3,4")
            self.assertEqual(evaluation.iloc[0].sequence_ratings, "5,5,5,5,5")

            checksum_lines = Path(processor.checksums_file()).read_text().splitlines()
            self.assertEqual(len(checksum_lines), 4)
            for line in checksum_lines:
                digest, filename = line.split("  ", 1)
                self.assertEqual(digest, _sha256(data_root / "kuairand-1k" / filename))

            metadata_path = Path(processor.metadata_json())
            original_metadata = metadata_path.read_text(encoding="utf-8")
            stripped_metadata = json.loads(original_metadata)
            del stripped_metadata["artifacts"]
            metadata_path.write_text(json.dumps(stripped_metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata schema mismatch"):
                processor.load_metadata()
            metadata_path.write_text(original_metadata, encoding="utf-8")

            original_checksums = Path(processor.checksums_file()).read_text(
                encoding="utf-8"
            )
            Path(processor.checksums_file()).write_text(
                "0" * 64 + "  metadata.json\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                processor.load_metadata()
            Path(processor.checksums_file()).write_text(
                original_checksums, encoding="utf-8"
            )

            official_processor = KuaiRandDataProcessor(
                archive_path=str(archive_path),
                output_root=str(data_root),
                validate_official=True,
            )
            with self.assertRaisesRegex(ValueError, "unexpected KuaiRand statistic"):
                official_processor.load_metadata()

            with mock.patch(
                "generative_recommenders.research.data.reco_dataset."
                "get_common_preprocessors",
                return_value={"kuairand-1k": processor},
            ):
                dataset = get_reco_dataset(
                    dataset_name="kuairand-1k",
                    max_sequence_length=KUAI_RAND_MAX_SEQUENCE_LENGTH,
                    chronological=True,
                )
            self.assertEqual(dataset.num_unique_items, 5)
            self.assertEqual(dataset.max_item_id, 5)
            self.assertEqual(dataset.all_item_ids, [1, 2, 3, 4, 5])
            self.assertEqual(len(dataset.train_dataset), 5)
            self.assertEqual(len(dataset.eval_dataset), 5)
            train_sample = dataset.train_dataset[0]
            eval_sample = dataset.eval_dataset[0]
            self.assertEqual(train_sample["historical_ids"][:3].tolist(), [1, 2, 3])
            self.assertEqual(int(train_sample["target_ids"]), 4)
            self.assertEqual(eval_sample["historical_ids"][:4].tolist(), [1, 2, 3, 4])
            self.assertEqual(int(eval_sample["target_ids"]), 5)

            with Path(processor.train_format_csv()).open(
                "a", encoding="utf-8"
            ) as output:
                output.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                processor.load_metadata()

    def test_dataset_uses_literal_sequence_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "malicious.csv"
            payload = "__import__('os').system('false')"
            pd.DataFrame(
                [
                    {
                        "user_id": 1,
                        "sequence_item_ids": payload,
                        "sequence_ratings": "5,5",
                        "sequence_timestamps": "1,2",
                    }
                ]
            ).to_csv(path, index=False)
            dataset = DatasetV2(
                ratings_file=str(path),
                padding_length=2,
                ignore_last_n=0,
                chronological=True,
            )
            with self.assertRaises((SyntaxError, ValueError)):
                dataset[0]

    def test_dataset_accepts_legacy_float_ratings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy.csv"
            pd.DataFrame(
                [
                    {
                        "user_id": 1,
                        "sequence_item_ids": "1,2",
                        "sequence_ratings": "3.5,5.0",
                        "sequence_timestamps": "1.0,2.0",
                    }
                ]
            ).to_csv(path, index=False)
            dataset = DatasetV2(
                ratings_file=str(path),
                padding_length=2,
                ignore_last_n=0,
                chronological=True,
            )
            sample = dataset[0]
            self.assertEqual(sample["historical_ratings"].tolist(), [3])
            self.assertEqual(int(sample["target_ratings"]), 5)

            dataset.ratings_frame.loc[0, "sequence_ratings"] = "'bad',5.0"
            dataset._cache.clear()
            with self.assertRaisesRegex(ValueError, "finite numbers"):
                dataset[0]

    def test_seen_filter_preserves_repeated_target(self) -> None:
        past_ids = torch.tensor([[1, 2, 1, 0], [3, 4, 5, 0]])
        targets = torch.tensor([[1], [5]])
        self.assertTrue(
            torch.equal(
                _seen_ids_excluding_target(past_ids, targets),
                torch.tensor([[0, 2, 0, 0], [3, 4, 0, 0]]),
            )
        )

    def test_candidate_filter_keeps_repeated_target_rankable(self) -> None:
        item_ids = torch.tensor([[1, 2, 3, 4]])
        item_embeddings = torch.tensor([[[4.0], [3.0], [2.0], [1.0]]])
        index = CandidateIndex(ids=item_ids, embeddings=item_embeddings)
        top_k = MIPSBruteForceTopK(
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
        invalid_ids = _seen_ids_excluding_target(
            past_ids=torch.tensor([[1, 2, 1]]),
            target_ids=torch.tensor([[1]]),
        )
        ranked_ids, _, _ = index.get_top_k_outputs(
            query_embeddings=torch.tensor([[1.0]]),
            k=2,
            top_k_module=top_k,
            invalid_ids=invalid_ids,
        )
        self.assertEqual(ranked_ids.tolist(), [[1, 3]])

    def test_preprocess_selector_preserves_legacy_default(self) -> None:
        self.assertEqual(_selected_datasets(None), list(LEGACY_DATASETS))
        self.assertEqual(_selected_datasets(["kuairand-1k"]), ["kuairand-1k"])
        self.assertEqual(_selected_datasets(["all"]), list(SELECTABLE_DATASETS))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _selected_datasets(["all", "ml-1m"])


if __name__ == "__main__":
    unittest.main()
