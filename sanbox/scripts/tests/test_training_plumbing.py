#!/usr/bin/env python3

import random
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import gin
import numpy as np
import torch
from generative_recommenders.research.data.eval import _avg, add_to_summary_writer
from generative_recommenders.research.trainer.data_loader import create_data_loader
from generative_recommenders.research.trainer.train import (
    _config_identities,
    _seed_everything,
    cleanup,
)
from scripts.qualify_safa_results import operative_config_identities
from scripts.snapshot import create_snapshot


REPO_ROOT = Path(__file__).resolve().parents[2]


class DeterminismTest(unittest.TestCase):
    def test_seed_everything_covers_python_numpy_and_torch(self) -> None:
        _seed_everything(43)
        first = (random.random(), np.random.random(), torch.rand(3))
        _seed_everything(43)
        second = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_sampler_order_is_reproducible_for_seed_and_epoch(self) -> None:
        dataset = torch.utils.data.TensorDataset(torch.arange(31))
        first_sampler, first_loader = create_data_loader(
            dataset,
            batch_size=4,
            world_size=1,
            rank=0,
            shuffle=True,
            num_workers=0,
            seed=44,
        )
        second_sampler, second_loader = create_data_loader(
            dataset,
            batch_size=4,
            world_size=1,
            rank=0,
            shuffle=True,
            num_workers=0,
            seed=44,
        )
        self.assertIsNotNone(first_sampler)
        self.assertIsNotNone(second_sampler)
        first_sampler.set_epoch(7)
        second_sampler.set_epoch(7)
        first_order = torch.cat([batch[0] for batch in first_loader])
        second_order = torch.cat([batch[0] for batch in second_loader])
        torch.testing.assert_close(first_order, second_order, rtol=0, atol=0)


class SingleProcessTest(unittest.TestCase):
    def test_cleanup_without_process_group_is_a_noop(self) -> None:
        with mock.patch(
            "generative_recommenders.research.trainer.train.dist.is_initialized",
            return_value=False,
        ), mock.patch(
            "generative_recommenders.research.trainer.train.dist.destroy_process_group"
        ) as destroy:
            cleanup()
        destroy.assert_not_called()

    def test_single_world_metrics_do_not_use_distributed_collectives(self) -> None:
        metrics = {"ndcg@10": torch.tensor([0.1, 0.3])}
        with mock.patch(
            "generative_recommenders.research.data.eval.dist.all_reduce"
        ) as all_reduce:
            average = _avg(metrics["ndcg@10"], world_size=1)
            add_to_summary_writer(
                writer=None,
                batch_id=0,
                metrics=metrics,
                prefix="eval",
                world_size=1,
            )
        all_reduce.assert_not_called()
        torch.testing.assert_close(average, torch.tensor(0.2))


class GinBindingsTest(unittest.TestCase):
    def tearDown(self) -> None:
        gin.clear_config()

    def test_repeated_cli_bindings_override_file_in_order(self) -> None:
        sys.modules.setdefault("fbgemm_gpu", types.ModuleType("fbgemm_gpu"))
        import main as training_main

        with tempfile.TemporaryDirectory() as temporary_directory:
            config = Path(temporary_directory) / "test.gin"
            config.write_text("train_fn.random_seed = 42\n", encoding="utf-8")
            with mock.patch.object(training_main, "train_fn") as train_fn:
                training_main.mp_train_fn(
                    rank=0,
                    world_size=1,
                    master_port=12345,
                    gin_config_file=str(config),
                    gin_bindings=[
                        "train_fn.random_seed=43",
                        "train_fn.random_seed=44",
                    ],
                )

            train_fn.assert_called_once_with(0, 1, 12345)
            self.assertEqual(gin.query_parameter("train_fn.random_seed"), 44)


class GinConfigIdentityTest(unittest.TestCase):
    @staticmethod
    def _config(mode: str, seed: int, learning_rate: float = 0.001) -> str:
        return (
            "# Parameters for hstu_encoder:\n"
            f"hstu_encoder.attention_mode = '{mode}'\n"
            "# Parameters for train_fn:\n"
            f"train_fn.learning_rate = {learning_rate}\n"
            f"train_fn.random_seed = {seed}\n"
        )

    def test_trainer_and_qualifier_share_config_identity_contract(self) -> None:
        experiment_identities = set()
        exact_identities = set()
        for mode in ("hstu", "safa"):
            for seed in (42, 43, 44):
                config = self._config(mode, seed)
                trainer_identity = _config_identities(
                    config, attention_mode=mode, random_seed=seed
                )
                qualifier_identity = operative_config_identities(
                    config, attention_mode=mode, random_seed=seed
                )
                self.assertEqual(trainer_identity, qualifier_identity)
                exact_identities.add(trainer_identity[0])
                experiment_identities.add(trainer_identity[1])

        self.assertEqual(len(exact_identities), 6)
        self.assertEqual(len(experiment_identities), 1)
        drifted_identity = _config_identities(
            self._config("hstu", 42, learning_rate=0.002),
            attention_mode="hstu",
            random_seed=42,
        )[1]
        self.assertNotIn(drifted_identity, experiment_identities)


class SlurmContractTest(unittest.TestCase):
    def test_full_array_is_one_shared_h200_per_paired_task(self) -> None:
        wrapper = (REPO_ROOT / "scripts/sbatch_safa_ab.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=h200", wrapper)
        self.assertIn("#SBATCH --qos=h200_mrs_shared", wrapper)
        self.assertIn("#SBATCH --gres=gpu:h200:1", wrapper)
        self.assertIn("#SBATCH --array=0-5", wrapper)
        self.assertIn("seeds=(42 43 44)", wrapper)
        self.assertIn("SLURM_ARRAY_TASK_ID / 2", wrapper)
        self.assertIn("SLURM_ARRAY_TASK_ID % 2", wrapper)
        self.assertIn('== "h200_dev"', wrapper)
        self.assertIn("GR_EXPECTED_EXPERIMENT_CONFIG_SHA256", wrapper)
        self.assertNotIn("master_port", wrapper)
        self.assertNotIn("SLURM_ARRAY_JOB_ID:-0", wrapper)

    def test_preflight_pins_canonical_experiment_identities(self) -> None:
        qualifier = (REPO_ROOT / "scripts/qualify_safa.sh").read_text(encoding="utf-8")
        self.assertIn("GR_CONFIG_IDENTITY_ONLY=1", qualifier)
        self.assertIn("experiment_config_ml-1m=", qualifier)
        self.assertIn("experiment_config_ml-20m=", qualifier)

    def test_submission_is_gated_on_qualification_for_both_datasets(self) -> None:
        submitter = (REPO_ROOT / "scripts/submit_safa_ab.sh").read_text(
            encoding="utf-8"
        )
        self.assertEqual(submitter.count('dependency="afterok:$qualification_job"'), 2)
        self.assertIn("GR_DATASET=ml-1m", submitter)
        self.assertIn("GR_DATASET=ml-20m", submitter)
        self.assertIn("--job-name=safa-ab-ml1m", submitter)
        self.assertIn("--job-name=safa-ab-ml20m", submitter)
        self.assertIn("postrun_ml1m=", submitter)
        self.assertIn("postrun_ml20m=", submitter)
        self.assertEqual(submitter.count("--expected-experiment-config-sha256"), 2)
        self.assertNotIn("WANDB_API_KEY=", submitter)

    def test_array_task_map_reaches_pinned_snapshot_launcher(self) -> None:
        expected_tasks = (
            (42, "hstu"),
            (42, "safa"),
            (43, "hstu"),
            (43, "safa"),
            (44, "hstu"),
            (44, "safa"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            snapshot = temporary_root / "snapshot"
            provenance = create_snapshot(
                REPO_ROOT,
                snapshot,
                commit_id="a" * 40,
                tree_id="b" * 40,
            )
            qualification_root = temporary_root / "qualifications"
            qualification_root.mkdir()
            marker = qualification_root / f"{provenance['source_manifest']}.passed"
            marker.write_text(
                "\n".join(
                    (
                        "status=passed",
                        "qualification_scope=preflight_only",
                        f"source_commit={provenance['source_commit']}",
                        f"source_tree={provenance['source_tree']}",
                        f"source_manifest={provenance['source_manifest']}",
                        f"experiment_config_ml-1m={'d' * 64}",
                        f"experiment_config_ml-20m={'e' * 64}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            fake_python = temporary_root / "python"
            fake_python.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["CAPTURE_PATH"]).write_text(json.dumps({
    "argv": sys.argv,
    "experiment_name": os.environ["GR_EXPERIMENT_NAME"],
    "wandb_name": os.environ["WANDB_NAME"],
    "wandb_group": os.environ["WANDB_RUN_GROUP"],
    "wandb_tags": os.environ["WANDB_TAGS"],
    "expected_experiment_config": os.environ["GR_EXPECTED_EXPERIMENT_CONFIG_SHA256"],
}), encoding="utf-8")
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            for task_id, (seed, arm) in enumerate(expected_tasks):
                with self.subTest(task_id=task_id):
                    capture = temporary_root / f"capture-{task_id}.json"
                    environment = {
                        **os.environ,
                        "CAPTURE_PATH": str(capture),
                        "GR_PYTHON": str(fake_python),
                        "GR_EXPECTED_SOURCE_MANIFEST": provenance["source_manifest"],
                        "GR_DATASET": "ml-1m",
                        "GR_DATA_ROOT": str(temporary_root / "data"),
                        "GR_EXPS_ROOT": str(temporary_root / "exps"),
                        "GR_CKPTS_ROOT": str(temporary_root / "ckpts"),
                        "GR_QUALIFICATION_ROOT": str(qualification_root),
                        "GR_WANDB_ENABLED": "0",
                        "SLURM_JOB_QOS": "h200_mrs_shared",
                        "SLURM_JOB_ID": "1000",
                        "SLURM_ARRAY_JOB_ID": "999",
                        "SLURM_ARRAY_TASK_ID": str(task_id),
                    }
                    subprocess.run(
                        [
                            "/bin/bash",
                            str(snapshot / "scripts/sbatch_safa_ab.sh"),
                        ],
                        check=True,
                        env=environment,
                        capture_output=True,
                        text=True,
                    )
                    captured = json.loads(capture.read_text(encoding="utf-8"))
                    arguments = captured["argv"]
                    expected_config = (
                        "hstu-matched-sampled-softmax-n128-large-final.gin"
                        if arm == "hstu"
                        else "safa-sampled-softmax-n128-large-final.gin"
                    )
                    self.assertTrue(arguments[2].endswith(expected_config))
                    self.assertIn(
                        f"--gin_bindings=train_fn.random_seed={seed}", arguments
                    )
                    self.assertNotIn("train_fn.wandb_run_name", " ".join(arguments))
                    expected_run_name = f"safa-ab-ml-1m-{arm}-seed{seed}-999"
                    self.assertEqual(captured["experiment_name"], expected_run_name)
                    self.assertEqual(captured["wandb_name"], expected_run_name)
                    self.assertEqual(
                        captured["wandb_group"],
                        f"safa-ab-ml-1m-{provenance['source_manifest'][:12]}",
                    )
                    self.assertEqual(
                        captured["wandb_tags"],
                        f"safa-ab,ml-1m,{arm},seed-{seed},parameter-matched",
                    )
                    self.assertEqual(captured["expected_experiment_config"], "d" * 64)


class TrainLauncherTest(unittest.TestCase):
    def test_external_roots_and_repeated_bindings_reach_training_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            fake_python = temporary_root / "python"
            capture = temporary_root / "capture.json"
            fake_python.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

Path(os.environ["CAPTURE_PATH"]).write_text(json.dumps({
    "argv": sys.argv,
    "cwd": os.getcwd(),
    "data_target": str(Path("tmp").resolve()),
    "pythonhashseed": os.environ["PYTHONHASHSEED"],
    "data_root": os.environ["GR_DATA_ROOT"],
    "exps_root": os.environ["GR_EXPS_ROOT"],
    "ckpts_root": os.environ["GR_CKPTS_ROOT"],
}), encoding="utf-8")
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            data_root = temporary_root / "data"
            exps_root = temporary_root / "exps"
            ckpts_root = temporary_root / "ckpts"
            environment = {
                **os.environ,
                "CAPTURE_PATH": str(capture),
                "GR_PYTHON": str(fake_python),
                "GR_CONDA_ENV": "",
                "GR_DATA_ROOT": str(data_root),
                "GR_EXPS_ROOT": str(exps_root),
                "GR_CKPTS_ROOT": str(ckpts_root),
                "GR_WANDB_ENABLED": "0",
            }
            bindings = [
                "--gin_bindings=train_fn.random_seed=43",
                "--gin_bindings",
                "train_fn.random_seed=44",
                "--gin_bindings=train_fn.num_epochs=1",
            ]
            subprocess.run(
                [
                    "/bin/bash",
                    str(REPO_ROOT / "scripts/train.sh"),
                    str(
                        REPO_ROOT
                        / "configs/ml-1m/hstu-matched-sampled-softmax-n128-large-final.gin"
                    ),
                    *bindings,
                ],
                check=True,
                env=environment,
                capture_output=True,
                text=True,
            )
            captured = json.loads(capture.read_text(encoding="utf-8"))
            self.assertEqual(captured["data_target"], str(data_root.resolve()))
            self.assertEqual(captured["pythonhashseed"], "44")
            self.assertEqual(captured["data_root"], str(data_root))
            self.assertEqual(captured["exps_root"], str(exps_root))
            self.assertEqual(captured["ckpts_root"], str(ckpts_root))
            self.assertEqual(captured["argv"][-4:], bindings)
            self.assertEqual(captured["argv"][1], str(REPO_ROOT / "main.py"))


if __name__ == "__main__":
    unittest.main()
