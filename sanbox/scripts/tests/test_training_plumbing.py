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
    _slurm_provenance,
    _wandb_finish,
    _wandb_initialize,
    _wandb_log,
    _wandb_requirement,
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


class SlurmProvenanceTest(unittest.TestCase):
    def test_all_six_canonical_tasks_are_recorded(self) -> None:
        expected_runs = (
            (42, "hstu"),
            (42, "safa"),
            (43, "hstu"),
            (43, "safa"),
            (44, "hstu"),
            (44, "safa"),
        )
        for task_id, (seed, arm) in enumerate(expected_runs):
            with self.subTest(task_id=task_id):
                environment = {
                    "GR_REQUIRE_SLURM_PROVENANCE": "1",
                    "SLURM_ARRAY_JOB_ID": "123456",
                    "SLURM_ARRAY_TASK_ID": str(task_id),
                    "SLURM_JOB_ID": str(123456 + task_id),
                    "SLURM_JOB_QOS": "h200_mrs_shared",
                    "SLURM_RESTART_COUNT": "0",
                    "SLURM_JOB_PARTITION": "h200",
                }
                with mock.patch.dict(os.environ, environment, clear=True):
                    provenance = _slurm_provenance(
                        attention_mode=arm,
                        random_seed=seed,
                    )
                self.assertEqual(
                    provenance,
                    {
                        "slurm_array_job_id": "123456",
                        "slurm_array_task_id": task_id,
                        "slurm_job_id": str(123456 + task_id),
                        "slurm_job_qos": "h200_mrs_shared",
                        "slurm_restart_count": 0,
                        "slurm_job_partition": "h200",
                    },
                )

    def test_required_provenance_rejects_missing_invalid_or_mismatched_values(
        self,
    ) -> None:
        valid = {
            "GR_REQUIRE_SLURM_PROVENANCE": "1",
            "SLURM_ARRAY_JOB_ID": "123456",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_JOB_ID": "123456",
            "SLURM_JOB_QOS": "h200_mrs_shared",
            "SLURM_RESTART_COUNT": "0",
            "SLURM_JOB_PARTITION": "h200",
        }
        mutations = (
            ("missing", "SLURM_JOB_ID", None),
            ("array ID", "SLURM_ARRAY_JOB_ID", "0"),
            ("task ID", "SLURM_ARRAY_TASK_ID", "6"),
            ("job ID", "SLURM_JOB_ID", "123_0"),
            ("QoS", "SLURM_JOB_QOS", "h200_dev"),
            ("restart", "SLURM_RESTART_COUNT", "-1"),
            ("partition", "SLURM_JOB_PARTITION", "debug"),
        )
        for label, key, value in mutations:
            with self.subTest(label=label):
                environment = dict(valid)
                if value is None:
                    del environment[key]
                else:
                    environment[key] = value
                with mock.patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(ValueError):
                        _slurm_provenance(attention_mode="hstu", random_seed=42)

        with mock.patch.dict(os.environ, valid, clear=True):
            with self.assertRaisesRegex(ValueError, "does not match"):
                _slurm_provenance(attention_mode="safa", random_seed=42)

    def test_scheduler_provenance_is_optional_outside_full_arrays(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"GR_REQUIRE_SLURM_PROVENANCE": "0"},
            clear=True,
        ):
            self.assertEqual(
                _slurm_provenance(attention_mode="hstu", random_seed=42), {}
            )


class RequiredWandbTest(unittest.TestCase):
    def test_requirement_flag_is_strict_and_requires_online_enabled_run(self) -> None:
        with mock.patch.dict(os.environ, {"GR_REQUIRE_WANDB": "1"}, clear=True):
            self.assertTrue(_wandb_requirement(wandb_enabled=True, wandb_mode="online"))
            with self.assertRaisesRegex(ValueError, "wandb_enabled"):
                _wandb_requirement(wandb_enabled=False, wandb_mode="online")
            with self.assertRaisesRegex(ValueError, "WANDB_MODE"):
                _wandb_requirement(wandb_enabled=True, wandb_mode="offline")
        with mock.patch.dict(os.environ, {"GR_REQUIRE_WANDB": "yes"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                _wandb_requirement(wandb_enabled=True, wandb_mode="online")

    def test_required_initialization_fails_closed(self) -> None:
        module_path = "generative_recommenders.research.trainer.train.wandb"
        with mock.patch(module_path, None):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                _wandb_initialize(required=True, init_kwargs={})

        failing_wandb = mock.Mock()
        failing_wandb.init.side_effect = OSError("network unavailable")
        with mock.patch(module_path, failing_wandb):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                _wandb_initialize(required=True, init_kwargs={})

        empty_wandb = mock.Mock()
        empty_wandb.init.return_value = None
        with mock.patch(module_path, empty_wandb):
            with self.assertRaisesRegex(RuntimeError, "returned no run"):
                _wandb_initialize(required=True, init_kwargs={})

    def test_required_logging_and_finalization_fail_closed(self) -> None:
        run = mock.Mock()
        run.log.side_effect = OSError("logging failed")
        with self.assertRaisesRegex(RuntimeError, "logging failed"):
            _wandb_log(run, {"metric": 1.0}, 7, required=True)
        _wandb_log(run, {"metric": 1.0}, 7, required=False)

        run.finish.side_effect = OSError("finish failed")
        with self.assertRaisesRegex(RuntimeError, "finalization failed"):
            _wandb_finish(run, required=True)
        _wandb_finish(run, required=False)


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
        self.assertIn("GR_REQUIRE_SLURM_PROVENANCE=1", wrapper)
        self.assertIn("GR_REQUIRE_WANDB=1", wrapper)
        self.assertIn("unset WANDB_MODE WANDB_DISABLED", wrapper)
        self.assertIn("WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID", wrapper)
        self.assertNotIn("master_port", wrapper)
        self.assertNotIn("SLURM_ARRAY_JOB_ID:-0", wrapper)

    def test_preflight_pins_canonical_experiment_identities(self) -> None:
        qualifier = (REPO_ROOT / "scripts/qualify_safa.sh").read_text(encoding="utf-8")
        self.assertIn("GR_CONFIG_IDENTITY_ONLY=1", qualifier)
        self.assertIn("experiment_config_ml-1m=", qualifier)
        self.assertIn("experiment_config_ml-20m=", qualifier)
        self.assertIn("unset WANDB_MODE WANDB_DISABLED", qualifier)
        self.assertIn("WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID", qualifier)
        self.assertIn('scontrol show job "$SLURM_JOB_ID"', qualifier)
        self.assertIn('actual_partition="$(sed -n', qualifier)
        self.assertIn('"$actual_partition" != "h200"', qualifier)
        self.assertIn("qualification_job_qos=$actual_qos", qualifier)
        self.assertIn("qualification_job_partition=$actual_partition", qualifier)

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
        self.assertEqual(submitter.count("--expected-array-job-id"), 2)
        self.assertIn("GR_CODE_SNAPSHOT=$snapshot", submitter)
        self.assertNotIn("WANDB_API_KEY=", submitter)

    def test_preflight_wrapper_reaches_exported_snapshot_when_spooled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            snapshot = temporary_root / "snapshot"
            scripts = snapshot / "scripts"
            scripts.mkdir(parents=True)
            (scripts / "qualify_safa.sh").write_text(
                "#!/usr/bin/env bash\nprintf 'snapshot=%s\\n' \"$GR_CODE_SNAPSHOT\"\n",
                encoding="utf-8",
            )
            spool = temporary_root / "spool"
            spool.mkdir()
            spooled_wrapper = spool / "slurm_script"
            spooled_wrapper.write_bytes(
                (REPO_ROOT / "scripts/sbatch_qualify_safa.sh").read_bytes()
            )

            completed = subprocess.run(
                ["/bin/bash", str(spooled_wrapper)],
                check=False,
                env={**os.environ, "GR_CODE_SNAPSHOT": str(snapshot)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, f"snapshot={snapshot}\n")

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
            spool = temporary_root / "spool"
            spool.mkdir()
            spooled_wrapper = spool / "slurm_script"
            spooled_wrapper.write_bytes(
                (snapshot / "scripts/sbatch_safa_ab.sh").read_bytes()
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
                        "qualification_job_id=998",
                        "qualification_job_qos=h200_mrs_shared",
                        "qualification_job_partition=h200",
                        "qualification_restart_count=0",
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
    "require_slurm": os.environ["GR_REQUIRE_SLURM_PROVENANCE"],
    "require_wandb": os.environ["GR_REQUIRE_WANDB"],
    "slurm_array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
    "slurm_array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_job_qos": os.environ["SLURM_JOB_QOS"],
    "slurm_restart_count": os.environ["SLURM_RESTART_COUNT"],
    "slurm_job_partition": os.environ["SLURM_JOB_PARTITION"],
    "wandb_mode": os.environ.get("WANDB_MODE"),
    "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
    "wandb_resume": os.environ.get("WANDB_RESUME"),
    "wandb_sweep_id": os.environ.get("WANDB_SWEEP_ID"),
}), encoding="utf-8")
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_bin = temporary_root / "bin"
            fake_bin.mkdir()
            fake_scontrol = fake_bin / "scontrol"
            fake_scontrol.write_text(
                "#!/usr/bin/env bash\n"
                'task_id="${3#*_}"\n'
                'job_id="$((1000 + task_id))"\n'
                "printf 'JobId=%s ArrayJobId=999 ArrayTaskId=%s "
                "Partition=h200 Restarts=0 QOS=h200_mrs_shared\\n' "
                '"$job_id" "$task_id"\n',
                encoding="utf-8",
            )
            fake_scontrol.chmod(0o755)

            for task_id, (seed, arm) in enumerate(expected_tasks):
                with self.subTest(task_id=task_id):
                    capture = temporary_root / f"capture-{task_id}.json"
                    environment = {
                        **os.environ,
                        "PATH": f"{fake_bin}:{os.environ['PATH']}",
                        "CAPTURE_PATH": str(capture),
                        "GR_PYTHON": str(fake_python),
                        "GR_CODE_SNAPSHOT": str(snapshot),
                        "GR_EXPECTED_SOURCE_MANIFEST": provenance["source_manifest"],
                        "GR_DATASET": "ml-1m",
                        "GR_DATA_ROOT": str(temporary_root / "data"),
                        "GR_EXPS_ROOT": str(temporary_root / "exps"),
                        "GR_CKPTS_ROOT": str(temporary_root / "ckpts"),
                        "GR_QUALIFICATION_ROOT": str(qualification_root),
                        "GR_WANDB_ENABLED": "0",
                        "SLURM_JOB_QOS": "h200_mrs_shared",
                        "SLURM_JOB_PARTITION": "h200",
                        "SLURM_JOB_ID": str(1000 + task_id),
                        "SLURM_ARRAY_JOB_ID": "999",
                        "SLURM_ARRAY_TASK_ID": str(task_id),
                        "WANDB_RUN_ID": "stale-run",
                        "WANDB_RESUME": "must",
                        "WANDB_SWEEP_ID": "stale-sweep",
                    }
                    subprocess.run(
                        [
                            "/bin/bash",
                            str(spooled_wrapper),
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
                    expected_run_name = f"safa-ab-ml-1m-{arm}-seed{seed}-999-r0"
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
                    self.assertEqual(captured["require_slurm"], "1")
                    self.assertEqual(captured["require_wandb"], "1")
                    self.assertEqual(captured["slurm_array_job_id"], "999")
                    self.assertEqual(captured["slurm_array_task_id"], str(task_id))
                    self.assertEqual(captured["slurm_job_id"], str(1000 + task_id))
                    self.assertEqual(captured["slurm_job_qos"], "h200_mrs_shared")
                    self.assertEqual(captured["slurm_restart_count"], "0")
                    self.assertEqual(captured["slurm_job_partition"], "h200")
                    self.assertIsNone(captured["wandb_mode"])
                    self.assertIsNone(captured["wandb_run_id"])
                    self.assertIsNone(captured["wandb_resume"])
                    self.assertIsNone(captured["wandb_sweep_id"])

            restart_capture = temporary_root / "capture-restart.json"
            fake_scontrol.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' "
                "'JobId=1005 ArrayJobId=999 ArrayTaskId=5 Partition=h200 "
                "Restarts=2 QOS=h200_mrs_shared'\n",
                encoding="utf-8",
            )
            restarted_environment = {
                **environment,
                "CAPTURE_PATH": str(restart_capture),
                "SLURM_RESTART_COUNT": "2",
            }
            subprocess.run(
                ["/bin/bash", str(spooled_wrapper)],
                check=True,
                env=restarted_environment,
                capture_output=True,
                text=True,
            )
            restarted = json.loads(restart_capture.read_text(encoding="utf-8"))
            self.assertTrue(restarted["experiment_name"].endswith("-999-r2"))
            self.assertEqual(restarted["wandb_name"], restarted["experiment_name"])
            self.assertEqual(restarted["slurm_restart_count"], "2")

            for key, value, message in (
                ("SLURM_RESTART_COUNT", "1", "SLURM_RESTART_COUNT disagrees"),
                ("SLURM_JOB_PARTITION", "debug", "SLURM_JOB_PARTITION disagrees"),
            ):
                with self.subTest(mismatched_scheduler_field=key):
                    mismatched_environment = {
                        **restarted_environment,
                        key: value,
                    }
                    mismatched = subprocess.run(
                        ["/bin/bash", str(spooled_wrapper)],
                        check=False,
                        env=mismatched_environment,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(mismatched.returncode, 0)
                    self.assertIn(message, mismatched.stderr)

            fake_scontrol.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' "
                "'JobId=1005 ArrayJobId=999 ArrayTaskId=5 Partition=h200 "
                "Restarts=0 QOS=h200_dev'\n",
                encoding="utf-8",
            )
            rejected_environment = {
                **environment,
                "SLURM_JOB_QOS": "h200_dev",
                "SLURM_RESTART_COUNT": "0",
            }
            rejected = subprocess.run(
                ["/bin/bash", str(spooled_wrapper)],
                check=False,
                env=rejected_environment,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("refusing experiment on h200_dev", rejected.stderr)


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
