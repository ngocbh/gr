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
Single-process (no ``mp.spawn``) launcher for the research trainer — for
interactive debugging.

Unlike ``main.py`` (which spawns one subprocess per GPU via ``mp.spawn``), this
runs ``train_fn`` directly in the current process with ``world_size=1``. That
means:
  * ``logging.info`` output actually shows up in your terminal;
  * ``breakpoint()`` / ``pdb`` / ``ipdb`` work;
  * exceptions give you a normal traceback instead of a mangled spawn error.

Run it ON A GPU (e.g. inside `srun --pty ... bash`); ``scripts/train_debug.sh``
wraps this with conda activation, env vars, and an interactive GPU allocation.

Usage:
  python -u -m generative_recommenders.research.trainer.debug_train \\
      --gin_config_file=configs/ml-20m/hstu-probe-dynamics.gin \\
      --gin_bindings=train_fn.num_epochs=1 \\
      --gin_bindings=create_data_loader.num_workers=0   # to pdb the data path
      --master_port=12355

Tips:
  * Add ``breakpoint()`` anywhere in the model/trainer and it will drop to pdb.
  * ``create_data_loader.num_workers=0`` keeps data loading in-process so you can
    step into the dataset too.
  * ``train_fn`` already enables ``torch.autograd.set_detect_anomaly(True)``, so
    NaN/Inf in the backward pass raises with the offending op's traceback.
"""

import logging
import os
import sys
from typing import List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # hide noisy tensorflow logs

import fbgemm_gpu  # noqa: F401  (registers torch.ops.fbgemm.* used by the model)
import gin
import torch
from absl import app, flags
from generative_recommenders.research.trainer.train import train_fn

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def _delete_flags(FLAGS, keys_to_delete: List[str]) -> None:  # pyre-ignore [2]
    """Drop flags if a transitive import already defined them (avoids
    absl DuplicateFlagError), mirroring main.py."""
    for key in [k for k in FLAGS._flags()]:
        if key in keys_to_delete:
            delattr(FLAGS, key)


_delete_flags(flags.FLAGS, ["gin_config_file", "master_port", "gin_bindings"])
flags.DEFINE_string("gin_config_file", None, "Path to the gin config file.")
flags.DEFINE_integer("master_port", 12355, "Master port for the 1-proc group.")
flags.DEFINE_multi_string(
    "gin_bindings",
    [],
    "Additional gin bindings, e.g. --gin_bindings=train_fn.num_epochs=1",
)
FLAGS = flags.FLAGS  # pyre-ignore [5]


def _main(argv) -> None:  # pyre-ignore [2]
    del argv
    if FLAGS.gin_config_file is not None:
        logging.info(f"Loading gin config from {FLAGS.gin_config_file}")
        gin.parse_config_file(FLAGS.gin_config_file)
    if FLAGS.gin_bindings:
        logging.info(f"Applying gin bindings: {list(FLAGS.gin_bindings)}")
        gin.parse_config(FLAGS.gin_bindings)

    assert torch.cuda.is_available(), (
        "No CUDA device visible. Run this on a GPU node, e.g.:\n"
        "  srun --partition=h200 --qos=h200_mrs_shared --gres=gpu:h200:1 "
        "--cpus-per-task=16 --mem=64G --time=04:00:00 --pty bash\n"
        "or just use scripts/train_debug.sh (it allocates one for you)."
    )
    logging.info(f"Running train_fn in-process on {torch.cuda.get_device_name(0)}")
    # rank=0, world_size=1, no mp.spawn -> single process, DDP over 1 rank.
    train_fn(rank=0, world_size=1, master_port=FLAGS.master_port)


def main() -> None:
    app.run(_main)


if __name__ == "__main__":
    main()
