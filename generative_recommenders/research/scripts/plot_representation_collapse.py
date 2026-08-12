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
Plot representation collapse across depth, in the style of Figure 3 of
"Hyper-Connections" (Zhu et al., 2024, https://arxiv.org/abs/2409.19606):

    y = cos(h^i, h^{i+1})   (cosine sim between a block's input and its output,
                             i.e. consecutive residual-stream states)
    x = layer index i
    solid line = median over tokens, shaded band = 5th-95th percentile.

Collapse shows up as the median rising toward 1.0 with depth (deep blocks stop
changing the representation). This reuses the same quantity the training probe
logs as ``probe/cos_in_out/layer_i``, but here we snapshot the *distribution*
over tokens from a trained checkpoint (eval mode, dropout off).

Rebuilds the model exactly as ``train_fn`` does (mirroring its construction),
loads the checkpoint, and runs a forward over a few eval batches with hooks on
each STU block. Supports overlaying multiple runs for comparison (e.g. vanilla
HSTU vs HSTU-mHC / hyper-connections).

Run on a GPU (see scripts/plot_collapse.sh). Examples:
  python -m generative_recommenders.research.scripts.plot_representation_collapse \\
      --gin_config_file=configs/ml-20m/hstu-probe-dynamics.gin \\
      --checkpoint=/checkpoints/ngocbh/longhstu/checkpoints/ml-20m-l200/..._last.pt \\
      --label="Residual (vanilla HSTU)" \\
      --output=/checkpoints/ngocbh/longhstu/plots/representation_collapse.png

  # overlay two runs (label,config,ckpt each):
  ... --run "Residual,configs/.../hstu-probe-dynamics.gin,/.../vanilla_last.pt" \\
      --run "mHC,configs/ml-1m/hstu-mhc-...gin,/.../mhc_last.pt"
"""

import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import fbgemm_gpu  # noqa: F401
import gin
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from absl import app, flags
from generative_recommenders.research.data.reco_dataset import get_reco_dataset
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    get_sequential_encoder,
)
from generative_recommenders.research.modeling.sequential.features import (
    movielens_seq_features_from_row,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.similarity_utils import (
    get_similarity_function,
)
from generative_recommenders.research.trainer.data_loader import create_data_loader

# Importing train_fn registers the @gin.configurable `train_fn` so the config's
# `train_fn.*` bindings parse and can be read back via gin.query_parameter.
from generative_recommenders.research.trainer.train import train_fn  # noqa: F401

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

flags.DEFINE_multi_string(
    "run",
    [],
    'A run to plot as "label,gin_config_file,checkpoint". Repeatable to overlay '
    "multiple runs. If omitted, --gin_config_file/--checkpoint/--label are used.",
)
flags.DEFINE_string("gin_config_file", None, "Config for the single-run case.")
flags.DEFINE_string("checkpoint", None, "Checkpoint for the single-run case.")
flags.DEFINE_string("label", "run", "Legend label for the single-run case.")
flags.DEFINE_integer("num_batches", 8, "Number of eval batches to aggregate.")
flags.DEFINE_integer("batch_size", 128, "Eval batch size.")
flags.DEFINE_string(
    "output",
    "/checkpoints/ngocbh/longhstu/plots/representation_collapse.png",
    "Output PNG path (a .npz with the raw stats is written alongside).",
)
FLAGS = flags.FLAGS

# Fig-3-style palette: red = residual/pre-norm (collapses), blue = the fix.
_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]


def _q(name: str, default=None):  # pyre-ignore [3,2]
    """Query a gin-bound train_fn parameter, falling back to train_fn's default."""
    try:
        return gin.query_parameter(f"train_fn.{name}")
    except (ValueError, KeyError):
        return default


def _build_model_and_loader(
    device: torch.device,
) -> Tuple[torch.nn.Module, object, int]:
    """Reconstruct the model + eval loader exactly as train_fn does."""
    dataset_name = _q("dataset_name", "ml-20m")
    max_sequence_length = _q("max_sequence_length", 200)
    item_embedding_dim = _q("item_embedding_dim", 240)
    main_module = _q("main_module", "HSTU")
    user_embedding_norm = _q("user_embedding_norm", "l2_norm")
    interaction_module_type = _q("interaction_module_type", "")
    gr_output_length = _q("gr_output_length", 10)
    dropout_rate = _q("dropout_rate", 0.2)

    dataset = get_reco_dataset(
        dataset_name=dataset_name,
        max_sequence_length=max_sequence_length,
        chronological=True,
    )
    embedding_module = LocalEmbeddingModule(
        num_items=dataset.max_item_id, item_embedding_dim=item_embedding_dim
    )
    interaction_module, _ = get_similarity_function(
        module_type=interaction_module_type,
        query_embedding_dim=item_embedding_dim,
        item_embedding_dim=item_embedding_dim,
    )
    output_postproc_module = (
        L2NormEmbeddingPostprocessor(embedding_dim=item_embedding_dim, eps=1e-6)
        if user_embedding_norm == "l2_norm"
        else LayerNormEmbeddingPostprocessor(embedding_dim=item_embedding_dim, eps=1e-6)
    )
    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=dataset.max_sequence_length + gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        dropout_rate=dropout_rate,
    )
    model = get_sequential_encoder(
        module_type=main_module,
        max_sequence_length=dataset.max_sequence_length,
        max_output_length=gr_output_length + 1,
        embedding_module=embedding_module,
        interaction_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        verbose=False,
    )
    model = model.to(device).eval()

    _, eval_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=FLAGS.batch_size,
        world_size=1,
        rank=0,
        shuffle=False,
        drop_last=False,
    )
    return model, eval_loader, gr_output_length


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    # Checkpoints are saved from a DDP-wrapped model -> strip the "module." prefix.
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logging.warning(f"  {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        logging.warning(f"  {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    logging.info(f"  loaded checkpoint (epoch={state.get('epoch', '?')})")


def _collect_cosines(
    model: torch.nn.Module, eval_loader: object, gr_output_length: int, device
) -> Dict[int, np.ndarray]:
    """Run forwards with hooks; return {layer_idx: 1D array of per-token cos}."""
    layers = model._hstu._attention_layers
    buckets: Dict[int, List[torch.Tensor]] = defaultdict(list)
    handles = []

    def make_hook(i: int):  # pyre-ignore [3]
        def hook(module, args, kwargs, output):  # pyre-ignore [2]
            x_in = kwargs.get("x", args[0] if args else None)
            x_out = output[0] if isinstance(output, tuple) else output
            cos = F.cosine_similarity(
                x_in.detach().float(), x_out.detach().float(), dim=-1
            )
            buckets[i].append(cos.cpu())

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i), with_kwargs=True))

    with torch.no_grad():
        for b, row in enumerate(iter(eval_loader)):
            if b >= FLAGS.num_batches:
                break
            seq_features, _, _ = movielens_seq_features_from_row(
                row, device=device, max_output_length=gr_output_length + 1
            )
            input_embeddings = model.get_item_embeddings(seq_features.past_ids)
            model(
                past_lengths=seq_features.past_lengths,
                past_ids=seq_features.past_ids,
                past_embeddings=input_embeddings,
                past_payloads=seq_features.past_payloads,
            )

    for h in handles:
        h.remove()
    return {i: torch.cat(v).numpy() for i, v in sorted(buckets.items())}


def _summarize(cos_by_layer: Dict[int, np.ndarray]):  # pyre-ignore [3]
    layers = sorted(cos_by_layer.keys())
    med = np.array([np.median(cos_by_layer[i]) for i in layers])
    p5 = np.array([np.percentile(cos_by_layer[i], 5) for i in layers])
    p95 = np.array([np.percentile(cos_by_layer[i], 95) for i in layers])
    return np.array(layers), med, p5, p95


def _main(argv) -> None:  # pyre-ignore [2]
    del argv
    assert torch.cuda.is_available(), "Run on a GPU (see scripts/plot_collapse.sh)."
    device = torch.device("cuda:0")

    # Resolve the list of (label, config, ckpt) runs to plot.
    runs: List[Tuple[str, str, str]] = []
    if FLAGS.run:
        for spec in FLAGS.run:
            parts = [p.strip() for p in spec.split(",")]
            assert len(parts) == 3, f'--run must be "label,config,ckpt"; got: {spec}'
            runs.append((parts[0], parts[1], parts[2]))
    else:
        assert FLAGS.gin_config_file and FLAGS.checkpoint, (
            "Provide --run, or both --gin_config_file and --checkpoint."
        )
        runs.append((FLAGS.label, FLAGS.gin_config_file, FLAGS.checkpoint))

    results = {}  # label -> (layers, med, p5, p95)
    for label, config, ckpt in runs:
        logging.info(f"[{label}] config={config}")
        # Each run may use a different config; reset + reparse gin between runs
        # (gin locks its config after the first configurable call, so unlock).
        with gin.unlock_config():
            gin.clear_config()
            gin.parse_config_file(config)
        model, eval_loader, gr_out = _build_model_and_loader(device)
        _load_checkpoint(model, ckpt)
        cos_by_layer = _collect_cosines(model, eval_loader, gr_out, device)
        layers, med, p5, p95 = _summarize(cos_by_layer)
        results[label] = (layers, med, p5, p95)
        logging.info(
            f"[{label}] median cos by layer: "
            + ", ".join(f"{l}:{m:.3f}" for l, m in zip(layers, med))
        )
        del model
        torch.cuda.empty_cache()

    # --- plot (Fig-3 style) ---
    plt.rcParams.update({"font.size": 12, "figure.dpi": 130})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (label, (layers, med, p5, p95)) in enumerate(results.items()):
        c = _COLORS[k % len(_COLORS)]
        ax.plot(layers, med, "-o", color=c, lw=2, ms=4, label=label)
        ax.fill_between(layers, p5, p95, color=c, alpha=0.18, linewidth=0)
    ax.set_xlabel("Layer index $i$")
    ax.set_ylabel(r"$\cos(\mathbf{h}^{i},\,\mathbf{h}^{i+1})$")
    ax.set_title("Representation collapse across depth (ML-20M HSTU)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.margins(x=0.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(FLAGS.output) or ".", exist_ok=True)
    fig.savefig(FLAGS.output, bbox_inches="tight")
    npz_path = os.path.splitext(FLAGS.output)[0] + ".npz"
    np.savez(
        npz_path,
        **{
            f"{label}::{arr}": val
            for label, (layers, med, p5, p95) in results.items()
            for arr, val in [
                ("layers", layers),
                ("median", med),
                ("p5", p5),
                ("p95", p95),
            ]
        },
    )
    logging.info(f"Saved plot -> {FLAGS.output}")
    logging.info(f"Saved raw stats -> {npz_path}")


def main() -> None:
    app.run(_main)


if __name__ == "__main__":
    main()
