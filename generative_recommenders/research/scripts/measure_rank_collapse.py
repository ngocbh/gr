# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# pyre-unsafe

"""
Measure TRUE rank collapse across depth, complementing
``plot_representation_collapse.py``.

That script measures ``cos(h^i, h^{i+1})`` -- how much each *layer* changes the
residual stream (which, in a residual net, can also just mean a layer learned to
do little). Here we measure the purer oversmoothing signals on each block's
*output* token matrix ``H in R^{L x D}`` (per user, real tokens only):

  * token-token cosine : mean off-diagonal of the normalized Gram matrix.
      -> 1.0 means all tokens in a sequence converge to the same direction
         (true representation / rank collapse).
  * effective rank     : exp(entropy of the normalized singular-value spectrum).
      Drops toward 1 as the representation loses dimensionality with depth.

Reuses ``plot_representation_collapse``'s model/checkpoint construction and its
``--run "label,config,ckpt"`` flags. Run on a GPU.

  python -m generative_recommenders.research.scripts.measure_rank_collapse \\
      --run "HSTU-large,configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin,<ckpt>" \\
      --num_batches 8
"""

import logging
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import gin
import numpy as np
import torch
import torch.nn.functional as F
from absl import app
from generative_recommenders.research.modeling.sequential.features import (
    movielens_seq_features_from_row,
)

# Reuse the exact model/loader construction, checkpoint loader, and CLI flags
# (--run/--num_batches/--batch_size) from the collapse-plot script.
from generative_recommenders.research.scripts.plot_representation_collapse import (
    _build_model_and_loader,
    _load_checkpoint,
    FLAGS,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def _token_metrics(x_out: torch.Tensor, offsets: torch.Tensor) -> Tuple[List[float], List[float]]:
    """Per-user token-token cosine + effective rank for a jagged [T, D] batch."""
    cos_vals: List[float] = []
    rank_vals: List[float] = []
    B = offsets.numel() - 1
    for b in range(B):
        s, e = int(offsets[b]), int(offsets[b + 1])
        L = e - s
        if L < 2:
            continue
        H = x_out[s:e].float()  # [L, D]
        Hn = F.normalize(H, dim=-1)
        G = Hn @ Hn.T  # [L, L]
        off = (G.sum() - torch.diagonal(G).sum()) / (L * (L - 1))
        cos_vals.append(float(off))
        sv = torch.linalg.svdvals(H)
        p = sv / (sv.sum() + 1e-9)
        eff = torch.exp(-(p * (p + 1e-12).log()).sum())
        rank_vals.append(float(eff))
    return cos_vals, rank_vals


def _collect(model, eval_loader, gr_out, device):  # pyre-ignore [2,3]
    layers = model._hstu._attention_layers
    tok_cos: Dict[int, List[float]] = defaultdict(list)
    eff_rank: Dict[int, List[float]] = defaultdict(list)
    handles = []

    def make_hook(i: int):  # pyre-ignore [3]
        def hook(module, args, kwargs, output):  # pyre-ignore [2]
            x_out = output[0] if isinstance(output, tuple) else output
            offs = kwargs.get("x_offsets")
            if offs is None:
                return
            c, r = _token_metrics(x_out.detach(), offs)
            tok_cos[i].extend(c)
            eff_rank[i].extend(r)

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i), with_kwargs=True))

    with torch.no_grad():
        for b, row in enumerate(iter(eval_loader)):
            if b >= FLAGS.num_batches:
                break
            sf, _, _ = movielens_seq_features_from_row(
                row, device=device, max_output_length=gr_out + 1
            )
            emb = model.get_item_embeddings(sf.past_ids)
            model(
                past_lengths=sf.past_lengths,
                past_ids=sf.past_ids,
                past_embeddings=emb,
                past_payloads=sf.past_payloads,
            )

    for h in handles:
        h.remove()
    return tok_cos, eff_rank


def _main(argv) -> None:  # pyre-ignore [2]
    del argv
    assert torch.cuda.is_available(), "Run on a GPU."
    device = torch.device("cuda:0")

    runs = []
    for spec in FLAGS.run:
        parts = [p.strip() for p in spec.split(",")]
        assert len(parts) == 3, f'--run must be "label,config,ckpt"; got {spec}'
        runs.append(tuple(parts))

    for label, config, ckpt in runs:
        with gin.unlock_config():
            gin.clear_config()
            gin.parse_config_file(config)
        model, eval_loader, gr_out = _build_model_and_loader(device)
        _load_checkpoint(model, ckpt)
        tok_cos, eff_rank = _collect(model, eval_loader, gr_out, device)
        layers = sorted(tok_cos.keys())
        tc = [float(np.median(tok_cos[i])) for i in layers]
        er = [float(np.median(eff_rank[i])) for i in layers]
        logging.info(
            f"[{label}] token-token cos by layer: "
            + ", ".join(f"{i}:{v:.3f}" for i, v in zip(layers, tc))
        )
        logging.info(
            f"[{label}] effective rank by layer:  "
            + ", ".join(f"{i}:{v:.2f}" for i, v in zip(layers, er))
        )
        del model
        torch.cuda.empty_cache()


def main() -> None:
    app.run(_main)


if __name__ == "__main__":
    main()
