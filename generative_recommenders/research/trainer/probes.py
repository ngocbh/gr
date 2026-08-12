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
Lightweight training-dynamics probe for the research HSTU encoder.

Attaches forward hooks to each ``SequentialTransductionUnitJagged`` block and,
on demand, records four per-layer scalars from the training forward pass:

  * ``input_norm``         -- mean L2 norm of the layer input x_in.
  * ``pre_residual_norm``  -- mean L2 norm of f(x), the block transform *before*
                              the residual add (the output of the block's ``_o``
                              projection).
  * ``post_residual_norm`` -- mean L2 norm of x_out = f(x) + x_in, the block
                              output *after* the residual add.
  * ``cos_in_out``         -- mean cosine similarity cos(x_in, x_out) per token.
                              Since x_out of layer i is the input of layer i+1,
                              this measures how much each block rotates the
                              representation; values approaching 1.0 across depth
                              indicate representational collapse / over-smoothing.

All quantities are computed on the jagged ``(sum_i N_i, D)`` activations (real
tokens only, no padding) under ``torch.no_grad`` on detached tensors, so the
probe never perturbs autograd. It is intended to run on rank 0 only.
"""

import logging
from typing import Dict, List

import torch
import torch.nn.functional as F


logger: logging.Logger = logging.getLogger(__name__)


class LayerProbe:
    """Forward-hook based probe over the HSTU block stack.

    Args:
        model: the (unwrapped) HSTU encoder, i.e. ``ddp_model.module``. It must
            expose ``_hstu._attention_layers`` (a ``ModuleList`` of
            ``SequentialTransductionUnitJagged``); otherwise ``AttributeError``
            is raised and the caller is expected to skip probing.
        enabled: initial capture state. Toggle with :meth:`set_enabled` right
            around the forward pass you want to observe.
    """

    def __init__(self, model: torch.nn.Module, enabled: bool = False) -> None:
        self.enabled: bool = enabled
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        # layer index -> scalar (0-dim) GPU tensor for the current step.
        self._input_norm: Dict[int, torch.Tensor] = {}
        self._pre_residual_norm: Dict[int, torch.Tensor] = {}
        self._post_residual_norm: Dict[int, torch.Tensor] = {}
        self._cos_in_out: Dict[int, torch.Tensor] = {}

        layers = model._hstu._attention_layers
        self._num_layers: int = len(layers)
        for i, layer in enumerate(layers):
            # x_in / x_out come from the block itself. The block is called with
            # keyword arguments, so request kwargs in the hook signature.
            self._handles.append(
                layer.register_forward_hook(
                    self._make_block_hook(i), with_kwargs=True
                )
            )
            # f(x) (pre-residual) is exactly the output of the block's ``_o``
            # projection, before ``+ x`` is applied in the block's forward.
            self._handles.append(layer._o.register_forward_hook(self._make_o_hook(i)))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def _make_o_hook(self, i: int):  # pyre-ignore [3]
        def hook(module, inputs, output) -> None:  # pyre-ignore [2]
            if not self.enabled:
                return
            with torch.no_grad():
                fx = output.detach().float()
                self._pre_residual_norm[i] = fx.norm(dim=-1).mean()

        return hook

    def _make_block_hook(self, i: int):  # pyre-ignore [3]
        def hook(module, args, kwargs, output) -> None:  # pyre-ignore [2]
            if not self.enabled:
                return
            with torch.no_grad():
                x_in = kwargs.get("x", args[0] if args else None)
                x_out = output[0] if isinstance(output, tuple) else output
                if x_in is None or x_out is None:
                    return
                x_in = x_in.detach().float()
                x_out = x_out.detach().float()
                self._input_norm[i] = x_in.norm(dim=-1).mean()
                self._post_residual_norm[i] = x_out.norm(dim=-1).mean()
                self._cos_in_out[i] = F.cosine_similarity(x_in, x_out, dim=-1).mean()

        return hook

    def collect(self) -> Dict[str, float]:
        """Return the captured scalars as a flat wandb-loggable dict.

        Uses a single host sync (one ``.tolist()``) for all scalars, then clears
        the per-step buffers. Keys are shaped ``probe/<metric>/layer_<ii>`` plus
        depth-averaged ``probe/summary/mean_<metric>`` for at-a-glance charts.
        """
        named_buffers = [
            ("input_norm", self._input_norm),
            ("pre_residual_norm", self._pre_residual_norm),
            ("post_residual_norm", self._post_residual_norm),
            ("cos_in_out", self._cos_in_out),
        ]

        keys: List[str] = []
        values: List[torch.Tensor] = []
        for metric, buffer in named_buffers:
            for i in range(self._num_layers):
                if i in buffer:
                    keys.append(f"probe/{metric}/layer_{i:02d}")
                    values.append(buffer[i])

        metrics: Dict[str, float] = {}
        if values:
            flat = torch.stack(values).cpu().tolist()  # single sync
            metrics = dict(zip(keys, flat))
            # Depth-averaged summaries (single-panel, immediately plottable).
            for metric, buffer in named_buffers:
                per_layer = [
                    metrics[f"probe/{metric}/layer_{i:02d}"]
                    for i in range(self._num_layers)
                    if i in buffer
                ]
                if per_layer:
                    metrics[f"probe/summary/mean_{metric}"] = sum(per_layer) / len(
                        per_layer
                    )

        self._input_norm.clear()
        self._pre_residual_norm.clear()
        self._post_residual_norm.clear()
        self._cos_in_out.clear()
        return metrics

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
