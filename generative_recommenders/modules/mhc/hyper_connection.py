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
Manifold-Constrained Hyper-Connections (mHC).

Hyper-connections (HC) generalize the residual connection: instead of a single
residual stream, the hidden state is widened into ``n`` parallel streams (the
"width-n manifold"). Each layer (a) READS a single input from the n streams via a
read map ``H_pre``, (b) MIXES the streams among themselves via a width map
``H_res``, and (c) WRITES its output back onto every stream via a write map
``H_post``::

    branch_input = H_pre   @ streams                      # (n,) . (n, d) -> (d,)
    streams      = H_res    @ streams  +  H_post (x) F(.)  # (n, n) . (n, d) + outer

mHC adds a *manifold constraint*: ``H_res`` is projected onto the Birkhoff polytope
(doubly-stochastic matrices: non-negative, every row and column sums to 1) via
Sinkhorn-Knopp normalization. Doubly-stochastic matrices are non-expansive and
closed under composition, so the depth-wise product of the per-layer ``H_res``
stays norm-preserving at any depth -- the property that keeps deep HC stable.
``H_pre`` / ``H_post`` are kept non-negative via sigmoid / 2*sigmoid (paper form).

This module implements the *static* mHC (the three maps are plain learnable
parameters, shared across positions and batch). Initialization makes the stack
behave like a plain residual network at step 0: ``H_res`` starts ~identity, each
layer's ``H_pre`` reads a single assigned stream, and ``H_post`` writes the full
branch output to every stream.

Ref: "mHC: Manifold-Constrained Hyper-Connections", https://arxiv.org/abs/2512.24880
and https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections.
"""

import torch
import torch.nn as nn


def sinkhorn_log(
    logits: torch.Tensor,
    num_iters: int = 20,
    tau: float = 0.05,
) -> torch.Tensor:
    """Project a square logit matrix onto the doubly-stochastic manifold.

    Runs Sinkhorn-Knopp in log space (numerically stable) entirely in float32,
    regardless of the surrounding autocast dtype. The returned matrix satisfies
    ``P @ 1 = 1`` and ``1^T @ P = 1`` (rows and columns each sum to 1) up to the
    iteration tolerance.

    Args:
        logits: (..., n, n) raw scores.
        num_iters: number of Sinkhorn iterations.
        tau: temperature; logits are divided by tau before normalization, so
            smaller tau yields a sharper (closer to permutation) result.

    Returns:
        (..., n, n) doubly-stochastic matrix, dtype float32.
    """
    log_k = logits.float() / tau
    # Row / column potentials f, g such that P_ij = exp(log_k_ij + f_i + g_j).
    # Target marginals are 1 (not 1/n), giving a doubly-stochastic matrix.
    f = torch.zeros(log_k.shape[:-1], dtype=log_k.dtype, device=log_k.device)
    g = torch.zeros(log_k.shape[:-1], dtype=log_k.dtype, device=log_k.device)
    for _ in range(num_iters):
        # Row normalize: sum_j exp(log_k_ij + f_i + g_j) = 1.
        f = -torch.logsumexp(log_k + g.unsqueeze(-2), dim=-1)
        # Column normalize: sum_i exp(log_k_ij + f_i + g_j) = 1.
        g = -torch.logsumexp(log_k + f.unsqueeze(-1), dim=-2)
    return torch.exp(log_k + f.unsqueeze(-1) + g.unsqueeze(-2))


def expand_streams(x: torch.Tensor, num_streams: int) -> torch.Tensor:
    """Widen a single residual stream into ``num_streams`` copies.

    (..., d) -> (..., num_streams, d). The new stream axis is the second-to-last.
    """
    return (
        x.unsqueeze(-2)
        .expand(*x.shape[:-1], num_streams, x.shape[-1])
        .contiguous()
    )


def reduce_streams(streams: torch.Tensor) -> torch.Tensor:
    """Collapse the n streams back into one by summing.

    (..., num_streams, d) -> (..., d).
    """
    return streams.sum(dim=-2)


class HyperConnection(nn.Module):
    """A single layer's static manifold-constrained hyper-connection.

    Holds the three learnable maps for one sublayer ``F``. Use it in two steps
    around the sublayer::

        branch_input, mixed = hc.width_connection(streams)   # read + mix
        branch_out = F(branch_input)                         # run the sublayer
        streams = hc.depth_connection(branch_out, mixed)     # write back

    where ``streams`` has shape ``(..., num_streams, d)`` and ``branch_input`` /
    ``branch_out`` have shape ``(..., d)``.
    """

    def __init__(
        self,
        num_streams: int,
        layer_index: int = 0,
        num_iters: int = 20,
        tau: float = 0.05,
        init_off_diag: float = -8.0,
        init_read: float = 12.0,
    ) -> None:
        super().__init__()
        assert num_streams >= 1, "num_streams must be >= 1"
        self._num_streams: int = num_streams
        self._num_iters: int = num_iters
        self._tau: float = tau

        # H_res (width mixing), projected to doubly-stochastic via Sinkhorn.
        # Init: 0 on the diagonal, very negative off-diagonal -> ~identity, so at
        # step 0 each stream passes through itself (plain residual).
        h_res = torch.full((num_streams, num_streams), float(init_off_diag))
        h_res.fill_diagonal_(0.0)
        self._h_res_logits: nn.Parameter = nn.Parameter(h_res)

        # H_pre (read), sigmoid. Init reads ~1.0 from the assigned stream
        # (layer_index % n) and ~0 from the rest, so the sublayer sees a single
        # clean stream at step 0.
        read_idx = layer_index % num_streams
        h_pre = torch.full((num_streams,), -float(init_read))
        h_pre[read_idx] = float(init_read)
        self._h_pre_logits: nn.Parameter = nn.Parameter(h_pre)

        # H_post (write), 2*sigmoid. Init logits 0 -> 2*sigmoid(0)=1, writing the
        # full branch output onto every stream at step 0.
        self._h_post_logits: nn.Parameter = nn.Parameter(torch.zeros(num_streams))

    @property
    def num_streams(self) -> int:
        return self._num_streams

    def h_res(self) -> torch.Tensor:
        """Doubly-stochastic width-mixing matrix, (n, n) float32."""
        if self._num_streams == 1:
            # Degenerates to the scalar 1 (plain residual); skip Sinkhorn.
            return torch.ones(
                (1, 1),
                dtype=torch.float32,
                device=self._h_res_logits.device,
            )
        return sinkhorn_log(
            self._h_res_logits, num_iters=self._num_iters, tau=self._tau
        )

    def width_connection(
        self, streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the sublayer input and mix the residual streams.

        Args:
            streams: (..., num_streams, d).

        Returns:
            branch_input: (..., d) -- the single input for the sublayer.
            mixed: (..., num_streams, d) -- streams after width mixing (H_res),
                to which the sublayer output is later added in depth_connection.
        """
        h_res = self.h_res().to(streams.dtype)
        h_pre = torch.sigmoid(self._h_pre_logits).to(streams.dtype)
        mixed = torch.einsum("st,...sd->...td", h_res, streams)
        branch_input = torch.einsum("s,...sd->...d", h_pre, streams)
        return branch_input, mixed

    def depth_connection(
        self, branch_out: torch.Tensor, mixed: torch.Tensor
    ) -> torch.Tensor:
        """Write the sublayer output back onto the (already mixed) streams.

        Args:
            branch_out: (..., d) -- the sublayer output.
            mixed: (..., num_streams, d) -- streams from width_connection.

        Returns:
            (..., num_streams, d) updated streams.
        """
        h_post = (2.0 * torch.sigmoid(self._h_post_logits)).to(branch_out.dtype)
        write = torch.einsum("...d,s->...sd", branch_out, h_post)
        return mixed + write
