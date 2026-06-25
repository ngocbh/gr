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

import unittest

import torch
from generative_recommenders.modules.mhc import (
    expand_streams,
    HyperConnection,
    reduce_streams,
    sinkhorn_log,
)


class SinkhornTest(unittest.TestCase):
    def test_doubly_stochastic(self) -> None:
        # tau=0.3 keeps the kernel well-conditioned so Sinkhorn converges to a
        # tight doubly-stochastic matrix (the sharp tau=0.05 regime needs many
        # more iters; the identity-at-init test below covers the default tau).
        torch.manual_seed(0)
        logits = torch.randn(5, 5)
        p = sinkhorn_log(logits, num_iters=200, tau=0.3)
        self.assertTrue(torch.all(p >= 0))
        torch.testing.assert_close(
            p.sum(dim=-1), torch.ones(5), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            p.sum(dim=-2), torch.ones(5), atol=1e-4, rtol=0
        )

    def test_batched(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(3, 4, 4)
        p = sinkhorn_log(logits, num_iters=200, tau=0.3)
        torch.testing.assert_close(
            p.sum(dim=-1), torch.ones(3, 4), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            p.sum(dim=-2), torch.ones(3, 4), atol=1e-4, rtol=0
        )

    def test_fp32_under_autocast_dtype(self) -> None:
        # logits provided as a low-precision tensor still normalize correctly.
        logits = (torch.randn(4, 4)).to(torch.float16)
        p = sinkhorn_log(logits, num_iters=200, tau=0.3)
        self.assertEqual(p.dtype, torch.float32)
        torch.testing.assert_close(
            p.sum(dim=-1), torch.ones(4), atol=1e-3, rtol=0
        )


class HyperConnectionTest(unittest.TestCase):
    def test_h_res_identity_at_init(self) -> None:
        hc = HyperConnection(num_streams=4, layer_index=0)
        h_res = hc.h_res()
        torch.testing.assert_close(
            h_res, torch.eye(4), atol=1e-3, rtol=0
        )

    def test_h_pre_reads_assigned_stream_at_init(self) -> None:
        for li in range(4):
            hc = HyperConnection(num_streams=4, layer_index=li)
            h_pre = torch.sigmoid(hc._h_pre_logits)
            # assigned stream ~1, others ~0
            self.assertGreater(h_pre[li % 4].item(), 0.99)
            others = [h_pre[j].item() for j in range(4) if j != li % 4]
            self.assertTrue(all(o < 0.01 for o in others))

    def test_h_post_writes_full_output_at_init(self) -> None:
        hc = HyperConnection(num_streams=4)
        h_post = 2.0 * torch.sigmoid(hc._h_post_logits)
        torch.testing.assert_close(
            h_post, torch.ones(4), atol=1e-5, rtol=0
        )

    def test_behaves_like_plain_residual_at_init(self) -> None:
        # At init, one round-trip through a layer should equal: every stream gets
        # branch_out added (H_post=1 to all, H_res=I), and the read is the
        # assigned stream. With all streams equal (fresh expand), the next read
        # equals stream + branch_out -- i.e. a standard residual update.
        torch.manual_seed(0)
        n, d, tokens = 4, 8, 6
        hc = HyperConnection(num_streams=n, layer_index=1)
        x = torch.randn(tokens, d)
        streams = expand_streams(x, n)

        branch_input, mixed = hc.width_connection(streams)
        # assigned stream is index 1; all streams equal -> branch_input == x
        torch.testing.assert_close(branch_input, x, atol=1e-4, rtol=0)
        # H_res ~ identity -> mixed == streams
        torch.testing.assert_close(mixed, streams, atol=1e-3, rtol=0)

        branch_out = torch.randn(tokens, d)
        streams2 = hc.depth_connection(branch_out, mixed)
        # every stream got branch_out added
        expected = streams + branch_out.unsqueeze(-2)
        torch.testing.assert_close(streams2, expected, atol=1e-3, rtol=0)

    def test_n1_collapses_to_plain_residual(self) -> None:
        torch.manual_seed(0)
        d, tokens = 8, 6
        hc = HyperConnection(num_streams=1, layer_index=0)
        x = torch.randn(tokens, d)
        streams = expand_streams(x, 1)  # (tokens, 1, d)

        branch_input, mixed = hc.width_connection(streams)
        torch.testing.assert_close(branch_input, x, atol=1e-4, rtol=0)
        torch.testing.assert_close(mixed, streams, atol=1e-5, rtol=0)

        branch_out = torch.randn(tokens, d)
        streams2 = hc.depth_connection(branch_out, mixed)
        out = reduce_streams(streams2)
        # plain residual: out == x + branch_out
        torch.testing.assert_close(out, x + branch_out, atol=1e-4, rtol=0)

    def test_expand_reduce_shapes(self) -> None:
        x = torch.randn(7, 16)
        s = expand_streams(x, 4)
        self.assertEqual(tuple(s.shape), (7, 4, 16))
        r = reduce_streams(s)
        self.assertEqual(tuple(r.shape), (7, 16))
        # expand makes identical copies -> reduce == n * x
        torch.testing.assert_close(r, 4.0 * x, atol=1e-5, rtol=0)

    def test_h_res_stays_doubly_stochastic_after_perturbation(self) -> None:
        torch.manual_seed(2)
        hc = HyperConnection(num_streams=4, num_iters=200, tau=0.3)
        with torch.no_grad():
            hc._h_res_logits.add_(torch.randn(4, 4))
        h_res = hc.h_res()
        torch.testing.assert_close(
            h_res.sum(dim=-1), torch.ones(4), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            h_res.sum(dim=-2), torch.ones(4), atol=1e-4, rtol=0
        )

    def test_gradients_flow(self) -> None:
        torch.manual_seed(3)
        n, d, tokens = 4, 8, 5
        hc = HyperConnection(num_streams=n)
        x = torch.randn(tokens, d, requires_grad=True)
        streams = expand_streams(x, n)
        branch_input, mixed = hc.width_connection(streams)
        branch_out = branch_input * 2.0
        streams2 = hc.depth_connection(branch_out, mixed)
        loss = reduce_streams(streams2).pow(2).sum()
        loss.backward()
        self.assertIsNotNone(hc._h_res_logits.grad)
        self.assertIsNotNone(hc._h_pre_logits.grad)
        self.assertIsNotNone(hc._h_post_logits.grad)
        self.assertTrue(torch.isfinite(hc._h_res_logits.grad).all())


if __name__ == "__main__":
    unittest.main()
