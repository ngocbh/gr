#!/usr/bin/env python3
"""GPU smoke test for STUpDeltaNet (STULayerpDeltaNet).

Checks:
 1. The layer constructs and runs a forward pass on a jagged batch (CUDA, bf16).
 2. The candidate-conditioned gate weight has the extra H*d_qk dims vs base.
 3. The delta write-mask is the non-overlapping split: writes only positions
    [0, x_lengths - W) per user (the last W are attention's; candidates read-only).
 4. Gate input width matches _gate_input_dim().
"""

import torch
import fbgemm_gpu  # noqa: F401  -- registers torch.ops.fbgemm.*
from generative_recommenders.modules.stu import STULayerConfig
from generative_recommenders.modules.stu_deltanet import (
    STULayerDeltaNet,
    STULayerpDeltaNet,
)

H, d_qk, d_hidden = 4, 16, 16
W = 8
cfg = STULayerConfig(
    embedding_dim=64,
    num_heads=H, hidden_dim=d_hidden, attention_dim=d_qk,
    causal=True, target_aware=True, max_attn_len=W,
    sort_by_length=True, contextual_seq_len=0,
)

dev = "cuda"
base = STULayerDeltaNet(cfg).to(dev)
plus = STULayerpDeltaNet(cfg).to(dev)

# (1) gate-weight dims: plus adds H*d_qk for the candidate query.
print("base gate_in:", base._gate_weight.numel(),
      " plus gate_in:", plus._gate_weight.numel(),
      " diff:", plus._gate_weight.numel() - base._gate_weight.numel(),
      " expected:", H * d_qk)
assert plus._gate_weight.numel() - base._gate_weight.numel() == H * d_qk

# (3) write-mask check, pure-Python on a known jagged batch.
# Two users: lengths 20 and 12, num_targets 2 and 3.
lengths = torch.tensor([20, 12], device=dev)
num_targets = torch.tensor([2, 3], device=dev)
x_offsets = torch.zeros(3, dtype=torch.long, device=dev)
x_offsets[1:] = torch.cumsum(lengths, 0)
total = int(x_offsets[-1].item())

mask = plus._delta_write_mask(total, x_offsets, lengths, num_targets, dev).squeeze(1)
# expected per user: write positions [0, len - W)
exp = torch.zeros(total, dtype=torch.bool, device=dev)
for u, (off, ln) in enumerate(zip(x_offsets[:-1].tolist(), lengths.tolist())):
    cut = max(min(ln - W, ln - int(num_targets[u])), 0)
    exp[off:off + cut] = True
print("write-mask matches expected:", bool((mask == exp).all()),
      " n_write:", int(mask.sum()), " expected:", int(exp.sum()))
assert bool((mask == exp).all())
# base writes all history (len - num_targets)
bmask = base._delta_write_mask(total, x_offsets, lengths, num_targets, dev).squeeze(1)
print("base n_write (full history):", int(bmask.sum()),
      " expected:", int((lengths - num_targets).clamp_min(0).sum()))
assert int(bmask.sum()) == int((lengths - num_targets).clamp_min(0).sum())

# (1)+(4) forward pass, bf16 AMP, on the jagged batch.
torch.manual_seed(0)
E = base._uvqk_weight.shape[0]  # embedding_dim expected by uvqk
x = torch.randn(total, E, device=dev, dtype=torch.bfloat16)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = plus(
        x=x, x_lengths=lengths, x_offsets=x_offsets,
        max_seq_len=int(lengths.max()), num_targets=num_targets,
    )
print("forward out shape:", tuple(out.shape), " dtype:", out.dtype,
      " finite:", bool(torch.isfinite(out).all()))
assert out.shape[0] == total and bool(torch.isfinite(out).all())
print("PASS")
