#!/usr/bin/env python3
"""Verify the bw-config filter fix: run a full HSTU forward+backward at the
dlrm_v3 head-dim-128 shape, letting the real backward autotuner run over the
filtered config set. Before the fix this SIGABRTs in LinearLayout::reshapeOuts;
after the fix it must complete and print OK.

Also asserts _get_bw_configs() no longer contains the 5 BLOCK_N=64/warps=8
configs that crash, leaving 23 of the original 28.
"""
import torch
from generative_recommenders.ops.triton import triton_hstu_attention as A

H = 4
DIM = 128
Z = 16
SEQ = 512
NUM_TARGETS = 8
CONTEXTUAL = 1


def check_prune() -> None:
    configs = A._get_bw_configs()
    n = len(configs)
    bad = [c for c in configs if c.kwargs.get("BLOCK_N") == 64 and c.num_warps == 8]
    print(f"total configs after filter: {n}", flush=True)
    print(f"BLOCK_N=64/warps=8 configs remaining: {len(bad)}", flush=True)
    assert len(bad) == 0, "crashing BLOCK_N=64/warps=8 configs still present"
    assert n == 23, f"expected 23 configs after filter, got {n}"
    print("prune-logic checks OK", flush=True)


def run_fwd_bwd() -> None:
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16
    L = Z * SEQ
    q = torch.randn(L, H, DIM, dtype=dtype, device=dev, requires_grad=True)
    k = torch.randn(L, H, DIM, dtype=dtype, device=dev, requires_grad=True)
    v = torch.randn(L, H, DIM, dtype=dtype, device=dev, requires_grad=True)
    seq_offsets = torch.arange(0, L + 1, SEQ, dtype=torch.int64, device=dev)
    num_targets = torch.full((Z,), NUM_TARGETS, dtype=torch.int64, device=dev)

    out = A.triton_hstu_mha(
        N=SEQ,
        alpha=1.0 / DIM,
        q=q, k=k, v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_attn_len=0,
        contextual_seq_len=CONTEXTUAL,
        sort_by_length=False,
        enable_tma=False,
    )
    loss = out.float().pow(2).sum()
    loss.backward()
    torch.cuda.synchronize()
    assert q.grad is not None and torch.isfinite(q.grad).all(), "bad dq"
    assert k.grad is not None and torch.isfinite(k.grad).all(), "bad dk"
    assert v.grad is not None and torch.isfinite(v.grad).all(), "bad dv"
    print("forward+backward (full autotune) OK", flush=True)


if __name__ == "__main__":
    print(f"torch.version.cuda={torch.version.cuda}", flush=True)
    check_prune()
    run_fwd_bwd()
    print("ALL OK", flush=True)
