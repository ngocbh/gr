#!/usr/bin/env python3
"""Bisect which HSTU backward autotune config triggers the triton 3.2.0
LinearLayout reshapeOuts SIGABRT for dlrm_v3 head-dim-128 shapes.

The assertion aborts the whole process (uncatchable), so each candidate
config is compiled in its own subprocess. Driver mode (default) forks one
child per config index and reports the exit code per index; worker mode
(--index N) overrides the backward autotuner to a single config and runs a
forward+backward to force that config to compile.
"""
import argparse
import os
import subprocess
import sys

H = 4
DIM = 128  # hstu_attn_qk_dim == hstu_attn_linear_dim for dlrm_v3
Z = 16  # batch
SEQ = 512  # per-sequence length (uniform for repro simplicity)
NUM_TARGETS = 8  # kuairand has multiple candidates
CONTEXTUAL = 1  # kuairand contextual user_id


def run_worker(index: int) -> None:
    import torch
    from generative_recommenders.ops.triton import triton_hstu_attention as A

    configs = A._get_bw_configs()
    if index >= len(configs):
        print(f"INDEX_OUT_OF_RANGE n={len(configs)}", flush=True)
        return
    cfg = configs[index]
    # Pin the backward autotuner to exactly this one config.
    A._hstu_attn_bwd.configs = [cfg]
    print(f"[worker {index}] cfg={cfg.kwargs} stages={cfg.num_stages} "
          f"warps={cfg.num_warps}", flush=True)

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
    print(f"[worker {index}] OK", flush=True)


def run_driver() -> None:
    import torch
    from generative_recommenders.ops.triton import triton_hstu_attention as A

    n = len(A._get_bw_configs())
    print(f"total bw configs: {n}", flush=True)
    print(f"torch.version.cuda={torch.version.cuda}", flush=True)
    bad, good, other = [], [], []
    for i in range(n):
        r = subprocess.run(
            [sys.executable, __file__, "--index", str(i)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        tag = "OK" if r.returncode == 0 else f"RC={r.returncode}"
        if r.returncode == 0:
            good.append(i)
        elif r.returncode == -6:  # SIGABRT
            bad.append(i)
        else:
            other.append((i, r.returncode))
        # Echo the child's own cfg line for context.
        cfgline = ""
        for line in r.stdout.decode(errors="replace").splitlines():
            if line.startswith(f"[worker {i}] cfg="):
                cfgline = line
                break
        print(f"  idx {i:2d}: {tag}  {cfgline}", flush=True)
    print("\n==== SUMMARY ====", flush=True)
    print(f"GOOD ({len(good)}): {good}", flush=True)
    print(f"SIGABRT ({len(bad)}): {bad}", flush=True)
    print(f"OTHER_FAIL ({len(other)}): {other}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=-1)
    args = ap.parse_args()
    if args.index >= 0:
        run_worker(args.index)
    else:
        run_driver()
