#!/usr/bin/env python3
"""Validate the gated delta rule implementations against each other.

1. CPU/float64: chunked (de-decay) == sequential reference, across chunk sizes
   and sequence lengths (incl. non-multiples of the chunk size).
2. GPU (if available): fla Triton kernel == sequential reference, in the
   batched and varlen (cu_seqlens) layouts. fla runs in fp32, so tolerances
   are looser than the float64 chunk check.
"""

import torch
from generative_recommenders.modules.stu_deltanet import (
    gated_delta_chunk,
    gated_delta_sequential,
)


def _rand_inputs(N, L, d_m, d_v, seed, gamma_bias, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(N, L, d_m, generator=g)
    k = torch.randn(N, L, d_m, generator=g)
    v = torch.randn(N, L, d_v, generator=g)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = torch.sigmoid(torch.randn(N, L, generator=g))
    gamma = torch.sigmoid(torch.randn(N, L, generator=g) + gamma_bias)
    to = lambda t: t.to(device=device, dtype=dtype)
    return to(q), to(k), to(v), to(beta), to(gamma)


def run(N, L, d_m, d_v, C, seed=0, gamma_bias=0.0):
    q, k, v, beta, gamma = _rand_inputs(
        N, L, d_m, d_v, seed, gamma_bias, "cpu", torch.float64
    )
    ref = gated_delta_sequential(q, k, v, beta, gamma)
    chk = gated_delta_chunk(q, k, v, beta, gamma, chunk_size=C)
    err = (ref - chk).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-9)
    print(
        f"N={N} L={L} d_m={d_m} d_v={d_v} C={C} gbias={gamma_bias}: "
        f"max_abs_err={err:.3e} rel={rel:.3e}"
    )
    return rel


def run_fla(N, L, d_m, d_v, seed=0, gamma_bias=3.0, varlen=False):
    """fla Triton kernel vs the sequential reference on GPU (fp32)."""
    from generative_recommenders.modules.stu_deltanet import gated_delta_fla

    dev = "cuda"
    q, k, v, beta, gamma = _rand_inputs(
        N, L, d_m, d_v, seed, gamma_bias, dev, torch.float32
    )
    # Reference: per-sequence sequential scan (fp32).
    ref = gated_delta_sequential(q, k, v, beta, gamma)  # [N, L, d_v]

    if varlen:
        # Pack the N sequences into one [1, N*L, ...] buffer with cu_seqlens.
        cu = torch.arange(0, (N + 1) * L, L, device=dev, dtype=torch.int32)
        qf = q.reshape(1, N * L, 1, d_m)
        kf = k.reshape(1, N * L, 1, d_m)
        vf = v.reshape(1, N * L, 1, d_v)
        bf = beta.reshape(1, N * L, 1)
        gf = gamma.reshape(1, N * L, 1)
        o = gated_delta_fla(qf, kf, vf, bf, gf, cu_seqlens=cu)
        o = o.reshape(N, L, d_v)
    else:
        # Batched layout: [N, L, H=1, .].
        o = gated_delta_fla(
            q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2),
            beta.unsqueeze(2), gamma.unsqueeze(2),
        ).squeeze(2)

    err = (ref - o).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-9)
    tag = "varlen" if varlen else "batched"
    print(
        f"[fla {tag}] N={N} L={L} d_m={d_m} d_v={d_v} gbias={gamma_bias}: "
        f"max_abs_err={err:.3e} rel={rel:.3e}"
    )
    return rel


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)  # tighten tolerance for the check
    # Strong-decay stress (gamma~0.5): de-decay form is ill-conditioned here, so
    # ~1e-6 float64 accumulation is expected. Real model inits gamma~0.95.
    worst = 0.0
    for (L, C) in [(16, 4), (64, 64), (100, 64), (257, 64), (512, 128)]:
        worst = max(worst, run(N=3, L=L, d_m=8, d_v=8, C=C))
    print(f"worst rel err (strong decay): {worst:.3e}")
    assert worst < 1e-4, "chunked delta diverges from sequential reference"

    # Realistic mild-decay regime (gamma~0.95): should be near machine precision.
    mild = 0.0
    for (L, C) in [(257, 64), (512, 128), (1024, 64)]:
        mild = max(mild, run(N=3, L=L, d_m=8, d_v=8, C=C, gamma_bias=3.0))
    print(f"worst rel err (mild decay):   {mild:.3e}")
    assert mild < 1e-9, "chunked delta inaccurate even under mild decay"

    # GPU: fla Triton kernel vs sequential reference (fp32). Skip if no CUDA.
    if torch.cuda.is_available():
        torch.set_default_dtype(torch.float32)
        fla_worst = 0.0
        for varlen in (False, True):
            for (L,) in [(128,), (257,), (512,), (1024,)]:
                fla_worst = max(
                    fla_worst,
                    run_fla(N=3, L=L, d_m=16, d_v=16, gamma_bias=3.0, varlen=varlen),
                )
        print(f"worst rel err (fla vs sequential, fp32): {fla_worst:.3e}")
        assert fla_worst < 1e-2, "fla kernel diverges from sequential reference"
    else:
        print("[skip] no CUDA -- fla kernel check not run")
    print("PASS")
