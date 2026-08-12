# Diagnostic: is the KDA slowness caused by per-batch Triton recompilation
# on the variable-length (cu_seqlens) path? Compare:
#   [packed]  varying total T each step  (current kda.py path)
#   [dense]   fixed [128, 211] each step (proposed fix)
# A large gap that persists for [packed] but vanishes after step 0 for [dense]
# confirms recompilation is the culprit.
import time
import torch
from fla.layers import KimiDeltaAttention

dev = "cuda"
D = 50


def mk():
    return KimiDeltaAttention(
        hidden_size=D, head_dim=24, num_heads=1, expand_v=1.0,
        mode="chunk", use_short_conv=True, conv_size=4,
    ).to(dev)


def bench_packed(steps=12):
    layer = mk(); layer.train()
    g = torch.Generator().manual_seed(0)
    for i in range(steps):
        lengths = torch.randint(20, 200, (128,), generator=g).to(torch.int32)
        cu = torch.zeros(129, dtype=torch.int32, device=dev)
        cu[1:] = torch.cumsum(lengths, 0).to(dev)
        T = int(cu[-1].item())
        x = torch.randn(1, T, D, device=dev, requires_grad=True)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, _, _ = layer(hidden_states=x, cu_seqlens=cu)
        o.float().sum().backward()
        torch.cuda.synchronize()
        print(f"[packed] step {i:2d} T={T:5d}  {time.time()-t0:.3f}s", flush=True)


def bench_dense(steps=12):
    layer = mk(); layer.train()
    for i in range(steps):
        x = torch.randn(128, 211, D, device=dev, requires_grad=True)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, _, _ = layer(hidden_states=x)
        o.float().sum().backward()
        torch.cuda.synchronize()
        print(f"[dense]  step {i:2d}         {time.time()-t0:.3f}s", flush=True)


print("=== PACKED (varying T, current kda.py path) ===", flush=True)
bench_packed()
print("=== DENSE (fixed [128,211], proposed fix) ===", flush=True)
bench_dense()
print("DONE", flush=True)
