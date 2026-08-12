"""Compare HSTU softmax-free (rel_bias) vs softmax (softmax_rel_bias) on ML-20M.

Tests HSTU's central claim: is the pointwise SiLU/N aggregation actually better
than softmax attention, all else equal? Both runs are identical except the
normalization. Reads the stable per-epoch full eval (eval_epoch/*). Run with the
gr-env python (has tensorboard).
"""
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def curve(d, tag):
    ev = sorted(glob.glob(os.path.join(d, "events.out.*")), key=os.path.getmtime)
    if not ev:
        return {}
    ea = EventAccumulator(ev[-1])
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return {}
    return {s.step: s.value for s in ea.Scalars(tag)}


def find():
    silu = smax = None
    for d in glob.glob("exps/ml-20m-l200/HSTU-b16-h8-dqk32-dv32-lsilud0.2-ad0.0*"):
        b = os.path.basename(d)
        # exclude KLA cores / other variants; keep only the two plain-HSTU runs
        if any(t in b for t in ("kda", "kla")):
            continue
        if "-softmax" in b:
            smax = d
        elif "_DotProduct" in b and "-norab" not in b:
            silu = d
    return silu, smax


silu, smax = find()
sh = curve(silu, "eval_epoch/hr@10") if silu else {}
sn = curve(silu, "eval_epoch/ndcg@10") if silu else {}
mh = curve(smax, "eval_epoch/hr@10") if smax else {}
mn = curve(smax, "eval_epoch/ndcg@10") if smax else {}

print("HSTU softmax-free (rel_bias) vs softmax (softmax_rel_bias) -- ML-20M, full eval")
if sh:
    e = max(sh)
    print(f"  softmax-FREE : ep{e+1:3d}  HR@10={sh[e]:.4f}  NDCG@10={sn.get(e,0):.4f}")
else:
    print("  softmax-FREE : (no data)")
if mh:
    e = max(mh)
    print(f"  softmax      : ep{e+1:3d}  HR@10={mh[e]:.4f}  NDCG@10={mn.get(e,0):.4f}")
    if sh:
        # compare at the softmax run's current epoch (fair, same-epoch)
        ce = max(k for k in sh if k <= e) if any(k <= e for k in sh) else None
        if ce is not None:
            dhr = mh[e] - sh[ce]
            print(
                f"  @ep{e+1}: softmax {mh[e]:.4f} vs softmax-free {sh[ce]:.4f}  "
                f"=> Δ={dhr:+.4f} HR@10 ({'softmax-free better' if dhr < 0 else 'softmax better'})"
            )
else:
    print("  softmax      : (no data yet)")
