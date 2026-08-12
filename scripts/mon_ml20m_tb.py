"""Read ML-20M mixer-comparison metrics straight from the TB event files.

The research trainer suppresses logging.info to the SLURM logs and writes eval
metrics only to tensorboard (exps/ml-20m-l200/<run>/events.*), so scraping the
.err files finds nothing. Run with the gr-env python (has tensorboard).
"""
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def short(n):
    for a, b in (
        ("-lsilud0.2-ad0.0", ""),
        ("_DotProduct_local-l2-eps1e-06_ssl-t0.05-n128-b128-lr0.001-wu0-wd0-2026-08-03", ""),
        ("HSTU-", ""),
    ):
        n = n.replace(a, b)
    return n or "HSTU-base"


def read():
    """Return {run: (hr_by_epoch, ndcg_by_epoch)} using the STABLE per-epoch
    full eval (eval_epoch/*), NOT the noisy per-interval partial eval (eval/*)."""
    runs = {}
    for d in sorted(glob.glob("exps/ml-20m-l200/*2026-08-03*")):
        ev = sorted(glob.glob(os.path.join(d, "events.out.*")), key=os.path.getmtime)
        if not ev:
            continue
        ea = EventAccumulator(ev[-1])
        ea.Reload()
        tg = ea.Tags().get("scalars", [])

        def series(t):
            return {s.step: s.value for s in ea.Scalars(t)} if t in tg else {}

        runs[short(os.path.basename(d))] = (
            series("eval_epoch/hr@10"),
            series("eval_epoch/ndcg@10"),
        )
    return runs


if __name__ == "__main__":
    runs = read()
    common = min((max(hr) for hr, _ in runs.values() if hr), default=0)
    print("README HSTU-large-20m (full eval): HR@10 0.3556 / NDCG@10 0.2098")
    print(f"{'run':42s} {'ep':>3s} | LATEST HR/NDCG | @ep{common} HR/NDCG")
    order = sorted(
        runs, key=lambda n: -(max(runs[n][0].values()) if runs[n][0] else 0)
    )
    for nm in order:
        hr, nd = runs[nm]
        if not hr:
            print(f"{nm:42s}  (no epoch eval yet)")
            continue
        e = max(hr)
        print(
            f"{nm:42s} {e + 1:3d} | {hr[e]:.4f} {nd.get(e, 0):.4f} | "
            f"{hr.get(common, 0):.4f} {nd.get(common, 0):.4f}"
        )
