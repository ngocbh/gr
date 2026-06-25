#!/usr/bin/env python3
"""Analyze the KuaiRand-27K max_seq_len sweep.

Reads final end-of-epoch eval metrics from each run's local wandb-summary.json
(runs `kr27k_len<L>_s<S>`, L in {512..16384} x seeds {1,2,3}), then writes:
  - seqlen_sweep_gauc.csv / seqlen_sweep_ne.csv : mean(+-std over seeds) tables
  - seqlen_sweep_gauc.png / seqlen_sweep_ne.png : metric vs seq_len, well-powered tasks

No network / no wandb API: wandb-summary.json is the local artifact wandb writes
per run (works for offline runs too). Output dir defaults to logs/seqlen_analysis.
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
             "#56B4E9", "#D55E00", "#F0E442", "#999999"]
NAME_RE = re.compile(r"len(\d+)_s(\d+)$")
KEY_RE = re.compile(r"eval_metric/lifetime_(gauc|ne|gauc_num_samples)/(.+)$")


def load(base: str) -> pd.DataFrame:
    rows = []
    for d in sorted(glob.glob(f"{base}/kr27k_len*_s*")):
        m = NAME_RE.search(d)
        if not m:
            continue
        seq_len, seed = int(m.group(1)), int(m.group(2))
        cand = sorted(glob.glob(os.path.join(d, "wandb", "run-*", "files", "wandb-summary.json")))
        if not cand:
            continue
        summ = json.load(open(cand[-1]))
        for k, v in summ.items():
            mm = KEY_RE.match(k)
            if mm:
                rows.append(dict(seq_len=seq_len, seed=seed,
                                 metric=mm.group(1), task=mm.group(2), value=v))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no kr27k_len* runs with wandb-summary.json under {base}")
    return df


def agg_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Wide table: rows=task, cols=seq_len, cells='mean+-std' over seeds."""
    g = df[df.metric == metric]
    mean = g.pivot_table(index="task", columns="seq_len", values="value", aggfunc="mean")
    std = g.pivot_table(index="task", columns="seq_len", values="value", aggfunc="std")
    out = mean.copy().astype(object)
    for t in mean.index:
        for c in mean.columns:
            out.loc[t, c] = f"{mean.loc[t, c]:.4f}+-{(std.loc[t, c] or 0):.4f}"
    return out, mean


def plot(df, metric, ylabel, title, well, peak, outpath, baseline=None):
    g = df[df.metric == metric]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, task in enumerate(well):
        sub = g[g.task == task].groupby("seq_len").value.agg(["mean", "std"]).reset_index()
        sub = sub.sort_values("seq_len")
        c = OKABE_ITO[i]
        x, mean, std = sub.seq_len.values, sub["mean"].values, np.nan_to_num(sub["std"].values)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.12, linewidth=0)
        ax.plot(x, mean, "-o", color=c, lw=2.2, ms=6, label=task)
    if baseline is not None:
        ax.axhline(baseline, ls=":", color="#999999", lw=1)
        ax.text(g.seq_len.min(), baseline, f" random ({baseline:.2f})",
                va="bottom", ha="left", fontsize=8, color="#999999")
    # peak annotation
    pt, px = peak
    pv = g[(g.task == pt) & (g.seq_len == px)].value.mean()
    ax.annotate("plateau/best zone", xy=(px, pv), xytext=(0, -34),
                textcoords="offset points", ha="center", fontsize=9, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#777777"))
    ax.set_xscale("log", base=2)
    xs = sorted(g.seq_len.unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s:,}" for s in xs])
    ax.set_xlabel("max sequence length (tokens, log2)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.grid(axis="y", color="#EBEBEB")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.text(0.0, -0.02, "Source: KuaiRand-27K seq-len sweep (local wandb summaries), "
             "1 epoch, end-of-epoch eval; low-volume tasks (n<250) excluded",
             fontsize=8, color="#999999")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get(
        "GR_EXPS_ROOT", "/checkpoints/ngocbh/longhstu/checkpoints/exps"))
    ap.add_argument("--out", default="logs/seqlen_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.base)
    cells = df[["seq_len", "seed"]].drop_duplicates().shape[0]
    print(f"loaded {cells} (seq_len, seed) cells from {args.base}")

    nsamp = df[df.metric == "gauc_num_samples"].groupby("task").value.mean().sort_values(ascending=False)
    well = nsamp[nsamp >= 800].index.tolist()
    print("\nmean eval sessions per task:")
    print(nsamp.round(0).to_string())
    print("\nwell-powered tasks (n>=800):", well)

    for metric, label in [("gauc", "gAUC"), ("ne", "NE")]:
        tbl, _ = agg_table(df, metric)
        csv = os.path.join(args.out, f"seqlen_sweep_{metric}.csv")
        tbl.to_csv(csv)
        print(f"\n=== eval {label} (mean+-std over 3 seeds) ===")
        print(tbl.to_string())
        print("wrote", csv)

    plot(df, "gauc", "eval gAUC (per-user AUC)",
         "Longer histories lift click & long-view ranking",
         well, ("long_view", 8192),
         os.path.join(args.out, "seqlen_sweep_gauc.png"), baseline=0.5)
    plot(df, "ne", "eval NE (lower = better)",
         "Calibrated loss improves on high-volume tasks",
         well, ("is_click", 4096),
         os.path.join(args.out, "seqlen_sweep_ne.png"))


if __name__ == "__main__":
    main()
