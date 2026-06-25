#!/usr/bin/env python3
"""Analyze the KuaiRand-27K sliding-window (max_attn_len) sweep.

Runs:
  - windowed:  kr27k_l<L>_w<W>_s<S>   L in {8192,16384}, W in {128..4096}, S in {1,2,3}
  - baseline:  kr27k_len<L>_s<S>      full causal attention (reused from the seq-len sweep)

Full attention == attending to the whole sequence, so each baseline is plotted as the
rightmost point at x = L (window "full"). For each L we get: does restricting the
attention window hurt eval quality vs full attention, and where does it saturate?

Reads local wandb-summary.json (no network). Tolerates not-yet-finished runs: uses
whatever has a summary. Writes per-metric figures (2 panels: L=8192, L=16384) + CSVs.
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
             "#56B4E9", "#D55E00", "#F0E442", "#999999"]
WIN_RE = re.compile(r"kr27k_l(\d+)_w(\d+)_s(\d+)$")
BASE_RE = re.compile(r"kr27k_len(\d+)_s(\d+)$")
KEY_RE = re.compile(r"eval_metric/lifetime_(gauc|ne|gauc_num_samples)/(.+)$")
SEQLENS = [8192, 16384]


def _summary(d):
    cand = sorted(glob.glob(os.path.join(d, "wandb", "run-*", "files", "wandb-summary.json")))
    return json.load(open(cand[-1])) if cand else None


def load(base):
    rows = []
    for d in sorted(glob.glob(f"{base}/kr27k_l*_w*_s*")):
        m = WIN_RE.search(d)
        if not m:
            continue
        L, W, S = int(m.group(1)), int(m.group(2)), int(m.group(3))
        summ = _summary(d)
        if summ is None:
            continue
        for k, v in summ.items():
            mm = KEY_RE.match(k)
            if mm:
                rows.append(dict(seq_len=L, window=W, full=False, seed=S,
                                 metric=mm.group(1), task=mm.group(2), value=v))
    # full-attention baselines: window := seq_len
    for L in SEQLENS:
        for d in sorted(glob.glob(f"{base}/kr27k_len{L}_s*")):
            m = BASE_RE.search(d)
            if not m:
                continue
            summ = _summary(d)
            if summ is None:
                continue
            S = int(m.group(2))
            for k, v in summ.items():
                mm = KEY_RE.match(k)
                if mm:
                    rows.append(dict(seq_len=L, window=L, full=True, seed=S,
                                     metric=mm.group(1), task=mm.group(2), value=v))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no attn-len runs with wandb-summary.json under {base}")
    return df


def plot(df, metric, ylabel, title, well, outpath, baseline=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    g = df[df.metric == metric]
    for ax, L in zip(axes, SEQLENS):
        gl = g[g.seq_len == L]
        xs = sorted(gl.window.unique())  # windows + the full point at x=L
        for i, task in enumerate(well):
            sub = (gl[gl.task == task].groupby("window").value
                   .agg(["mean", "std"]).reindex(xs).reset_index())
            c = OKABE_ITO[i]
            x = sub.window.values
            mean = sub["mean"].values
            std = np.nan_to_num(sub["std"].values)
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.12, linewidth=0)
            ax.plot(x, mean, "-o", color=c, lw=2.2, ms=6, label=task)
        ax.axvline(L, ls=":", color="#BBBBBB", lw=1)  # marks the full-attention point
        if baseline is not None:
            ax.axhline(baseline, ls=":", color="#999999", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([("full\n%d" % v) if v == L else f"{v}" for v in xs], fontsize=9)
        ax.set_xlabel("attention window (max_attn_len; rightmost = full attention)")
        ax.set_title(f"max_seq_len = {L:,}", fontsize=12, loc="left", color="#333333")
        ax.grid(axis="y", color="#EBEBEB")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.06, ha="left")
    fig.text(0.06, -0.02, "Source: KuaiRand-27K sliding-window sweep (local wandb summaries), "
             "1 epoch, end-of-epoch eval; low-volume tasks (n<250) excluded",
             fontsize=8, color="#999999")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get(
        "GR_EXPS_ROOT", "/checkpoints/ngocbh/longhstu/checkpoints/exps"))
    ap.add_argument("--out", default="logs/attnlen_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.base)
    cov = (df[["seq_len", "window", "seed"]].drop_duplicates()
           .groupby(["seq_len", "window"]).size().rename("seeds").reset_index())
    print("seed coverage per (seq_len, window):")
    print(cov.to_string(index=False))

    nsamp = df[df.metric == "gauc_num_samples"].groupby("task").value.mean().sort_values(ascending=False)
    well = nsamp[nsamp >= 800].index.tolist()
    print("\nwell-powered tasks (n>=800):", well)

    for metric, label in [("gauc", "gAUC"), ("ne", "NE")]:
        g = df[df.metric == metric]
        tbl = g.pivot_table(index=["seq_len", "task"], columns="window", values="value", aggfunc="mean")
        csv = os.path.join(args.out, f"attnlen_sweep_{metric}.csv")
        tbl.to_csv(csv)
        print(f"\n=== eval {label} mean (rows: seq_len/task, cols: window; last col per L = full) ===")
        print(tbl.loc[(slice(None), well), :].round(4).to_string())
        print("wrote", csv)

    plot(df, "gauc", "eval gAUC (per-user AUC)",
         "Sliding-window attention vs full attention (gAUC)", well,
         os.path.join(args.out, "attnlen_sweep_gauc.png"), baseline=0.5)
    plot(df, "ne", "eval NE (lower = better)",
         "Sliding-window attention vs full attention (NE)", well,
         os.path.join(args.out, "attnlen_sweep_ne.png"))


if __name__ == "__main__":
    main()
