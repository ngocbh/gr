#!/usr/bin/env python3
"""Gather the gated-DeltaNet sliding-window sweep -> CSV + bento plots.

Parses lifetime NE / GAUC per task from logs/slurm/o_<jobid>.out for the
DeltaNet sweep {128, 256, 512} x 10 seeds (max_seq_len=16384), aggregates
mean +/- std across seeds, runs a paired t-test against the STU_PYTORCH
baseline at the matching window, writes a CSV, and renders bento line plots
(DeltaNet vs baseline) over window size.
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def paired_t(dv, bv):
    """Paired t-statistic and two-sided p for dv-bv. df=n-1, normal approx p."""
    d = dv - bv
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0, 1.0
    t = d.mean() / (sd / np.sqrt(n))
    # two-sided p via survival of |t| under t-dist approximated by normal for df>=9
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    return t, p

REPO = "/storage/home/ngocbh/project/gr"
SLURM = os.path.join(REPO, "logs", "slurm")

WINDOWS = [128, 256, 512]
TASKS = ["is_click", "long_view", "is_profile_enter", "is_like",
         "is_comment", "is_follow", "is_forward", "is_hate"]
PLOT_TASKS = ["is_click", "long_view", "is_profile_enter", "is_like"]

ne_re = re.compile(r"eval metric/lifetime_ne/(\w+):\s*([0-9.eE+-]+)")
gauc_re = re.compile(r"eval metric/lifetime_gauc/(\w+):\s*([0-9.eE+-]+)")


def load_jobs(path):
    jobs = {}
    with open(path) as f:
        for line in f:
            jid, name, W, seed = line.split()
            jobs[(int(W), int(seed))] = int(jid)
    return jobs


def parse_job(jid):
    path = os.path.join(SLURM, f"o_{jid}.out")
    ne, gauc = {}, {}
    with open(path) as f:
        for line in f:
            m = ne_re.search(line)
            if m:
                ne[m.group(1)] = float(m.group(2))
            m = gauc_re.search(line)
            if m:
                gauc[m.group(1)] = float(m.group(2))
    return {"ne": ne, "gauc": gauc}


def collect(jobs):
    """vals[metric][task][W] = {seed: value} (seed-keyed for pairing)."""
    vals = {"ne": defaultdict(lambda: defaultdict(dict)),
            "gauc": defaultdict(lambda: defaultdict(dict))}
    for (W, seed), jid in jobs.items():
        if W not in WINDOWS:
            continue
        res = parse_job(jid)
        for metric in ("ne", "gauc"):
            for task, val in res[metric].items():
                vals[metric][task][W][seed] = val
    return vals


delta = collect(load_jobs(os.path.join(REPO, "logs", "kr27k_deltanet_jobids.txt")))
base = collect(load_jobs(os.path.join(REPO, "logs", "kr27k_stu_pytorch_window_jobids.txt")))


def seeds_array(d):  # dict seed->val -> array ordered by seed
    return np.array([d[s] for s in sorted(d)])


# ---- CSV: per task/metric/window, DeltaNet vs baseline + paired t-test ----
rows = []
for task in TASKS:
    for metric, label in (("ne", "lifetime_ne"), ("gauc", "lifetime_gauc")):
        for W in WINDOWS:
            dd = delta[metric][task].get(W, {})
            bb = base[metric][task].get(W, {})
            if not dd or not bb:
                continue
            common = sorted(set(dd) & set(bb))
            dv = np.array([dd[s] for s in common])
            bv = np.array([bb[s] for s in common])
            row = {
                "task": task, "metric": label, "window": W, "n": len(common),
                "deltanet": f"{dv.mean():.4f} +/- {dv.std():.4f}",
                "baseline": f"{bv.mean():.4f} +/- {bv.std():.4f}",
                "delta_minus_base": f"{(dv - bv).mean():+.4f}",
            }
            if len(common) >= 2 and (dv - bv).std() > 0:
                t, p = paired_t(dv, bv)
                row["t"] = f"{t:+.3f}"
                row["p"] = f"{p:.3f}"
                row["sig95"] = "*" if abs(t) > 2.262 else ""
            else:
                row["t"] = row["p"] = row["sig95"] = ""
            rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(REPO, "logs", "kr27k_deltanet_window_results.csv")
df.to_csv(csv_path, index=False)
print(f"[wrote] {csv_path}")
print(df.to_string(index=False))

# ---- Bento line plots: DeltaNet vs baseline, well-powered tasks ----
exec(open(os.path.expanduser(
    "~/.claude/agent-market/plugins/10x-data-scientist/skills/visualization/"
    "bento-plotly/references/bento_style_template.py"
)).read())

x = [str(W) for W in WINDOWS]


def hex_to_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def mean_std(src, metric, task):
    mean = np.array([seeds_array(src[metric][task][W]).mean() for W in WINDOWS])
    std = np.array([seeds_array(src[metric][task][W]).std() for W in WINDOWS])
    return mean, std


for metric, mlabel, fname, yr in (
    ("ne", "Lifetime NE (lower is better)", "ne", None),
    ("gauc", "Lifetime GAUC (higher is better)", "gauc", None),
):
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=PLOT_TASKS,
        horizontal_spacing=0.10, vertical_spacing=0.14,
    )
    for i, task in enumerate(PLOT_TASKS):
        r, c = i // 2 + 1, i % 2 + 1
        for src, nm, color in ((delta, "DeltaNet", OKABE_ITO[1]),
                               (base, "HSTU baseline", OKABE_ITO[0])):
            mean, std = mean_std(src, metric, task)
            band = hex_to_rgba(color, 0.15)
            show = (i == 0)
            fig.add_trace(go.Scatter(
                x=x + x[::-1], y=list(mean + std) + list((mean - std)[::-1]),
                fill="toself", fillcolor=band, mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False, legendgroup=nm,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=x, y=mean, name=nm, legendgroup=nm, showlegend=show,
                mode="lines+markers", line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                hovertemplate=f"<b>{nm} / {task}</b><br>window=%{{x}}"
                              f"<br>{mlabel.split()[1]}: %{{y:.4f}}<extra></extra>",
            ), row=r, col=c)
    fig.update_xaxes(title_text="Attention window")
    for ann in fig.layout.annotations:
        ann.y = ann.y  # keep subplot titles
    fig.update_layout(
        title=dict(text=f"<b>Gated DeltaNet vs HSTU baseline</b>"
                        f"<br><sub>{mlabel}; max_seq_len=16384, KuaiRand-27K, "
                        f"mean +/- std over 10 seeds</sub>"),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(r=180, t=120),
        width=1000, height=760,
    )
    add_source(fig, "Internal KuaiRand-27K eval, FY26")
    png = os.path.join(REPO, "logs", f"kr27k_deltanet_window_{fname}.png")
    fig.write_image(png, scale=2)
    print(f"[wrote] {png}")
