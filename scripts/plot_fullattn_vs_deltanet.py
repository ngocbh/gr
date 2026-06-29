#!/usr/bin/env python3
"""Bar plot: full HSTU attention vs sliding-window + gated DeltaNet.

X-axis = configuration:
    Full attn (HSTU)  -- no window, sees all 16384 history
    SLW=128 + DeltaNet
    SLW=256 + DeltaNet
    SLW=512 + DeltaNet
Two panels (NE, GAUC), bars grouped by well-powered task, mean +/- std / 10 seeds.
Question: can a small local window + DeltaNet long memory match full attention?
"""

import os
import re
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = "/storage/home/ngocbh/project/gr"
SLURM = os.path.join(REPO, "logs", "slurm")

PLOT_TASKS = ["is_click", "long_view", "is_profile_enter", "is_like"]
ne_re = re.compile(r"eval metric/lifetime_ne/(\w+):\s*([0-9.eE+-]+)")
gauc_re = re.compile(r"eval metric/lifetime_gauc/(\w+):\s*([0-9.eE+-]+)")

# Full-attention HSTU baseline (max_seq_len=16384, no window): seq-len sweep batch.
FULL_JOBS = [1469665, 1469666, 1469667, 1469668, 1469670,
             1469671, 1469672, 1469673, 1469674, 1469675]


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


def load_jobs(path):
    jobs = {}
    with open(path) as f:
        for line in f:
            jid, name, W, seed = line.split()
            jobs[(int(W), int(seed))] = int(jid)
    return jobs


delta_jobs = load_jobs(os.path.join(REPO, "logs", "kr27k_deltanet_jobids.txt"))

# config -> metric -> task -> [vals over seeds]
CONFIGS = ["Full attn (HSTU)", "SLW=128\n+DeltaNet",
           "SLW=256\n+DeltaNet", "SLW=512\n+DeltaNet"]
vals = {cfg: {"ne": defaultdict(list), "gauc": defaultdict(list)} for cfg in CONFIGS}

for jid in FULL_JOBS:
    res = parse_job(jid)
    for metric in ("ne", "gauc"):
        for task, v in res[metric].items():
            vals["Full attn (HSTU)"][metric][task].append(v)

for W, cfg in ((128, CONFIGS[1]), (256, CONFIGS[2]), (512, CONFIGS[3])):
    for seed in range(1, 11):
        jid = delta_jobs[(W, seed)]
        res = parse_job(jid)
        for metric in ("ne", "gauc"):
            for task, v in res[metric].items():
                vals[cfg][metric][task].append(v)

# ---- bento bar plot ----
exec(open(os.path.expanduser(
    "~/.claude/agent-market/plugins/10x-data-scientist/skills/visualization/"
    "bento-plotly/references/bento_style_template.py"
)).read())

xlabels = [c.replace("\n", "<br>") for c in CONFIGS]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Lifetime NE (lower is better)",
                    "Lifetime GAUC (higher is better)"),
    horizontal_spacing=0.10,
)

for i, task in enumerate(PLOT_TASKS):
    color = OKABE_ITO[i]
    ne_mean = [np.mean(vals[c]["ne"][task]) for c in CONFIGS]
    ne_std = [np.std(vals[c]["ne"][task]) for c in CONFIGS]
    gauc_mean = [np.mean(vals[c]["gauc"][task]) for c in CONFIGS]
    gauc_std = [np.std(vals[c]["gauc"][task]) for c in CONFIGS]
    fig.add_trace(go.Bar(
        x=xlabels, y=ne_mean, name=task, legendgroup=task, marker_color=color,
        error_y=dict(type="data", array=ne_std, visible=True),
        hovertemplate=f"<b>{task}</b><br>%{{x}}<br>NE: %{{y:.4f}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=xlabels, y=gauc_mean, name=task, legendgroup=task, showlegend=False,
        marker_color=color,
        error_y=dict(type="data", array=gauc_std, visible=True),
        hovertemplate=f"<b>{task}</b><br>%{{x}}<br>GAUC: %{{y:.4f}}<extra></extra>",
    ), row=1, col=2)

fig.update_xaxes(type="category")
fig.update_yaxes(range=[0.70, 1.00], row=1, col=1)
fig.update_yaxes(range=[0.50, 0.60], row=1, col=2)
for ann in fig.layout.annotations:
    ann.y = 1.0

fig.update_layout(
    title=dict(text="<b>Can local window + DeltaNet match full attention?</b>"
                    "<br><sub>Full HSTU attention vs sliding-window + gated DeltaNet; "
                    "KuaiRand-27K, mean +/- std over 10 seeds</sub>"),
    barmode="group",
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    margin=dict(r=160, t=120),
    width=1150, height=560,
)
add_source(fig, "Internal KuaiRand-27K eval, FY26")
png = os.path.join(REPO, "logs", "kr27k_fullattn_vs_deltanet_bar.png")
fig.write_image(png, scale=2)
print(f"[wrote] {png}")

# quick text table
for metric in ("ne", "gauc"):
    print(f"\n== {metric.upper()} ==")
    print(f"{'task':18s}" + "".join(f"{c.replace(chr(10),' '):>22s}" for c in CONFIGS))
    for task in PLOT_TASKS:
        line = f"{task:18s}"
        for c in CONFIGS:
            m, s = np.mean(vals[c][metric][task]), np.std(vals[c][metric][task])
            line += f"{f'{m:.4f}+/-{s:.4f}':>22s}"
        print(line)
