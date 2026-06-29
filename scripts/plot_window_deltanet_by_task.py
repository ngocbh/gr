#!/usr/bin/env python3
"""Bar plot grouped by task; one bar per method (color), + an Average group.

Methods (x within each task group), ordering each SLW baseline next to its
+DeltaNet counterpart for direct comparison:
    Full attn | SLW=512 | SLW=512+DN | SLW=256 | SLW=256+DN | SLW=128 | SLW=128+DN
Groups = 4 well-powered tasks + "Average" (mean across those tasks, per seed).
Two panels: lifetime NE and GAUC, mean +/- std over 10 seeds.
"""

import os
import re
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = "/storage/home/ngocbh/project/gr"
SLURM = os.path.join(REPO, "logs", "slurm")

TASKS = ["is_click", "long_view", "is_profile_enter", "is_like"]
SEEDS = list(range(1, 11))
ne_re = re.compile(r"eval metric/lifetime_ne/(\w+):\s*([0-9.eE+-]+)")
gauc_re = re.compile(r"eval metric/lifetime_gauc/(\w+):\s*([0-9.eE+-]+)")

FULL_JOBS = {s: j for s, j in zip(SEEDS, [
    1469665, 1469666, 1469667, 1469668, 1469670,
    1469671, 1469672, 1469673, 1469674, 1469675])}


def parse_job(jid):
    ne, gauc = {}, {}
    with open(os.path.join(SLURM, f"o_{jid}.out")) as f:
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


base_jobs = load_jobs(os.path.join(REPO, "logs", "kr27k_stu_pytorch_window_jobids.txt"))
delta_jobs = load_jobs(os.path.join(REPO, "logs", "kr27k_deltanet_jobids.txt"))

# method -> seed -> jobid
METHODS = ["Full attn", "SLW=512", "SLW=512 +DN", "SLW=256", "SLW=256 +DN",
           "SLW=128", "SLW=128 +DN"]
method_jobs = {
    "Full attn": FULL_JOBS,
    "SLW=512": {s: base_jobs[(512, s)] for s in SEEDS},
    "SLW=512 +DN": {s: delta_jobs[(512, s)] for s in SEEDS},
    "SLW=256": {s: base_jobs[(256, s)] for s in SEEDS},
    "SLW=256 +DN": {s: delta_jobs[(256, s)] for s in SEEDS},
    "SLW=128": {s: base_jobs[(128, s)] for s in SEEDS},
    "SLW=128 +DN": {s: delta_jobs[(128, s)] for s in SEEDS},
}

# data[method][metric][task] = list over seeds (seed-ordered)
data = {m: {"ne": defaultdict(list), "gauc": defaultdict(list)} for m in METHODS}
for m in METHODS:
    for s in SEEDS:
        res = parse_job(method_jobs[m][s])
        for metric in ("ne", "gauc"):
            for task in TASKS:
                data[m][metric][task].append(res[metric][task])


def stat(method, metric, task):
    a = np.array(data[method][metric][task])
    return a.mean(), a.std()


def avg_stat(method, metric):
    # per-seed average across tasks, then mean/std over seeds
    per_seed = np.mean([data[method][metric][t] for t in TASKS], axis=0)
    return per_seed.mean(), per_seed.std()


exec(open(os.path.expanduser(
    "~/.claude/agent-market/plugins/10x-data-scientist/skills/visualization/"
    "bento-plotly/references/bento_style_template.py"
)).read())

GROUPS = TASKS + ["Average"]
# 7 distinct colors (Okabe-Ito has 8); +DN variants share family visually by order
COLORS = [OKABE_ITO[i] for i in [7, 0, 4, 1, 2, 5, 3]]
# 7: black (Full), 0/4 blue pair (512), 1/2 orange-green (256), 5/3 (128)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Lifetime NE (lower is better)",
                    "Lifetime GAUC (higher is better)"),
    horizontal_spacing=0.09,
)

for mi, method in enumerate(METHODS):
    color = COLORS[mi]
    for col, metric in ((1, "ne"), (2, "gauc")):
        means, stds = [], []
        for g in GROUPS:
            if g == "Average":
                mu, sd = avg_stat(method, metric)
            else:
                mu, sd = stat(method, metric, g)
            means.append(mu)
            stds.append(sd)
        fig.add_trace(go.Bar(
            x=GROUPS, y=means, name=method, legendgroup=method,
            showlegend=(col == 1), marker_color=color,
            error_y=dict(type="data", array=stds, visible=True, thickness=1),
            hovertemplate=f"<b>{method}</b><br>%{{x}}<br>"
                          f"{metric.upper()}: %{{y:.4f}}<extra></extra>",
        ), row=1, col=col)

fig.update_xaxes(type="category")
fig.update_yaxes(range=[0.70, 1.00], row=1, col=1)
fig.update_yaxes(range=[0.50, 0.60], row=1, col=2)
for ann in fig.layout.annotations:
    ann.y = 1.0

fig.update_layout(
    title=dict(text="<b>HSTU windows vs +gated DeltaNet, by task</b>"
                    "<br><sub>Full attn &amp; sliding-window HSTU with/without DeltaNet long "
                    "memory; KuaiRand-27K, mean +/- std over 10 seeds</sub>"),
    barmode="group", bargap=0.25, bargroupgap=0.0,
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    margin=dict(r=160, t=120),
    width=1400, height=560,
)
add_source(fig, "Internal KuaiRand-27K eval, FY26; 4 well-powered tasks")
png = os.path.join(REPO, "logs", "kr27k_window_deltanet_by_task_bar.png")
fig.write_image(png, scale=2)
print(f"[wrote] {png}")

for metric in ("ne", "gauc"):
    print(f"\n== {metric.upper()} (mean over seeds) ==")
    hdr = f"{'method':14s}" + "".join(f"{g:>17s}" for g in GROUPS)
    print(hdr)
    for m in METHODS:
        line = f"{m:14s}"
        for g in GROUPS:
            mu, _ = (avg_stat(m, metric) if g == "Average" else stat(m, metric, g))
            line += f"{mu:>17.4f}"
        print(line)
