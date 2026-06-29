#!/usr/bin/env python3
"""Grouped bars by task; 3 colors (Full attn / Sliding window / +DeltaNet).

Within each task block the bars are laid out with manual x-positions so each
window's (baseline, +DN) pair sits together with a gap before the next window:

    Full | [512 512+DN]   [256 256+DN]   [128 128+DN]

Window size is printed on top of each bar. Sliding-window bars share one color
across all sizes; +DeltaNet bars share another. Groups = 4 well-powered tasks +
Average. Two panels: lifetime NE and GAUC, mean +/- std over 10 seeds.
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
GROUPS = TASKS + ["Average"]
SEEDS = list(range(1, 11))
SW = [512, 256, 128]

ne_re = re.compile(r"eval metric/lifetime_ne/(\w+):\s*([0-9.eE+-]+)")
gauc_re = re.compile(r"eval metric/lifetime_gauc/(\w+):\s*([0-9.eE+-]+)")

FULL_JOBS = {s: j for s, j in zip(SEEDS, [
    1469665, 1469666, 1469667, 1469668, 1469670,
    1469671, 1469672, 1469673, 1469674, 1469675])}

COL_FULL = "#888888"   # neutral gray
COL_SW = "#0072B2"     # blue  -- sliding window (all sizes)
COL_DN = "#D55E00"     # vermillion -- sliding window + DeltaNet (all sizes)

# local x-offset of each bar inside a task block; pairs adjacent, gap between.
LOCAL = {
    "Full attn": 0.0,
    "SLW=512": 1.6, "SLW=512 +DN": 2.6,
    "SLW=256": 4.1, "SLW=256 +DN": 5.1,
    "SLW=128": 6.6, "SLW=128 +DN": 7.6,
}
STRIDE = 10.0
CENTER = 3.8  # task-label tick position within a block


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

method_jobs = {"Full attn": FULL_JOBS}
for w in SW:
    method_jobs[f"SLW={w}"] = {s: base_jobs[(w, s)] for s in SEEDS}
    method_jobs[f"SLW={w} +DN"] = {s: delta_jobs[(w, s)] for s in SEEDS}

# data[method][metric][task] = list over seeds
data = {m: {"ne": defaultdict(list), "gauc": defaultdict(list)} for m in method_jobs}
for m in method_jobs:
    for s in SEEDS:
        res = parse_job(method_jobs[m][s])
        for metric in ("ne", "gauc"):
            for task in TASKS:
                data[m][metric][task].append(res[metric][task])


def val(method, metric, group):
    if group == "Average":
        per_seed = np.mean([data[method][metric][t] for t in TASKS], axis=0)
        return per_seed.mean(), per_seed.std()
    a = np.array(data[method][metric][group])
    return a.mean(), a.std()


exec(open(os.path.expanduser(
    "~/.claude/agent-market/plugins/10x-data-scientist/skills/visualization/"
    "bento-plotly/references/bento_style_template.py"
)).read())

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Lifetime NE (lower is better)",
                    "Lifetime GAUC (higher is better)"),
    horizontal_spacing=0.08,
)


def series(metric):
    """Build per-category x/y/err/text arrays for one metric."""
    cats = {k: {"x": [], "y": [], "e": [], "t": []}
            for k in ("Full attn", "Sliding window", "Sliding window + DN")}
    for gi, g in enumerate(GROUPS):
        b = gi * STRIDE
        mu, sd = val("Full attn", metric, g)
        c = cats["Full attn"]
        c["x"].append(b + LOCAL["Full attn"]); c["y"].append(mu)
        c["e"].append(sd); c["t"].append("Full")
        for w in SW:
            mu, sd = val(f"SLW={w}", metric, g)
            c = cats["Sliding window"]
            c["x"].append(b + LOCAL[f"SLW={w}"]); c["y"].append(mu)
            c["e"].append(sd); c["t"].append(str(w))
            mu, sd = val(f"SLW={w} +DN", metric, g)
            c = cats["Sliding window + DN"]
            c["x"].append(b + LOCAL[f"SLW={w} +DN"]); c["y"].append(mu)
            c["e"].append(sd); c["t"].append(str(w))
    return cats


CAT_COLOR = {"Full attn": COL_FULL, "Sliding window": COL_SW,
             "Sliding window + DN": COL_DN}

for col, metric in ((1, "ne"), (2, "gauc")):
    cats = series(metric)
    for name, d in cats.items():
        fig.add_trace(go.Bar(
            x=d["x"], y=d["y"], name=name, legendgroup=name,
            showlegend=(col == 1), marker_color=CAT_COLOR[name], width=0.9,
            text=d["t"], textposition="outside", textfont=dict(size=8),
            cliponaxis=False,
            error_y=dict(type="data", array=d["e"], visible=True, thickness=1),
            hovertemplate=f"<b>{name}</b><br>window=%{{text}}<br>"
                          f"{metric.upper()}: %{{y:.4f}}<extra></extra>",
        ), row=1, col=col)

tickvals = [gi * STRIDE + CENTER for gi in range(len(GROUPS))]
for col in (1, 2):
    fig.update_xaxes(tickvals=tickvals, ticktext=GROUPS, row=1, col=col)
fig.update_yaxes(range=[0.70, 1.00], row=1, col=1)
fig.update_yaxes(range=[0.50, 0.60], row=1, col=2)
for ann in fig.layout.annotations:
    ann.y = 1.0

fig.update_layout(
    title=dict(text="<b>Sliding window vs +gated DeltaNet, by task</b>"
                    "<br><sub>Window size labeled on each bar; KuaiRand-27K, "
                    "mean +/- std over 10 seeds (4 well-powered tasks)</sub>"),
    barmode="overlay",
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    margin=dict(r=170, t=120),
    width=1500, height=560,
)
add_source(fig, "Internal KuaiRand-27K eval, FY26")
png = os.path.join(REPO, "logs", "kr27k_window_deltanet_3color_bar.png")
fig.write_image(png, scale=2)
print(f"[wrote] {png}")
