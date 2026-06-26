#!/usr/bin/env python3
"""Gather STU_PYTORCH seq-length sweep results -> CSV + bar plot.

Parses lifetime NE / GAUC per task from logs/slurm/o_<jobid>.out across the
5-point seq-length sweep {512, 2048, 4096, 8192, 16384} x 10 seeds, aggregates
mean +/- std across seeds, writes a CSV, and renders a bento-styled bar plot.
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = "/storage/home/ngocbh/project/gr"
SLURM = os.path.join(REPO, "logs", "slurm")

# (seq_len, seed) -> jobid
JOBS = {}
# 16384 batch
for seed, jid in zip(
    range(1, 11),
    [1469665, 1469666, 1469667, 1469668, 1469670,
     1469671, 1469672, 1469673, 1469674, 1469675],
):
    JOBS[(16384, seed)] = jid
# 512/2048/4096/8192 from seqlen jobids file
with open(os.path.join(REPO, "logs", "kr27k_stu_pytorch_seqlen_jobids.txt")) as f:
    for line in f:
        jid, name, L, seed = line.split()
        JOBS[(int(L), int(seed))] = int(jid)

SEQLENS = [512, 2048, 4096, 8192, 16384]
TASKS = ["is_click", "long_view", "is_profile_enter", "is_like",
         "is_comment", "is_follow", "is_forward", "is_hate"]

ne_re = re.compile(r"eval metric/lifetime_ne/(\w+):\s*([0-9.eE+-]+)")
gauc_re = re.compile(r"eval metric/lifetime_gauc/(\w+):\s*([0-9.eE+-]+)")


def parse_job(jid):
    """Return {'ne': {task: val}, 'gauc': {task: val}} for last epoch in log."""
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


# values[metric][task][seqlen] = list over seeds
values = {"ne": defaultdict(lambda: defaultdict(list)),
          "gauc": defaultdict(lambda: defaultdict(list))}

for (L, seed), jid in JOBS.items():
    res = parse_job(jid)
    for metric in ("ne", "gauc"):
        for task, val in res[metric].items():
            values[metric][task][L].append(val)

# ---- Build CSV (mean +/- std, one column per seq_len) ----
rows = []
for task in TASKS:
    for metric, label in (("ne", "lifetime_ne"), ("gauc", "lifetime_gauc")):
        row = {"task": task, "metric": label}
        for L in SEQLENS:
            vals = values[metric][task].get(L, [])
            if vals:
                row[f"seq{L}"] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
                row[f"seq{L}_n"] = len(vals)
            else:
                row[f"seq{L}"] = ""
                row[f"seq{L}_n"] = 0
        rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(REPO, "logs", "kr27k_stu_pytorch_seqlen_results.csv")
df.to_csv(csv_path, index=False)
print(f"[wrote] {csv_path}")
print(df.to_string(index=False))

# ---- Bar plot: well-powered tasks (n>=800), NE + GAUC, seq_len on x-axis ----
exec(open(os.path.expanduser(
    "~/.claude/agent-market/plugins/10x-data-scientist/skills/visualization/"
    "bento-plotly/references/bento_style_template.py"
)).read())

PLOT_TASKS = ["is_click", "long_view", "is_profile_enter", "is_like"]
x = [str(L) for L in SEQLENS]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Lifetime NE (lower is better)",
                    "Lifetime GAUC (higher is better)"),
    horizontal_spacing=0.10,
)

for i, task in enumerate(PLOT_TASKS):
    color = OKABE_ITO[i]
    ne_mean = [np.mean(values["ne"][task][L]) for L in SEQLENS]
    ne_std = [np.std(values["ne"][task][L]) for L in SEQLENS]
    gauc_mean = [np.mean(values["gauc"][task][L]) for L in SEQLENS]
    gauc_std = [np.std(values["gauc"][task][L]) for L in SEQLENS]

    fig.add_trace(go.Bar(
        x=x, y=ne_mean, name=task, legendgroup=task,
        marker_color=color,
        error_y=dict(type="data", array=ne_std, visible=True),
        hovertemplate=f"<b>{task}</b><br>seq_len=%{{x}}<br>NE: %{{y:.4f}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=x, y=gauc_mean, name=task, legendgroup=task, showlegend=False,
        marker_color=color,
        error_y=dict(type="data", array=gauc_std, visible=True),
        hovertemplate=f"<b>{task}</b><br>seq_len=%{{x}}<br>GAUC: %{{y:.4f}}<extra></extra>",
    ), row=1, col=2)

fig.update_xaxes(title_text="Max sequence length", type="category", row=1, col=1)
fig.update_xaxes(title_text="Max sequence length", type="category", row=1, col=2)
fig.update_yaxes(range=[0.70, 1.02], row=1, col=1)
fig.update_yaxes(range=[0.5, 0.62], row=1, col=2)

# Push subplot titles down so they don't collide with the title/subtitle row.
for ann in fig.layout.annotations:
    ann.y = 1.0

fig.update_layout(
    title=dict(text="<b>Sequence-length sweep</b>"
                    "<br><sub>HSTU full-attention, KuaiRand-27K, mean +/- std over 10 seeds</sub>"),
    barmode="group",
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    margin=dict(r=180, t=130),
    width=1100, height=560,
)
png_path = os.path.join(REPO, "logs", "kr27k_stu_pytorch_seqlen_bar.png")
fig.write_image(png_path, scale=2)
print(f"[wrote] {png_path}")

# ---- Line plot: same panels, trend across seq_len with std error bands ----
figl = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Lifetime NE (lower is better)",
                    "Lifetime GAUC (higher is better)"),
    horizontal_spacing=0.10,
)


def hex_to_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


MARKERS = ["circle", "square", "diamond", "triangle-up"]

for i, task in enumerate(PLOT_TASKS):
    color = OKABE_ITO[i]
    band = hex_to_rgba(color, 0.15)
    symbol = MARKERS[i % len(MARKERS)]
    for col, metric, mlabel in ((1, "ne", "NE"), (2, "gauc", "GAUC")):
        mean = np.array([np.mean(values[metric][task][L]) for L in SEQLENS])
        std = np.array([np.std(values[metric][task][L]) for L in SEQLENS])
        show = (col == 1)
        # std band (upper then lower, filled) -- lines only, no markers
        figl.add_trace(go.Scatter(
            x=x + x[::-1], y=list(mean + std) + list((mean - std)[::-1]),
            fill="toself", fillcolor=band, mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False, legendgroup=task,
        ), row=1, col=col)
        figl.add_trace(go.Scatter(
            x=x, y=mean, name=task, legendgroup=task, showlegend=show,
            mode="lines+markers", line=dict(color=color, width=2),
            marker=dict(size=8, color=color, symbol=symbol),
            hovertemplate=f"<b>{task}</b><br>seq_len=%{{x}}<br>{mlabel}: %{{y:.4f}}<extra></extra>",
        ), row=1, col=col)

figl.update_xaxes(title_text="Max sequence length", type="category", row=1, col=1)
figl.update_xaxes(title_text="Max sequence length", type="category", row=1, col=2)
figl.update_yaxes(range=[0.70, 1.02], row=1, col=1)
figl.update_yaxes(range=[0.50, 0.62], row=1, col=2)

for ann in figl.layout.annotations:
    ann.y = 1.0

figl.update_layout(
    title=dict(text="<b>Sequence-length sweep</b>"
                    "<br><sub>HSTU full-attention, KuaiRand-27K, mean +/- std over 10 seeds</sub>"),
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    margin=dict(r=180, t=130),
    width=1100, height=560,
)

line_path = os.path.join(REPO, "logs", "kr27k_stu_pytorch_seqlen_line.png")
figl.write_image(line_path, scale=2)
print(f"[wrote] {line_path}")
