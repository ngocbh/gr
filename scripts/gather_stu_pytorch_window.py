#!/usr/bin/env python3
"""Gather STU_PYTORCH sliding-window sweep results -> CSV + bar/line plots.

Parses lifetime NE / GAUC per task from logs/slurm/o_<jobid>.out across the
sliding-window sweep {128, 256, 512} x 10 seeds (all at max_seq_len=16384),
aggregates mean +/- std across seeds, writes a CSV, and renders bento-styled
bar + line plots with window size on the x-axis.
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

# (window, seed) -> jobid, read from the window jobids file.
# window=0 is the sentinel for full causal attention; those runs are the
# max_seq_len=16384 batch from the seq-length sweep (same model, no window).
JOBS = {}
with open(os.path.join(REPO, "logs", "kr27k_stu_pytorch_window_jobids.txt")) as f:
    for line in f:
        jid, name, W, seed = line.split()
        JOBS[(int(W), int(seed))] = int(jid)

# Full-attention (window=0) point: seq-len sweep's 16384 batch.
for seed, jid in zip(
    range(1, 11),
    [1469665, 1469666, 1469667, 1469668, 1469670,
     1469671, 1469672, 1469673, 1469674, 1469675],
):
    JOBS[(0, seed)] = jid

# 8192 point: seq-len sweep's max_seq_len=8192 full-attention batch (NOT a
# window=8192 run -- it caps total history at 8192 with full attention).
for seed, jid in zip(
    range(1, 11),
    [1469695, 1469696, 1469697, 1469698, 1469699,
     1469700, 1469701, 1469702, 1469703, 1469704],
):
    JOBS[(8192, seed)] = jid

WINDOWS = [128, 256, 512, 8192, 0]
WLABEL = {128: "128", 256: "256", 512: "512", 8192: "8192", 0: "Full"}
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


# values[metric][task][window] = list over seeds
values = {"ne": defaultdict(lambda: defaultdict(list)),
          "gauc": defaultdict(lambda: defaultdict(list))}

for (W, seed), jid in JOBS.items():
    res = parse_job(jid)
    for metric in ("ne", "gauc"):
        for task, val in res[metric].items():
            values[metric][task][W].append(val)

# ---- Build CSV (mean +/- std, one column per window) ----
rows = []
for task in TASKS:
    for metric, label in (("ne", "lifetime_ne"), ("gauc", "lifetime_gauc")):
        row = {"task": task, "metric": label}
        for W in WINDOWS:
            vals = values[metric][task].get(W, [])
            col = f"w{WLABEL[W]}" if W else "full"
            if vals:
                row[col] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
                row[f"{col}_n"] = len(vals)
            else:
                row[col] = ""
                row[f"{col}_n"] = 0
        rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(REPO, "logs", "kr27k_stu_pytorch_window_results.csv")
df.to_csv(csv_path, index=False)
print(f"[wrote] {csv_path}")
print(df.to_string(index=False))

# ---- Bar plot: well-powered tasks (n>=800), NE + GAUC, window on x-axis ----
exec(open(os.path.expanduser(
    "~/.claude/agent-market/plugins/10x-data-scientist/skills/visualization/"
    "bento-plotly/references/bento_style_template.py"
)).read())

PLOT_TASKS = ["is_click", "long_view", "is_profile_enter", "is_like"]
x = [WLABEL[W] for W in WINDOWS]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Lifetime NE (lower is better)",
                    "Lifetime GAUC (higher is better)"),
    horizontal_spacing=0.10,
)

for i, task in enumerate(PLOT_TASKS):
    color = OKABE_ITO[i]
    ne_mean = [np.mean(values["ne"][task][W]) for W in WINDOWS]
    ne_std = [np.std(values["ne"][task][W]) for W in WINDOWS]
    gauc_mean = [np.mean(values["gauc"][task][W]) for W in WINDOWS]
    gauc_std = [np.std(values["gauc"][task][W]) for W in WINDOWS]

    fig.add_trace(go.Bar(
        x=x, y=ne_mean, name=task, legendgroup=task,
        marker_color=color,
        error_y=dict(type="data", array=ne_std, visible=True),
        hovertemplate=f"<b>{task}</b><br>window=%{{x}}<br>NE: %{{y:.4f}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=x, y=gauc_mean, name=task, legendgroup=task, showlegend=False,
        marker_color=color,
        error_y=dict(type="data", array=gauc_std, visible=True),
        hovertemplate=f"<b>{task}</b><br>window=%{{x}}<br>GAUC: %{{y:.4f}}<extra></extra>",
    ), row=1, col=2)

fig.update_xaxes(title_text="Attention window (Full = no window)", type="category", row=1, col=1)
fig.update_xaxes(title_text="Attention window (Full = no window)", type="category", row=1, col=2)
fig.update_yaxes(range=[0.70, 1.02], row=1, col=1)
fig.update_yaxes(range=[0.5, 0.62], row=1, col=2)

# Push subplot titles down so they don't collide with the title/subtitle row.
for ann in fig.layout.annotations:
    ann.y = 1.0

fig.update_layout(
    title=dict(text="<b>Sliding-window sweep</b>"
                    "<br><sub>HSTU local attention, max_seq_len=16384, KuaiRand-27K, mean +/- std over 10 seeds</sub>"),
    barmode="group",
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    margin=dict(r=180, t=130),
    width=1100, height=560,
)
png_path = os.path.join(REPO, "logs", "kr27k_stu_pytorch_window_bar.png")
fig.write_image(png_path, scale=2)
print(f"[wrote] {png_path}")

# ---- Line plot: same panels, trend across window with std error bands ----
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
        mean = np.array([np.mean(values[metric][task][W]) for W in WINDOWS])
        std = np.array([np.std(values[metric][task][W]) for W in WINDOWS])
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
            hovertemplate=f"<b>{task}</b><br>window=%{{x}}<br>{mlabel}: %{{y:.4f}}<extra></extra>",
        ), row=1, col=col)

figl.update_xaxes(title_text="Attention window (Full = no window)", type="category", row=1, col=1)
figl.update_xaxes(title_text="Attention window (Full = no window)", type="category", row=1, col=2)
figl.update_yaxes(range=[0.70, 1.02], row=1, col=1)
figl.update_yaxes(range=[0.50, 0.62], row=1, col=2)

for ann in figl.layout.annotations:
    ann.y = 1.0

figl.update_layout(
    title=dict(text="<b>Sliding-window sweep</b>"
                    "<br><sub>HSTU local attention, max_seq_len=16384, KuaiRand-27K, mean +/- std over 10 seeds</sub>"),
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    margin=dict(r=180, t=130),
    width=1100, height=560,
)

line_path = os.path.join(REPO, "logs", "kr27k_stu_pytorch_window_line.png")
figl.write_image(line_path, scale=2)
print(f"[wrote] {line_path}")
