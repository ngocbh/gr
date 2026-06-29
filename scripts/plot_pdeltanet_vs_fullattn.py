#!/usr/bin/env python3
"""STUpDeltaNet (non-overlapping split) vs full-attention baselines, by task.

Question: can a W-token attention window plus a gated-DeltaNet long memory
(STUpDeltaNet) recover the accuracy of full attention over the whole 16k
history? So per window W in {256,512,1024} we compare:

    Full attn @W   (STU_PYTORCH, max_seq_len=W, no long memory)
    STUpDeltaNet @W (attn covers recent W; delta summarizes history older than W)

against the Full attn @16k reference (single bar, shared across windows).

Layout within each task block (manual x-positions; pair per window, gap between):

    Full@16k | [FA@256 pDN@256]  [FA@512 pDN@512]  [FA@1024 pDN@1024]

Window size printed on each bar. 3 colors: Full attn @W (gray),
STUpDeltaNet @W (vermillion), Full attn @16k (blue). Groups = 4 well-powered
tasks + Average. Two panels: lifetime NE and GAUC, mean +/- std over 10 seeds.
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
WINDOWS = [256, 512, 1024]

ne_re = re.compile(r"eval metric/lifetime_ne/(\w+):\s*([0-9.eE+-]+)")
gauc_re = re.compile(r"eval metric/lifetime_gauc/(\w+):\s*([0-9.eE+-]+)")

# Full attention over the entire 16k history (STU_PYTORCH, max_seq_len=16384).
FULL16K_JOBS = {s: j for s, j in zip(SEEDS, [
    1469665, 1469666, 1469667, 1469668, 1469670,
    1469671, 1469672, 1469673, 1469674, 1469675])}

COL_FA = "#888888"    # gray       -- full attn @W (short context)
COL_PDN = "#D55E00"   # vermillion -- STUpDeltaNet @W (W attn + delta memory)
COL_REF = "#0072B2"   # blue       -- full attn @16k reference

# local x-offset inside a task block; window pairs adjacent, gap between.
LOCAL = {
    "Full @16k": 0.0,
    "FA@256": 1.6, "pDN@256": 2.6,
    "FA@512": 4.1, "pDN@512": 5.1,
    "FA@1024": 6.6, "pDN@1024": 7.6,
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


def load_jobs(path, keep_w=None):
    """Map (W, seed) -> jobid; optionally restrict to a single window."""
    jobs = {}
    with open(path) as f:
        for line in f:
            jid, name, W, seed = line.split()
            W, seed = int(W), int(seed)
            if keep_w is not None and W != keep_w:
                continue
            jobs[(W, seed)] = int(jid)
    return jobs


pdelta_jobs = load_jobs(os.path.join(REPO, "logs", "kr27k_pdeltanet_jobids.txt"))
fa_short_jobs = load_jobs(os.path.join(REPO, "logs", "kr27k_fullattn_short_jobids.txt"))
# Full attn @512 comes from the earlier seqlen sweep (max_seq_len=512).
fa512_jobs = load_jobs(
    os.path.join(REPO, "logs", "kr27k_stu_pytorch_seqlen_jobids.txt"), keep_w=512)

fa_jobs = {**fa_short_jobs, **fa512_jobs}  # full attn @ {256,512,1024}

method_jobs = {"Full @16k": FULL16K_JOBS}
for w in WINDOWS:
    method_jobs[f"FA@{w}"] = {s: fa_jobs[(w, s)] for s in SEEDS}
    method_jobs[f"pDN@{w}"] = {s: pdelta_jobs[(w, s)] for s in SEEDS}

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
    cats = {k: {"x": [], "y": [], "e": [], "t": []}
            for k in ("Full attn @16k", "Full attn @W", "STUpDeltaNet @W")}
    for gi, g in enumerate(GROUPS):
        b = gi * STRIDE
        mu, sd = val("Full @16k", metric, g)
        c = cats["Full attn @16k"]
        c["x"].append(b + LOCAL["Full @16k"]); c["y"].append(mu)
        c["e"].append(sd); c["t"].append("16k")
        for w in WINDOWS:
            mu, sd = val(f"FA@{w}", metric, g)
            c = cats["Full attn @W"]
            c["x"].append(b + LOCAL[f"FA@{w}"]); c["y"].append(mu)
            c["e"].append(sd); c["t"].append(str(w))
            mu, sd = val(f"pDN@{w}", metric, g)
            c = cats["STUpDeltaNet @W"]
            c["x"].append(b + LOCAL[f"pDN@{w}"]); c["y"].append(mu)
            c["e"].append(sd); c["t"].append(str(w))
    return cats


CAT_COLOR = {"Full attn @16k": COL_REF, "Full attn @W": COL_FA,
             "STUpDeltaNet @W": COL_PDN}

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
fig.update_yaxes(range=[0.78, 0.98], row=1, col=1)
fig.update_yaxes(range=[0.50, 0.60], row=1, col=2)
for ann in fig.layout.annotations:
    ann.y = 1.0

fig.update_layout(
    title=dict(text="<b>STUpDeltaNet vs full attention: does delta memory "
                    "recover full-context accuracy?</b>"
                    "<br><sub>Attn window W (256/512/1024) + gated-delta long "
                    "memory vs full attn @W and full attn @16k; KuaiRand-27K, "
                    "mean +/- std over 10 seeds</sub>"),
    barmode="overlay",
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    margin=dict(r=190, t=120),
    width=1500, height=560,
)
add_source(fig, "Internal KuaiRand-27K eval, FY26")
png = os.path.join(REPO, "logs", "kr27k_pdeltanet_vs_fullattn_bar.png")
fig.write_image(png, scale=2)
print(f"[wrote] {png}")

# ---- terse numeric summary (Average over the 4 well-powered tasks) ----
print("\nAverage over 4 well-powered tasks (mean +/- std, n=10):")
print(f"{'method':<16} {'NE':>16} {'GAUC':>16}")
ref_ne = val("Full @16k", "ne", "Average")
ref_gauc = val("Full @16k", "gauc", "Average")
print(f"{'Full @16k':<16} {ref_ne[0]:.4f}+/-{ref_ne[1]:.4f}  "
      f"{ref_gauc[0]:.4f}+/-{ref_gauc[1]:.4f}")
for w in WINDOWS:
    for tag in (f"FA@{w}", f"pDN@{w}"):
        ne = val(tag, "ne", "Average")
        gauc = val(tag, "gauc", "Average")
        print(f"{tag:<16} {ne[0]:.4f}+/-{ne[1]:.4f}  "
              f"{gauc[0]:.4f}+/-{gauc[1]:.4f}")
