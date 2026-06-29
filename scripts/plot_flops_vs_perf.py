#!/usr/bin/env python3
"""Pareto plot: compute (FLOPs) vs performance, two modes.

Usage:
  python scripts/plot_flops_vs_perf.py [task] [--encode]

  task      optional single task name (e.g. is_click); default = 4-task average.
  --encode  full-sequence ENCODE FLOPs (cold / training-time: every history token
            re-projected, attention over the whole sequence). Default (no flag) is
            per-candidate INFERENCE FLOPs with a warm cache (KV cache + fixed-size
            recurrent delta state), i.e. steady-state decode cost.

Each method is one point. Two marker series:
  * Full attention @W           (STU_PYTORCH, max_seq_len=W: sees only last W)
  * Full attention @W + Long-term Mem  (STUpDeltaNet: window-W attn + gated-delta
                                        memory over history older than W)

Dims: heads H=4, d_qk=128, d_v=128, d_model=512, 5 layers. FLOPs = 2 x MACs.
Effective history length L_eff = mean(min(real_len, cap)) on a 2001-user sample
(real mean ~12k, median ~8.6k): cap 256->256.0, 512->511.2, 1024->1015.6,
16384->9239.2.
"""

import os
import re
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = "/storage/home/ngocbh/project/gr"
SLURM = os.path.join(REPO, "logs", "slurm")

args = [a for a in sys.argv[1:]]
ENCODE = "--encode" in args
args = [a for a in args if not a.startswith("--")]

ALL_WELL_POWERED = ["is_click", "long_view", "is_profile_enter", "is_like"]
if args:
    TASKS = [args[0]]
    TASK_DESC = args[0]
    OUT_SUFFIX = f"_{args[0]}"
else:
    TASKS = ALL_WELL_POWERED
    TASK_DESC = "4 well-powered tasks"
    OUT_SUFFIX = ""

SEEDS = list(range(1, 11))
WINDOWS = [256, 512, 1024]

ne_re = re.compile(r"eval metric/lifetime_ne/(\w+):\s*([0-9.eE+-]+)")
gauc_re = re.compile(r"eval metric/lifetime_gauc/(\w+):\s*([0-9.eE+-]+)")

FULL16K_JOBS = {s: j for s, j in zip(SEEDS, [
    1469665, 1469666, 1469667, 1469668, 1469670,
    1469671, 1469672, 1469673, 1469674, 1469675])}

L_EFF_CAP = {256: 256.0, 512: 511.2, 1024: 1015.6, 16384: 9239.2}
L_EFF_FULL = L_EFF_CAP[16384]


# ---------------- eval parsing ----------------
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
fa512_jobs = load_jobs(
    os.path.join(REPO, "logs", "kr27k_stu_pytorch_seqlen_jobids.txt"), keep_w=512)
fa_jobs = {**fa_short_jobs, **fa512_jobs}

method_jobs = {"FA@16384": FULL16K_JOBS}
for w in WINDOWS:
    method_jobs[f"FA@{w}"] = {s: fa_jobs[(w, s)] for s in SEEDS}
    method_jobs[f"LTM@{w}"] = {s: pdelta_jobs[(w, s)] for s in SEEDS}


def avg_perf(jobs):
    ne_seed, gauc_seed = [], []
    for s in SEEDS:
        res = parse_job(jobs[s])
        ne_seed.append(np.mean([res["ne"][t] for t in TASKS]))
        gauc_seed.append(np.mean([res["gauc"][t] for t in TASKS]))
    ne_seed, gauc_seed = np.array(ne_seed), np.array(gauc_seed)
    return ne_seed.mean(), ne_seed.std(), gauc_seed.mean(), gauc_seed.std()


# ---------------- FLOPs models ----------------
H, DQK, DV, DM, LAYERS = 4, 128, 128, 512, 5
LIN_PER_TOK = 2 * (DM * (2 * DV + 2 * DQK) * H + (DV * H) * DM)
ATTN_PER_PAIR = 2 * ((DQK + DV) * H)
DELTA_PER_TOK = 2 * (4 * DQK * DV * H)   # full write+read per token (encode)
DELTA_READ = 2 * (DQK * DV * H)          # fixed-state read (inference)


def full_pairs(n):
    return n * (n + 1) / 2.0


def window_pairs(n, w):
    if n <= w:
        return full_pairs(n)
    return w * (w + 1) / 2.0 + (n - w) * w


def flops_fa_encode(cap):
    le = L_EFF_CAP[cap]
    return LAYERS * (LIN_PER_TOK * le + ATTN_PER_PAIR * full_pairs(le))


def flops_ltm_encode(w):
    le = L_EFF_FULL
    return LAYERS * (LIN_PER_TOK * le + ATTN_PER_PAIR * window_pairs(le, w)
                     + DELTA_PER_TOK * le)


def flops_fa_infer(attn_keys):
    return LAYERS * (LIN_PER_TOK + ATTN_PER_PAIR * attn_keys)


def flops_ltm_infer(w):
    return LAYERS * (LIN_PER_TOK + ATTN_PER_PAIR * w + DELTA_READ)


# assemble points
points = {}
for w in WINDOWS:
    nem, nes, gam, gas = avg_perf(method_jobs[f"FA@{w}"])
    flops = flops_fa_encode(w) if ENCODE else flops_fa_infer(w)
    points[f"FA@{w}"] = dict(series="Full attention @W", w=w, flops=flops,
                             ne=nem, ne_e=nes, gauc=gam, gauc_e=gas)
nem, nes, gam, gas = avg_perf(method_jobs["FA@16384"])
flops = flops_fa_encode(16384) if ENCODE else flops_fa_infer(L_EFF_FULL)
points["FA@16384"] = dict(series="Full attention @W", w=16384, flops=flops,
                          ne=nem, ne_e=nes, gauc=gam, gauc_e=gas)
for w in WINDOWS:
    nem, nes, gam, gas = avg_perf(method_jobs[f"LTM@{w}"])
    flops = flops_ltm_encode(w) if ENCODE else flops_ltm_infer(w)
    points[f"LTM@{w}"] = dict(series="Full attention @W + Long-term Mem", w=w,
                             flops=flops, ne=nem, ne_e=nes, gauc=gam, gauc_e=gas)

# ---------------- mode-specific labels ----------------
if ENCODE:
    UNIT, UNIT_LABEL = 1e9, "GFLOPs / user"
    X_TITLE = "Full-sequence encode cost (GFLOPs / user, log)"
    SUBTITLE = ("Full-sequence encode FLOPs (cold / training-time) vs "
                f"performance; KuaiRand-27K, mean +/- std over 10 seeds, {TASK_DESC}")
    MODE_SUFFIX = "_encode"
    HOVER_UNIT = "GFLOP"
else:
    UNIT, UNIT_LABEL = 1e6, "MFLOPs / candidate"
    X_TITLE = "Inference cost (MFLOPs / candidate, log)"
    SUBTITLE = ("Per-candidate inference FLOPs (warm cache) vs performance; "
                f"KuaiRand-27K, mean +/- std over 10 seeds, {TASK_DESC}")
    MODE_SUFFIX = ""
    HOVER_UNIT = "MFLOP/candidate"

# ---------------- plot ----------------
exec(open(os.path.expanduser(
    "~/.claude/agent-market/plugins/10x-data-scientist/skills/visualization/"
    "bento-plotly/references/bento_style_template.py"
)).read())

SER = {
    "Full attention @W": dict(color="#888888", symbol="circle"),
    "Full attention @W + Long-term Mem": dict(color="#D55E00", symbol="diamond"),
}


def wlabel(w):
    return "16k" if w == 16384 else str(w)


fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Lifetime NE (lower is better)",
                    "Lifetime GAUC (higher is better)"),
    horizontal_spacing=0.10,
)

for col, metric, emetric in ((1, "ne", "ne_e"), (2, "gauc", "gauc_e")):
    for sname, style in SER.items():
        pts = sorted([p for p in points.values() if p["series"] == sname],
                     key=lambda p: p["flops"])
        fig.add_trace(go.Scatter(
            x=[p["flops"] / UNIT for p in pts],
            y=[p[metric] for p in pts],
            error_y=dict(type="data", array=[p[emetric] for p in pts],
                         visible=True, thickness=1, width=3),
            mode="lines+markers+text",
            name=sname, legendgroup=sname, showlegend=(col == 1),
            line=dict(color=style["color"], width=1.5, dash="dot"),
            marker=dict(color=style["color"], symbol=style["symbol"], size=12,
                        line=dict(width=1, color="white")),
            text=[f" W={wlabel(p['w'])}" for p in pts],
            textposition="top center", textfont=dict(size=9, color=style["color"]),
            cliponaxis=False,
            customdata=[[wlabel(p["w"]), p["flops"] / UNIT] for p in pts],
            hovertemplate=(f"<b>{sname}</b><br>W=%{{customdata[0]}}<br>"
                           f"FLOPs: %{{customdata[1]:.1f}} {HOVER_UNIT}<br>"
                           f"{metric.upper()}: %{{y:.4f}}<extra></extra>"),
        ), row=1, col=col)

for col in (1, 2):
    fig.update_xaxes(type="log", title_text=X_TITLE, row=1, col=col)
fig.update_yaxes(title_text="Lifetime NE", row=1, col=1)
fig.update_yaxes(title_text="Lifetime GAUC", row=1, col=2)
for ann in fig.layout.annotations:
    ann.y = 1.0

fig.update_layout(
    title=dict(text="<b>Long-term memory: full-context quality at short-window "
                    f"compute cost</b><br><sub>{SUBTITLE}</sub>",
               y=0.97, yanchor="top"),
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    margin=dict(r=260, t=150),
    width=1500, height=580,
)
png = os.path.join(REPO, "logs",
                   f"kr27k_flops_vs_perf_pareto{MODE_SUFFIX}{OUT_SUFFIX}.png")
fig.write_image(png, scale=2)
print(f"[wrote] {png}\n")

print(f"mode={'ENCODE (full-seq)' if ENCODE else 'INFERENCE (per-candidate)'}  "
      f"task={TASK_DESC}")
print(f"{'method':<10} {'W':>6} {UNIT_LABEL:>16} {'NE':>16} {'GAUC':>16}")
for name in ["FA@256", "FA@512", "FA@1024", "FA@16384",
             "LTM@256", "LTM@512", "LTM@1024"]:
    p = points[name]
    print(f"{name:<10} {wlabel(p['w']):>6} {p['flops']/UNIT:>16.2f} "
          f"{p['ne']:.4f}+/-{p['ne_e']:.4f}  {p['gauc']:.4f}+/-{p['gauc_e']:.4f}")
