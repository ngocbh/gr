# AGENTS.md

This file provides guidance to coding agents and contributors working in this repository.

## What this repo is

Generative Recommenders (GR) — code for *"Actions Speak Louder than Words: Trillion-Parameter
Sequential Transducers for Generative Recommendations"* (ICML'24). It reformulates classical DLRM
recommendation as a generative sequence-modeling problem, centered on the **HSTU** architecture
(Hierarchical Sequential Transduction Unit) and the M-FALCON inference algorithm.

There are **two largely independent code paths** in one repo. Know which one you are in before editing:

1. **`generative_recommenders/research/`** — the public *academic* experiments from the paper
   (traditional sequential retrieval on MovieLens / Amazon Reviews). Self-contained, PyTorch-level,
   no custom kernels required. Entry point: root **`main.py`** driven by **`configs/*.gin`**.

2. **`generative_recommenders/{modules,ops,dlrm_v3}/`** — the *production-grade* ranking stack
   (DLRM-v3) with hardware-optimized kernels (Triton + CUDA/CUTLASS) and TorchRec sharding.
   Entry points: **`dlrm_v3/train/train_ranker.py`** and **`dlrm_v3/inference/main.py`**.

The two paths each have their own HSTU implementation (`research/modeling/sequential/hstu.py` vs
`modules/stu.py` + `modules/hstu_transducer.py`). Do not assume a change in one propagates to the other.

## Common commands

Everything runs from the repo root. A convenience wrapper `scripts/train.sh` and a SLURM wrapper
`scripts/wrapper.sh` exist (see "Local conventions" below); the raw commands are:

```bash
# Setup
pip3 install -r requirements.txt        # or: pip install -e .

# --- Research path (paper reproduction) ---
mkdir -p tmp/ && python3 preprocess_public_data.py     # download + preprocess ml-1m/ml-20m/amzn-books
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345
tensorboard --logdir exps/ --port 24001 --bind_all     # logs written to exps/ by default

# --- DLRM-v3 path (production ranking) ---
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/train/train_ranker.py \
    --dataset debug --mode train
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/inference/main.py --dataset debug

# --- Synthetic MovieLens-3B via fractal expansion ---
python3 run_fractal_expansion.py --input-csv-file tmp/ml-20m/ratings.csv \
    --write-dataset True --output-prefix tmp/ml-3b/
```

### Tests

Tests use **`unittest` + `hypothesis`** (property-based). **Most kernel/module tests require a GPU** —
they self-skip via `@unittest.skipIf(*gpu_unavailable)` when none is present, so a clean run on CPU
can be misleadingly empty.

```bash
python -m pytest generative_recommenders/ops/tests/layer_norm_test.py                # one file
python -m pytest generative_recommenders/ops/tests/layer_norm_test.py -k LayerNorm   # one test
python -m unittest generative_recommenders.modules.tests.stu_test                    # also works
```

### Building the CUDA attention kernels (`ops/cpp`, optional, needs the CUTLASS submodule)

Only needed for the H100/H200 FlashAttention-V3-based HSTU attention. Pure Triton/PyTorch paths work without it.

```bash
git submodule update --init --recursive          # pulls ops/cpp/cutlass
cd generative_recommenders/ops/cpp && python setup.py install
```

Build is controlled by `FLASH_ATTENTION_*` env vars (see `ops/cpp/setup.py`), e.g. `FLASH_ATTENTION_FORCE_BUILD=TRUE`.

## Architecture — the concepts that span multiple files

**gin-config drives everything.** Model, encoder, loss, data loader, and hyperparameters are all
selected/wired through `@gin.configurable` functions bound in `.gin` files — there is almost no
argument passing in Python. To understand a run, read its `.gin` first. Research configs live in
`configs/<dataset>/`; DLRM-v3 configs in `dlrm_v3/{train,inference}/gin/` (selected by the
`--dataset` key mapped in `SUPPORTED_CONFIGS` inside `train_ranker.py`). Override any binding on the
CLI via `--gin_bindings=fn.param=value`.

**Multi-backend kernel dispatch (`HammerKernel`).** The heart of `ops/` and `modules/`. Ops are
implemented multiple times and selected at runtime by the `HammerKernel` enum
(`PYTORCH`, `TRITON`, `TLX`, `TRITON_CC`, `CUDA`, `CUTEDSL`) — defined in `generative_recommenders/common.py`.
Layout of an op like HSTU attention:
- `ops/hstu_attention.py` — the **dispatcher** that branches on the kernel enum.
- `ops/pytorch/pt_*.py` — reference PyTorch implementation (correctness baseline for tests).
- `ops/triton/triton_*.py` — Triton kernels (the large files; the efficiency work).
- `ops/cpp/*.{cpp,cu}` + `ops/cpp/hstu_attention/` — CUDA/CUTLASS kernels (SM90 FlashAttention-V3 style).

Modules subclass **`HammerModule`** (in `common.py`), which owns `hammer_kernel` selection and the
`inference`/`triton_cc` toggles. When adding an op, add all relevant backends and a test comparing
the Triton/CUDA output against the PyTorch reference (this is the dominant test pattern).

**Jagged tensors are pervasive.** Sequences are variable-length and stored jagged (values +
`seq_offsets`) rather than padded dense, to avoid wasting compute on long-tail sequences. See
`ops/jagged_tensors.py`, `ops/{triton,pytorch}/*jagged*`, and `ops/cpp/*jagged*`. Most kernels take
`seq_offsets` and `max_seq_len`; a large amount of complexity exists purely to operate on this layout.

**HSTU building blocks.**
- Research path: `research/modeling/sequential/hstu.py` and variants `hstu_attnres.py`,
  `hstu_mhc.py`, `hstu_neutreno.py`; `sasrec.py` is the baseline. Models are constructed by
  `@gin.configurable` factories in `research/modeling/sequential/encoder_utils.py`
  (`hstu_encoder`, `sasrec_encoder`, etc.).
- Production path: `modules/stu.py` (`STULayer`/`STUStack` and NeuTRENO/AttnRes/mHC subclasses),
  composed by `modules/hstu_transducer.py`, with the full ranking model in `modules/dlrm_hstu.py`.
  Input handling in `modules/{preprocessors,contextual_interleave_preprocessor,action_encoder,content_encoder}.py`;
  multi-task heads in `modules/multitask_module.py`.

**Retrieval vs ranking.** The research path is retrieval (candidate generation): similarity /
top-k modules live in `research/indexing/` and `research/rails/` (MIPS brute force + MoL — Mixture
of Logits — similarity). DLRM-v3 is ranking (scoring a candidate set) with dense/sparse predict
modules under `dlrm_v3/inference/`.

**Training loops.** Both paths use `torch.multiprocessing.spawn` with one process per GPU and DDP;
research uses plain `DistributedDataParallel` (`research/trainer/train.py`, `train_fn` is the gin
entry), DLRM-v3 uses TorchRec sharding / DMP (`dlrm_v3/train/utils.py`).

## Local conventions (this checkout's customizations over upstream)

This is a SLURM cluster checkout (H200 GPUs, conda env `gr`). Some files here are **not in upstream**:

- **`scripts/train.sh [research|dlrm_v3] <config-or-dataset> [args...]`** — unified launcher. Sources
  `.env` for defaults (`GR_CONDA_ENV`, `GR_DATA_ROOT`, `GR_CKPTS_ROOT`, `GR_EXPS_ROOT`,
  `GR_WANDB_ENABLED`, `WANDB_*`). Prefer this over invoking `main.py` directly for consistency.
- **`scripts/wrapper.sh`** — `sbatch scripts/wrapper.sh <command...>` to submit to SLURM
  (defaults: partition `h200`, `--gres=gpu:h200:4`, override on the CLI).
- GPU experiments in this checkout use the shared H200 QoS
  **`h200_mrs_shared`**. Do not submit them to `h200_dev`.
- **`scripts/submit_attention_experiments.sh [core|ab|fohstu|fohstu-repl|fosoftmax|softmax-canonical|moments|hybrid|lift-ml20|tanh|signed-additive|signed-lift]`** creates a read-only,
  checksummed source snapshot and submits the controlled attention arrays from it.
  `core` submits only the corrected softmax A/B and FoHSTU baseline arrays;
  mechanism-specific arrays remain explicit modes. `lift-ml20` submits the seed-42
  ML-20M full FoHSTU, local FoHSTU W32, and hybrid/LIFT W32 target-regression array.
  `tanh` submits the seed-42 ML-20M HSTU attention-score SiLU-to-tanh ablation;
  it has the exact HSTU named trainable inventory (38,913,120 total, 5,255,776 mixer).
  A separate fixed `0.5 * tanh` arm is intentionally omitted: HSTU immediately
  LayerNorms the attention output, removing any fixed positive scale up to epsilon.
  `signed-lift` submits the parameter-matched ML-1M W32 local, identity-tail,
  signed-tanh-tail, and abs-tanh-tail screen over seeds 42--44.
  Use `pueue` for detached submission/monitoring; the daemon runs on the login node.
- **`scripts/inspect_hybrid_gains.py CHECKPOINT...`** reports the learned per-layer/head
  `rho` and bounded tail gain `alpha = 2 tanh(rho / 2)` for local/hybrid FoHSTU runs.
- **Weights & Biases** integration was added to `research/trainer/train.py` (`train_fn.wandb_*`
  gin params). Enable with `GR_WANDB_ENABLED=1` via `train.sh`, or
  `--gin_bindings=train_fn.wandb_enabled=True`. It is optional and degrades gracefully if `wandb`
  is not installed.
- **`scripts/*`** contains many experiment sweep/analysis helpers (seqlen, attnlen, deltanet, mhc,
  stu_pytorch sweeps + plotting) and `stu_deltanet.py` / `dynamic_stu.py` module variants — these
  are research explorations layered on top of the paper code.

## Code style

- Python 3.10+. Files carry a **Pyre** header — `# pyre-strict` (new/typed code) or `# pyre-unsafe`.
  Match the header of the file you edit and keep annotations valid under it.
- Formatting/imports follow Meta tooling (**black** + **usort**): 4-space indent, sorted imports.
  (The "2 spaces / 80 char" line in `CONTRIBUTING.md` is boilerplate and does **not** match the
  actual Python here — follow the surrounding code.)
- Kernel files under `ops/triton/` are very large and autotuned; prefer targeted edits and rely on
  the PyTorch-reference comparison tests to validate changes.
