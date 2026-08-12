#!/usr/bin/env bash
# Probe HSTU-large training dynamics on ML-20M (research path, single H200).
#
# Submit:  sbatch scripts/sbatch_probe_ml20m.sh
# Watch:   tail -f /checkpoints/ngocbh/longhstu/logs/slurm/probe_<jobid>.out
#
# Trains the frozen ML-20M HSTU-large config with per-layer dynamics probing
# (input / pre-residual / post-residual norms + cos(x_in,x_out) collapse signal)
# streamed to wandb, and keeps only the last checkpoint.

#SBATCH --job-name=hstu-probe-ml20m
#SBATCH --partition=h200
#SBATCH --qos=h200_mrs_shared
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=/checkpoints/ngocbh/longhstu/logs/slurm/probe_%j.out
#SBATCH --error=/checkpoints/ngocbh/longhstu/logs/slurm/probe_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"

# --- experiment I/O locations (survive train.sh's `.env` sourcing: not set there) ---
export GR_DATA_ROOT=/checkpoints/ngocbh/longhstu/datasets
export GR_CKPTS_ROOT=/checkpoints/ngocbh/longhstu/checkpoints
export GR_EXPS_ROOT=/checkpoints/ngocbh/longhstu/exps
export GR_WANDB_ENABLED=1

# Frozen ML-20M config recommends expandable_segments to avoid fragmentation OOM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$GR_CKPTS_ROOT" "$GR_EXPS_ROOT" /checkpoints/ngocbh/longhstu/logs/slurm

echo "=== slurm job ==="
date
hostname
echo "job_id: ${SLURM_JOB_ID:-N/A}  gpus: ${SLURM_GPUS_ON_NODE:-N/A}  cuda_visible: ${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo "GR_DATA_ROOT=$GR_DATA_ROOT"
echo "GR_CKPTS_ROOT=$GR_CKPTS_ROOT"
echo "GR_EXPS_ROOT=$GR_EXPS_ROOT"

# train.sh handles: conda activate gr, proxy unset (wandb egress), wandb gin
# bindings from env, and exec of main.py (world_size = torch.cuda.device_count()).
# The trailing --gin_bindings set the wandb run name/tags for this experiment.
# Derive a per-job master port so DDP's localhost store doesn't collide when the
# shared QOS packs multiple jobs onto one node.
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))
echo "MASTER_PORT=$MASTER_PORT"

bash scripts/train.sh research configs/ml-20m/hstu-probe-dynamics.gin \
  --master_port="$MASTER_PORT" \
  "--gin_bindings=train_fn.wandb_run_name='ml20m-hstu-probe-dynamics'" \
  "--gin_bindings=train_fn.wandb_tags=['ml-20m','hstu-large','probe','dynamics']"

echo "=== done ==="
date
