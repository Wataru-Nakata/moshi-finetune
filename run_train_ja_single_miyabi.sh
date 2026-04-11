#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -W group_list=gj18
#PBS -r y
#PBS -N moshi_train_ja_1n
#PBS -j oe

# Single-node (1 GH200) debug / smoke-test training job. Use this first to
# verify config and data before submitting the multi-node variant.

set -euo pipefail

CONFIG=example/moshi_7B_ja_wds.yaml

PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
cd "${PROJECT_DIR}"

export PATH="${HOME}/.local/bin:/work/gj18/e43001/miniconda3/bin:${PATH}"
# miniconda libs are needed for libopus.so.0 (pulled in by sphn at runtime).
export LD_LIBRARY_PATH="/work/gj18/e43001/miniconda3/lib:/work/gj18/e43001/miniconda3/bin:${LD_LIBRARY_PATH:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# Pick up wandb credentials from ~/.netrc or the env. Set WANDB_API_KEY
# in your shell profile (never commit it).
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_DIR="${PROJECT_DIR}/runs"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Config: ${CONFIG}"

nvidia-smi --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 60 > "${PROJECT_DIR}/logs/gpu_train_single.log" 2>&1 &
GPU_LOG_PID=$!

uv run torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    train.py "${CONFIG}"

kill ${GPU_LOG_PID} 2>/dev/null || true
echo "Job finished: $(date)"
