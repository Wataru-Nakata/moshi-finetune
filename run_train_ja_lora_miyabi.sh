#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -W group_list=gj18
#PBS -r y
#PBS -N moshi_lora_ja
#PBS -j oe

set -euo pipefail

CONFIG=example/moshi_7B_ja_wds.yaml
PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
cd "${PROJECT_DIR}"

export PATH="${HOME}/.local/bin:/work/gj18/e43001/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH="/work/gj18/e43001/miniconda3/lib:/work/gj18/e43001/miniconda3/bin:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export TORCHDYNAMO_DISABLE=1
export CC=/usr/bin/gcc
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_DIR="${PROJECT_DIR}/runs"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Config: ${CONFIG}"

nvidia-smi --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 60 > "${PROJECT_DIR}/logs/gpu_train_lora.log" 2>&1 &
GPU_LOG_PID=$!

uv run torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    train.py "${CONFIG}"

kill ${GPU_LOG_PID} 2>/dev/null || true
echo "Job finished: $(date)"
