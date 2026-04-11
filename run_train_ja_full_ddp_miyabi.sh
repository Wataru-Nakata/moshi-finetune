#!/bin/bash
#PBS -q regular-g
#PBS -l select=8
#PBS -l walltime=48:00:00
#PBS -W group_list=gj18
#PBS -r y
#PBS -N moshi_train_ja_ddp
#PBS -j oe

# Full fine-tuning of Moshi-7B on Japanese podcast dialogue using DDP
# (replicated model per GH200, not FSDP).
#
# Each Miyabi-G node has 1 GH200 GPU (120 GB), so world_size == NUM_NODES.
# Moshi-7B in bf16 + AdamW + gradient checkpointing fits with headroom,
# which is why DDP is preferred over FSDP here.

set -euo pipefail

NUM_NODES=8
GPUS_PER_NODE=1
CONFIG=example/moshi_7B_ja_wds_full_ddp.yaml

PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
cd "${PROJECT_DIR}"

export PATH="${HOME}/.local/bin:/work/gj18/e43001/miniconda3/bin:${PATH}"
# miniconda libs are needed for libopus.so.0 (pulled in by sphn at runtime).
export LD_LIBRARY_PATH="/work/gj18/e43001/miniconda3/lib:/work/gj18/e43001/miniconda3/bin:${LD_LIBRARY_PATH:-}"

# NCCL / torch tuning for GH200
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^lo,docker
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# wandb credentials propagate via env; set WANDB_API_KEY in your shell.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_DIR="${PROJECT_DIR}/runs"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs"

MASTER_ADDR=$(head -n 1 "${PBS_NODEFILE}" | awk '{print $1}')
MASTER_PORT=29501

echo "Job started: $(date)"
echo "NUM_NODES=${NUM_NODES} GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "CONFIG=${CONFIG}"
cat "${PBS_NODEFILE}" | sort -u

mpirun -np ${NUM_NODES} --map-by ppr:1:node --max-restarts 0 \
    -x PATH -x LD_LIBRARY_PATH -x LIBRARY_PATH -x CPATH \
    -x NCCL_IB_DISABLE -x NCCL_DEBUG \
    -x NCCL_SOCKET_IFNAME -x PYTORCH_CUDA_ALLOC_CONF -x OMP_NUM_THREADS \
    -x TOKENIZERS_PARALLELISM \
    -x WANDB_API_KEY -x WANDB_DIR \
    bash -c '
    set +e
    NODE_RANK=${OMPI_COMM_WORLD_RANK}
    LOG_FILE='"${PROJECT_DIR}"'/logs/train_ddp_node${NODE_RANK}.log

    echo "Node ${NODE_RANK}/'"${NUM_NODES}"' started on $(hostname) at $(date)" | tee -a "${LOG_FILE}"

    nvidia-smi --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv -l 60 > '"${PROJECT_DIR}"'/logs/gpu_train_ddp_${NODE_RANK}.log 2>&1 &
    GPU_LOG_PID=$!

    uv run torchrun \
        --nnodes '"${NUM_NODES}"' \
        --nproc_per_node '"${GPUS_PER_NODE}"' \
        --node_rank ${NODE_RANK} \
        --master_addr '"${MASTER_ADDR}"' \
        --master_port '"${MASTER_PORT}"' \
        train.py '"${CONFIG}"' 2>&1 | tee -a "${LOG_FILE}" \
        || echo "Node ${NODE_RANK} training failed with exit code $? at $(date)"

    kill ${GPU_LOG_PID} 2>/dev/null || true
    echo "Node ${NODE_RANK} finished at $(date)" | tee -a "${LOG_FILE}"
    exit 0
'

echo "Job finished: $(date)"
