#!/bin/bash
#PBS -q regular-g
#PBS -l select=8
#PBS -l walltime=48:00:00
#PBS -W group_list=gj18
#PBS -r y
#PBS -N moshi_lora_ja_8n
#PBS -j oe

set -euo pipefail

NUM_NODES=8
GPUS_PER_NODE=1
CONFIG=example/moshi_7B_ja_wds.yaml

PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
cd "${PROJECT_DIR}"

export PATH="${HOME}/.local/bin:/work/gj18/e43001/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH="/work/gj18/e43001/miniconda3/lib:/work/gj18/e43001/miniconda3/bin:${LD_LIBRARY_PATH:-}"

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ibP2s2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export CC=/usr/bin/gcc

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_DIR="${PROJECT_DIR}/runs"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/runs"

MASTER_ADDR=$(head -n 1 "${PBS_NODEFILE}" | awk '{print $1}')
MASTER_PORT=29500

echo "Job started: $(date)"
echo "NUM_NODES=${NUM_NODES}"
echo "MASTER_ADDR=${MASTER_ADDR}"
cat "${PBS_NODEFILE}" | sort -u

mpirun -np ${NUM_NODES} --map-by ppr:1:node --max-restarts 0 \
    bash -c '
    set +e
    export TORCHDYNAMO_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=4
    export TOKENIZERS_PARALLELISM=false
    export CC=/usr/bin/gcc
    NODE_RANK=${OMPI_COMM_WORLD_RANK}
    LOG_FILE='"${PROJECT_DIR}"'/logs/train_lora_node${NODE_RANK}.log

    echo "Node ${NODE_RANK}/'"${NUM_NODES}"' on $(hostname)" | tee -a "${LOG_FILE}"

    nvidia-smi --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv -l 60 > '"${PROJECT_DIR}"'/logs/gpu_train_lora_${NODE_RANK}.log 2>&1 &
    GPU_LOG_PID=$!

    uv run torchrun \
        --nnodes '"${NUM_NODES}"' \
        --nproc_per_node '"${GPUS_PER_NODE}"' \
        --node_rank ${NODE_RANK} \
        --master_addr '"${MASTER_ADDR}"' \
        --master_port '"${MASTER_PORT}"' \
        train.py '"${CONFIG}"' 2>&1 | tee -a "${LOG_FILE}" \
        || echo "Node ${NODE_RANK} failed with exit code $? at $(date)"

    kill ${GPU_LOG_PID} 2>/dev/null || true
    echo "Node ${NODE_RANK} finished at $(date)" | tee -a "${LOG_FILE}"
    exit 0
'

echo "Job finished: $(date)"
