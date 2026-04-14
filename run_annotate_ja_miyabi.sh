#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -W group_list=gj18
#PBS -J 0-7
#PBS -r y
#PBS -N annotate_wds_ja
#PBS -j oe

# Whisper annotation of podcast_crawl webdataset shards for moshi-finetune.
#
# Each array task owns a disjoint subset of shards via
#   --shard $PBS_ARRAY_INDEX --num-shards 64
# so all 64 tasks can run concurrently against the same input/output roots.
#
# Output: one .jsonl file per input .tar.gz, mirrored under TRANSCRIPT_DIR.
# Safe to re-run: existing transcripts are skipped.

set -euo pipefail

# Number of parallel Whisper processes PER NODE. Each loads its own model
# and owns a disjoint shard subset. Whisper medium uses ~6 GB GPU memory,
# so 16 processes × 6 GB = 96 GB fits in GH200's 120 GB.
PROCS_PER_NODE=2

# Total logical shards = PBS array size × PROCS_PER_NODE.
# For a single-node test: NUM_NODES=1, TOTAL_SHARDS=16.
# For 64-node production: set #PBS -J 0-63 and NUM_NODES=64, TOTAL_SHARDS=1024.
NUM_NODES=8
NODE_INDEX=${PBS_ARRAY_INDEX:-0}
TOTAL_SHARDS=$((NUM_NODES * PROCS_PER_NODE))

PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
AUDIO_DIR=/work/gj18/e43001/github.com/podcast_crawl/data/wds_ja_filtered
TRANSCRIPT_DIR=/work/gj18/e43001/github.com/podcast_crawl/data/wds_ja_transcripts

cd "${PROJECT_DIR}"

export PATH="${HOME}/.local/bin:/work/gj18/e43001/miniconda3/bin:${PATH}"
NVIDIA_BASE=/work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9
CT2_DIR=/work/gj18/e43001/github.com/CTranslate2/install
export LD_LIBRARY_PATH="${CT2_DIR}/lib:${CT2_DIR}/lib64:${NVIDIA_BASE}/math_libs/13.0/targets/sbsa-linux/lib:${NVIDIA_BASE}/cuda/lib64:/work/gj18/e43001/miniconda3/lib:/work/gj18/e43001/miniconda3/bin:${LD_LIBRARY_PATH:-}"

LOG_DIR=${PROJECT_DIR}/logs
mkdir -p "${LOG_DIR}"
GPU_LOG="${LOG_DIR}/gpu_annotate_${NODE_INDEX}.log"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Node: ${NODE_INDEX}/${NUM_NODES}, procs_per_node: ${PROCS_PER_NODE}, total_shards: ${TOTAL_SHARDS}"
echo "Audio: ${AUDIO_DIR}"
echo "Transcripts: ${TRANSCRIPT_DIR}"

nvidia-smi --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 60 > "${GPU_LOG}" 2>&1 &
GPU_LOG_PID=$!

# Launch PROCS_PER_NODE parallel Whisper processes, each with a unique shard index.
PIDS=()
BASE_SHARD=$((NODE_INDEX * PROCS_PER_NODE))
for i in $(seq 0 $((PROCS_PER_NODE - 1))); do
    SHARD_IDX=$((BASE_SHARD + i))
    ${PROJECT_DIR}/.venv/bin/python3 scripts/annotate_wds.py \
        --input "${AUDIO_DIR}" \
        --output "${TRANSCRIPT_DIR}" \
        --lang ja \
        --whisper-model medium \
        --batch-size 128 \
        --compute-type float16 \
        --shard "${SHARD_IDX}" \
        --num-shards "${TOTAL_SHARDS}" \
        --local-rank 0 \
        > "${LOG_DIR}/annotate_${NODE_INDEX}_${i}.log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} processes: ${PIDS[*]}"

# Wait for all to finish.
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

kill ${GPU_LOG_PID} 2>/dev/null || true
echo "Job finished: $(date), failures: ${FAIL}/${PROCS_PER_NODE}"
