#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gj18
#PBS -N tokenize_quick
#PBS -j oe

set -euo pipefail

PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
PODCAST_DIR=/work/gj18/e43001/github.com/podcast_crawl

INPUT_DIR=${PODCAST_DIR}/data/wds_ja_filtered
TRANSCRIPTS_DIR=${PODCAST_DIR}/data/wds_ja_transcripts
OUTPUT_DIR=${PROJECT_DIR}/data/tokens_quick

cd "${PROJECT_DIR}"

export PATH="${HOME}/.local/bin:/work/gj18/e43001/miniconda3/bin:${PATH}"
NVIDIA_BASE=/work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9
CT2_DIR=/work/gj18/e43001/github.com/CTranslate2/install
export LD_LIBRARY_PATH="${CT2_DIR}/lib:${CT2_DIR}/lib64:${NVIDIA_BASE}/math_libs/13.0/targets/sbsa-linux/lib:${NVIDIA_BASE}/cuda/lib64:/work/gj18/e43001/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export TORCHDYNAMO_DISABLE=1

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Job started: $(date)"
echo "Host: $(hostname)"

# 8 procs, 1 shard each
PROCS=8
PIDS=()
for i in $(seq 0 $((PROCS - 1))); do
    ${PROJECT_DIR}/.venv/bin/python3 scripts/tokenize_wds.py \
        --input "${INPUT_DIR}" \
        --transcripts "${TRANSCRIPTS_DIR}" \
        --output "${OUTPUT_DIR}" \
        --duration-sec 300 \
        --shard ${i} \
        --num-shards ${PROCS} \
        --max-shards 1 \
        > "${PROJECT_DIR}/logs/tokenize_quick_${i}.log" 2>&1 &
    PIDS+=($!)
done
echo "Launched ${#PIDS[@]} processes"

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done
echo "Failures: ${FAIL}/${PROCS}"

echo "=== OUTPUT ==="
find "${OUTPUT_DIR}" -name "*.jsonl" | head -10
for f in $(find "${OUTPUT_DIR}" -name "*.jsonl" | head -3); do
    echo "$f: $(wc -l < $f) lines"
done

echo "=== GPU MEMORY ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "Job finished: $(date)"
