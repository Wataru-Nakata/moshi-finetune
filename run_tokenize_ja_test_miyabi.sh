#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -W group_list=gj18
#PBS -N tokenize_test
#PBS -j oe

set -euo pipefail

PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
PODCAST_DIR=/work/gj18/e43001/github.com/podcast_crawl

INPUT_DIR=${PODCAST_DIR}/data/wds_ja_filtered
TRANSCRIPTS_DIR=${PODCAST_DIR}/data/wds_ja_transcripts
OUTPUT_DIR=${PROJECT_DIR}/data/tokens_test

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

# Launch 8 parallel tokenize processes on 1 GPU (mimi ~1GB each, 8x ~8GB)
PROCS=8
TOTAL_SHARDS=${PROCS}
PIDS=()
for i in $(seq 0 $((PROCS - 1))); do
    ${PROJECT_DIR}/.venv/bin/python3 scripts/tokenize_wds.py \
        --input "${INPUT_DIR}" \
        --transcripts "${TRANSCRIPTS_DIR}" \
        --output "${OUTPUT_DIR}" \
        --duration-sec 300 \
        --shard ${i} \
        --num-shards ${TOTAL_SHARDS} \
        \
        > "${PROJECT_DIR}/logs/tokenize_test_${i}.log" 2>&1 &
    PIDS+=($!)
done
echo "Launched ${#PIDS[@]} processes"

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done
echo "=== TOKENIZE DONE (failures: ${FAIL}/${PROCS}) ==="

echo "=== TOKENIZE DONE ==="
echo "Output files:"
find "${OUTPUT_DIR}" -name "*.jsonl" | head -5
echo "Lines in first JSONL:"
find "${OUTPUT_DIR}" -name "*.jsonl" | head -1 | xargs wc -l
echo "First line preview:"
find "${OUTPUT_DIR}" -name "*.jsonl" | head -1 | xargs head -1 | python3 -c "
import sys, json
rec = json.loads(sys.stdin.read())
print('key:', rec['key'])
print('n_codebooks:', len(rec['audio_tokens_ch0main']))
print('n_frames:', len(rec['audio_tokens_ch0main'][0]) if rec['audio_tokens_ch0main'] else 0)
print('#align_ch0:', len(rec['alignments_ch0']))
print('#align_ch1:', len(rec['alignments_ch1']))
print('duration:', rec['duration_sec'])
print('real_frames:', rec['num_real_frames'])
"

echo "Job finished: $(date)"
