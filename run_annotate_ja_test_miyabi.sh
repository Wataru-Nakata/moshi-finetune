#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gj18
#PBS -N annotate_wds_ja_test
#PBS -j oe

# Smoke test for scripts/annotate_wds.py: transcribe 5 samples from
# 1 shard and exit. Confirms that:
#   - Whisper medium loads on GH200 with the ARM torch build
#   - sphn can decode podcast_crawl's 24 kHz stereo MP3s
#   - output JSONL is well-formed with non-empty Japanese alignments

set -euo pipefail

PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
PODCAST_DIR=/work/gj18/e43001/github.com/podcast_crawl

# Use an existing finished job's output. 1594387 was the first successful
# JA crawl and has plenty of shards.
INPUT_DIR=${PODCAST_DIR}/data/wds_ja/1594387
OUTPUT_DIR=${PROJECT_DIR}/data/transcripts_test

cd "${PROJECT_DIR}"

export PATH="${HOME}/.local/bin:/work/gj18/e43001/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH="/work/gj18/e43001/miniconda3/lib:/work/gj18/e43001/miniconda3/bin:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
LOG_FILE=${PROJECT_DIR}/logs/annotate_test.log

echo "Job started: $(date)" | tee "${LOG_FILE}"
echo "Host: $(hostname)" | tee -a "${LOG_FILE}"

NVIDIA_BASE=/work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9
CT2_DIR=/work/gj18/e43001/github.com/CTranslate2/install
export LD_LIBRARY_PATH="${CT2_DIR}/lib:${CT2_DIR}/lib64:${NVIDIA_BASE}/math_libs/13.0/targets/sbsa-linux/lib:${NVIDIA_BASE}/cuda/lib64:${LD_LIBRARY_PATH}"

# Diagnostic: verify ctranslate2 CUDA support before running
${PROJECT_DIR}/.venv/bin/python3 -c "
import ctranslate2
print('ct2 version:', ctranslate2.__version__)
print('ct2 file:', ctranslate2.__file__)
print('cuda devices:', ctranslate2.get_cuda_device_count())
" 2>&1

${PROJECT_DIR}/.venv/bin/python3 scripts/annotate_wds.py \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --lang ja \
    --whisper-model medium \
    --batch-size 128 \
    --compute-type float16 \
    --max-shards 1 \
    --max-samples-per-shard 50 \
    2>&1 | tee -a "${LOG_FILE}"

echo "Job finished: $(date)" | tee -a "${LOG_FILE}"
echo "--- OUTPUT FILES ---" | tee -a "${LOG_FILE}"
find "${OUTPUT_DIR}" -type f | tee -a "${LOG_FILE}"
echo "--- FIRST TRANSCRIPT ---" | tee -a "${LOG_FILE}"
find "${OUTPUT_DIR}" -name "*.jsonl" | head -1 | xargs -I{} head -2 {} | tee -a "${LOG_FILE}"
