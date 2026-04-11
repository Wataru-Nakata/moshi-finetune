#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -W group_list=gj18
#PBS -J 0-63
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

NUM_SHARDS=64

PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
AUDIO_DIR=/work/gj18/e43001/github.com/podcast_crawl/data/wds_ja_filtered
TRANSCRIPT_DIR=/work/gj18/e43001/github.com/podcast_crawl/data/wds_ja_transcripts

cd "${PROJECT_DIR}"

export PATH="${HOME}/.local/bin:/work/gj18/e43001/miniconda3/bin:${PATH}"
# miniconda libs are needed for libopus.so.0 (pulled in by sphn at runtime).
export LD_LIBRARY_PATH="/work/gj18/e43001/miniconda3/lib:/work/gj18/e43001/miniconda3/bin:${LD_LIBRARY_PATH:-}"

LOG_DIR=${PROJECT_DIR}/logs
mkdir -p "${LOG_DIR}"
GPU_LOG="${LOG_DIR}/gpu_annotate_${PBS_ARRAY_INDEX}.log"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Shard: ${PBS_ARRAY_INDEX}/${NUM_SHARDS}"
echo "Audio: ${AUDIO_DIR}"
echo "Transcripts: ${TRANSCRIPT_DIR}"

nvidia-smi --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 60 > "${GPU_LOG}" 2>&1 &
GPU_LOG_PID=$!

uv run python3 scripts/annotate_wds.py \
    --input "${AUDIO_DIR}" \
    --output "${TRANSCRIPT_DIR}" \
    --lang ja \
    --whisper-model medium \
    --shard "${PBS_ARRAY_INDEX}" \
    --num-shards "${NUM_SHARDS}" \
    --local-rank 0

kill ${GPU_LOG_PID} 2>/dev/null || true
echo "Job finished: $(date)"
