#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gj18
#PBS -N moshi_dialogue
#PBS -j oe

set -euo pipefail

PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
cd "${PROJECT_DIR}"

export PATH="${HOME}/.local/bin:/work/gj18/e43001/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH="/work/gj18/e43001/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export TORCHDYNAMO_DISABLE=1

echo "Job started: $(date)"
echo "Host: $(hostname)"

# Base model dialogue
${PROJECT_DIR}/.venv/bin/python3 scripts/moshi_dialogue.py \
    --hf-repo kyutai/moshiko-pytorch-bf16 \
    --duration 30 \
    --output ${PROJECT_DIR}/data/dialogue_base.wav \
    --seed 42

echo "=== Base model done ==="

# Fine-tuned model dialogue
${PROJECT_DIR}/.venv/bin/python3 scripts/moshi_dialogue.py \
    --hf-repo kyutai/moshiko-pytorch-bf16 \
    --lora-path ${PROJECT_DIR}/runs/moshi_ja_wds/checkpoints/checkpoint_000500/consolidated/lora.safetensors \
    --duration 30 \
    --output ${PROJECT_DIR}/data/dialogue_finetuned.wav \
    --seed 42

echo "=== Fine-tuned model done ==="

echo "Job finished: $(date)"
