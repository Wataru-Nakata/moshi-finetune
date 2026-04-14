#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -W group_list=gj18
#PBS -N build_ct2
#PBS -j oe

set -euo pipefail

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

NVIDIA_BASE=/work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9
export CUDA_HOME=$NVIDIA_BASE/cuda
MATH_LIBS=$NVIDIA_BASE/math_libs/13.0/targets/sbsa-linux
export PATH="${HOME}/.local/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${MATH_LIBS}/lib:/work/gj18/e43001/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${MATH_LIBS}/lib:${LIBRARY_PATH:-}"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

PROJECT_DIR=/work/gj18/e43001/github.com/moshi-finetune
CT2_DIR=/work/gj18/e43001/github.com/CTranslate2

echo "Job started: $(date)"
echo "Host: $(hostname)"

# Build C++ library
cd ${CT2_DIR}/build
make -j$(nproc) 2>&1
make install 2>&1

echo "C++ build done: $(date)"

# Install Python bindings into moshi-finetune venv
cd ${PROJECT_DIR}
uv pip install pybind11 2>&1
export CTranslate2_ROOT=${CT2_DIR}/install
export CMAKE_PREFIX_PATH=${CT2_DIR}/install
export CPATH=${CT2_DIR}/install/include:${CPATH:-}
export LIBRARY_PATH=${CT2_DIR}/install/lib:${CT2_DIR}/install/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${CT2_DIR}/install/lib:${CT2_DIR}/install/lib64:${LD_LIBRARY_PATH:-}
uv pip install ${CT2_DIR}/python --reinstall --no-build-isolation 2>&1

echo "Python install done: $(date)"

# Verify — use venv python directly so uv doesn't re-sync and overwrite
${PROJECT_DIR}/.venv/bin/python3 -c "
import ctranslate2
print('ctranslate2 version:', ctranslate2.__version__)
print('cuda devices:', ctranslate2.get_cuda_device_count())
print('supported compute types (cuda):', ctranslate2.get_supported_compute_types('cuda'))
"

echo "Job finished: $(date)"
