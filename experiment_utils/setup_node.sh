#!/usr/bin/env bash
# setup_node.sh — provision ONE node for the StreamInfer artifact.
#
#   bash experiment_utils/setup_node.sh
#
# Idempotent: creates the `streaminfer` conda env if missing, installs Python
# deps, applies the vLLM patch, builds disagmoe_c, and loads the gdrdrv kernel
# module. Run once on the head node AND on every worker node. Mirrors
# StreamInfer/readme.md but automated. Assumes system prerequisites from that
# readme are already present (CUDA, apt libnccl2/libnccl-dev, libzmq3-dev +
# cppzmq-dev, gdrcopy under /usr/local/gdrcopy, UCX).
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/throughput-itl/config.sh"
log(){ echo "$(date '+%H:%M:%S') [setup $(hostname -s)] $*"; }

source "$MINICONDA/etc/profile.d/conda.sh"

# 1. conda env
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  log "creating conda env $CONDA_ENV (python 3.12.8) + torch + vllm ..."
  conda create -n "$CONDA_ENV" python=3.12.8 -y
  conda activate "$CONDA_ENV"
  pip install torch==2.6.0 torchvision torchaudio
  pip install vllm==0.8.2
else
  log "conda env $CONDA_ENV already exists"
  conda activate "$CONDA_ENV"
fi

# 2. Python deps + submodules
cd "$REPO_DIR"
log "pip install -r requirements.txt"
pip install -r requirements.txt
if [ -f .gitmodules ] && [ -d .git ]; then
  git submodule update --init --recursive third_party/cutlass third_party/cereal \
      third_party/NVTX third_party/pybind11 2>/dev/null || true
fi

# 3. vLLM patch (apply once; skip if already applied)
PATCH="$REPO_DIR/patches/vllm_0.8.2.patch"
SITE="$(python -c "import os, site; print(next(p for p in site.getsitepackages() if os.path.isdir(os.path.join(p,'vllm'))))")"
if git -C "$SITE" apply -R --check "$PATCH" >/dev/null 2>&1; then
  log "vLLM patch already applied"
elif git -C "$SITE" apply --check "$PATCH" >/dev/null 2>&1; then
  log "applying vLLM patch to $SITE"
  git -C "$SITE" apply "$PATCH"
else
  log "WARNING: vLLM patch neither applies nor is already applied — check vllm version"
fi

# 4. Build disagmoe_c (against apt NCCL)
log "building disagmoe_c (make pip) ..."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-/usr/include}"
export NCCL_LIBRARY_DIR="${NCCL_LIBRARY_DIR:-/usr/lib/x86_64-linux-gnu}"
export ZMQ_HOME="${ZMQ_HOME:-/usr}"
export GDRCOPY_HOME="${GDRCOPY_HOME:-/usr/local/gdrcopy}"
export C_INCLUDE_PATH="${C_INCLUDE_PATH:-/usr/include}"
export CPP_INCLUDE_PATH="${CPP_INCLUDE_PATH:-/usr/include}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$NCCL_LIBRARY_DIR:${LIBRARY_PATH:-}"
make pip

# 5. gdrdrv kernel module
if ! lsmod | grep -qw gdrdrv; then
  if [ -d "$HOME/gdrcopy" ]; then
    log "loading gdrdrv kernel module ..."
    (cd "$HOME/gdrcopy" && sudo bash ./insmod.sh) || log "WARNING: could not load gdrdrv"
  else
    log "WARNING: gdrdrv not loaded and ~/gdrcopy not found"
  fi
fi

# 6. Verify
log "verifying imports ..."
python -c "import disagmoe_c; print('disagmoe_c OK')"
python -c "import vllm; print('vllm', vllm.__version__)"
log "NODE SETUP COMPLETE"
