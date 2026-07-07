#!/usr/bin/env bash
# Distributed training launcher for UniverSat-based models.
#
# Usage:
#   bash projects/universat/tools/dist_train.sh \
#        projects/universat/configs/base_universat.py \
#        <num_gpus> \
#        [optional arguments passed to tools/train.py]
#
# Examples:
#   # 4-GPU single-node training
#   bash projects/universat/tools/dist_train.sh \
#        projects/universat/configs/base_universat.py 4
#
#   # 4-GPU training with resume
#   bash projects/universat/tools/dist_train.sh \
#        projects/universat/configs/base_universat.py 4 \
#        --resume-from work_dirs/base_universat/latest.pth
#
#   # 2-node training (run on each node with appropriate NODE_RANK)
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<ip> \
#   bash projects/universat/tools/dist_train.sh \
#        projects/universat/configs/base_universat.py 8
#
# The script activates the ``universat`` conda environment, sets PYTHONPATH,
# and launches the standard MMDet ``tools/train.py`` via PyTorch DDP.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMDET_TOOLS_DIR="$(cd "${PROJECT_DIR}/../../../tools" && pwd)"

CONFIG="${1:-}"
GPUS="${2:-}"
shift 2 || true

if [[ -z "${CONFIG}" || -z "${GPUS}" ]]; then
    echo "Error: both config file and number of GPUs are required."
    echo "Usage: $0 <config.py> <num_gpus> [extra args for tools/train.py]"
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: config file not found: ${CONFIG}"
    exit 1
fi

if ! [[ "${GPUS}" =~ ^[0-9]+$ ]]; then
    echo "Error: number of GPUs must be an integer, got: ${GPUS}"
    exit 1
fi

if [[ ! -f "${MMDET_TOOLS_DIR}/train.py" ]]; then
    echo "Error: cannot find mmdetection tools/train.py at ${MMDET_TOOLS_DIR}"
    echo "Make sure this script is placed under projects/universat/tools/"
    exit 1
fi

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------
ENV_NAME="universat"

if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
elif command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base)"
else
    echo "Error: cannot locate conda installation."
    exit 1
fi

if [[ ! -d "${CONDA_BASE}/envs/${ENV_NAME}" ]]; then
    echo "Error: conda environment '${ENV_NAME}' not found."
    echo "Create it first with:"
    echo "  conda env create -f ${PROJECT_DIR}/environment.yml"
    exit 1
fi

# ---------------------------------------------------------------------------
# Distributed settings
# ---------------------------------------------------------------------------
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
PORT="${PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# ---------------------------------------------------------------------------
# Select PyTorch distributed launcher
# ---------------------------------------------------------------------------
# Prefer ``torchrun`` (PyTorch >= 1.10), fall back to ``torch.distributed.launch``.
LAUNCHER=""
if conda run -n "${ENV_NAME}" which torchrun &>/dev/null; then
    LAUNCHER="torchrun"
else
    LAUNCHER="torch.distributed.launch"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "============================================================"
echo "UniverSat distributed training"
echo "  config:      ${CONFIG}"
echo "  gpus/node:   ${GPUS}"
echo "  nodes:       ${NNODES}"
echo "  node rank:   ${NODE_RANK}"
echo "  master addr: ${MASTER_ADDR}"
echo "  master port: ${PORT}"
echo "  launcher:    ${LAUNCHER}"
echo "  project:     ${PROJECT_DIR}"
echo "  env:         ${ENV_NAME}"
echo "  extra:       ${*}"
echo "============================================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ "${LAUNCHER}" == "torchrun" ]]; then
    torchrun \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${PORT}" \
        --nproc_per_node="${GPUS}" \
        "${MMDET_TOOLS_DIR}/train.py" \
        "${CONFIG}" \
        --launcher pytorch \
        "$@"
else
    python -m torch.distributed.launch \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${PORT}" \
        --nproc_per_node="${GPUS}" \
        "${MMDET_TOOLS_DIR}/train.py" \
        "${CONFIG}" \
        --launcher pytorch \
        "$@"
fi
