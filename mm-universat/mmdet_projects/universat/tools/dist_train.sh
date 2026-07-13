#!/usr/bin/env bash
# Distributed training launcher for UniverSat-based models.
#
# This project lives OUTSIDE the MMDetection repository as a custom project
# under ``universat_run/mmdet_projects/universat/``. This script sets PYTHONPATH
# so that MMDetection can load it without modifying the MMDetection source tree.
#
# Usage:
#   bash universat_run/mmdet_projects/universat/tools/dist_train.sh \
#        universat_run/mmdet_projects/universat/configs/base_universat.py \
#        <num_gpus> \
#        [optional arguments passed to tools/train.py]
#
# Examples:
#   # 4-GPU single-node training
#   bash universat_run/mmdet_projects/universat/tools/dist_train.sh \
#        universat_run/mmdet_projects/universat/configs/base_universat.py 4
#
#   # 4-GPU training with resume
#   bash universat_run/mmdet_projects/universat/tools/dist_train.sh \
#        universat_run/mmdet_projects/universat/configs/base_universat.py 4 \
#        --resume-from work_dirs/base_universat/latest.pth
#
#   # 2-node training (run on each node with appropriate NODE_RANK)
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<ip> \
#   bash universat_run/mmdet_projects/universat/tools/dist_train.sh \
#        universat_run/mmdet_projects/universat/configs/base_universat.py 8
#
# Environment:
#   MMDET_ROOT  (optional)  Path to the MMDetection repository/install root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                          # .../universat/
MMDET_PROJECTS="$(cd "${PROJECT_DIR}/.." && pwd)"                      # .../mmdet_projects/
UNIVERSAT_ROOT="$(cd "${MMDET_PROJECTS}/../.." && pwd)"                # UniverSat repo root

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
    echo "  conda env create -f UniverSat/environment.yaml"
    exit 1
fi

# Locate MMDetection root.
MMDET_ROOT="${MMDET_ROOT:-}"
if [[ -z "${MMDET_ROOT}" ]]; then
    MMDET_ROOT="$(python -c "import mmdet, os; print(os.path.dirname(os.path.dirname(mmdet.__file__)))" 2>/dev/null || true)"
fi
if [[ -z "${MMDET_ROOT}" || ! -f "${MMDET_ROOT}/tools/train.py" ]]; then
    echo "Error: cannot find MMDetection tools/train.py."
    echo "Please set MMDET_ROOT to your MMDetection repository or installation root, e.g.:"
    echo "  export MMDET_ROOT=/path/to/mmdetection"
    exit 1
fi

# Distributed settings
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
PORT="${PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# Prefer torchrun (PyTorch >= 1.10), fall back to torch.distributed.launch.
LAUNCHER=""
if conda run -n "${ENV_NAME}" which torchrun &>/dev/null; then
    LAUNCHER="torchrun"
else
    LAUNCHER="torch.distributed.launch"
fi

echo "============================================================"
echo "UniverSat distributed training"
echo "  config:         ${CONFIG}"
echo "  gpus/node:      ${GPUS}"
echo "  nodes:          ${NNODES}"
echo "  node rank:      ${NODE_RANK}"
echo "  master addr:    ${MASTER_ADDR}"
echo "  master port:    ${PORT}"
echo "  launcher:       ${LAUNCHER}"
echo "  project:        ${PROJECT_DIR}"
echo "  mmdet_projects: ${MMDET_PROJECTS}"
echo "  mmdet_root:     ${MMDET_ROOT}"
echo "  env:            ${ENV_NAME}"
echo "  extra:          ${*}"
echo "============================================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${MMDET_PROJECTS}:${PYTHONPATH:-}"

if [[ "${LAUNCHER}" == "torchrun" ]]; then
    torchrun \
        --nnodes="${NNODES}" \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${PORT}" \
        --nproc_per_node="${GPUS}" \
        "${MMDET_ROOT}/tools/train.py" \
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
        "${MMDET_ROOT}/tools/train.py" \
        "${CONFIG}" \
        --launcher pytorch \
        "$@"
fi
