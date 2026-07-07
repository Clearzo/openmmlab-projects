#!/usr/bin/env bash
# Distributed training launcher for UniverSat-based segmentors (MMSeg).
#
# Usage:
#   bash projects/universat/tools/dist_train.sh \
#        projects/universat/configs/base_universat_seg.py \
#        <num_gpus> \
#        [optional arguments]
#
# Examples:
#   bash projects/universat/tools/dist_train.sh \
#        projects/universat/configs/base_universat_seg.py 4
#
#   bash projects/universat/tools/dist_train.sh \
#        projects/universat/configs/base_universat_seg.py 4 \
#        --resume-from work_dirs/base_universat_seg/latest.pth

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMSEG_TOOLS_DIR="$(cd "${PROJECT_DIR}/../../../tools" && pwd)"

CONFIG="${1:-}"
GPUS="${2:-}"
shift 2 || true

if [[ -z "${CONFIG}" || -z "${GPUS}" ]]; then
    echo "Error: both config file and number of GPUs are required."
    echo "Usage: $0 <config.py> <num_gpus> [extra args]"
    exit 1
fi

if ! [[ "${GPUS}" =~ ^[0-9]+$ ]]; then
    echo "Error: number of GPUs must be an integer, got: ${GPUS}"
    exit 1
fi

if [[ ! -f "${MMSEG_TOOLS_DIR}/train.py" ]]; then
    echo "Error: cannot find mmsegmentation tools/train.py at ${MMSEG_TOOLS_DIR}"
    exit 1
fi

ENV_NAME="universat"
if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
    CONDA_BASE="$(conda info --base)"
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
PORT="${PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if conda run -n "${ENV_NAME}" which torchrun &>/dev/null; then
    LAUNCHER="torchrun"
else
    LAUNCHER="torch.distributed.launch"
fi

echo "============================================================"
echo "UniverSat distributed segmentation training (MMSeg)"
echo "  config:      ${CONFIG}"
echo "  gpus/node:   ${GPUS}"
echo "  nodes:       ${NNODES}"
echo "  node rank:   ${NODE_RANK}"
echo "  master:      ${MASTER_ADDR}:${PORT}"
echo "  launcher:    ${LAUNCHER}"
echo "  project:     ${PROJECT_DIR}"
echo "  env:         ${ENV_NAME}"
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
        "${MMSEG_TOOLS_DIR}/train.py" \
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
        "${MMSEG_TOOLS_DIR}/train.py" \
        "${CONFIG}" \
        --launcher pytorch \
        "$@"
fi
