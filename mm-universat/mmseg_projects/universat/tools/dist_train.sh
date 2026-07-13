#!/usr/bin/env bash
# Distributed training of a UniverSat-based segmentor.
#
# This project lives OUTSIDE the MMSegmentation repository as a custom project
# under ``universat_run/mmseg_projects/universat/``. This script sets PYTHONPATH
# so that MMSegmentation can load it without modifying the MMSegmentation source
# tree.
#
# Usage:
#   bash universat_run/mmseg_projects/universat/tools/dist_train.sh \
#        universat_run/mmseg_projects/universat/configs/base_universat_seg.py \
#        <num_gpus> \
#        [optional_checkpoint.pth]
#
# Environment:
#   MMSEG_ROOT  (optional)  Path to the MMSegmentation repository/install root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                          # .../universat/
MMSEG_PROJECTS="$(cd "${PROJECT_DIR}/.." && pwd)"                      # .../mmseg_projects/
UNIVERSAT_ROOT="$(cd "${MMSEG_PROJECTS}/../.." && pwd)"                # UniverSat repo root

CONFIG="${1:-}"
GPUS="${2:-1}"
CKPT="${3:-}"

if [[ -z "${CONFIG}" ]]; then
    echo "Error: config file is required."
    echo "Usage: $0 <config.py> <num_gpus> [checkpoint]"
    exit 1
fi

ENV_NAME="universat"
if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
    CONDA_BASE="$(conda info --base)"
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Locate MMSegmentation root.
MMSEG_ROOT="${MMSEG_ROOT:-}"
if [[ -z "${MMSEG_ROOT}" ]]; then
    MMSEG_ROOT="$(python -c "import mmseg, os; print(os.path.dirname(os.path.dirname(mmseg.__file__)))" 2>/dev/null || true)"
fi
if [[ -z "${MMSEG_ROOT}" || ! -f "${MMSEG_ROOT}/tools/train.py" ]]; then
    echo "Error: cannot find MMSegmentation tools/train.py."
    echo "Please set MMSEG_ROOT to your MMSegmentation repository or installation root, e.g.:"
    echo "  export MMSEG_ROOT=/path/to/mmsegmentation"
    exit 1
fi

PORT="${PORT:-29500}"

EXTRA_ARGS=()
if [[ -n "${CKPT}" ]]; then
    EXTRA_ARGS+=("--resume-from" "${CKPT}")
fi

echo "============================================================"
echo "UniverSat distributed segmentation training (MMSeg)"
echo "  config:         ${CONFIG}"
echo "  gpus:           ${GPUS}"
echo "  project:        ${PROJECT_DIR}"
echo "  mmseg_projects: ${MMSEG_PROJECTS}"
echo "  mmseg_root:     ${MMSEG_ROOT}"
echo "  env:            ${ENV_NAME}"
echo "============================================================"

export PYTHONPATH="${MMSEG_PROJECTS}:${PYTHONPATH:-}"

if conda run -n "${ENV_NAME}" which torchrun &>/dev/null; then
    torchrun \
        --nproc_per_node="${GPUS}" \
        --master_port="${PORT}" \
        "${MMSEG_ROOT}/tools/train.py" \
        "${CONFIG}" \
        --launcher pytorch \
        "${EXTRA_ARGS[@]}"
else
    python -m torch.distributed.launch \
        --nproc_per_node="${GPUS}" \
        --master_port="${PORT}" \
        "${MMSEG_ROOT}/tools/train.py" \
        "${CONFIG}" \
        --launcher pytorch \
        "${EXTRA_ARGS[@]}"
fi
