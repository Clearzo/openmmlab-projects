#!/usr/bin/env bash
# Train a UniverSat-based segmentor on PASTIS-R in an MMSegmentation project.
#
# Usage:
#   bash projects/universat/PASTIS_R/tools/train.sh \
#        projects/universat/PASTIS_R/configs/ft_universat_pastisr.py \
#        [optional_checkpoint.pth] \
#        [optional_work_dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMSEG_TOOLS_DIR="$(cd "${PROJECT_DIR}/../../../tools" && pwd)"

CONFIG="${1:-}"
CKPT="${2:-}"
WORK_DIR="${3:-}"

if [[ -z "${CONFIG}" ]]; then
    echo "Error: config file is required."
    echo "Usage: $0 <config.py> [checkpoint] [work_dir]"
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: config file not found: ${CONFIG}"
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

EXTRA_ARGS=()
if [[ -n "${CKPT}" ]]; then
    EXTRA_ARGS+=("--resume-from" "${CKPT}")
fi
if [[ -z "${WORK_DIR}" ]]; then
    CONFIG_BASENAME="$(basename "${CONFIG}" .py)"
    EXTRA_ARGS+=("--work-dir" "./work_dirs/${CONFIG_BASENAME}")
else
    EXTRA_ARGS+=("--work-dir" "${WORK_DIR}")
fi

echo "============================================================"
echo "UniverSat PASTIS-R segmentation training (MMSeg)"
echo "  config:   ${CONFIG}"
echo "  project:  ${PROJECT_DIR}"
echo "  mmseg:    ${MMSEG_TOOLS_DIR}"
echo "  env:      ${ENV_NAME}"
echo "============================================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python "${MMSEG_TOOLS_DIR}/train.py" \
    "${CONFIG}" \
    "${EXTRA_ARGS[@]}"
