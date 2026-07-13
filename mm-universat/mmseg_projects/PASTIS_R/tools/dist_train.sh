#!/usr/bin/env bash
# Distributed training of a UniverSat-based segmentor on PASTIS-R.
#
# Usage:
#   bash projects/universat/PASTIS_R/tools/dist_train.sh \
#        projects/universat/PASTIS_R/configs/ft_universat_pastisr.py \
#        <num_gpus> \
#        [optional_checkpoint.pth]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMSEG_TOOLS_DIR="$(cd "${PROJECT_DIR}/../../../tools" && pwd)"

CONFIG="${1:-}"
GPUS="${2:-1}"
CKPT="${3:-}"

if [[ -z "${CONFIG}" ]]; then
    echo "Error: config file is required."
    echo "Usage: $0 <config.py> <num_gpus> [checkpoint]"
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

PORT="${PORT:-29500}"

EXTRA_ARGS=()
if [[ -n "${CKPT}" ]]; then
    EXTRA_ARGS+=("--resume-from" "${CKPT}")
fi

echo "============================================================"
echo "UniverSat PASTIS-R distributed segmentation training (MMSeg)"
echo "  config:   ${CONFIG}"
echo "  gpus:     ${GPUS}"
echo "  project:  ${PROJECT_DIR}"
echo "  mmseg:    ${MMSEG_TOOLS_DIR}"
echo "  env:      ${ENV_NAME}"
echo "============================================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if conda run -n "${ENV_NAME}" which torchrun &>/dev/null; then
    torchrun \
        --nproc_per_node="${GPUS}" \
        --master_port="${PORT}" \
        "${MMSEG_TOOLS_DIR}/train.py" \
        "${CONFIG}" \
        --launcher pytorch \
        "${EXTRA_ARGS[@]}"
else
    python -m torch.distributed.launch \
        --nproc_per_node="${GPUS}" \
        --master_port="${PORT}" \
        "${MMSEG_TOOLS_DIR}/train.py" \
        "${CONFIG}" \
        --launcher pytorch \
        "${EXTRA_ARGS[@]}"
fi
