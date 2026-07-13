#!/usr/bin/env bash
# Distributed linear-probe semantic segmentation on PASTIS-R.
#
# This project lives OUTSIDE the MMSegmentation repository as a custom project
# pair (``universat/`` + ``PASTIS_R/``) under ``universat_run/mmseg_projects/``.
#
# Usage:
#   bash universat_run/mmseg_projects/PASTIS_R/tools/dist_run_linear_probe.sh \
#        universat_run/mmseg_projects/PASTIS_R/configs/linear_probe_universat_pastisr.py \
#        <num_gpus>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                          # .../PASTIS_R/
MMSEG_PROJECTS="$(cd "${PROJECT_DIR}/.." && pwd)"                      # .../mmseg_projects/
UNIVERSAT_ROOT="$(cd "${MMSEG_PROJECTS}/../.." && pwd)"               # UniverSat repo root

CONFIG="${1:-}"
GPUS="${2:-1}"

if [[ -z "${CONFIG}" ]]; then
    echo "Error: config file is required."
    echo "Usage: $0 <config.py> <num_gpus>"
    exit 1
fi

ENV_NAME="universat"
if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
    CONDA_BASE="$(conda info --base)"
fi

PORT="${PORT:-29500}"

echo "============================================================"
echo "UniverSat PASTIS-R distributed linear probe"
echo "  config:         ${CONFIG}"
echo "  gpus:           ${GPUS}"
echo "  project:        ${PROJECT_DIR}"
echo "  mmseg_projects: ${MMSEG_PROJECTS}"
echo "  repo:           ${UNIVERSAT_ROOT}"
echo "  env:            ${ENV_NAME}"
echo "============================================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${MMSEG_PROJECTS}:${PYTHONPATH:-}"

cd "${UNIVERSAT_ROOT}"

if conda run -n "${ENV_NAME}" which torchrun &>/dev/null; then
    torchrun \
        --nproc_per_node="${GPUS}" \
        --master_port="${PORT}" \
        "${PROJECT_DIR}/tools/linear_probe.py" \
        "${CONFIG}" \
        --launcher pytorch
else
    python -m torch.distributed.launch \
        --nproc_per_node="${GPUS}" \
        --master_port="${PORT}" \
        "${PROJECT_DIR}/tools/linear_probe.py" \
        "${CONFIG}" \
        --launcher pytorch
fi
