#!/usr/bin/env bash
# Distributed linear-probe semantic segmentation on PASTIS-R.
#
# Usage:
#   bash projects/universat/PASTIS_R/tools/dist_run_linear_probe.sh \
#        projects/universat/PASTIS_R/configs/linear_probe_universat_pastisr.py \
#        <num_gpus>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMSEG_ROOT="$(cd "${PROJECT_DIR}/../../../.." && pwd)"

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
echo "  config:   ${CONFIG}"
echo "  gpus:     ${GPUS}"
echo "  project:  ${PROJECT_DIR}"
echo "  mmseg:    ${MMSEG_ROOT}"
echo "  env:      ${ENV_NAME}"
echo "============================================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/..:${PYTHONPATH:-}"

cd "${MMSEG_ROOT}"

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
