#!/usr/bin/env bash
# Run UniverSat linear-probe semantic segmentation on PASTIS-R.
#
# Usage:
#   bash projects/universat/PASTIS_R/tools/run_linear_probe.sh \
#        projects/universat/PASTIS_R/configs/linear_probe_universat_pastisr.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNIVERSAT_ROOT="$(cd "${PROJECT_DIR}/../../.." && pwd)"

CONFIG="${1:-}"
if [[ -z "${CONFIG}" ]]; then
    echo "Error: config file is required."
    echo "Usage: $0 <config.py>"
    exit 1
fi

ENV_NAME="universat"
if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
    CONDA_BASE="$(conda info --base)"
fi

echo "============================================================"
echo "UniverSat PASTIS-R linear probe"
echo "  config:   ${CONFIG}"
echo "  project:  ${PROJECT_DIR}"
echo "  repo:     ${UNIVERSAT_ROOT}"
echo "  env:      ${ENV_NAME}"
echo "============================================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${UNIVERSAT_ROOT}:${PYTHONPATH:-}"

cd "${UNIVERSAT_ROOT}"
python "${PROJECT_DIR}/tools/linear_probe.py" "${CONFIG}"
