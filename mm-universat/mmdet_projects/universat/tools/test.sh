#!/usr/bin/env bash
# Test / inference a UniverSat-based model in an OpenMMLab project.
#
# Usage:
#   bash projects/universat/tools/test.sh \
#        projects/universat/configs/base_universat.py \
#        work_dirs/base_universat/latest.pth \
#        [--out results.pkl] \
#        [--eval bbox] \
#        [--show-dir vis/]
#
# The script activates the ``universat`` conda environment, adds the project
# directory to PYTHONPATH, and delegates to the standard MMDet ``tools/test.py``.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMDET_TOOLS_DIR="$(cd "${PROJECT_DIR}/../../../tools" && pwd)"

CONFIG="${1:-}"
CKPT="${2:-}"
shift 2 || true

if [[ -z "${CONFIG}" || -z "${CKPT}" ]]; then
    echo "Error: both config and checkpoint are required."
    echo "Usage: $0 <config.py> <checkpoint.pth> [extra args for tools/test.py]"
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: config file not found: ${CONFIG}"
    exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
    echo "Error: checkpoint not found: ${CKPT}"
    exit 1
fi

if [[ ! -f "${MMDET_TOOLS_DIR}/test.py" ]]; then
    echo "Error: cannot find mmdetection tools/test.py at ${MMDET_TOOLS_DIR}"
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
# Run
# ---------------------------------------------------------------------------
echo "============================================================"
echo "UniverSat testing / inference"
echo "  config:   ${CONFIG}"
echo "  checkpoint: ${CKPT}"
echo "  project:  ${PROJECT_DIR}"
echo "  mmdet:    ${MMDET_TOOLS_DIR}"
echo "  env:      ${ENV_NAME}"
echo "  extra:    ${*}"
echo "============================================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python "${MMDET_TOOLS_DIR}/test.py" \
    "${CONFIG}" \
    "${CKPT}" \
    "$@"
