#!/usr/bin/env bash
# Train a UniverSat-based detector/segmentor in an OpenMMLab project.
#
# This project lives OUTSIDE the MMDetection repository as a custom project
# under ``universat_run/mmdet_projects/universat/``. This script sets PYTHONPATH
# so that MMDetection can load it without modifying the MMDetection source tree.
#
# Usage:
#   bash universat_run/mmdet_projects/universat/tools/train.sh \
#        universat_run/mmdet_projects/universat/configs/base_universat.py \
#        [optional_checkpoint.pth] \
#        [optional_work_dir]
#
# Environment:
#   MMDET_ROOT  (optional)  Path to the MMDetection repository/install root.
#                           If not set, the script tries to auto-detect it from
#                           the installed ``mmdet`` package.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                          # .../universat/
MMDET_PROJECTS="$(cd "${PROJECT_DIR}/.." && pwd)"                      # .../mmdet_projects/
UNIVERSAT_ROOT="$(cd "${MMDET_PROJECTS}/../.." && pwd)"                # UniverSat repo root

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

EXTRA_ARGS=()
if [[ -n "${CKPT}" ]]; then
    EXTRA_ARGS+=("--resume-from" "${CKPT}")
fi
if [[ -n "${WORK_DIR}" ]]; then
    EXTRA_ARGS+=("--work-dir" "${WORK_DIR}")
fi

# Auto-generate work_dir based on config name if not provided.
if [[ -z "${WORK_DIR}" ]]; then
    CONFIG_BASENAME="$(basename "${CONFIG}" .py)"
    EXTRA_ARGS+=("--work-dir" "${UNIVERSAT_ROOT}/work_dirs/${CONFIG_BASENAME}")
fi

echo "============================================================"
echo "UniverSat training"
echo "  config:         ${CONFIG}"
echo "  project:        ${PROJECT_DIR}"
echo "  mmdet_projects: ${MMDET_PROJECTS}"
echo "  mmdet_root:     ${MMDET_ROOT}"
echo "  env:            ${ENV_NAME}"
echo "  extra:          ${EXTRA_ARGS[*]}"
echo "============================================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${MMDET_PROJECTS}:${PYTHONPATH:-}"

python "${MMDET_ROOT}/tools/train.py" \
    "${CONFIG}" \
    "${EXTRA_ARGS[@]}"
