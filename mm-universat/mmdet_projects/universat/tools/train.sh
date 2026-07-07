#!/usr/bin/env bash
# Train a UniverSat-based detector/segmentor in an OpenMMLab project.
#
# Usage:
#   bash projects/universat/tools/train.sh \
#        projects/universat/configs/base_universat.py \
#        [optional_checkpoint.pth] \
#        [optional_work_dir]
#
# The script assumes it lives inside ``projects/universat/tools/`` of an
# OpenMMLab repository (e.g. mmdetection). It activates the ``universat``
# conda environment, adds the project directory to PYTHONPATH, and delegates
# to the standard MMDet ``tools/train.py``.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMDET_TOOLS_DIR="$(cd "${PROJECT_DIR}/../../../tools" && pwd)"

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

if [[ ! -f "${MMDET_TOOLS_DIR}/train.py" ]]; then
    echo "Error: cannot find mmdetection tools/train.py at ${MMDET_TOOLS_DIR}"
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

# Activate the environment in a subshell-friendly way.
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
# Build command
# ---------------------------------------------------------------------------
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
    EXTRA_ARGS+=("--work-dir" "./work_dirs/${CONFIG_BASENAME}")
fi

echo "============================================================"
echo "UniverSat training"
echo "  config:   ${CONFIG}"
echo "  project:  ${PROJECT_DIR}"
echo "  mmdet:    ${MMDET_TOOLS_DIR}"
echo "  env:      ${ENV_NAME}"
echo "  extra:    ${EXTRA_ARGS[*]}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python "${MMDET_TOOLS_DIR}/train.py" \
    "${CONFIG}" \
    "${EXTRA_ARGS[@]}"
