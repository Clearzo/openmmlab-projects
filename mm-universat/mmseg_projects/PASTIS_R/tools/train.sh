#!/usr/bin/env bash
# Train a UniverSat-based segmentor on PASTIS-R.
#
# This project lives OUTSIDE the MMSegmentation repository. The custom projects
# ``universat/`` and ``PASTIS_R/`` are placed side-by-side under
# ``universat_run/mmseg_projects/``. This script sets PYTHONPATH so that
# MMSegmentation can load them without modifying the MMSegmentation source tree.
#
# Usage:
#   bash universat_run/mmseg_projects/PASTIS_R/tools/train.sh \
#        universat_run/mmseg_projects/PASTIS_R/configs/ft_universat_pastisr.py \
#        [optional_checkpoint.pth] \
#        [optional_work_dir]
#
# Environment:
#   MMSEG_ROOT  (optional)  Path to the MMSegmentation repository/install root.
#                           If not set, the script tries to auto-detect it from
#                           the installed ``mmseg`` package.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                          # .../PASTIS_R/
MMSEG_PROJECTS="$(cd "${PROJECT_DIR}/.." && pwd)"                      # .../mmseg_projects/
UNIVERSAT_ROOT="$(cd "${MMSEG_PROJECTS}/../.." && pwd)"               # UniverSat repo root

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

EXTRA_ARGS=()
if [[ -n "${CKPT}" ]]; then
    EXTRA_ARGS+=("--resume-from" "${CKPT}")
fi
if [[ -z "${WORK_DIR}" ]]; then
    CONFIG_BASENAME="$(basename "${CONFIG}" .py)"
    EXTRA_ARGS+=("--work-dir" "${UNIVERSAT_ROOT}/work_dirs/${CONFIG_BASENAME}")
else
    EXTRA_ARGS+=("--work-dir" "${WORK_DIR}")
fi

echo "============================================================"
echo "UniverSat PASTIS-R segmentation training (MMSeg)"
echo "  config:       ${CONFIG}"
echo "  project:      ${PROJECT_DIR}"
echo "  mmseg_projects: ${MMSEG_PROJECTS}"
echo "  mmseg_root:   ${MMSEG_ROOT}"
echo "  env:          ${ENV_NAME}"
echo "============================================================"

export PYTHONPATH="${MMSEG_PROJECTS}:${PYTHONPATH:-}"

python "${MMSEG_ROOT}/tools/train.py" \
    "${CONFIG}" \
    "${EXTRA_ARGS[@]}"
