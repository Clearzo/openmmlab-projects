#!/usr/bin/env bash
# Test / inference a UniverSat-based segmentor in an MMSegmentation project.
#
# This project lives OUTSIDE the MMSegmentation repository as a custom project
# under ``universat_run/mmseg_projects/universat/``. This script sets PYTHONPATH
# so that MMSegmentation can load it without modifying the MMSegmentation source
# tree.
#
# Usage:
#   bash universat_run/mmseg_projects/universat/tools/test.sh \
#        universat_run/mmseg_projects/universat/configs/base_universat_seg.py \
#        work_dirs/base_universat_seg/latest.pth \
#        [--out results.pkl] \
#        [--eval mIoU] \
#        [--show-dir vis/]
#
# Environment:
#   MMSEG_ROOT  (optional)  Path to the MMSegmentation repository/install root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                          # .../universat/
MMSEG_PROJECTS="$(cd "${PROJECT_DIR}/.." && pwd)"                      # .../mmseg_projects/
UNIVERSAT_ROOT="$(cd "${MMSEG_PROJECTS}/../.." && pwd)"                # UniverSat repo root

CONFIG="${1:-}"
CKPT="${2:-}"
shift 2 || true

if [[ -z "${CONFIG}" || -z "${CKPT}" ]]; then
    echo "Error: both config and checkpoint are required."
    echo "Usage: $0 <config.py> <checkpoint.pth> [extra args]"
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
if [[ -z "${MMSEG_ROOT}" || ! -f "${MMSEG_ROOT}/tools/test.py" ]]; then
    echo "Error: cannot find MMSegmentation tools/test.py."
    echo "Please set MMSEG_ROOT to your MMSegmentation repository or installation root, e.g.:"
    echo "  export MMSEG_ROOT=/path/to/mmsegmentation"
    exit 1
fi

echo "============================================================"
echo "UniverSat segmentation testing (MMSeg)"
echo "  config:         ${CONFIG}"
echo "  checkpoint:     ${CKPT}"
echo "  project:        ${PROJECT_DIR}"
echo "  mmseg_projects: ${MMSEG_PROJECTS}"
echo "  mmseg_root:     ${MMSEG_ROOT}"
echo "  env:            ${ENV_NAME}"
echo "============================================================"

export PYTHONPATH="${MMSEG_PROJECTS}:${PYTHONPATH:-}"

python "${MMSEG_ROOT}/tools/test.py" \
    "${CONFIG}" \
    "${CKPT}" \
    "$@"
