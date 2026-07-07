#!/usr/bin/env bash
# Test / inference a UniverSat-based segmentor in an MMSegmentation project.
#
# Usage:
#   bash projects/universat/tools/test.sh \
#        projects/universat/configs/base_universat_seg.py \
#        work_dirs/base_universat_seg/latest.pth \
#        [--out results.pkl] \
#        [--eval mIoU] \
#        [--show-dir vis/]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MMSEG_TOOLS_DIR="$(cd "${PROJECT_DIR}/../../../tools" && pwd)"

CONFIG="${1:-}"
CKPT="${2:-}"
shift 2 || true

if [[ -z "${CONFIG}" || -z "${CKPT}" ]]; then
    echo "Error: both config and checkpoint are required."
    echo "Usage: $0 <config.py> <checkpoint.pth> [extra args]"
    exit 1
fi

if [[ ! -f "${MMSEG_TOOLS_DIR}/test.py" ]]; then
    echo "Error: cannot find mmsegmentation tools/test.py at ${MMSEG_TOOLS_DIR}"
    exit 1
fi

ENV_NAME="universat"
if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
    CONDA_BASE="$(conda info --base)"
fi

echo "============================================================"
echo "UniverSat segmentation testing (MMSeg)"
echo "  config:     ${CONFIG}"
echo "  checkpoint: ${CKPT}"
echo "  project:    ${PROJECT_DIR}"
echo "  mmseg:      ${MMSEG_TOOLS_DIR}"
echo "  env:        ${ENV_NAME}"
echo "============================================================"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python "${MMSEG_TOOLS_DIR}/test.py" \
    "${CONFIG}" \
    "${CKPT}" \
    "$@"
