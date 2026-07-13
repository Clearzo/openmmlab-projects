"""Project-specific training entrypoint.

This thin wrapper ensures the custom projects under ``universat_run/mmdet_projects/``
are on ``sys.path`` before delegating to the standard OpenMMLab ``tools/train.py``.

The project is designed to live OUTSIDE the MMDetection repository. Set
``MMDET_ROOT`` to point to your MMDetection installation if it cannot be
auto-detected.

Usage::

    python universat_run/mmdet_projects/universat/tools/train.py \
        universat_run/mmdet_projects/universat/configs/base_universat.py [OPTIONS]
"""

import os
import sys


def _find_mmdet_root():
    """Return MMDetection repository/install root."""
    if os.environ.get("MMDET_ROOT"):
        return os.environ["MMDET_ROOT"]
    try:
        import mmdet

        return os.path.dirname(os.path.dirname(mmdet.__file__))
    except Exception as exc:
        raise RuntimeError(
            "Cannot locate MMDetection. Please set MMDET_ROOT environment variable, e.g.\n"
            "  export MMDET_ROOT=/path/to/mmdetection"
        ) from exc


def main():
    # Add mmdet_projects/ to PYTHONPATH so that ``universat`` and other custom
    # projects can be imported by MMDetection's registry mechanism.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mmdet_projects = os.path.dirname(os.path.dirname(script_dir))
    if mmdet_projects not in sys.path:
        sys.path.insert(0, mmdet_projects)

    # Import after adjusting the path to trigger registry registration.
    import universat  # noqa: F401

    # Locate and import the standard OpenMMLab train entrypoint.
    mmdet_root = _find_mmdet_root()
    tools_dir = os.path.join(mmdet_root, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    from train import main as mmdet_train  # noqa: E402
    sys.argv = ["train.py"] + sys.argv[1:]
    mmdet_train()


if __name__ == "__main__":
    main()
