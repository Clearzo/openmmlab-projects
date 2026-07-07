"""Project-specific training entrypoint.

This thin wrapper ensures the ``universat`` project package is on the path
before delegating to the standard OpenMMLab ``tools/train.py``.

Usage::

    python projects/universat/tools/train.py projects/universat/configs/base_universat.py [OPTIONS]
"""

import os
import sys


def main():
    # Add the project directory to PYTHONPATH so that
    # ``universat.models`` and ``universat.datasets`` can be imported.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import after adjusting the path.
    import universat  # noqa: F401

    # Delegate to the standard OpenMMLab train entrypoint.
    # This assumes the command is run from the repository root.
    from tools.train import main as mmdet_train
    sys.argv = ["tools/train.py"] + sys.argv[1:]
    mmdet_train()


if __name__ == "__main__":
    main()
