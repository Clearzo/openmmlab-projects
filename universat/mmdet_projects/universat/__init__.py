"""UniverSat project for OpenMMLab."""

# Models are always available (pure PyTorch fallback if MMDet is missing).
from .models import *  # noqa: F401,F403

# Datasets require mmdet; allow graceful import failure during standalone tests.
try:
    from .datasets import *  # noqa: F401,F403
except ImportError:  # pragma: no cover
    pass
