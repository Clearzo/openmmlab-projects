"""UniverSat project for MMSegmentation."""

from .models import *  # noqa: F401,F403

try:
    from .datasets import *  # noqa: F401,F403
except ImportError:  # pragma: no cover
    pass
