"""UniverSat models for MMSegmentation."""

from .backbones import *  # noqa: F401,F403

try:
    from .decode_heads import *  # noqa: F401,F403
except ImportError:  # pragma: no cover
    pass
