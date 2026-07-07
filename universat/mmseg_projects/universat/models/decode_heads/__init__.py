"""UniverSat decode heads."""

from .universat_lp_head import UniverSatLinearProbeHead
from .universat_seg_head import UniverSatSegHead

__all__ = ["UniverSatSegHead", "UniverSatLinearProbeHead"]
