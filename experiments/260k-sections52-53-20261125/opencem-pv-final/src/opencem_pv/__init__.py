"""Daily measured PV trajectories, using the pinned official OpenCEM interpolator."""
from .api import PVDatabase, PVDay, DataGapError, EnergyReport

__version__ = "1.0.0"
__all__ = ["PVDatabase", "PVDay", "DataGapError", "EnergyReport"]
