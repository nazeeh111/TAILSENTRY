"""Rare-event research engine. Model uncertainty is separate from simulation error."""
__version__ = "0.1.1.dev0"
from .models import QuadraticModel
from .engine import estimate
__all__ = ["QuadraticModel", "estimate"]
