"""Executable comparison-method package for the IresFM study."""

from .configs import METHOD_SPECS, get_method_spec
from .models import build_model

__all__ = ["METHOD_SPECS", "build_model", "get_method_spec"]
