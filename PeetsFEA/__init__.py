"""
PeetsFEA
========

Top-level package exposing optimization (`peetspareto`) and simulation helpers.
"""

from importlib import metadata

from . import peetspareto

try:
    __version__ = metadata.version("PeetsFEA")
except metadata.PackageNotFoundError:  # pragma: no cover - occurs in editable installs
    __version__ = "0.0.0"

__all__ = ["__version__", "peetspareto"]
