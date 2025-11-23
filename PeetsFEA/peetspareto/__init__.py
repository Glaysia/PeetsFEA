"""
Optimization utilities for transformer design workflows.

The `pcbpcb` subpackage houses the PCB-on-PCB NSGA-II optimizer that replaces
the legacy EVDD notebook. The `litzpcb` subpackage is a thin wrapper that keeps
the EVDD_litz_PCB_v2 notebook logic runnable without touching `legacy_codes`.
"""

from . import pcbpcb, litzpcb

__all__ = ["pcbpcb", "litzpcb"]
