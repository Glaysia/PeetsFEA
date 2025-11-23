"""
Legacy EVDD_litz_PCB NSGA-II runner (thin wrapper around the notebook logic).
"""

from .runtime import LitzPCBResult, load_models, run_litzpcb_nsga2

__all__ = ["run_litzpcb_nsga2", "LitzPCBResult", "load_models"]
