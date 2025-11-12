"""
PCB-on-PCB transformer optimization workflows.

This package houses the NSGA-II orchestration, provenance writers,
and public API entry points that replace the legacy notebook.
"""

from .api import generate_pareto_front, run_pareto
from .config import (
    ExportConfig,
    OptimizationLoopConfig,
    ParetoRunConfig,
    PCBPCBModelConfig,
    ModelArtifactMetadata,
    ModelArtifactSelectionError,
    default_legacy_config,
)
from .io import records_to_dataframe, write_export_bundle
from .model import PCBPCBModel, PCBPCBPrediction, PredictionProvenance
from .optimizer import ParetoOptimizer, ParetoResult
from .schemas import DesignVector

__all__ = [
    "ExportConfig",
    "OptimizationLoopConfig",
    "ParetoRunConfig",
    "PCBPCBModel",
    "PCBPCBModelConfig",
    "PCBPCBPrediction",
    "PredictionProvenance",
    "DesignVector",
    "ModelArtifactMetadata",
    "ModelArtifactSelectionError",
    "default_legacy_config",
    "ParetoOptimizer",
    "ParetoResult",
    "generate_pareto_front",
    "run_pareto",
    "records_to_dataframe",
    "write_export_bundle",
]
