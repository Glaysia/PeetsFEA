"""Public API entry points for PCB-on-PCB optimization runs."""

from __future__ import annotations

from typing import Mapping

from .config import ParetoRunConfig
from .io import write_export_bundle
from .optimizer import ParetoOptimizer, ParetoResult
from .protocols import CandidateEncoder, ObjectiveAggregator, PredictionService, SimulationHook


def run_pareto(
    *,
    config: ParetoRunConfig,
    encoder: CandidateEncoder,
    prediction_service: PredictionService,
    aggregator: ObjectiveAggregator | None = None,
    simulation_hook: SimulationHook | None = None,
) -> ParetoResult:
    """
    Executes NSGA-II and returns an in-memory result bundle without writing artifacts.
    """

    optimizer = ParetoOptimizer(
        encoder=encoder,
        prediction_service=prediction_service,
        objectives=config.objectives,
        aggregator=aggregator,
        simulation_hook=simulation_hook,
    )
    return optimizer.run(config)


def generate_pareto_front(
    *,
    config: ParetoRunConfig,
    encoder: CandidateEncoder,
    prediction_service: PredictionService,
    aggregator: ObjectiveAggregator | None = None,
    simulation_hook: SimulationHook | None = None,
    extra_provenance: Mapping[str, object] | None = None,
) -> ParetoResult:
    """
    High-level API that runs NSGA-II and persists CSV + provenance artifacts.
    """

    result = run_pareto(
        config=config,
        encoder=encoder,
        prediction_service=prediction_service,
        aggregator=aggregator,
        simulation_hook=simulation_hook,
    )
    write_export_bundle(result, extra_provenance=extra_provenance)
    return result
