"""
NSGA-II orchestration for the PCB-on-PCB transformer problem.

This module stays agnostic to the concrete schema/model layers by relying
on the protocols declared under `pcbpcb.protocols`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .config import ParetoRunConfig
from .protocols import (
    CandidateEncoder,
    EvaluationRecord,
    ObjectiveAggregator,
    ObjectiveDefinition,
    PredictionService,
    SimulationHook,
    TransformerSpec,
)


@dataclass(slots=True)
class ParetoResult:
    """Container for everything produced by a Pareto optimization run."""

    config: ParetoRunConfig
    decision_vectors: np.ndarray
    front_records: Sequence[EvaluationRecord]
    all_records: Sequence[EvaluationRecord]
    runtime_seconds: float
    timestamp: datetime
    model_metadata: dict[str, str]
    algorithm_metadata: dict[str, str | int | float]

    def objective_matrix(self) -> np.ndarray:
        """Returns a (#front, #objectives) array ordered like config.objectives."""
        rows: list[list[float]] = []
        for record in self.front_records:
            rows.append([float(record.objectives[obj.name]) for obj in self.config.objectives])
        return np.asarray(rows, dtype=float)


class ParetoOptimizer:
    """
    Runs NSGA-II using a provided candidate encoder and prediction service.

    The optimizer focuses strictly on search orchestration; schema/model/simulation
    specifics live behind the injected protocols.
    """

    def __init__(
        self,
        encoder: CandidateEncoder,
        prediction_service: PredictionService,
        objectives: Sequence[ObjectiveDefinition],
        aggregator: ObjectiveAggregator | None = None,
        simulation_hook: SimulationHook | None = None,
    ) -> None:
        if not objectives:
            raise ValueError("objectives must not be empty")
        self.encoder = encoder
        self.prediction_service = prediction_service
        self.objectives = list(objectives)
        self.objective_names = [obj.name for obj in self.objectives]
        self.aggregator = aggregator or self._build_default_aggregator()
        self.simulation_hook = simulation_hook
        self._evaluation_log: list[EvaluationRecord] = []

    def run(self, config: ParetoRunConfig) -> ParetoResult:
        """Executes NSGA-II and returns the resulting Pareto data."""

        if [obj.name for obj in config.objectives] != self.objective_names:
            raise ValueError("Config objectives must match optimizer objectives")

        problem = _NSGAProblem(
            encoder=self.encoder,
            objective_count=len(self.objectives),
            evaluate_vectors=self._evaluate_vectors,
        )
        algorithm = NSGA2(
            pop_size=config.loop.population_size,
            n_offsprings=config.loop.n_offsprings,
            eliminate_duplicates=config.loop.eliminate_duplicates,
        )
        termination = get_termination("n_gen", config.loop.n_generations)
        start = perf_counter()
        result = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=termination,
            seed=config.loop.seed,
            verbose=False,
        )
        runtime_seconds = perf_counter() - start

        decision_vectors = np.asarray(result.X if result.X is not None else [], dtype=float)
        front_records: list[EvaluationRecord] = []
        if decision_vectors.size:
            _, front_records = self._evaluate_vectors(decision_vectors)

        timestamp = datetime.now(tz=UTC)
        algorithm_metadata = {
            "algorithm": "NSGA-II",
            "population_size": config.loop.population_size,
            "n_generations": config.loop.n_generations,
            "seed": config.loop.seed,
        }
        model_metadata = {
            "model_version": getattr(self.prediction_service, "model_version", "unknown"),
            "artifact_checksum": getattr(self.prediction_service, "artifact_checksum", "unknown"),
        }
        return ParetoResult(
            config=config,
            decision_vectors=decision_vectors,
            front_records=front_records,
            all_records=tuple(self._evaluation_log),
            runtime_seconds=runtime_seconds,
            timestamp=timestamp,
            model_metadata=model_metadata,
            algorithm_metadata=algorithm_metadata,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _evaluate_vectors(
        self, decision_vectors: Sequence[Sequence[float]]
    ) -> tuple[np.ndarray, list[EvaluationRecord]]:
        """Evaluates a batch of decision vectors and returns pymoo-ready F matrix."""

        specs: list[TransformerSpec] = [
            self.encoder.decode(tuple(float(v) for v in vector))
            for vector in decision_vectors
        ]
        predictions = self.prediction_service.predict_batch(specs)
        if len(predictions) != len(specs):
            raise ValueError("predict_batch returned mismatched length")

        records: list[EvaluationRecord] = []
        rows: list[list[float]] = []
        for spec, prediction in zip(specs, predictions):
            measures = self.simulation_hook.evaluate(spec) if self.simulation_hook else {}
            provenance: dict[str, str] = {
                "model_version": getattr(self.prediction_service, "model_version", "unknown"),
                "artifact_checksum": getattr(self.prediction_service, "artifact_checksum", "unknown"),
            }
            objectives = dict(
                self.aggregator(spec, prediction, measures or None, provenance)
            )
            self._ensure_objectives_complete(objectives)
            record = EvaluationRecord(
                spec=spec,
                objectives=objectives,
                measures={**prediction, **measures},
                provenance=provenance,
            )
            records.append(record)
            rows.append(self._objective_row(objectives))

        self._evaluation_log.extend(records)
        return np.asarray(rows, dtype=float), records

    def _objective_row(self, objectives: dict[str, float]) -> list[float]:
        row: list[float] = []
        for definition in self.objectives:
            value = float(objectives[definition.name])
            weighted = value * float(definition.weight)
            row.append(weighted if definition.minimize else -weighted)
        return row

    def _ensure_objectives_complete(self, objectives: dict[str, float]) -> None:
        missing = [name for name in self.objective_names if name not in objectives]
        if missing:
            raise KeyError(f"Objective aggregator missing keys: {missing}")

    def _build_default_aggregator(self) -> ObjectiveAggregator:
        objective_names = tuple(self.objective_names)

        def _aggregator(spec: TransformerSpec, predictions, measures=None, mutable_provenance=None):
            del spec, mutable_provenance  # unused, but part of the contract
            container = measures or predictions
            return {name: float(container[name]) for name in objective_names}

        return _aggregator


class _NSGAProblem(Problem):
    """Thin pymoo Problem wrapper that calls back into ParetoOptimizer."""

    def __init__(
        self,
        encoder: CandidateEncoder,
        objective_count: int,
        evaluate_vectors: Callable[[Sequence[Sequence[float]]], tuple[np.ndarray, list[EvaluationRecord]]],
    ) -> None:
        super().__init__(
            n_var=encoder.dimension,
            n_obj=objective_count,
            xl=np.asarray(encoder.lower_bounds, dtype=float),
            xu=np.asarray(encoder.upper_bounds, dtype=float),
        )
        self._evaluate_vectors = evaluate_vectors

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        matrix, _records = self._evaluate_vectors(np.atleast_2d(X))
        out["F"] = matrix
