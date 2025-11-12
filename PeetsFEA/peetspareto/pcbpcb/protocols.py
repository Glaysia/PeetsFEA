"""
Shared protocols that decouple Agent C's optimizer from
Agent A/B implementations.

These interfaces provide the minimum surface we need to wire
schema objects, model services, and downstream simulation hooks
without importing concrete classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Protocol, Sequence, runtime_checkable

ParameterValue = float | int | str | bool | None
ParameterDict = Mapping[str, ParameterValue]


@runtime_checkable
class TransformerSpec(Protocol):
    """
    Minimal surface that Agent A's schema objects should expose.

    Implementations should be immutable and validated; we only rely on
    two capabilities:
    - an identifier string for provenance
    - a total ordering of features for model consumption
    """

    @property
    def identifier(self) -> str: ...

    def to_feature_vector(self) -> Sequence[float]: ...

    def to_parameters(self) -> ParameterDict: ...


@runtime_checkable
class PredictionService(Protocol):
    """Agent B's LightGBM wrapper should satisfy this protocol."""

    model_version: str
    artifact_checksum: str

    def predict_batch(self, specs: Sequence[TransformerSpec]) -> Sequence[Mapping[str, float]]: ...


@runtime_checkable
class SimulationHook(Protocol):
    """
    Optional hook for Agent C to blend AEDT scores in the future.

    Implementations may be expensive; optimizers should call them
    sparingly (e.g., only along the Pareto front).
    """

    def evaluate(self, spec: TransformerSpec) -> Mapping[str, float]: ...


@runtime_checkable
class CandidateEncoder(Protocol):
    """
    Converts between bounded decision vectors and validated transformer specs.
    """

    dimension: int
    lower_bounds: Sequence[float]
    upper_bounds: Sequence[float]

    def decode(self, vector: Sequence[float]) -> TransformerSpec: ...

    def encode(self, spec: TransformerSpec) -> Sequence[float]: ...


@dataclass(slots=True, frozen=True)
class ObjectiveDefinition:
    """Describes a single scalar objective optimized by NSGA-II."""

    name: str
    minimize: bool = True
    weight: float = 1.0
    description: str | None = None


@dataclass(slots=True, frozen=True)
class EvaluationRecord:
    """
    Canonical representation of one evaluated candidate.

    `objectives` should include every objective defined in the optimizer config.
    `measures` may carry additional metrics (loss breakdown, temps, etc.).
    `provenance` is free-form metadata propagated to exporters.
    """

    spec: TransformerSpec
    objectives: Mapping[str, float]
    measures: Mapping[str, float]
    provenance: Mapping[str, Any]


class ObjectiveAggregator(Protocol):
    """
    Converts raw prediction/measured values into the objective scalars
    used by NSGA-II.
    """

    def __call__(
        self,
        spec: TransformerSpec,
        predictions: Mapping[str, float],
        measures: Mapping[str, float] | None = None,
        mutable_provenance: MutableMapping[str, Any] | None = None,
    ) -> Mapping[str, float]: ...
