"""
Runtime glue that mirrors the legacy NSGA-II notebook behavior.

This module bundles:
- a `LegacyCandidateEncoder` that reproduces the integer search space
- default objective definitions + aggregator (total loss & volume)
- a zero-config `run_pcbpcb_nsga2` helper for quick notebook parity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from .api import generate_pareto_front
from .config import ParetoRunConfig
from .model import PCBPCBModel
from .optimizer import ParetoResult
from .protocols import (
    CandidateEncoder,
    ObjectiveAggregator,
    ObjectiveDefinition,
    PredictionService,
    TransformerSpec,
)
from .schemas import (
    DEFAULT_DESIGN_VECTOR,
    DESIGN_FIELD_ORDER,
    OPERATING_LIMITS,
    VARIABLE_SPECS,
    DesignVector,
    SchemaValidationError,
)

DEFAULT_OBJECTIVES: tuple[ObjectiveDefinition, ...] = (
    ObjectiveDefinition(
        name="total_loss",
        minimize=True,
        description="Total predicted copper + core losses (W)",
    ),
    ObjectiveDefinition(
        name="volume",
        minimize=True,
        description="Computed transformer volume (mm^3)",
    ),
)


def default_objectives() -> tuple[ObjectiveDefinition, ...]:
    """Return the canonical notebook objectives (total loss & volume)."""

    return DEFAULT_OBJECTIVES


def default_objective_aggregator() -> ObjectiveAggregator:
    """
    Aggregate LightGBM predictions into the two canonical objectives.
    """

    def _aggregate(
        spec: TransformerSpec,
        predictions: Mapping[str, float],
        measures: Mapping[str, float] | None = None,
        mutable_provenance: MutableMapping[str, float] | None = None,
    ) -> Mapping[str, float]:
        if not isinstance(spec, DesignVector):
            raise TypeError(
                "default_objective_aggregator expects DesignVector specs. "
                "Provide a custom aggregator if you need a different schema."
            )
        if "total_loss" not in predictions:
            raise KeyError(
                "Prediction payload missing 'total_loss'. Ensure the model service "
                "returns LightGBM totals before calling the optimizer."
            )
        volume = float(spec.geometry.volume)
        if mutable_provenance is not None and hasattr(mutable_provenance, "__setitem__"):
            mutable_provenance.setdefault("volume_mm3", volume)
        return {
            "total_loss": float(predictions["total_loss"]),
            "volume": volume,
        }

    return _aggregate


@dataclass(slots=True)
class LegacyCandidateEncoder(CandidateEncoder):
    """Encodes/decodes the 22-variable integer search space from the notebook."""

    dimension: int = len(DESIGN_FIELD_ORDER)
    lower_bounds: tuple[float, ...] = tuple(
        float(VARIABLE_SPECS[name].raw_min) for name in DESIGN_FIELD_ORDER
    )
    upper_bounds: tuple[float, ...] = tuple(
        float(VARIABLE_SPECS[name].raw_max) for name in DESIGN_FIELD_ORDER
    )

    def decode(self, vector: Sequence[float]) -> DesignVector:
        if len(vector) != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} decision variables, received {len(vector)}"
            )
        snapped: list[int] = []
        for idx, raw_value in enumerate(vector):
            name = DESIGN_FIELD_ORDER[idx]
            spec = VARIABLE_SPECS[name]
            quantized = _snap_to_grid(raw_value, spec.raw_min, spec.raw_max, spec.raw_step)
            if name in OPERATING_LIMITS:
                lower, upper = OPERATING_LIMITS[name]
                lower_raw = _snap_to_grid(spec.encode(lower), spec.raw_min, spec.raw_max, spec.raw_step)
                upper_raw = _snap_to_grid(spec.encode(upper), spec.raw_min, spec.raw_max, spec.raw_step)
                quantized = int(min(max(quantized, lower_raw), upper_raw))
            snapped.append(quantized)

        try:
            return DesignVector.from_raw(snapped)
        except SchemaValidationError:
            return _repair_design_vector(snapped)

    def encode(self, spec: TransformerSpec) -> Sequence[float]:
        if not isinstance(spec, DesignVector):
            raise TypeError("LegacyCandidateEncoder only encodes DesignVector specs.")
        return spec.to_raw()


def run_pcbpcb_nsga2(
    *,
    config: ParetoRunConfig | None = None,
    prediction_service: PredictionService | None = None,
    encoder: CandidateEncoder | None = None,
    aggregator: ObjectiveAggregator | None = None,
    extra_provenance: Mapping[str, object] | None = None,
) -> ParetoResult:
    """
    Execute the PCB-to-PCB NSGA-II workflow using notebook-equivalent defaults.

    Args:
        config:
            Optional `ParetoRunConfig`. When omitted, a default config using the
            `default_objectives()` list, 64-population, and 25 generations is used.
        prediction_service:
            Override the LightGBM inference service. Defaults to `PCBPCBModel()`
            which auto-loads the legacy artifacts bundled in the repo.
        encoder:
            Optional override for the candidate encoder. Defaults to
            `LegacyCandidateEncoder`.
        aggregator:
            Optional objective aggregator; defaults to `default_objective_aggregator`.
        extra_provenance:
            Additional metadata to merge into the provenance JSON payload.
    """

    run_config = config or ParetoRunConfig(objectives=default_objectives())
    service = prediction_service or PCBPCBModel()
    candidate_encoder = encoder or LegacyCandidateEncoder()
    agg = aggregator or default_objective_aggregator()

    return generate_pareto_front(
        config=run_config,
        encoder=candidate_encoder,
        prediction_service=service,
        aggregator=agg,
        extra_provenance=extra_provenance,
    )


def _repair_design_vector(raw_values: Sequence[int]) -> DesignVector:
    """
    Attempt to coerce an infeasible decision vector back into the valid manifold.

    The repair heuristic walks the defaults and applies per-field updates,
    skipping any change that would violate derived constraints.
    """

    spec = DEFAULT_DESIGN_VECTOR
    for name, raw in zip(DESIGN_FIELD_ORDER, raw_values):
        spec_meta = VARIABLE_SPECS[name]
        target_value = spec_meta.decode(raw)
        try:
            spec = spec.with_updates(**{name: target_value})
        except SchemaValidationError:
            continue
    return spec


def _snap_to_grid(value: float, lower: int, upper: int, step: int) -> int:
    clamped = min(max(float(value), float(lower)), float(upper))
    if step <= 0:
        return int(round(clamped))
    relative = (clamped - lower) / step
    snapped = lower + round(relative) * step
    return int(min(max(snapped, lower), upper))
