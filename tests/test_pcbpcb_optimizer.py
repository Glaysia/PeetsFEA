from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from PeetsFEA.peetspareto.pcbpcb import (
    ExportConfig,
    OptimizationLoopConfig,
    ParetoRunConfig,
    generate_pareto_front,
    records_to_dataframe,
)
from PeetsFEA.peetspareto.pcbpcb.protocols import ObjectiveAggregator, ObjectiveDefinition, TransformerSpec


@dataclass(frozen=True)
class _Spec:
    identifier: str
    values: tuple[float, float]

    def to_feature_vector(self) -> Sequence[float]:
        return self.values

    def to_parameters(self) -> Mapping[str, float]:
        return {"x": self.values[0], "y": self.values[1]}


class _Encoder:
    dimension = 2
    lower_bounds = (-1.0, -1.0)
    upper_bounds = (1.0, 1.0)

    def decode(self, vector: Sequence[float]) -> TransformerSpec:
        x, y = vector
        return _Spec(identifier=f"{x:.3f}:{y:.3f}", values=(float(x), float(y)))

    def encode(self, spec: TransformerSpec) -> Sequence[float]:
        assert isinstance(spec, _Spec)
        return spec.values


class _Model:
    model_version = "test-model"
    artifact_checksum = "checksum"

    def predict_batch(self, specs: Sequence[TransformerSpec]):
        outputs = []
        for spec in specs:
            assert isinstance(spec, _Spec)
            x, y = spec.values
            outputs.append(
                {
                    "loss": x**2 + y**2,
                    "gap": x - y,
                    "stress": abs(x) + abs(y),
                }
            )
        return outputs


def _aggregator() -> ObjectiveAggregator:
    def _inner(spec: TransformerSpec, predictions, measures=None, mutable_provenance=None):
        del spec, measures, mutable_provenance
        return {"loss": predictions["loss"], "gap": predictions["gap"]}

    return _inner


def test_generate_pareto_front_creates_artifacts(tmp_path: Path) -> None:
    objectives = [
        ObjectiveDefinition(name="loss", minimize=True),
        ObjectiveDefinition(name="gap", minimize=False),
    ]
    config = ParetoRunConfig(
        objectives=objectives,
        loop=OptimizationLoopConfig(population_size=6, n_generations=3, seed=2),
        export=ExportConfig(directory=tmp_path / "exports"),
    )
    result = generate_pareto_front(
        config=config,
        encoder=_Encoder(),
        prediction_service=_Model(),
        aggregator=_aggregator(),
    )

    assert result.front_records, "expected non-empty Pareto front"
    assert result.decision_vectors.shape[1] == 2
    assert result.objective_matrix().shape == (len(result.front_records), len(objectives))

    csv_path = config.export.csv_path()
    prov_path = config.export.provenance_path()
    assert csv_path.exists(), "expected Pareto CSV to be written"
    assert prov_path.exists(), "expected provenance JSON to be written"

    df = records_to_dataframe(result.front_records)
    assert {"obj__loss", "obj__gap"}.issubset(df.columns)
