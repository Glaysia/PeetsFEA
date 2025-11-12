from __future__ import annotations

import pytest

from PeetsFEA.peetspareto.pcbpcb.config import (
    ExportConfig,
    ModelArtifactSelectionError,
    OptimizationLoopConfig,
    ParetoRunConfig,
    default_legacy_model_root,
)
from PeetsFEA.peetspareto.pcbpcb.runtime import (
    LegacyCandidateEncoder,
    default_objective_aggregator,
    default_objectives,
    run_pcbpcb_nsga2,
)
from PeetsFEA.peetspareto.pcbpcb import schemas


def test_legacy_candidate_encoder_roundtrip(design_defaults: schemas.DesignVector) -> None:
    encoder = LegacyCandidateEncoder()
    encoded = tuple(encoder.encode(design_defaults))
    decoded = encoder.decode(encoded)

    assert decoded.to_raw() == encoded
    assert decoded.identifier.startswith("pcbpcb-")


def test_default_objective_aggregator_uses_geometry(design_defaults: schemas.DesignVector) -> None:
    aggregator = default_objective_aggregator()
    provenance: dict[str, float] = {}

    result = aggregator(design_defaults, {"total_loss": 12.5}, None, provenance)

    assert pytest.approx(result["total_loss"], rel=1e-9) == 12.5
    assert pytest.approx(result["volume"], rel=1e-9) == design_defaults.geometry.volume
    assert pytest.approx(provenance["volume_mm3"], rel=1e-9) == design_defaults.geometry.volume


def test_run_pcbpcb_nsga2_smoke(tmp_path) -> None:
    try:
        default_legacy_model_root()
    except ModelArtifactSelectionError as exc:
        pytest.skip(f"Legacy LightGBM artifacts unavailable: {exc}")

    export = ExportConfig(directory=tmp_path / "pareto")
    config = ParetoRunConfig(
        objectives=default_objectives(),
        loop=OptimizationLoopConfig(population_size=4, n_generations=2, seed=5),
        export=export,
    )

    result = run_pcbpcb_nsga2(config=config)

    assert result.front_records, "expected run to produce a Pareto front"
    assert export.csv_path().exists()
    assert export.provenance_path().exists()
