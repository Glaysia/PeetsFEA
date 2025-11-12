from __future__ import annotations

import pytest

from PeetsFEA.peetspareto.pcbpcb import config, schemas


def test_load_design_defaults() -> None:
    loaded = config.load_design_defaults()
    assert loaded == schemas.DEFAULT_DESIGN_VECTOR


def test_merge_overrides_respects_schema(design_defaults: schemas.DesignVector) -> None:
    updated = config.merge_overrides(design_defaults, {"freq": 145.0})
    assert updated.freq == 145.0

    with pytest.raises(config.ConfigError):
        config.merge_overrides(design_defaults, {"unknown": 123})

    with pytest.raises(schemas.SchemaValidationError):
        config.merge_overrides(design_defaults, {"freq": 50.0})


def test_decision_vector_helper_round_trip() -> None:
    decoded = config.decision_vector_to_design(schemas.DEFAULT_DECISION_VECTOR)
    assert decoded.to_raw() == schemas.DEFAULT_DECISION_VECTOR
