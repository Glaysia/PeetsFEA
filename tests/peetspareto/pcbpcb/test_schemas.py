from __future__ import annotations

import pytest

from PeetsFEA.peetspareto.pcbpcb import schemas


def test_design_vector_round_trip(design_defaults: schemas.DesignVector) -> None:
    raw = design_defaults.to_raw()
    rebuilt = schemas.DesignVector.from_raw(raw)
    assert rebuilt.to_raw() == raw


def test_frequency_window_enforced(design_defaults: schemas.DesignVector) -> None:
    with pytest.raises(schemas.SchemaValidationError):
        design_defaults.with_updates(freq=90.0)


def test_window_length_constraint(design_defaults: schemas.DesignVector) -> None:
    with pytest.raises(schemas.SchemaValidationError):
        design_defaults.with_updates(l2=5.0)


def test_default_decision_vector_matches_embedded_defaults() -> None:
    decoded = schemas.decision_vector_to_design(schemas.DEFAULT_DECISION_VECTOR)
    assert decoded.to_raw() == schemas.DEFAULT_DECISION_VECTOR
