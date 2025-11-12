from __future__ import annotations

import pytest

from PeetsFEA.peetspareto.pcbpcb import config, schemas


def test_load_design_defaults(design_defaults: schemas.DesignVector) -> None:
    # Fixture already exercises load_design_defaults via the real TOML file.
    assert design_defaults.freq >= schemas.OPERATING_LIMITS["freq"][0]


def test_merge_overrides_respects_schema(design_defaults: schemas.DesignVector) -> None:
    updated = config.merge_overrides(design_defaults, {"freq": 145.0})
    assert updated.freq == 145.0

    with pytest.raises(config.ConfigError):
        config.merge_overrides(design_defaults, {"unknown": 123})

    with pytest.raises(schemas.SchemaValidationError):
        config.merge_overrides(design_defaults, {"freq": 50.0})
