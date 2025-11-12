from __future__ import annotations

import pytest

from PeetsFEA.peetspareto.pcbpcb import config, schemas


@pytest.fixture(scope="session")
def design_defaults() -> schemas.DesignVector:
    """Shared, validated design vector loaded from the repo defaults."""
    return config.load_design_defaults()
