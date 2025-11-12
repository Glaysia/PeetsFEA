from __future__ import annotations

import pytest

from PeetsFEA.peetspareto.pcbpcb import schemas


@pytest.fixture(scope="session")
def design_defaults() -> schemas.DesignVector:
    """Shared, validated design vector matching the embedded notebook defaults."""
    return schemas.DEFAULT_DESIGN_VECTOR
