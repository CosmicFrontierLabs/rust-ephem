"""Fixtures for saa_constraint tests."""

import pytest


@pytest.fixture
def saa_polygon() -> list[tuple[float, float]]:
    """Simple rectangular SAA polygon for testing."""
    return [
        (-90.0, -50.0),  # Southwest
        (-40.0, -50.0),  # Southeast
        (-40.0, 0.0),  # Northeast
        (-90.0, 0.0),  # Northwest
    ]
