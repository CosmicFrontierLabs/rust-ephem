"""Fixtures for airmass_astropy tests."""

from datetime import datetime, timedelta

import pytest


@pytest.fixture
def test_times() -> list[datetime]:
    """Generate a sequence of test times."""
    start = datetime(2025, 1, 15, 0, 0, 0)
    return [start + timedelta(hours=i) for i in range(6)]
