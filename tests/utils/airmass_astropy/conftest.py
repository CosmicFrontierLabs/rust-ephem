"""Fixtures for airmass_astropy tests."""

from datetime import datetime, timedelta
from typing import List

import pytest


@pytest.fixture
def test_times() -> List[datetime]:
    """Generate a sequence of test times."""
    start = datetime(2025, 1, 15, 0, 0, 0)
    return [start + timedelta(hours=i) for i in range(6)]
