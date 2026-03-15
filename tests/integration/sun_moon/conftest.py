"""Fixtures for integration_test_sun_moon tests."""

from datetime import datetime, timezone
from typing import Any

import pytest

import rust_ephem

# Test TLE for NORAD ID 28485 (2004-047A)
TLE1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
TLE2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"


@pytest.fixture
def single_timestamp_ephem() -> Any:
    """Create a single-timestamp TLE ephemeris."""
    test_time = datetime(2025, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.TLEEphemeris(TLE1, TLE2, test_time, test_time, 1)


@pytest.fixture
def multi_timestamp_ephem() -> Any:
    """Create a multi-timestamp TLE ephemeris."""
    begin = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.TLEEphemeris(TLE1, TLE2, begin, end, 3600)
