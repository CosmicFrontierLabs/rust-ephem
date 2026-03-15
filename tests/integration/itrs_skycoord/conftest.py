"""Fixtures for integration_test_itrs_skycoord tests."""

from datetime import datetime, timezone
from typing import Any

import pytest

import rust_ephem

# Test TLE for NORAD ID 28485 (2004-047A)
TLE1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
TLE2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

# Test times
BEGIN = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2025, 10, 14, 0, 30, 0, tzinfo=timezone.utc)
STEP_SIZE = 60  # 1 minute

# Position/velocity tolerance
TOLERANCE = 1e-9


@pytest.fixture
def ephem() -> Any:
    """Create a multi-point TLE ephemeris."""
    return rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)


@pytest.fixture
def ephem_single_point() -> Any:
    """Create a single-point TLE ephemeris."""
    return rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, BEGIN, 1)
