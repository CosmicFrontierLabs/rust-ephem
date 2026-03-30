"""Fixtures for test_boresight_offset_constraint tests."""

from datetime import datetime, timezone

import pytest

import rust_ephem


@pytest.fixture
def tle_ephem() -> rust_ephem.TLEEphemeris:
    """Create a TLE ephemeris for testing."""
    tle1 = "1 28485U 04047A   25317.24527149  .00068512  00000+0  12522-2 0  9999"
    tle2 = "2 28485  20.5556  25.5469 0004740 206.7882 153.2316 15.47667717153136"
    begin = datetime(2025, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 9, 23, 3, 0, 0, tzinfo=timezone.utc)
    step_s = 600
    return rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_s)
