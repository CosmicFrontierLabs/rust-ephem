"""Fixtures for index_method tests."""

from datetime import datetime

import pytest

from rust_ephem import GroundEphemeris, TLEEphemeris


@pytest.fixture
def tle_ephemeris() -> TLEEphemeris:
    """Create a TLE ephemeris for testing"""
    tle1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
    tle2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0)
    return TLEEphemeris(tle1, tle2, begin, end, step_size=60)


@pytest.fixture
def ground_ephemeris() -> GroundEphemeris:
    """Create a ground ephemeris for testing"""
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0)
    return GroundEphemeris(30.2672, -97.7431, 150.0, begin, end, step_size=60)
