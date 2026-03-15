"""Fixtures for binary_search tests."""

from datetime import datetime

import pytest

from rust_ephem import TLEEphemeris

_TLE1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
_TLE2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"


@pytest.fixture(scope="module")
def eph_small_range() -> TLEEphemeris:
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0)
    return TLEEphemeris(_TLE1, _TLE2, begin, end, step_size=60)


@pytest.fixture(scope="module")
def eph_large_range() -> TLEEphemeris:
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 2, 0, 0, 0)
    return TLEEphemeris(_TLE1, _TLE2, begin, end, step_size=10)
