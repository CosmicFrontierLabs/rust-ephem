"""Pytest configuration for utils tests."""

from datetime import datetime, timezone
from typing import List, Tuple

import pytest

import rust_ephem


@pytest.fixture(scope="module")
def ensure_planetary_data() -> None:
    """Ensure planetary ephemeris is loaded once for all tests"""
    import os

    test_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "test_data", "de440s.bsp"
    )

    # If file exists locally, use it without downloading
    if os.path.exists(test_data_path):
        rust_ephem.ensure_planetary_ephemeris(
            py_path=test_data_path, download_if_missing=False
        )
    else:
        # File doesn't exist, allow download (will happen once per machine)
        rust_ephem.ensure_planetary_ephemeris(
            py_path=test_data_path, download_if_missing=True
        )


@pytest.fixture
def tle_ephemeris(ensure_planetary_data: None) -> rust_ephem.TLEEphemeris:
    """Create a TLEEphemeris instance for testing"""
    tle1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
    tle2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"
    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size=600)


@pytest.fixture
def ground_ephemeris(ensure_planetary_data: None) -> rust_ephem.GroundEphemeris:
    """Create a GroundEphemeris instance for testing (Mauna Kea Observatory)"""
    latitude = 19.8207  # degrees
    longitude = -155.4681  # degrees
    height = 4207  # meters
    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.GroundEphemeris(
        latitude, longitude, height, begin, end, step_size=600
    )


@pytest.fixture
def spice_ephemeris(ensure_planetary_data: None) -> rust_ephem.SPICEEphemeris:
    """Create a SPICEEphemeris instance for testing"""
    import os

    # Use the same test data path as ensure_planetary_data
    test_data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "test_data", "de440s.bsp"
    )

    # If file doesn't exist in test_data, try cache directory as fallback
    if not os.path.exists(test_data_path):
        test_data_path = os.path.expanduser("~/.cache/rust_ephem/de440s.bsp")

    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    # Use Moon (NAIF ID 301) as the observer
    return rust_ephem.SPICEEphemeris(test_data_path, 301, begin, end, step_size=600)


@pytest.fixture
def test_times() -> List[datetime]:
    """Generate a sequence of test times."""
    from datetime import timedelta

    start = datetime(2025, 1, 15, 0, 0, 0)
    return [start + timedelta(hours=i) for i in range(6)]


@pytest.fixture
def tle_lines() -> Tuple[str, str]:
    """Provide TLE data for testing"""
    return (
        "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995",
        "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530",
    )


@pytest.fixture(scope="module")
def eph_small_range() -> rust_ephem.TLEEphemeris:
    from rust_ephem import TLEEphemeris

    TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0)
    return TLEEphemeris(TLE1, TLE2, begin, end, step_size=60)


@pytest.fixture(scope="module")
def eph_large_range() -> rust_ephem.TLEEphemeris:
    from rust_ephem import TLEEphemeris

    TLE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    begin = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 2, 0, 0, 0)
    return TLEEphemeris(TLE1, TLE2, begin, end, step_size=10)
