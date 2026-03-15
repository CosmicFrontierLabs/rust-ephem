"""Fixtures for angular_radius tests."""

import os
from datetime import datetime, timezone
from typing import Any

import pytest

import rust_ephem

# Constants for angular radius calculations
SUN_RADIUS_KM = 695700.0  # IAU 2015 Resolution B3 nominal solar radius
MOON_RADIUS_KM = 1737.4


@pytest.fixture(scope="module")
def ensure_planetary_data() -> None:
    """Ensure planetary ephemeris is loaded once for all tests"""
    test_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "test_data",
        "de440s.bsp",
    )
    if os.path.exists(test_data_path):
        rust_ephem.ensure_planetary_ephemeris(
            py_path=test_data_path, download_if_missing=False
        )
    else:
        rust_ephem.ensure_planetary_ephemeris(
            py_path=test_data_path, download_if_missing=True
        )


@pytest.fixture
def tle_ephemeris(ensure_planetary_data: Any) -> rust_ephem.TLEEphemeris:
    """Create a TLEEphemeris instance for testing"""
    tle1 = "1 25544U 98067A   25315.25818480  .00012468  00000-0  22984-3 0  9991"
    tle2 = "2 25544  51.6338 298.3179 0004133  57.8977 302.2413 15.49525392537972"
    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.TLEEphemeris(tle1, tle2, begin, end, step_size=600)


@pytest.fixture
def ground_ephemeris(ensure_planetary_data: Any) -> rust_ephem.GroundEphemeris:
    """Create a GroundEphemeris instance for testing (Mauna Kea Observatory)"""
    latitude = 19.8207
    longitude = -155.4681
    height = 4207
    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.GroundEphemeris(
        latitude, longitude, height, begin, end, step_size=600
    )


@pytest.fixture
def spice_ephemeris(ensure_planetary_data: Any) -> rust_ephem.SPICEEphemeris:
    """Create a SPICEEphemeris instance for testing"""
    test_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "test_data",
        "de440s.bsp",
    )
    if not os.path.exists(test_data_path):
        test_data_path = os.path.expanduser("~/.cache/rust_ephem/de440s.bsp")
    begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
    return rust_ephem.SPICEEphemeris(test_data_path, 301, begin, end, step_size=600)
