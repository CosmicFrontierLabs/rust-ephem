"""Fixtures for integration_test_gcrs tests."""

from datetime import datetime, timezone
from typing import Any

import pytest

# Try to import required modules
try:
    import astropy.units as u  # type: ignore[import-untyped]
    from astropy.coordinates import (  # type: ignore[import-untyped]
        GCRS,
        TEME,
        CartesianDifferential,
        CartesianRepresentation,
    )
    from astropy.time import Time  # type: ignore[import-untyped]

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

import rust_ephem

# Test TLE for NORAD ID 28485 (2004-047A)
# This is a LEO satellite with inclination ~20.5 degrees, useful for testing
# coordinate transformations across different orbital geometries
TLE1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
TLE2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

# Test times spanning 24 hours
TEST_TIMES: list[datetime] = [
    datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 10, 14, 6, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 10, 14, 12, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 10, 14, 18, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 10, 15, 0, 0, 0, tzinfo=timezone.utc),
]

# Tolerance for position accuracy (in km)
# Without equation-of-equinoxes correction this case is ~0.09 km off, so keep
# the threshold tight enough to catch that regression.
POSITION_TOLERANCE_KM = 0.01

# Step size for single-point tests (arbitrary since begin==end)
SINGLE_POINT_STEP_SIZE = 1


@pytest.fixture(params=TEST_TIMES)
def single_time(request: Any) -> Any:  # noqa: D401 - simple fixture
    """Provide a single test time from TEST_TIMES via parametrization."""
    return request.param


@pytest.fixture
def tle_ephemeris(single_time: Any) -> Any:
    """Create a TLE ephemeris for the given time."""
    return rust_ephem.TLEEphemeris(
        TLE1, TLE2, single_time, single_time, SINGLE_POINT_STEP_SIZE
    )


@pytest.fixture
def teme_coordinates(tle_ephemeris: Any, single_time: Any) -> Any:
    """Create TEME coordinates from the ephemeris data."""
    teme_pos = tle_ephemeris.teme_pv.position[0]
    teme_vel = tle_ephemeris.teme_pv.velocity[0]

    t: Time = Time(single_time.isoformat().replace("+00:00", "Z"), scale="utc")
    return TEME(
        CartesianRepresentation(
            x=teme_pos[0] * u.km,
            y=teme_pos[1] * u.km,
            z=teme_pos[2] * u.km,
            differentials=CartesianDifferential(
                d_x=teme_vel[0] * u.km / u.s,
                d_y=teme_vel[1] * u.km / u.s,
                d_z=teme_vel[2] * u.km / u.s,
            ),
        ),
        obstime=t,
    )


@pytest.fixture
def gcrs_coordinates(teme_coordinates: Any, single_time: Any) -> Any:
    """Transform TEME coordinates to GCRS."""
    t: Time = Time(single_time.isoformat().replace("+00:00", "Z"), scale="utc")
    return teme_coordinates.transform_to(GCRS(obstime=t))
