#!/usr/bin/env python3
"""Integration tests for TEME -> GCRS transformation accuracy.

Refactored so that:
    - All tests reside in a test class
    - Each test function performs exactly ONE assertion
    - Per–timestamp accuracy is validated via parametrization

The original single test that looped over times with multiple asserts has been
split for clearer failure reporting and to satisfy the single-assert rule.
"""

import sys
from datetime import datetime, timezone

import numpy as np
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

try:
    import rust_ephem  # type: ignore[import-untyped]

    RUST_EPHEM_AVAILABLE = True
except ImportError:
    RUST_EPHEM_AVAILABLE = False

# Test TLE for NORAD ID 28485 (2004-047A)
# This is a LEO satellite with inclination ~20.5 degrees, useful for testing
# coordinate transformations across different orbital geometries
TLE1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
TLE2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

# Test times spanning 24 hours
TEST_TIMES = [
    datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 10, 14, 6, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 10, 14, 12, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 10, 14, 18, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 10, 15, 0, 0, 0, tzinfo=timezone.utc),
]

# Tolerance for position accuracy (in km)
# Set to 0.2 km to catch regressions; typical accuracy is ~0.1 km
POSITION_TOLERANCE_KM = 0.2

# Step size for single-point tests (arbitrary since begin==end)
SINGLE_POINT_STEP_SIZE = 1


@pytest.fixture(params=TEST_TIMES)
def single_time(request):  # noqa: D401 - simple fixture
    """Provide a single test time from TEST_TIMES via parametrization."""
    return request.param


class TestGCRSTransformation:
    """Per‑timestamp TEME→GCRS accuracy tests (single assertion each)."""

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_position_error_within_tolerance(self, single_time):
        """GCRS position error for a single timestamp is below tolerance."""
        ephem = rust_ephem.TLEEphemeris(
            TLE1, TLE2, single_time, single_time, SINGLE_POINT_STEP_SIZE
        )
        teme_pos = ephem.teme_pv.position[0]
        teme_vel = ephem.teme_pv.velocity[0]
        rust_gcrs_pos = ephem.gcrs_pv.position[0]

        t = Time(single_time.isoformat().replace("+00:00", "Z"), scale="utc")
        teme_coord = TEME(
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
        gcrs_coord = teme_coord.transform_to(GCRS(obstime=t))
        astropy_gcrs_pos = np.array(
            [
                gcrs_coord.cartesian.x.to(u.km).value,
                gcrs_coord.cartesian.y.to(u.km).value,
                gcrs_coord.cartesian.z.to(u.km).value,
            ]
        )
        pos_error_km = np.linalg.norm(rust_gcrs_pos - astropy_gcrs_pos)
        assert pos_error_km < POSITION_TOLERANCE_KM


if __name__ == "__main__":  # pragma: no cover - optional direct execution
    if not (RUST_EPHEM_AVAILABLE and ASTROPY_AVAILABLE):
        missing = []
        if not RUST_EPHEM_AVAILABLE:
            missing.append("rust_ephem")
        if not ASTROPY_AVAILABLE:
            missing.append("astropy")
        print("Missing dependencies: ", ", ".join(missing))
        sys.exit(1)
    # Run via pytest to leverage parametrization
    raise SystemExit(pytest.main([__file__, "-v"]))
