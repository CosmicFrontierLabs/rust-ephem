#!/usr/bin/env python3
"""Integration tests for SkyCoord output functionality with single-assert tests."""

import sys
import time
from datetime import datetime, timezone

import numpy as np
import pytest

try:
    import rust_ephem  # type: ignore[import-untyped]

    RUST_EPHEM_AVAILABLE = True
except ImportError:
    RUST_EPHEM_AVAILABLE = False

try:
    import astropy.units as u  # type: ignore[import-untyped]
    from astropy.coordinates import (  # type: ignore[import-untyped]
        GCRS,
        CartesianDifferential,
        CartesianRepresentation,
        SkyCoord,
    )
    from astropy.time import Time  # type: ignore[import-untyped]

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Test TLE for NORAD ID 28485 (2004-047A)
TLE1 = "1 28485U 04047A   25287.56748435  .00035474  00000+0  70906-3 0  9995"
TLE2 = "2 28485  20.5535 247.0048 0005179 187.1586 172.8782 15.44937919148530"

# Test times
BEGIN = datetime(2025, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2025, 10, 14, 1, 40, 0, tzinfo=timezone.utc)
STEP_SIZE = 60  # 1 minute = 101 points

# Position/velocity tolerance (in km and km/s)
POSITION_TOLERANCE = 1e-6  # Very tight tolerance since we're comparing our own data
VELOCITY_TOLERANCE = 1e-6


class TestObsGeo:
    """Single-assert tests for obsgeoloc/obsgeovel."""

    @pytest.fixture
    def ephem(self):
        return rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, BEGIN, 1)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_has_obsgeoloc_attr(self, ephem):
        assert hasattr(ephem, "obsgeoloc")

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_has_obsgeovel_attr(self, ephem):
        assert hasattr(ephem, "obsgeovel")

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_obsgeoloc_shape_matches_gcrs(self, ephem):
        assert ephem.obsgeoloc.shape == ephem.gcrs_pv.position.shape

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_obsgeovel_shape_matches_gcrs(self, ephem):
        assert ephem.obsgeovel.shape == ephem.gcrs_pv.velocity.shape

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_obsgeoloc_equals_gcrs_position(self, ephem):
        assert np.allclose(ephem.obsgeoloc, ephem.gcrs_pv.position)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_obsgeovel_equals_gcrs_velocity(self, ephem):
        assert np.allclose(ephem.obsgeovel, ephem.gcrs_pv.velocity)


class TestGCRSSkyCoord:
    """Single-assert tests for gcrs basic behavior."""

    @pytest.fixture
    def ephem(self):
        return rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)

    @pytest.fixture
    def skycoord(self, ephem):
        return ephem.gcrs

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_type_is_skycoord(self, skycoord):
        assert isinstance(skycoord, SkyCoord)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_length_matches_timestamps(self, ephem, skycoord):
        assert len(skycoord) == len(ephem.timestamp)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_frame_is_gcrs(self, skycoord):
        assert isinstance(skycoord.frame, GCRS)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_has_velocity(self, skycoord):
        assert skycoord.velocity is not None


class TestGCRSSkyCoordAccuracy:
    """Per-index, single-assert accuracy checks for gcrs."""

    @pytest.fixture
    def ephem(self):
        return rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)

    @pytest.fixture
    def skycoord(self, ephem):
        return ephem.gcrs

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    @pytest.mark.parametrize(
        "i", range(int((END - BEGIN).total_seconds() // STEP_SIZE) + 1)
    )
    def test_position_matches_gcrs(self, ephem, skycoord, i):
        expected_pos = ephem.gcrs_pv.position[i]
        actual_pos = skycoord[i].cartesian.xyz.to(u.km).value
        assert np.allclose(expected_pos, actual_pos, rtol=POSITION_TOLERANCE)

    @pytest.mark.skipif(
        not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available"
    )
    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    @pytest.mark.parametrize(
        "i", range(int((END - BEGIN).total_seconds() // STEP_SIZE) + 1)
    )
    def test_velocity_matches_gcrs(self, ephem, skycoord, i):
        expected_vel = ephem.gcrs_pv.velocity[i]
        actual_vel = skycoord[i].velocity.d_xyz.to(u.km / u.s).value
        assert np.allclose(expected_vel, actual_vel, rtol=VELOCITY_TOLERANCE)


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_performance():
    """Test that gcrs_to_skycoord() is significantly faster than manual loops."""
    print("\n" + "=" * 80)
    print("Test 4: Performance Comparison")
    print("=" * 80)

    ephem = rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)

    # Method 1: Manual loop (old way)
    print("Running manual loop method...")
    start = time.time()
    skycoords_manual = []
    for i in range(len(ephem.timestamp)):
        pos = ephem.gcrs_pv.position[i] * u.km
        vel = ephem.gcrs_pv.velocity[i] * u.km / u.s
        t = Time(ephem.timestamp[i], scale="utc")

        cart_diff = CartesianDifferential(d_x=vel[0], d_y=vel[1], d_z=vel[2])
        cart_rep = CartesianRepresentation(
            x=pos[0], y=pos[1], z=pos[2], differentials=cart_diff
        )

        skycoords_manual.append(SkyCoord(cart_rep, frame=GCRS(obstime=t)))
    manual_time = time.time() - start

    # Method 2: gcrs property (new way)
    print("Running gcrs property...")
    start = time.time()
    _ = ephem.gcrs
    vectorized_time = time.time() - start

    # Calculate speedup
    # Guard against extremely small (sub-timer resolution) vectorized_time causing divide-by-zero
    speedup = manual_time / max(vectorized_time, 1e-9)
    time_saved = manual_time - vectorized_time

    print("\n✓ Performance test completed")
    print(
        f"  Manual loop: {manual_time:.3f}s ({manual_time / len(ephem.timestamp) * 1000:.2f}ms per point)"
    )
    print(
        f"  gcrs property: {vectorized_time:.3f}s ({vectorized_time / len(ephem.timestamp) * 1000:.2f}ms per point)"
    )
    print(f"  Speedup: {speedup:.1f}x faster")
    print(f"  Time saved: {time_saved:.3f}s for {len(ephem.timestamp)} points")

    # Assert significant speedup (at least 10x faster)
    assert speedup > 10, f"Expected >10x speedup, got {speedup:.1f}x"


def main():  # pragma: no cover
    return pytest.main([__file__, "-v"])


if __name__ == "__main__":
    sys.exit(main())
