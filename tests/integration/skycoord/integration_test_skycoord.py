#!/usr/bin/env python3
"""Integration tests for SkyCoord output functionality with single-assert tests."""

import sys
import time
from typing import Any

import numpy as np
import pytest
from astropy.coordinates.representation.cartesian import (  # type: ignore[import-untyped]
    CartesianDifferential,
    CartesianRepresentation,
)
from astropy.time.core import Time  # type: ignore[import-untyped]

import rust_ephem

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

from .conftest import (
    BEGIN,
    END,
    POSITION_TOLERANCE,
    STEP_SIZE,
    TLE1,
    TLE2,
    VELOCITY_TOLERANCE,
)


class TestObsGeo:
    """Single-assert tests for obsgeoloc/obsgeovel."""

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_has_obsgeoloc_attr(self: Any, single_point_ephem: Any) -> None:
        assert hasattr(single_point_ephem, "obsgeoloc")

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_has_obsgeovel_attr(self: Any, single_point_ephem: Any) -> None:
        assert hasattr(single_point_ephem, "obsgeovel")

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_obsgeoloc_shape_matches_gcrs(self: Any, single_point_ephem: Any) -> None:
        assert (
            single_point_ephem.obsgeoloc.shape
            == single_point_ephem.gcrs_pv.position.shape
        )

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_obsgeovel_shape_matches_gcrs(self: Any, single_point_ephem: Any) -> None:
        assert (
            single_point_ephem.obsgeovel.shape
            == single_point_ephem.gcrs_pv.velocity.shape
        )

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_obsgeoloc_equals_gcrs_position(self: Any, single_point_ephem: Any) -> None:
        assert np.allclose(
            single_point_ephem.obsgeoloc, single_point_ephem.gcrs_pv.position
        )

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_obsgeovel_equals_gcrs_velocity(self: Any, single_point_ephem: Any) -> None:
        assert np.allclose(
            single_point_ephem.obsgeovel, single_point_ephem.gcrs_pv.velocity
        )


class TestGCRSSkyCoord:
    """Single-assert tests for gcrs basic behavior."""

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_type_is_skycoord(self: Any, gcrs_skycoord: Any) -> None:
        assert isinstance(gcrs_skycoord, SkyCoord)

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_length_matches_timestamps(
        self: Any, multi_point_ephem: Any, gcrs_skycoord: Any
    ) -> None:
        assert len(gcrs_skycoord) == len(multi_point_ephem.timestamp)

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_frame_is_gcrs(self: Any, gcrs_skycoord: Any) -> None:
        assert isinstance(gcrs_skycoord.frame, GCRS)

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    def test_has_velocity(self: Any, gcrs_skycoord: Any) -> None:
        assert gcrs_skycoord.velocity is not None


class TestGCRSSkyCoordAccuracy:
    """Per-index, single-assert accuracy checks for gcrs."""

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    @pytest.mark.parametrize(
        "i", range(int((END - BEGIN).total_seconds() // STEP_SIZE) + 1)
    )
    def test_position_matches_gcrs(
        self: Any, multi_point_ephem: Any, gcrs_skycoord: Any, i: Any
    ) -> None:
        expected_pos = multi_point_ephem.gcrs_pv.position[i]
        actual_pos = gcrs_skycoord[i].cartesian.xyz.to(u.km).value
        assert np.allclose(expected_pos, actual_pos, rtol=POSITION_TOLERANCE)

    @pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
    @pytest.mark.parametrize(
        "i", range(int((END - BEGIN).total_seconds() // STEP_SIZE) + 1)
    )
    def test_velocity_matches_gcrs(
        self: Any, multi_point_ephem: Any, gcrs_skycoord: Any, i: Any
    ) -> None:
        expected_vel = multi_point_ephem.gcrs_pv.velocity[i]
        actual_vel = gcrs_skycoord[i].velocity.d_xyz.to(u.km / u.s).value
        assert np.allclose(expected_vel, actual_vel, rtol=VELOCITY_TOLERANCE)


@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_performance() -> None:
    """Test that gcrs_to_skycoord() is significantly faster than manual loops."""
    print("\n" + "=" * 80)
    print("Test 4: Performance Comparison")
    print("=" * 80)

    ephem = rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)

    # Method 1: Manual loop (old way)
    print("Running manual loop method...")
    start: float = time.time()
    skycoords_manual = []
    for i in range(len(ephem.timestamp)):
        pos = ephem.gcrs_pv.position[i] * u.km
        vel = ephem.gcrs_pv.velocity[i] * u.km / u.s
        t: Time = Time(ephem.timestamp[i], scale="utc")

        cart_diff: CartesianDifferential = CartesianDifferential(
            d_x=vel[0], d_y=vel[1], d_z=vel[2]
        )
        cart_rep: CartesianRepresentation = CartesianRepresentation(
            x=pos[0], y=pos[1], z=pos[2], differentials=cart_diff
        )

        skycoords_manual.append(SkyCoord(cart_rep, frame=GCRS(obstime=t)))
    manual_time: float = time.time() - start

    # Method 2: gcrs property (new way)
    print("Running gcrs property...")
    start2: float = time.time()
    _ = ephem.gcrs
    vectorized_time: float = time.time() - start2

    # Calculate speedup
    # Guard against extremely small (sub-timer resolution) vectorized_time causing divide-by-zero
    speedup: float = manual_time / max(vectorized_time, 1e-9)
    time_saved: float = manual_time - vectorized_time

    print("\n✓ Performance test completed")
    print(
        f"  Manual loop: {manual_time:.3f}s ({manual_time / len(ephem.timestamp) * 1000:.2f}ms per point)"
    )
    print(
        f"  gcrs property: {vectorized_time:.3f}s ({vectorized_time / len(ephem.timestamp) * 1000:.2f}ms per point)"
    )
    print(f"  Speedup: {speedup:.1f}x faster")
    print(f"  Time saved: {time_saved:.3f}s for {len(ephem.timestamp)} points")

    # Assert significant speedup (at least 3x faster)
    # Note: Expected speedup is 3-10x depending on parallelization settings:
    # - Sequential (RUST_EPHEM_PARALLEL=1 or unset): ~3-4x speedup
    # - Multi-threaded (RUST_EPHEM_PARALLEL>1): ~8-10x speedup
    # CI runs with sequential execution by default, so we use 3x as the threshold.
    assert speedup > 3, f"Expected >3x speedup, got {speedup:.1f}x"


def main() -> int | pytest.ExitCode:  # pragma: no cover
    return pytest.main([__file__, "-v"])


if __name__ == "__main__":
    sys.exit(main())
