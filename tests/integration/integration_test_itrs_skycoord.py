#!/usr/bin/env python3
"""
Integration test for itrs property.

This test verifies that:
1. itrs property exists and works
2. The returned SkyCoord objects have ITRS frame
3. The positions and velocities match expected values from ITRS data

Requirements:
    pip install astropy numpy

Build and install the Rust module first:
    maturin build --release
    pip install target/wheels/*.whl
"""

from datetime import datetime, timezone

import numpy as np
import pytest

try:
    import rust_ephem  # type: ignore[import-untyped]

    RUST_EPHEM_AVAILABLE = True
except ImportError:
    RUST_EPHEM_AVAILABLE = False

try:
    from astropy.coordinates import ITRS, SkyCoord  # type: ignore[import-untyped]

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

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
def ephem():
    return rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, END, STEP_SIZE)


@pytest.fixture()
def ephem_single_point():
    return rust_ephem.TLEEphemeris(TLE1, TLE2, BEGIN, BEGIN, 1)


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_itrs_to_skycoord_is_skycoord(ephem):
    """Test that itrs property returns valid SkyCoord with ITRS frame."""

    # Get ITRS SkyCoord
    itrs_skycoord = ephem.itrs

    # Check type
    assert isinstance(itrs_skycoord, SkyCoord), (
        f"Expected SkyCoord, got {type(itrs_skycoord)}"
    )


def test_itrs_to_skycoord_check_len(ephem):
    """Test that itrs property returns valid SkyCoord with ITRS frame."""

    # Get ITRS SkyCoord
    itrs_skycoord = ephem.itrs

    # Check length
    expected_len = len(ephem.timestamp)
    assert len(itrs_skycoord) == expected_len, (
        f"Expected {expected_len} coords, got {len(itrs_skycoord)}"
    )


def test_itrs_to_skycoord_check_frame(ephem):
    """Test that itrs property returns valid SkyCoord with ITRS frame."""

    # Get ITRS SkyCoord
    itrs_skycoord = ephem.itrs

    # Check frame
    assert isinstance(itrs_skycoord.frame, ITRS), (
        f"Expected ITRS frame, got {type(itrs_skycoord.frame)}"
    )


def test_itrs_to_skycoord_has_velocity(ephem):
    """Test that itrs property returns valid SkyCoord with ITRS frame."""

    # Get ITRS SkyCoord
    itrs_skycoord = ephem.itrs

    # Check has velocity
    assert itrs_skycoord.velocity is not None, "SkyCoord missing velocity"


def test_itrs_to_skycoord_check_positions(ephem):
    """Test that itrs property returns valid SkyCoord with ITRS frame."""

    # Get ITRS SkyCoord
    itrs_skycoord = ephem.itrs

    # Verify ITRS positions match for all time points
    for i in range(len(itrs_skycoord)):
        expected_pos = ephem.itrs_pv.position[i]
        actual_pos = itrs_skycoord[i].cartesian.xyz.to("km").value
        assert np.allclose(expected_pos, actual_pos, rtol=TOLERANCE), (
            f"ITRS position mismatch at index {i}"
        )


@pytest.mark.parametrize(
    "index", range(31)
)  # 31 time points (0-30 minutes, 1 min step)
def test_itrs_to_skycoord_check_velocities(ephem, index):
    """Test that itrs property returns valid SkyCoord with ITRS frame."""

    # Get ITRS SkyCoord
    itrs_skycoord = ephem.itrs

    # Verify ITRS velocity matches at this time point
    expected_vel = ephem.itrs_pv.velocity[index]
    actual_vel = itrs_skycoord[index].velocity.d_xyz.to("km/s").value
    assert np.allclose(expected_vel, actual_vel, rtol=TOLERANCE), (
        f"ITRS velocity mismatch at index {index}"
    )


@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
@pytest.mark.skipif(not RUST_EPHEM_AVAILABLE, reason="rust_ephem module not available")
@pytest.mark.skipif(not ASTROPY_AVAILABLE, reason="astropy not available")
def test_itrs_frame_conversion(ephem_single_point):
    """Test that ITRS SkyCoord can be converted to other frames."""
    # Get ITRS SkyCoord
    itrs_skycoord = ephem_single_point.itrs

    # Test conversion to GCRS

    gcrs_from_itrs = itrs_skycoord.transform_to("gcrs")

    # Compare with directly computed GCRS
    gcrs_direct = ephem_single_point.gcrs
    gcrs_pos_direct = gcrs_direct[0].cartesian.xyz.to("km").value
    gcrs_pos_from_itrs = gcrs_from_itrs[0].cartesian.xyz.to("km").value

    # Should match within reasonable tolerance (frame transformations may have small differences)
    assert np.allclose(gcrs_pos_direct, gcrs_pos_from_itrs, rtol=1e-6, atol=0.2)
